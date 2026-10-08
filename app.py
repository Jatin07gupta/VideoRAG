"""
VideoRAG: Production-Grade YouTube Transcript RAG System
Author: Jatin Gupta (https://github.com/Jatin07gupta)
"""

import streamlit as st
import os
from langchain_community.document_loaders import YoutubeLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_community.document_compressors.flashrank_rerank import FlashrankRerank
try:
    from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever
except ImportError:
    from langchain_classic.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# --- PAGE CONFIG ---
st.set_page_config(page_title="YouTube RAG Pro", page_icon="🎥")
st.title("🎥 YouTube RAG: Production-Grade Summarizer")

# --- SIDEBAR: CONFIGURATION ---
with st.sidebar:
    st.header("⚙️ Configuration")
    # API Key Input
    api_key = st.text_input("Google API Key", type="password")
    if api_key:
        os.environ["GOOGLE_API_KEY"] = api_key

    # Video URL Input
    video_url = st.text_input("Enter YouTube URL")
    process_btn = st.button("Analyze Video")

# --- SESSION STATE ---
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- 1. PROCESSING FUNCTION (Layers 1-6) ---
def process_video(url):
    with st.spinner("📥 Downloading & Processing Transcript..."):
        try:
            # Layer 1: Ingestion (Robust)
            try:
                loader = YoutubeLoader.from_youtube_url(url, add_video_info=False, language=["en", "en-US"])
                raw_docs = loader.load()
            except:
                # Fallback to translation
                loader = YoutubeLoader.from_youtube_url(url, add_video_info=False, translation="en")
                raw_docs = loader.load()

            st.success(f"✅ Loaded {len(raw_docs[0].page_content)} characters.")

            # Layer 2: Semantic Chunking
            embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
            text_splitter = SemanticChunker(embeddings, breakpoint_threshold_type="standard_deviation")
            docs = text_splitter.split_documents(raw_docs)

            # Fallback for Chunking
            if len(docs) < 2:
                st.warning("⚠️ Semantic split weak. Using fallback overlap strategy.")
                splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                docs = splitter.split_documents(raw_docs)

            st.info(f"✂️ Generated {len(docs)} chunks.")

            # Layer 3 & 4: Hybrid Search (Vector + Keyword)
            vectorstore = FAISS.from_documents(docs, embeddings)
            vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
            keyword_retriever = BM25Retriever.from_documents(docs)
            keyword_retriever.k = 10

            ensemble_retriever = EnsembleRetriever(
                retrievers=[vector_retriever, keyword_retriever],
                weights=[0.5, 0.5]
            )

            # Layer 6: Reranking (Flashrank)
            compressor = FlashrankRerank(top_n=3)
            production_retriever = ContextualCompressionRetriever(
                base_compressor=compressor,
                base_retriever=ensemble_retriever
            )

            st.session_state.retriever = production_retriever
            st.success("✅ System Ready! Ask questions below.")

        except Exception as e:
            st.error(f"Error processing video: {e}")

# Trigger Processing
if process_btn and video_url and api_key:
    process_video(video_url)

# --- 2. CHAT INTERFACE (Layer 7) ---
if st.session_state.retriever:
    # Display Chat History
    for role, message in st.session_state.chat_history:
        with st.chat_message(role):
            st.markdown(message)

    # User Input
    question = st.chat_input("Ask about the video...")
    if question:
        # User Message
        st.session_state.chat_history.append(("user", question))
        with st.chat_message("user"):
            st.markdown(question)

        # Generate Response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)

                template = """
                Answer strictly based on the context. Cite sources/quotes.
                Context: {context}
                Question: {question}
                """
                prompt = ChatPromptTemplate.from_template(template)

                def format_docs(docs):
                    return "\n\n".join(d.page_content for d in docs)

                chain = (
                    {"context": st.session_state.retriever | format_docs, "question": RunnablePassthrough()}
                    | prompt
                    | llm
                    | StrOutputParser()
                )

                response = chain.invoke(question)
                st.markdown(response)
                st.session_state.chat_history.append(("assistant", response))
