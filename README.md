# VideoRAG: Production-Grade YouTube Transcript RAG System

A sophisticated Retrieval-Augmented Generation (RAG) system for analyzing YouTube videos with semantic chunking, hybrid search, and intelligent reranking.

## Features

- **Layer 1: Document Processing** - Robust YouTube transcript ingestion with automatic language fallback
- **Layer 2: Semantic Chunking** - Intelligent text splitting using embeddings with fallback strategies
- **Layer 3-4: Hybrid Search** - Combines vector search (FAISS) + keyword search (BM25) for comprehensive retrieval
- **Layer 6: Reranking** - FlashRank-based document reranking for top 3 most relevant results
- **Layer 7: Generation** - Google Gemini AI powered responses with source citations

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Jatin07gupta/VideoRAG.git
cd VideoRAG
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your Google API Key:
   - Get your API key from [Google AI Studio](https://aistudio.google.com/app/apikey)
   - You'll enter it in the Streamlit app sidebar

## Usage

Run the Streamlit app:
```bash
streamlit run app.py
```

1. Enter your Google API Key in the sidebar
2. Paste a YouTube URL
3. Click "Analyze Video"
4. Ask questions about the video content

## Architecture

```
Video URL
    ↓
[Layer 1] YouTube Loader → Transcript Extraction
    ↓
[Layer 2] Semantic Chunker → Intelligent Text Splitting
    ↓
[Layer 3-4] Hybrid Retrieval
    ├─→ Vector Search (FAISS)
    └─→ Keyword Search (BM25)
    ↓
[Layer 6] FlashRank Reranker
    ↓
[Layer 7] Gemini LLM → Answer Generation
    ↓
User Response + Citations
```

## Requirements

- Python 3.8+
- Google API Key for Gemini
- Internet connection for YouTube access

## Project Structure

```
videoRAG/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── .gitignore            # Git ignore rules
└── notebooks/            # Jupyter notebooks (optional)
```

## Technologies Used

- **LangChain** - RAG orchestration
- **FAISS** - Vector similarity search
- **BM25** - Keyword search
- **FlashRank** - Document reranking
- **Google Gemini** - LLM for generation
- **Streamlit** - Web UI framework

## Author

- **Jatin Gupta**
  - **GitHub**: [@Jatin07gupta](https://github.com/Jatin07gupta)
  - **Institution**: Indian Institute of Technology, Jodhpur (M.Tech CSE)
  - **Email**: [jatngupta0710@gmail.com](mailto:jatngupta0710@gmail.com)

## License

MIT License - Feel free to use and modify!
