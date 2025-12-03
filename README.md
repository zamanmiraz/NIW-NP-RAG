# NIW-NP-RAG

Retrieval-Augmented Generation (RAG) pipeline for analyzing USCIS Administrative Appeals Office (AAO) case PDFs with a focus on National Interest Waiver (NIW) decisions. The repository contains tools extract and preprocess text, build semantic embeddings, index with FAISS, and expose a RAG query service.

## Architecture

![NIW-NP-RAG Architecture](/assets/architecture.jpg)
---

## Features

- **PDF Crawling & Download**: Automated crawler with polite throttling, deduplication, and pause/stop controls
- **Text Extraction**: Robust PDF parsing with OCR fallback for scanned documents
- **Semantic Chunking**: Intelligent document splitting using embedding-based boundaries
- **Vector Embeddings**: High-quality semantic embeddings via sentence-transformers or cloud providers
- **FAISS Indexing**: Fast approximate nearest-neighbor search for semantic retrieval
- **RAG Pipeline**: Retrieve relevant context and generate answers using LLM prompts
- **Web UI**: Streamlit interface for interactive queries and document exploration
- **REST API**: FastAPI service for programmatic access and integration

### Retrieval Performance

Evaluated on a held-out test set of NIW case queries:

| Metric | Score |
|--------|-------|
| Recall@K | 72.00% |
| Precision@K | 20.80% |
| Mean Reciprocal Rank (MRR) | 0.70 |
| Normalized Discounted Cumulative Gain (nDCG@K) | 0.6957 |

## Quick Start
 
[![Query Demo](/assets/query.gif)](/assets/query.gif)

### Prerequisites
- Python 3.10+ (recommended to use a virtual environment)
- Git

### Setup
1. Clone the repository and navigate to the project folder.
2. Create and activate a virtual environment.
3. Install the required dependencies:

```powershell
pip install -r requirements.txt
```

> **Note:** Obtain the data and place it in the `data/` folder. Contact me for access if needed.

Run the API and UI
```powershell
# API
uvicorn niw_np_rag.app.main:app --host 0.0.0.0 --port 8000

# Streamlit UI (in another terminal)
streamlit run streamlit_app.py
```

---

## Folder structure

``` text
NIW-NP-RAG/                         # Project root directory
│
├── assets/                         # Static files, images, or supporting assets
│
├── niw_np_rag/                     # Main Python package
│   ├── app/                        # Application modules (e.g., RAG pipeline, API)
│   ├── config/                     # Configuration files and environment settings
│   └── scripts/                    # Utility or preprocessing scripts
│
├── data/                           # (Optional) Raw and processed data storage
│
├── test/                           # Test scripts and small development environment
│
├── streamlit_app.py                # Streamlit UI entry point
│
└── README.md                       # Project documentation

```

---

## Configuration

- Save credentials or API keys (LLM, embedding providers) in environment variables or a config file not checked into source control.
- Example environment variables:
  - GOOGLE_API_KEY
  - OPENAI_API_KEY
  - VECTOR_STORE_PATH

Place provider-specific configuration under `niw_np_rag/config/` and do not commit secrets.

---


## Contributing

1. Fork the repo and create a feature branch
2. Keep changes small and unit-tested
3. Open a pull request with a description and testing steps
4. Use `black`/`ruff` for style (if configured)

---

## License & attribution

Specify project license here (e.g., MIT). Acknowledge any third-party tools, libraries, and data sources used.

---

## Contact

For questions, open an issue or contact the maintainer via the GitHub repository.