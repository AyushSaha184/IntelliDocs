# IntelliDocs - Enterprise-Scale RAG Assistant

IntelliDocs is an AI-powered enterprise-scale Retrieval-Augmented Generation (RAG) system designed to intelligently process and query over local documents and knowledge bases. Built for speed and scalability, the system handles document ingestion, chunking, high-performance vector retrieval, and LLM-powered answer generation — all through a clean, web interface.

**[🚀 Live Demo](https://enterprise-ai-assistant-rn3m.onrender.com)**

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [Build Command](#build-command)
  - [Query Command](#query-command)
  - [API Command](#api-command)
  - [Test Command](#test-command)
- [Configuration](#configuration)
  - [Environment Variables](#environment-variables)
  - [Command-line Flags](#command-line-flags)
- [How IntelliDocs Works](#how-intellidocs-works)
  - [Parallel Processing Pipeline](#parallel-processing-pipeline)
  - [Extension-Aware Chunking](#extension-aware-chunking)
  - [Embedding & Caching](#embedding--caching)
  - [Session Isolation](#session-isolation)
- [Architecture](#architecture)
- [Requirements](#requirements)
- [Troubleshooting](#troubleshooting)
- [License](#license)

## Project Overview

IntelliDocs enables organizations to turn their local document repositories into interactive knowledge bases. It features a **Parallel Processing Pipeline** that optimizes CPU and GPU usage for large-scale document ingestion. By combining dense vector retrieval with optional NVIDIA reranking APIs, it achieves state-of-the-art accuracy in retrieval tasks. The system is built around session isolation, allowing multiple users to independently upload, process, and query their own documents securely.

Specialized for:

- Company policies (long, formal documents)

- HR documents (rule-based and structured)

- FAQs (short Q&A format)

- Financial summaries (number-heavy content)

- Product documentation (technical text and details about previous projects)

- CSV structured data (tabular format)

- Website content (marketing and general information)

## Features

### Intelligent Ingestion

- **Multi-Format Support**: Ingest PDFs, Word documents, Excel sheets, CSVs, Markdown, JSON, HTML, source code files, and Jupyter notebooks.
- **Parallel Pipeline**: Asynchronous document loading (threads), multi-process chunking, and batched GPU/API embedding for more than 10 Documents.
- **Auto-Tuning**: Automatically detects document count and system resources to choose the optimal pipeline (sequential vs. parallel).

### Extension-Aware Chunking

- **Smart Routing**: Each file type uses the most appropriate chunking strategy automatically.
- **Strategy Library**: Semantic, header-based (Markdown), DOM-aware (HTML), code-structure (Python/JS/Java), row-group (CSV/Excel), key-path (JSON/YAML), and cell-aware (Jupyter) chunkers.
- **Q&A CSV Detection**: Automatically detects query/answer column structure and applies per-row chunking for semantic retrieval.
- **Table-Aware Chunking**: Detects and preserves tables as atomic units within text documents.

### Advanced Retrieval

- **Dense Vector Search**: High-performance similarity search using FAISS (Flat, IVF, and HNSW indices supported).
- **NVIDIA Reranking**: Optional integration with `nv-rerank-qa-mistral-4b:1` for precision reranking after initial retrieval.
- **SQLite Embedding Cache**: Avoids redundant API calls by persisting embeddings on disk, saving cost and time.

### Enterprise Ready

- **Session Isolation**: Each user session has its own document store, vector index, and chunk metadata — no cross-user data leakage.
- **FastAPI Backend**: Robust, session-aware REST API with background task management and concurrency control (max 5 concurrent processing sessions).
- **Auto Cleanup**: Background scheduler deletes sessions after 2 hours from creation or 1 hour of inactivity.
- **Disk Safety**: Upload endpoint checks available disk space before accepting files and enforces a 20MB per-file limit.
- **Modern UI**: Clean, responsive web interface built with React 19, Vite, and Tailwind CSS v4.
- **Two-Level Response Cache**: SQLite-backed retrieval cache and LLM cache to avoid redundant computation within a session.

## Project Structure

```text
.
├── backend/
│   ├── api/
│   │   └── routes.py           # FastAPI route definitions (upload, ask, status, process, health)
│   ├── database/
│   │   └── models.py           # SQLAlchemy session model; PostgreSQL with SQLite fallback
│   ├── rag/
│   │   ├── IngestSession.py    # Session-aware ingestion pipeline (parallel file loading + retry)
│   │   ├── Embedder.py         # Embedding service wrapper
│   │   └── retriever.py        # Retriever wrapper
│   ├── services/
│   │   ├── session_service.py  # Session creation, isolation, and lifecycle management
│   │   ├── rag_service_session.py  # Session-scoped QueryHandler with shared LLM/reranker singletons
│   │   ├── cleanup_scheduler.py    # Background session and cache expiry scheduler
│   │   └── cleanup_storage.py      # Manual storage cleanup CLI utility
│   └── main.py                 # FastAPI app entrypoint (lifespan, CORS, static serving)
├── config/
│   └── config.py               # Central environment-aware configuration
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── ChatInput.jsx       # Message input with multi-file upload and process trigger
│   │   │   ├── ChatMessage.jsx     # Message bubble with markdown, source citations, copy button
│   │   │   ├── FileUpload.jsx      # File picker with validation (type + size)
│   │   │   ├── FilePreviews.jsx    # Pre-upload file preview and batch upload UI
│   │   │   └── UploadedFiles.jsx   # Post-upload file status with collapsible panel
│   │   ├── services/
│   │   │   └── api.js              # API client (upload, process, status, ask, health)
│   │   └── App.jsx                 # Main app with session state and polling logic
│   └── index.html
├── src/
│   ├── modules/
│   │   ├── Loader.py           # Multi-format document loader with PDF streaming and CSV detection
│   │   ├── Chunking.py         # Extension-aware text chunker with all strategies and PostgreSQL persistence
│   │   ├── Embeddings.py       # NVIDIA BGE-M3 embedding service with disk cache
│   │   ├── VectorStore.py      # FAISS vector store with atomic save and cached reverse mapping
│   │   ├── Retriever.py        # Dense retriever with optional NVIDIA reranker
│   │   ├── QueryGeneration.py  # QueryHandler: retrieval + LLM response + two-level cache
│   │   ├── LLM.py              # LLM providers (OpenRouter, Gemini, HuggingFace)
│   │   ├── ParallelPipeline.py # Three-stage streaming pipeline (threads → processes → GPU)
│   │   └── QueryCache.py       # SQLite-backed retrieval and LLM response caches
│   └── utils/
│       └── Logger.py           # Structlog-based logger with JSON file + console output
├── data/                       # Runtime data (vector indices, session dirs, cache) — gitignored
├── logs/                       # Application logs — gitignored
├── main.py                     # Unified CLI entrypoint (build / query / api / test modes)
├── requirements.txt            # Full dependency list
├── requirements-render.txt     # Slim dependency list for Render.com (no local ML models)
├── Dockerfile                  # Multi-stage build (Node frontend → Python backend)
├── docker-compose.yml          # PostgreSQL + app container orchestration
├── server.bat                  # Local Windows startup script (PostgreSQL + backend + frontend)
└── .env.example                # Environment variable template
```

## Installation

### Prerequisites

- Python 3.10+
- Node.js 18+
- PostgreSQL (or Docker)
- NVIDIA API Key
- OpenRouter API Key

### Steps

1. **Clone & Setup Backend**:

   ```bash
   git clone <repo_url>
   cd Enterprise-ai-assistant
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Setup Frontend**:

   ```bash
   cd frontend
   npm install
   ```

3. **Database Setup**:
   Create a PostgreSQL database:

   ```sql
   CREATE DATABASE rag_db;
   ```

4. **Environment Configuration**:

   ```bash
   cp .env.example .env
   # Edit .env with your credentials and API keys
   ```

5. **Quick Start (Windows)**:

   ```bash
   .\server.bat
   ```

6. **Quick Start (Docker)**:

   ```bash
   docker-compose up --build
   ```

## Usage

### Build Command

Build or update the vector store by ingesting documents from `data/documents/`. Automatically selects sequential or parallel mode based on document count (threshold: 11 documents).

```bash
python main.py --build
```

Force a specific pipeline:

```bash
python main.py --build --parallel     # Force parallel (recommended for 50+ docs)
python main.py --build --sequential   # Force sequential
python main.py --build --incremental  # Add new documents without clearing existing data
```

### Query Command

Start an interactive CLI chat session against the built vector store.

```bash
python main.py --query
```

### API Command

Start the production FastAPI server (serves both the REST API and the React frontend).

```bash
python main.py --api
```

Access the UI at `http://localhost:8000` and API docs at `http://localhost:8000/docs`.

### Test Command

Run a single test query against the built vector store and print results to the console.

```bash
python main.py --test-query "What are the capabilities of this system?"
```

## Configuration

### Environment Variables

Key variables for your `.env` file:

```bash
# Embedding Configuration
NVIDIA_API_KEY=your_nvidia_api_key_here
EMBEDDING_PROVIDER=nvidia               # nvidia | gemini | hf-inference | lm-studio | local
EMBEDDING_MODEL=baai/bge-m3
EMBEDDING_NORMALIZE=true
EMBEDDING_TIMEOUT=120.0
EMBEDDING_MAX_RETRIES=3

# Retrieval Configuration
RETRIEVAL_MODE=dense                    # dense | hybrid

# Reranker Configuration
USE_RERANKER=true
RERANKER_MODEL=nv-rerank-qa-mistral-4b:1
MIN_CHUNKS_TO_RERANK=8
TOP_K_AFTER_RERANK=5

# LLM Configuration
LLM_PROVIDER=openrouter                 # openrouter | gemini | hf-inference
OPENROUTER_API_KEY=your_openrouter_api_key_here
LLM_MODEL=nvidia/nemotron-3-nano-30b-a3b:free
LLM_TEMPERATURE=0.7
LLM_MAX_TOKENS=1000

# PostgreSQL Configuration
DATABASE_URL=postgresql://user:password@localhost:5432/rag_db
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=rag_db
POSTGRES_USER=your_db_user
POSTGRES_PASSWORD=your_db_password
```

### Command-line Flags

`main.py` supports runtime overrides:

- `--device`: `cpu` or `cuda` for local embeddings.
- `--top-k`: Number of chunks to retrieve per query (default: 5).
- `--chunk-size`: Target token size for text chunks (default: 600).
- `--chunk-overlap`: Token overlap between chunks (default: 90).
- `--strategy`: Chunking strategy (`fixed_size`, `semantic`, `sliding_window`, `sentence`, `paragraph`).
- `--vector-store-type`: FAISS index type (`flat`, `ivf`, `hnsw`).
- `--api-port`: Port for the API server (default: 8000).
- `--loader-threads`: Number of loader threads in parallel mode (default: 8).
- `--incremental`: Keep existing database; only add new documents.

### Docker Compose (Recommended)

The `docker-compose.yml` orchestrates a PostgreSQL database and the app container with persistent storage volumes.

```bash
docker-compose up --build
```

Set `OPENROUTER_API_KEY` in `.env` before running. The app will be available at `http://localhost:8000`.

### Render.com

An example blueprint is provided in `render.example.yaml`. Key settings for Render deployment:

- Set `EMBEDDING_PROVIDER=nvidia` to use the NVIDIA embedding API (no local model required, fits within Render free-tier RAM limits).
- Use `requirements-render.txt` (slim build, no PyTorch).
- Provision a Render PostgreSQL database and attach `DATABASE_URL` automatically.
- Mount a persistent disk at `/app/data` to preserve session data across restarts.

## How IntelliDocs Works

### Parallel Processing Pipeline

IntelliDocs uses a three-stage concurrent architecture to handle large datasets:

1. **Loading (I/O Bound)**: A `ThreadPoolExecutor` loads files from disk in parallel, with format-specific handlers for PDF (PyMuPDF with OCR fallback), DOCX (python-docx with unstructured fallback), Excel, CSV, and plain text.
2. **Chunking (CPU Bound)**: A `ProcessPoolExecutor` bypasses the GIL to chunk documents in parallel using extension-aware routing.
3. **Embedding (GPU/API Bound)**: A concurrent `ThreadPoolExecutor` runs multiple embedding API calls in flight simultaneously (default: 6 workers), writing results to FAISS as each batch completes. For small datasets (≤5 batches), a sequential path is used to reduce overhead.

The auto-detection threshold (11 documents) switches the build from sequential to parallel mode. For incremental builds (`--incremental`), existing data is preserved and only new documents are processed.

### Extension-Aware Chunking

The `TextChunker` routes each file to a dedicated chunking strategy based on file extension:

| Extension | Strategy | Description |
|---|---|---|
| `.pdf`, `.docx`, `.txt` | Semantic | Sentence-boundary splits with 15% overlap; table blocks kept atomic |
| `.md` | Header-based | Respects heading hierarchy; falls back to fixed-size for oversized sections |
| `.html`, `.htm` | DOM-aware | BeautifulSoup extraction with header hierarchy tracking |
| `.py`, `.js`, `.java`, etc. | Code-structure | AST-based function and class extraction |
| `.csv`, `.tsv` | Row-group | Auto-detects Q&A, Document, or Bulk structure; vectorized for performance |
| `.json`, `.yaml` | Key-path | Flattens nested structures into dot-path key-value chunks |
| `.ipynb` | Cell-aware | Treats code and markdown cells separately |

### Embedding & Caching

Identical text segments are never embedded twice. The system uses a **SHA-256 content-addressable cache** stored in a SQLite database (`data/cache/embedding_cache.db`). At startup, cached embeddings are loaded into memory so subsequent builds retrieve them instantly without any API calls. New embeddings are persisted back in a single batch transaction per ingestion cycle.

### Session Isolation

Each user upload creates a fully isolated session with its own directories:

```
data/sessions/{session_id}/
    documents/      ← Only this user's uploaded files
    chunks/         ← Chunk metadata JSON for this session
    vector_store/   ← FAISS index for this session
```

Shared singletons (LLM, embedding service, reranker) are initialized once and reused across sessions to minimize memory and startup cost. A background `CleanupScheduler` runs every 10 minutes, deleting sessions older than 2 hours or idle for more than 1 hour, and pruning the retrieval and LLM caches.

## Architecture

The system is built on modular, swappable components:

- **Loader** (`src/modules/Loader.py`): Multi-format document parsing with streaming for large PDFs and CSVs, scanned PDF detection, and OCR fallback.
- **Chunking** (`src/modules/Chunking.py`): Extension-aware chunker with 8 strategies, PostgreSQL persistence via `executemany` batch inserts, and a thread-safe 50K-entry token count cache.
- **Embeddings** (`src/modules/Embeddings.py`): Abstract embedding service with NVIDIA BGE-M3 as primary, disk-backed SQLite cache, in-memory deduplication, and retry with exponential backoff.
- **VectorStore** (`src/modules/VectorStore.py`): FAISS-powered storage with O(1) reverse ID mapping, atomic file saves (write-to-.tmp then `os.replace`), and batch-add support.
- **Retriever** (`src/modules/Retriever.py`): Dense search with optional NVIDIA API reranking; retrieves `k×3` candidates before reranking to top-k.
- **QueryGeneration** (`src/modules/QueryGeneration.py`): `QueryHandler` orchestrates retrieval, context formatting, LLM call, and two-level (retrieval + LLM) SQLite caching.
- **LLM** (`src/modules/LLM.py`): Provider-agnostic LLM layer; strips `<think>` reasoning blocks from model output automatically.
- **ParallelPipeline** (`src/modules/ParallelPipeline.py`): Three-stage streaming pipeline with bounded queues and backpressure; standalone helpers (`load_documents_parallel`, `chunk_documents_parallel`, `embed_chunks`) can be used independently.
- **Backend API** (`backend/`): FastAPI app with session lifecycle management, concurrency semaphore (max 5 concurrent ingest jobs), disk-space guard, and background cleanup.
- **Frontend** (`frontend/`): React 19 + Vite SPA with multi-file upload, processing status polling, Markdown rendering, collapsible source citations, and auto-scroll.

### Architect Diagram

![Architecture Diagram](Demo/Architecture_Diagram.png)

### Data Flow Diagram

![Data Flow Diagram](Demo/DataFlow_Diagram.jpeg)

## Requirements

- Python 3.10+
- Node.js 18+
- PostgreSQL (SQLite fallback available for local development)
- NVIDIA API Key (optional — for NVIDIA embeddings and reranking)
- Google Gemini or OpenRouter API Key (for LLM responses)
- Docker (optional — for containerized deployment)
- Tesseract (optional — for OCR on scanned PDFs)

## Troubleshooting

**PostgreSQL Connection Errors**

Ensure the database is running and credentials in `.env` match. The system will automatically fall back to SQLite if `DATABASE_URL` is not set or is a localhost URL on a remote server (e.g., Render).

**NVIDIA API Rate Limits or Timeouts**

Increase `EMBEDDING_TIMEOUT` and `EMBEDDING_MAX_RETRIES` in `.env`. The embedding service retries with exponential backoff automatically.

**Scanned PDFs Produce No Text**

Install OCR dependencies: `pip install pytesseract Pillow`. Then install the Tesseract binary for your OS. IntelliDocs will automatically detect scanned pages and apply OCR.

**OOM Errors During Embedding**

If running with a local GPU model, reduce `EMBEDDING_BATCH_SIZE` or set `--device cpu`. For the NVIDIA API path, the IngestSession pipeline automatically falls back to sequential batching if concurrent embedding workers are overloaded.

**Low Disk Space**

Run the manual cleanup utility: `python -m backend.services.cleanup_storage`. This shows a storage breakdown and lets you delete old sessions and documents interactively.

**Session Processing Stuck**

If a session stays in `processing` status for more than 2 minutes, check the application logs in `logs/`. Common causes are embedding API errors (check API key and quota) or PostgreSQL lock contention (run `python clear_db.py` to reset tables).

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
