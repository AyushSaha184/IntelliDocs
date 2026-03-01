# IntelliDocs - Enterprise-Scale RAG Assistant

IntelliDocs is an AI-powered enterprise-scale Retrieval-Augmented Generation (RAG) system designed to intelligently process and query over local documents and knowledge bases. Built for speed and scalability, the system handles document ingestion, chunking, high-performance vector retrieval, and LLM-powered answer generation.

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
    - [Embedding Caching](#embedding-caching)
- [Architecture](#architecture)
- [Requirements](#requirements)
- [Troubleshooting](#troubleshooting)
- [License](#license)

## Project Overview

IntelliDocs enables organizations to turn their local document repositories into interactive knowledge bases. It features a unique **Parallel Processing Pipeline** that optimizes CPU and GPU usage for multi-million document ingestion. By combining dense vector retrieval with NVIDIA's latest reranking APIs, it achieves state-of-the-art accuracy in retrieval tasks. The system is designed with session isolation, allowing multiple users to manage independent knowledge bases securely.

## Features

### Intelligent Ingestion
- **Multi-Format Support**: Ingest PDFs, Word documents, Excel sheets, CSVs, Markdown, JSON, and source code files.
- **Parallel Pipeline**: Asynchronous document loading, multi-process chunking, and batched GPU/API embedding.
- **Auto-Tuning**: Automatically detects VRAM and system resources to tune batch sizes for maximum throughput.

### Advanced Retrieval
- **Dense Vector Search**: High-performance similarity search using FAISS (Flat, IVF, and HNSW indices supported).
- **NVIDIA Reranking**: Integration with `nv-rerank-qa-mistral-4b:1` for ultimate precision after initial retrieval.
- **SQLite Embedding Cache**: Avoids redundant API calls by persisting embeddings on disk, saving costs and time.

### Enterprise Ready
- **Session Isolation**: Each user session has its own document store, vector index, and processing logs.
- **FastAPI Backend**: Robust, session-aware REST API with background task management.
- **Modern UI**: Clean, responsive interface built with React 19, Vite, and Tailwind CSS v4.

## Project Structure

```text
.
├── backend/            # FastAPI application logic
│   ├── api/            # API route definitions
│   ├── database/       # PostgreSQL models and migrations
│   ├── rag/            # Backend-specific RAG wrappers
│   └── services/       # Core business logic (Session mgmt, RAG service)
├── config/             # Centralized configuration
│   └── config.py       # Environment-aware settings
├── data/               # Persistent storage (Vector indices, caches, docs)
├── frontend/           # React 19 + Vite frontend application
│   ├── src/            # Frontend source code
│   └── static/         # Compiled frontend assets
├── src/                # Shared core RAG modules
│   ├── modules/        # Modular RAG components (Loader, Chunking, etc.)
│   └── utils/          # Shared utilities (Logger)
├── main.py             # Integrated CLI and API entry point
├── requirements.txt    # Backend Python dependencies
├── Dockerfile          # Containerization for deployment
└── docker-compose.yml  # Multi-container orchestration
```

## Installation

### Prerequisites
- Python 3.10+
- Node.js 18+
- PostgreSQL
- NVIDIA API Key (Optional, for Reranking/Embeddings)
- Google Gemini or OpenRouter API Key

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
   Create a PostgreSQL database named `rag_db`.

4. **Environment Configuration**:
   ```bash
   cp .env.example .env
   # Edit .env with your PostgreSQL credentials and API keys
   ```

## Usage

### Build Command
Build or update the vector store by ingesting documents from `data/documents/`.
```bash
python main.py --build
```

### Query Command
Start an interactive CLI chat session.
```bash
python main.py --query
```

### API Command
Start the production FastAPI server.
```bash
python main.py --api
```

### Test Command
Test retrieval performance for a specific query.
```bash
python main.py --test "What are the capabilities of this system?"
```

## Configuration

### Environment Variables
Key configuration options in your `.env` file:
- `LLM_PROVIDER`: `gemini`, `openrouter`, or `hf-inference`.
- `EMBEDDING_PROVIDER`: `nvidia`, `gemini`, `lm-studio`, or `local`.
- `USE_RERANKER`: Enable/Disable NVIDIA API reranking.
- `POSTGRES_DB`: PostgreSQL database name.

### Command-line Flags
`main.py` supports several runtime overrides:
- `--chunk-size`: Target size for text chunks.
- `--top-k`: Number of documents to retrieve.
- `--device`: `cpu` or `cuda` for local embeddings.

## How IntelliDocs Works

### Parallel Processing Pipeline
IntelliDocs uses a three-stage concurrent architecture to handle large datasets:
1. **Loading (I/O Bound)**: Utilizes `ThreadPoolExecutor` to stream files from disk.
2. **Chunking (CPU Bound)**: Uses `ProcessPoolExecutor` to bypass the GIL and chunk text in parallel.
3. **Embedding (GPU/API Bound)**: Employs a single-worker batching queue to maximize GPU throughput or manage API rate limits.

### Embedding Caching
Identical text segments are never embedded twice. The system uses a **SHA-256 content-addressable cache** stored in a SQLite database (`data/cache/embedding_cache.db`). This ensures that even if you rebuild your index, previously processed chunks are retrieved from disk instantly.

## Architecture
The system is built on modular, swappable components located in `src/modules/`:
- **Loader**: Multi-format document parsing using PyMuPDF and Unstructured.
- **Chunking**: Recursive character splitting with overlap management.
- **Embeddings**: Abstract provider interface for NVIDIA, Gemini, and Local models.
- **VectorStore**: FAISS-powered storage with metadata management.
- **Retriever**: Dense search with optional reranking integration.

## Troubleshooting
- **PostgreSQL Connection**: Ensure the database is running and credentials in `.env` match.
- **NVIDIA API Limits**: If reranking fails, check your API quota and model availability.
- **OOM Errors**: If running locally on GPU, the system will automatically attempt to reduce batch sizes, but you may need to set `--device cpu` if VRAM is insufficient.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
