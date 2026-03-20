# IntelliDocs - Enterprise-Scale RAG Assistant

IntelliDocs is an AI-powered enterprise-scale Retrieval-Augmented Generation (RAG) system that turns local document repositories into interactive knowledge bases. It features a **Multi-Agent Orchestration Pipeline** that routes every query through a Planner → Router → Retriever → Synthesizer → Validator chain, a **Low-Latency Streaming Pipeline** with asynchronous post-stream safety validation, a **Hybrid Search Pipeline** combining dense vector search and BM25 sparse retrieval, a **Human-in-the-Loop Review System** for high-confidence deployments, and a **full authentication layer** via Supabase (Google OAuth + email/password) with guest access.

**[🚀 Live Demo](https://intelli-docs-five.vercel.app)**

### Architecture Diagram

![Architecture Diagram](https://i.postimg.cc/Bn5gcDDq/Architecture-Diagram.png)

### Data Flow Diagram

![Data Flow Diagram](https://i.postimg.cc/Zn1cjxjg/Data-Flow-Diagram.png)

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
  - [Multi-Agent Orchestration](#multi-agent-orchestration)
  - [Conditional Query Routing](#conditional-query-routing)
  - [Human-in-the-Loop Review](#human-in-the-loop-review)
  - [Hybrid Search Pipeline](#hybrid-search-pipeline)
  - [Parallel Processing Pipeline](#parallel-processing-pipeline)
  - [Extension-Aware Chunking](#extension-aware-chunking)
  - [Four-Level Caching](#four-level-caching)
  - [Metadata Enrichment](#metadata-enrichment)
  - [Session Isolation](#session-isolation)
  - [Authentication & Multi-Chat](#authentication--multi-chat)
- [Architecture](#architecture)
- [API Reference](#api-reference)
- [Requirements](#requirements)
- [Troubleshooting](#troubleshooting)
- [License](#license)

## Project Overview

IntelliDocs enables organizations to turn local document repositories into interactive, queryable knowledge bases. At its core is a **Multi-Agent Orchestration Pipeline**: every query is classified by a `QueryPlanner`, routed by a `ConditionalRouter` to the appropriate execution path (direct LLM, single-agent, or multi-agent), processed by `RetrieverAgent` + `SynthesizerAgent`, validated for hallucination by `ValidatorAgent`, and optionally escalated to human review via the `Gatekeeper`.

Retrieval uses a **Hybrid Search** core that fuses dense vector search (FAISS) with BM25 sparse keyword search via Reciprocal Rank Fusion, then optionally applies NVIDIA neural reranking. A **three-stage Parallel Processing Pipeline** distributes ingestion work across I/O threads, CPU processes, and concurrent embedding API calls. A **two-tier query cache** (retrieval + LLM response) eliminates redundant computation, and rolling **Evaluation Metrics** expose precision, recall, and latency per query.

Specialized for:

- Company policies (long, formal documents)
- HR documents (rule-based and structured)
- FAQs (short Q&A format)
- Financial summaries (number-heavy content)
- Product documentation (technical text)
- CSV structured data (tabular format)
- Website content (marketing and general information)
- Source code repositories

## Features

### 🤖 Multi-Agent Orchestration (New)

- **QueryPlanner**: Uses the LLM to classify every query into one of six types — `trivial`, `factual`, `summarization`, `analytical`, `comparative`, `multi_hop` — and decomposes multi-hop queries into sub-queries. Falls back to a heuristic classifier when the LLM is unavailable.
- **ConditionalRouter**: Routes the plan to the fastest viable execution path: `DIRECT_LLM` (trivial — no retrieval), `SINGLE_AGENT` (factual/summarization), `MULTI_AGENT` (analytical/comparative/multi-hop), or `HUMAN_REVIEW` (low confidence).
- **RetrieverAgent**: Wraps the hybrid RAG retriever for use inside the agent pipeline; checks the retrieval cache before calling FAISS+BM25.
- **SynthesizerAgent**: Builds the full prompt (system prompt + compressed context + query) and calls the LLM. Supports **streaming token generation** and provider-specific **KV Caching** (via OpenRouter `cache_control` hints) for drastically lower perceived latency.
- **ValidatorAgent**: Uses a secondary fast LLM (e.g., Cerebras Llama 3.1) to verify answer groundedness. In streaming mode, this runs **asynchronously post-stream**, redacting the response in the UI if hallucinations are detected after the fact.
- **Context Compression**: Two-stage compression keeps prompts within `TOKEN_BUDGET`: (1) cheap truncation of least-relevant chunks; (2) LLM summarization only when still >20% over budget after truncation.
- **Max Agent Steps**: Orchestrator enforces a 5-step cap and a configurable timeout to prevent runaway pipelines.
- **Human-Approved Answer Cache**: Corrected answers stored by human reviewers are first-checked on every subsequent identical query (with a freshness guard per session).

### 🔀 Conditional Query Routing

| Query Type | Route | Behaviour |
|---|---|---|
| `trivial` (greetings, simple yes/no) | `DIRECT_LLM` | LLM called directly — zero retrieval overhead |
| `factual` (simple, single-hop) | `SINGLE_AGENT` | One hybrid retrieval → synthesize → validate |
| `summarization` | `SINGLE_AGENT` | Same path, `top_k` raised to 10 for broader coverage |
| `analytical` / `comparative` | `MULTI_AGENT` | Full pipeline: retrieve → synthesize → validate |
| `multi_hop` | `MULTI_AGENT` | Parallel per-sub-query retrieval (fan-out capped at `k=3`) |
| Avg retrieval score < 0.4 | `HUMAN_REVIEW` | Answer escalated to the human review queue |

### 👤 Human-in-the-Loop Review

- **Gatekeeper**: Checks three conditions after retrieval — sensitivity (password/salary/key patterns), groundedness (from `ValidatorAgent`), and confidence (avg retrieval score vs. threshold). Escalates when any check fails.
- **Sensitivity Filter**: Pre-compiled regex patterns detect prompt-injection and PII-adjacent queries at near-zero cost.
- **Review Queue**: Pending reviews stored in SQLite with full query, answer, confidence, and reason. Exposed via `GET /reviews/pending`.
- **Approve or Correct**: Human reviewers can approve the AI answer (`POST /reviews/{id}/approve`) or supply a corrected answer (`POST /reviews/{id}/correct`). Corrected answers are cached and replayed on future identical queries.
- **Audit Log**: Every escalation is written to an `audit_log` table for compliance traceability.

### 🔍 Hybrid Search

- **Dense + Sparse Fusion**: Combines FAISS vector search (semantic understanding) with BM25 keyword search (exact term matching) for higher recall across diverse query types.
- **Reciprocal Rank Fusion (RRF)**: Merges dense and sparse rankings using the industry-standard RRF formula (`1/(k+rank)`), producing a unified, calibrated score without needing a trained fusion head.
- **BM25 Persistent Index**: The BM25 index is built during ingestion and saved as `bm25_index.pkl` alongside the FAISS index — zero rebuild cost on restart.
- **Graceful Degradation**: If BM25 is unavailable or the index is empty, the pipeline falls back to dense-only search transparently.

### 🎯 NVIDIA Neural Reranking

- **nv-rerank-qa-mistral-4b:1**: Optional precision reranking via the NVIDIA Reranking API after hybrid candidate retrieval.
- **Adaptive Threshold**: Reranking only activates when the candidate set exceeds `MIN_CHUNKS_TO_RERANK` (default: 8), avoiding unnecessary API calls on small result sets.
- **Connection Reuse**: Uses a persistent `requests.Session` to reduce per-request TLS/TCP overhead.
- **Configurable**: Enable/disable via `USE_RERANKER` env var.

### ⚡ Low-Latency Streaming (New)

- **SSE Pipeline**: End-to-end streaming from the LLM provider (OpenRouter/Cerebras) through the FastAPI backend to the React frontend via Server-Sent Events.
- **KV Prompt Caching**: Automatically restructures prompts to place static system instructions and retrieval context first, leveraging OpenRouter's KV cache to skip processing of 1000+ repetitive tokens.
- **Redact-and-Replace Safety**: If the asynchronous `ValidatorAgent` determines a streamed answer is ungrounded, the frontend immediately hides the message and displays a safety warning, preventing users from reading AI hallucinations.

### ⚡ Parallel Processing Pipeline

- **Three-Stage Streaming**: Document loading (ThreadPoolExecutor) → chunking (ProcessPoolExecutor) → embedding (concurrent API workers), connected by bounded queues for backpressure.
- **Auto-Tuning**: Automatically selects sequential vs. parallel mode based on document count (threshold: 11 documents). Automatically detects GPU VRAM and tunes `batch_size` and `pre_batch_size`.
- **Concurrent Embedding**: Up to 6 embedding API calls in-flight simultaneously (I/O-bound optimization); FAISS writes are serialized in the main thread — no locking needed.
- **OOM Recovery**: Automatically halves batch size and retries up to 3 times on GPU out-of-memory errors.
- **Incremental Ingestion**: `--incremental` flag preserves existing vector store and appends only new documents.

### 📄 Extension-Aware Chunking

- **Smart Routing**: Each file type uses the most appropriate chunking strategy automatically.
- **8 Strategies**: Semantic, header-based (Markdown), DOM-aware (HTML), code-structure (Python/JS/Java/Go/Rust/C/C#), row-group (CSV/Excel), key-path (JSON/YAML), cell-aware (Jupyter notebooks), and fixed-size fallback.
- **Q&A CSV Detection**: Automatically detects query/answer column structures and applies per-row chunking for optimal semantic retrieval.
- **Table-Aware**: Detects and preserves tables as atomic units within text documents.

### 💾 Four-Level Caching

| Level | What's Cached | Strategy |
|---|---|---|
| **KV Cache** | Prompt Prefix (System+Context) | Provider-side (OpenRouter) ephemeral KV storage |
| **Retrieval Cache** | Retrieved chunk sets | Redis: session + normalized query + top_k |
| **LLM Cache** | Final LLM responses | Redis: session + query hash |
| **Human-Approved** | Corrected answers | SQLite: exact query match (High Confidence) |

- All caches are **session-scoped** — no cross-user pollution.
- Background `CleanupScheduler` evicts stale cache entries on a 10-minute cycle.

### 🧠 Metadata Enrichment

- **Background Worker**: Runs asynchronously after ingestion completes — never blocks the user session.
- **True LLM Batching**: Multiple chunks are sent in a single LLM call for summary generation, reducing API round trips.
- **Three Enrichment Types**: Per-chunk LLM summaries (2-3 sentences), TF-scored keyword extraction (no LLM needed), and hypothetical question generation (HyDE-style for improved retrieval).
- **Resumable**: Only processes chunks with `enrichment_status='pending'` — safe to restart mid-run.
- **Optional**: Controlled by `ENABLE_CHUNK_ENRICHMENT=1` environment variable.

### 📊 Evaluation & Observability

- **Per-Query Metrics**: Retrieval precision/recall, answer relevance score, latency (retrieval, LLM, total), and route taken are recorded after every query.
- **Rolling Summary**: `GET /eval/summary` returns a configurable window of aggregated metrics for monitoring dashboards.
- **Structured Logging**: Per-module rotating log files with console + file handlers via `src/utils/Logger.py`.

### 🏢 Enterprise Ready

- **Session Isolation**: Each user upload creates its own document store, FAISS/Qdrant index, BM25 index, and chunk metadata directory — zero cross-user data leakage.
- **FastAPI Backend**: Robust, session-aware REST API with concurrency control (max 5 concurrent ingest jobs), disk-space guard (20MB per file, 4GB minimum free), and background lifecycle management.
- **Dual Vector Backend**: Automatically selects FAISS (local/Docker) or Qdrant (managed/remote) at startup based on configuration. Override with `VECTOR_BACKEND=faiss|qdrant`.
- **Storage Abstraction**: `StorageService` supports both local filesystem and Supabase Storage backends for uploaded files — same API, swappable at runtime.
- **Cascade Deletes**: `CascadeService` atomically removes chats, documents, messages, and vector store data together. A background `CleanupWorker` processes delete jobs and sweeps TTL-expired guest sessions on a 60-second cycle.
- **Multi-Chat Persistence**: Logged-in users get a persistent sidebar of named chats with full message and document history, synced across browser tabs via `BroadcastChannel`.
- **Guest Isolation**: Guests get a fresh isolated chat context on every "New Chat" — no data leakage between guest sessions. Limits: 5 documents per chat, no chat history.
- **Document Upload Quotas**: Guest — 5 docs per chat; Logged-in — 15 docs per chat, 40 docs per account. Enforced atomically at the database level with `SELECT ... FOR UPDATE`.
- **Auto Cleanup**: Background worker runs every 60 seconds — deletes TTL-expired guest sessions and processes queued cascade delete jobs.
- **Modern UI**: React 19 + Vite frontend with multi-file upload, per-document processing progress bar, failed-document indicators, Markdown rendering, collapsible source citations, auto-scroll, and a **Report Bug** button.

## Project Structure

```text
.
├── backend/
│   ├── api/
│   │   └── routes.py               # All FastAPI routes: upload, ask, status, process, health,
│   │                               #   chat CRUD, messages, documents, guest sessions, review,
│   │                               #   eval, stress-test (merged from chat_routes.py)
│   ├── auth/
│   │   └── supabase_auth.py        # Supabase JWT verification; anonymous guest passthrough
│   ├── database/
│   │   ├── models.py               # SQLAlchemy models: Session, Chat, ChatMessage, Document,
│   │   │                           #   CleanupJob; PostgreSQL with SQLite fallback
│   │   └── migrations/             # Alembic migration scripts
│   ├── rag/
│   │   ├── IngestSession.py        # Session ingestion: parallel file load + retry embeddings
│   │   │                           #   + BM25 build + FAISS/Qdrant store + progress tracking
│   │   ├── Ingest.py               # Global CLI ingestion pipeline (--build path)
│   │   ├── Embedder.py             # Embedding service wrapper (session path)
│   │   ├── Generator.py            # LLM generation wrapper (session path)
│   │   ├── retriever.py            # Retriever wrapper (session path)
│   │   └── Vector_Store.py         # FAISS/Qdrant write helper (session path)
│   ├── services/
│   │   ├── session_service.py      # Session creation, isolation, status, and lifecycle
│   │   ├── chat_service.py         # Chat/message/document CRUD; quota enforcement
│   │   │                           #   (guest: 5 docs, per-chat: 15, account: 40)
│   │   ├── cascade_service.py      # Atomic cascade deletes + background CleanupWorker (TTL sweep)
│   │   ├── storage_service.py      # Storage abstraction: LocalStorageService / SupabaseStorageService
│   │   ├── rag_service_session.py  # Session-scoped QueryHandler with shared LLM/reranker singletons
│   │   ├── rag_service.py          # Global index QueryHandler (CLI / legacy path)
│   │   ├── cleanup_scheduler.py    # Legacy cleanup scheduler (session TTL expiry)
│   │   └── cleanup_storage.py      # Manual storage cleanup CLI utility
│   └── main.py                     # FastAPI app entrypoint (lifespan, CORS, router registration)
├── config/
│   └── config.py                   # Central environment-aware configuration (all providers,
│                                   #   DB, Qdrant, Supabase, vector backend auto-selection)
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── App.jsx              # Main app: auth state, chat state, upload/process/ask flows,
│   │   │   │                        #   BroadcastChannel cross-tab sync, Report Bug dropdown
│   │   │   ├── Sidebar.jsx          # Chat history sidebar with rename/delete (logged-in users)
│   │   │   ├── ChatInput.jsx        # Message input with multi-file drop zone and process trigger
│   │   │   ├── ChatMessage.jsx      # Message bubble: Markdown, source citations, copy button,
│   │   │   │                        #   grounding badge, streaming indicator
│   │   │   ├── FileUpload.jsx       # File picker with type + size validation
│   │   │   ├── FilePreviews.jsx     # Pre-upload file preview and batch upload UI
│   │   │   └── UploadedFiles.jsx    # Post-upload panel: per-doc progress bar, failed indicators,
│   │   │                            #   collapsible on completion
│   │   ├── services/
│   │   │   ├── api.js               # API client: guest session, chat CRUD, upload, process,
│   │   │   │                        #   status, ask stream, health
│   │   │   └── supabase.js          # Supabase client initialisation
│   │   └── main.jsx                 # Vite entry point
│   └── index.html
├── src/
│   ├── agents/
│   │   ├── Orchestrator.py         # AgentOrchestrator: planner → router → retrieval → synthesis
│   │   │                           #   → validation; step cap; human-approved answer cache
│   │   ├── Planner.py              # QueryPlanner: LLM-based classification (6 types) + sub-query
│   │   │                           #   decomposition; heuristic fallback
│   │   ├── Router.py               # ConditionalRouter: DIRECT_LLM / SINGLE_AGENT /
│   │   │                           #   MULTI_AGENT / HUMAN_REVIEW
│   │   ├── RetrieverAgent.py       # Agent wrapper around RAGRetriever; retrieval cache check
│   │   ├── SynthesizerAgent.py     # Context-to-answer generation; KV prompt caching hints
│   │   ├── ValidatorAgent.py       # Groundedness / hallucination checker (secondary LLM)
│   │   ├── HumanValidation.py      # Gatekeeper + ReviewManager: sensitivity, groundedness,
│   │   │                           #   confidence checks; escalation, approve, correct workflow
│   │   ├── Tools.py                # Agent tool definitions (search, summarise, etc.)
│   │   └── BaseAgent.py            # Abstract base with AgentTask / AgentResult dataclasses
│   ├── evaluation/
│   │   └── Evaluator.py            # Per-query metrics: precision, recall, latency, route;
│   │                               #   rolling summary API
│   ├── modules/
│   │   ├── Loader.py               # Multi-format document loader: PDF streaming + OCR fallback,
│   │   │                           #   python-docx / unstructured DOCX, streaming CSV
│   │   ├── Chunking.py             # Extension-aware chunker (8 strategies); 50K-entry token cache
│   │   ├── DocumentParser.py       # Structure analyzer with parse result caching
│   │   ├── Embeddings.py           # Provider-agnostic embedding service (NVIDIA / Gemini /
│   │   │                           #   HF-Inference / LM Studio / local); exponential backoff retry
│   │   ├── VectorStore.py          # FAISS vector store (Flat/IVF/HNSW); O(1) reverse ID map;
│   │   │                           #   atomic file saves; metadata archive on overwrite
│   │   ├── QdrantStore.py          # Qdrant session store: upsert, search, delete-by-session;
│   │   │                           #   auto-creates collection; falls back to FAISS on init error
│   │   ├── Retriever.py            # BM25Retriever + NvidiaReranker + RAGRetriever
│   │   │                           #   (dense + sparse + RRF + optional rerank)
│   │   ├── QueryCache.py           # Session-scoped retrieval cache + LLM response cache (SQLite)
│   │   ├── QueryGeneration.py      # Legacy QueryHandler for global index / CLI path
│   │   ├── MetadataEnricher.py     # Async background enricher: TF keywords + batched LLM
│   │   │                           #   summaries + HyDE questions; resumable; semaphore-capped
│   │   ├── LLM.py                  # Provider-agnostic LLM: Gemini, HuggingFace, NVIDIA,
│   │   │                           #   LM Studio; strips <think> blocks automatically
│   │   └── ParallelPipeline.py     # Three-stage streaming pipeline (threads → processes → API)
│   └── utils/
│       ├── Logger.py               # Rotating logger with console + file handlers
│       └── llm_provider.py         # Shared LLM/reranker singleton factory
├── data/                           # Runtime data (gitignored)
│   ├── cache/                      # SQLite caches: retrieval, LLM response
│   ├── .parse_cache/               # Document parse cache (avoids re-parsing unchanged files)
│   ├── vector_store/               # Global FAISS index + BM25 index (CLI --build path)
│   ├── documents/                  # Ingested documents (global --build path)
│   ├── chunks/                     # Chunk metadata JSON (global --build path)
│   └── sessions/                   # Per-session isolated stores (API path)
│       └── {session_id}/
│           ├── documents/          # Session-scoped uploaded files
│           ├── chunks/             # Session chunk metadata + chunks_metadata_prev.json rollback
│           └── vector_store/       # Session FAISS/Qdrant index + BM25 index
├── logs/                           # Application logs — gitignored
├── main.py                         # Unified CLI entrypoint (build / query / api / test modes)
├── requirements.txt                # Full dev dependencies
├── requirements-render.txt         # Slim Linux/Docker dependencies (no PyTorch)
├── Dockerfile                      # Multi-stage build (Node frontend → Python backend)
├── docker-compose.yml              # PostgreSQL + app container orchestration
├── server.bat                      # Windows one-command launcher
└── .env.example                    # Environment variable template
```

## Installation

### Prerequisites

- Python 3.10+
- Node.js 20+
- PostgreSQL (or Docker)
- NVIDIA API Key (embeddings + reranking)
- OpenRouter API KEY (LLM responses)

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

Ingest documents from `data/documents/` into the vector store. Automatically selects sequential or parallel mode based on document count (threshold: 11).

```bash
python main.py --build
```

Force a specific pipeline or mode:

```bash
python main.py --build --parallel       # Force parallel (recommended for 50+ docs)
python main.py --build --sequential     # Force sequential
python main.py --build --incremental    # Add new documents without clearing existing data
```

### Query Command

Start an interactive CLI chat session with hybrid search against the built vector store.

```bash
python main.py --query
```

### API Command

Start the FastAPI server (serves REST API + React frontend).

```bash
python main.py --api
```

Access at `http://localhost:8000` — API docs at `http://localhost:8000/docs`.

### Test Command

Run a single test query and print ranked sources + LLM response.

```bash
python main.py --test-query "What are the capabilities of this system?"
```

## Configuration

### Environment Variables

```bash
NVIDIA_API_KEY=your_nvidia_api_key_here
EMBEDDING_PROVIDER=nvidia        
EMBEDDING_MODEL=baai/bge-m3
EMBEDDING_NORMALIZE=true
EMBEDDING_TIMEOUT=120.0
EMBEDDING_MAX_RETRIES=3
REDIS_URL=redis://localhost:6379/0
REDIS_PREFIX=rag
REDIS_DEFAULT_TTL_SECONDS=1800
REDIS_RETRIEVAL_TTL_SECONDS=1800
REDIS_LLM_TTL_SECONDS=1800
REDIS_EMBEDDING_TTL_SECONDS=604800

RETRIEVAL_MODE=hybrid 

USE_RERANKER=true
RERANKER_MODEL=nv-rerank-qa-mistral-4b:1
MIN_CHUNKS_TO_RERANK=8     
TOP_K_AFTER_RERANK=5

LLM_PROVIDER=openrouter
OPENROUTER_API_KEY=your_openrouter_api_key_here
LLM_MODEL=nvidia/nemotron-3-nano-30b-a3b:free
LLM_TEMPERATURE=0.7
LLM_MAX_TOKENS=1000

JUDGE_PROVIDER=cerebras 
CEREBRAS_API_KEY=your_cerebras_api_key_here

DATABASE_URL=postgresql://user:password@localhost:5432/rag_db
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=rag_db
POSTGRES_USER=your_db_user
POSTGRES_PASSWORD=your_db_password
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_ANON_KEY=your_supabase_anon_key
SUPABASE_SERVICE_ROLE_KEY=your_supabase_service_role_key
SUPABASE_STORAGE_BUCKET=documents
VECTOR_BACKEND=auto
QDRANT_URL=https://your-cluster.qdrant.io
QDRANT_API_KEY=your_qdrant_api_key
QDRANT_COLLECTION=chunk_embeddings
QDRANT_DISTANCE=cosine
QDRANT_TIMEOUT_SECONDS=10
```

### Command-line Flags

- `--device`: `cpu` or `cuda` for local embedding models.
- `--top-k`: Chunks to retrieve per query (default: 5).
- `--chunk-size`: Target token size per chunk (default: 600).
- `--chunk-overlap`: Token overlap between chunks (default: 90).
- `--strategy`: Chunking strategy (`fixed_size`, `semantic`, `sliding_window`, `sentence`, `paragraph`).
- `--vector-store-type`: FAISS index type (`flat`, `ivf`, `hnsw`).
- `--api-port`: API server port (default: 8000).
- `--loader-threads`: Loader threads in parallel mode (default: 8).
- `--incremental`: Keep existing data; only add new documents.

### Docker Compose

```bash
docker-compose up --build
```

Set API keys in `.env` before running. App available at `http://localhost:8000`.

### Render.com

Deploy via `render.yaml` (blueprint auto-configures DB, env vars, and disk):

- Uses `requirements-render.txt` (no PyTorch — fits Render free-tier RAM).
- `EMBEDDING_PROVIDER=nvidia` — no local model required.
- Persistent disk mounted at `/app/data` (7 GB) — preserves vector store and BM25 index across restarts.
- Set `ENABLE_CHUNK_ENRICHMENT=1` to enable background metadata enrichment (uses additional LLM tokens).

## How IntelliDocs Works

### Multi-Agent Orchestration

Every `POST /ask` request passes through `AgentOrchestrator.run()`, which coordinates a chain of specialized agents:

![Multi-Agent Orchestration](https://i.postimg.cc/g0YKMJqm/Multi-Agent.png)

**Guardrails:**

- Maximum 5 agent steps and configurable timeout per request.
- Multi-hop fan-out capped at `k=3` per sub-query to control latency.
- Context compression: cheap truncation first, costly LLM summarization only when still >20% over budget.

### Conditional Query Routing

`QueryPlanner` uses the LLM to classify the query (with a heuristic fallback when the LLM is unavailable). `ConditionalRouter` maps the classification to an execution path:

| Query Type | Route | Description |
|---|---|---|
| `trivial` | `DIRECT_LLM` | Greetings, simple yes/no — zero retrieval overhead |
| `factual` | `SINGLE_AGENT` | One hybrid retrieval → synthesize → validate |
| `summarization` | `SINGLE_AGENT` | Same path, `top_k` raised to 10 for broader coverage |
| `analytical` | `MULTI_AGENT` | Full pipeline with context aggregation |
| `comparative` | `MULTI_AGENT` | Multi-source retrieval + comparative synthesis |
| `multi_hop` | `MULTI_AGENT` | Parallel per-sub-query retrieval, fan-out capped at k=3 |
| avg score < 0.4 | `HUMAN_REVIEW` | Escalated to the pending review queue |

### Human-in-the-Loop Review

`Gatekeeper.should_escalate()` checks three conditions after retrieval:

1. **Sensitivity**: Pre-compiled regex detects password, salary, API-key, or prompt-injection patterns.
2. **Groundedness**: `ValidatorAgent` reports whether the answer is supported by the context.
3. **Confidence**: Average retrieval score below 0.4.

If any condition is met the answer is stored in the review queue. Human reviewers use the frontend `ReviewPanel` (or raw API calls) to approve or correct. Corrected answers are persisted and returned on future identical queries via `_lookup_approved_answer()` with a session freshness guard.

### Hybrid Search Pipeline

IntelliDocs retrieves documents through a four-stage hybrid pipeline:

1. **Dense Retrieval**: Query is embedded and searched via FAISS (up to `k×3` candidates).
2. **Sparse Retrieval**: Query is tokenized (stopword-filtered) and scored against the pre-built BM25 index.
3. **RRF Fusion**: Each candidate's rank from dense and sparse lists is combined: `score += 1/(60 + rank)`. Higher combined score = better match.
4. **Reranking**: If the candidate count exceeds `MIN_CHUNKS_TO_RERANK`, results are reranked by NVIDIA's `nv-rerank-qa-mistral-4b:1` model and trimmed to `top_k`.

### Parallel Processing Pipeline

Three concurrent stages minimize end-to-end ingestion time:

1. **Loading (I/O Bound)**: `ThreadPoolExecutor` loads files in parallel with format-specific handlers — PyMuPDF for PDFs (OCR fallback via Tesseract), python-docx for Word, openpyxl for Excel, and UTF-8 text for code/markup files.
2. **Chunking (CPU Bound)**: `ProcessPoolExecutor` bypasses the GIL to chunk documents in parallel. Child processes re-import `TextChunker` fresh to avoid stale bytecode.
3. **Embedding (API Bound)**: Up to 6 `ThreadPoolExecutor` workers send concurrent API calls. Results are collected and batch-written to FAISS. The BM25 index is built from all chunk texts after embedding completes.

Auto-detection switches to the parallel pipeline at 11+ documents. Small datasets use the sequential path to avoid process-spawn overhead.

### Extension-Aware Chunking

`TextChunker` routes each file to the optimal strategy based on extension:

| Extension | Strategy | Description |
|---|---|---|
| `.pdf`, `.docx`, `.txt` | Semantic | Sentence-boundary splits, 15% overlap, tables kept atomic |
| `.md` | Header-based | Respects heading hierarchy; fixed-size fallback for oversized sections |
| `.html`, `.htm` | DOM-aware | BeautifulSoup extraction with header hierarchy tracking |
| `.py`, `.js`, `.java`, `.go`, `.rs`, etc. | Code-structure | AST-based function and class boundary detection |
| `.csv`, `.tsv` | Row-group | Auto-detects Q&A / document / bulk structure; vectorized parsing |
| `.json`, `.yaml`, `.yml` | Key-path | Flattens nested structures into dot-path key-value chunks |
| `.ipynb` | Cell-aware | Separates code cells from markdown cells |

### Three-Level Caching

Retrieval and LLM caches are session-scoped, preventing any cross-user collisions. Cache keys are SHA-256 hashed. The background `CascadeService` `CleanupWorker` evicts stale entries and expired guest sessions on a 60-second cycle.

### Metadata Enrichment

When `ENABLE_CHUNK_ENRICHMENT=1`, a background thread runs after each ingestion:

1. Fetches chunks with `enrichment_status='pending'` from the database in configurable batch sizes.
2. Uses **TF-based keyword extraction** (no LLM required) for all chunks.
3. Batches multiple chunk summaries into a **single LLM call** using a structured prompt (separated by `---`).
4. Generates **hypothetical questions** per chunk (HyDE-style) to improve query-chunk semantic alignment.
5. Updates the database with summaries, keywords, and questions in a single commit per batch.

The enricher is **resumable** — restarting the server picks up from where it left off without re-processing already-enriched chunks. A semaphore caps concurrent enrichment workers at 3 to prevent LLM quota exhaustion.

### Session Isolation

Every chat — whether guest or logged-in — gets its own fully isolated `session_id`. No two chats share a vector index, chunk store, or document directory.

**Per-chat on-disk layout:**

```
data/sessions/{session_id}/
    documents/                       ← Only this chat's uploaded files
    chunks/                          ← Chunk metadata JSON
    │   chunks_metadata_prev.json    ← Rollback snapshot (restored on failed re-ingest)
    vector_store/                    ← FAISS index + BM25 index
                                       (or Qdrant collection scoped by session_id)
```
**Shared singletons:** LLM, embedding service, and reranker are initialized once at server startup and safely reused across all sessions (read-only operations). The `CleanupWorker` runs every 60 seconds, processing deletion jobs for cascade-deleted chats and evicting TTL-expired guest session data.

## Architecture

### Agent Layer

| Component | File | Responsibility |
|---|---|---|
| **AgentOrchestrator** | `src/agents/Orchestrator.py` | Top-level coordinator; enforces step cap, timeout, caching, and human-approved answer lookup |
| **QueryPlanner** | `src/agents/Planner.py` | LLM-based query classification (6 types) + multi-hop decomposition; heuristic fallback |
| **ConditionalRouter** | `src/agents/Router.py` | Routes plans to DIRECT_LLM / SINGLE_AGENT / MULTI_AGENT / HUMAN_REVIEW |
| **RetrieverAgent** | `src/agents/RetrieverAgent.py` | Agent wrapper around `RAGRetriever`; checks retrieval cache first |
| **SynthesizerAgent** | `src/agents/SynthesizerAgent.py` | Builds final prompt (system + context + query) and calls LLM |
| **ValidatorAgent** | `src/agents/ValidatorAgent.py` | Secondary LLM call to check answer groundedness; catches hallucinations |
| **Gatekeeper / HumanValidation** | `src/agents/HumanValidation.py` | Sensitivity + groundedness + confidence checks; review queue; approve/correct workflow |
| **Tools** | `src/agents/Tools.py` | Agent tool definitions (search, summarise, etc.) |
| **BaseAgent** | `src/agents/BaseAgent.py` | Abstract base with `AgentTask` / `AgentResult` dataclasses |

### Core Modules

| Component | File | Responsibility |
|---|---|---|
| **Loader** | `src/modules/Loader.py` | Multi-format parsing, PDF streaming, scanned PDF detection, OCR fallback |
| **Chunking** | `src/modules/Chunking.py` | 8-strategy extension-aware chunker; 50K-entry token cache |
| **DocumentParser** | `src/modules/DocumentParser.py` | Structure analysis with parse result caching |
| **Embeddings** | `src/modules/Embeddings.py` | Provider-agnostic embedding service (NVIDIA/Gemini/HF/local); exponential backoff retry |
| **VectorStore** | `src/modules/VectorStore.py` | FAISS (Flat/IVF/HNSW); O(1) reverse ID map; atomic file saves; metadata archive |
| **QdrantStore** | `src/modules/QdrantStore.py` | Qdrant session-scoped store; upsert, search, delete-by-session; FAISS fallback |
| **Retriever** | `src/modules/Retriever.py` | `BM25Retriever` + `NvidiaReranker` + `RAGRetriever` (dense+sparse+RRF+rerank) |
| **QueryCache** | `src/modules/QueryCache.py` | Session-scoped retrieval cache + LLM response cache (SQLite) |
| **QueryGeneration** | `src/modules/QueryGeneration.py` | Legacy `QueryHandler` for global index / CLI path |
| **MetadataEnricher** | `src/modules/MetadataEnricher.py` | Async background enricher: TF keywords + batched LLM summaries + HyDE questions |
| **LLM** | `src/modules/LLM.py` | Provider-agnostic LLM; strips `<think>` reasoning blocks automatically |
| **ParallelPipeline** | `src/modules/ParallelPipeline.py` | Three-stage streaming pipeline; standalone helpers for sequential reuse |
| **Evaluator** | `src/evaluation/` | Per-query metrics: precision, recall, latency, route; rolling summary API |

### Backend Services

| Component | File | Responsibility |
|---|---|---|
| **SessionService** | `backend/services/session_service.py` | Session creation, isolation, directory layout, status updates, lifecycle |
| **ChatService** | `backend/services/chat_service.py` | Chat/message/document CRUD; quota enforcement (guest 5, per-chat 15, account 40) |
| **CascadeService** | `backend/services/cascade_service.py` | Atomic cascade deletes + `CleanupWorker` (60s cycle, TTL sweep, job queue) |
| **StorageService** | `backend/services/storage_service.py` | Storage abstraction: `LocalStorageService` / `SupabaseStorageService` |
| **RagServiceSession** | `backend/services/rag_service_session.py` | Session-scoped `QueryHandler`; shared LLM/reranker singletons |
| **SupabaseAuth** | `backend/auth/supabase_auth.py` | JWT verification; anonymous guest passthrough |
| **IngestSession** | `backend/rag/IngestSession.py` | Session ingestion: parallel load + retry embeddings + BM25 + FAISS/Qdrant + progress tracking |
| **Backend API** | `backend/api/routes.py` | All FastAPI routes: session, chat, guest, ask/stream, review, eval, stress-test |
| **Frontend** | `frontend/` | React 19 + Vite SPA; Supabase auth; multi-chat sidebar; multi-file upload; progress tracking |

## API Reference

### Document / Session Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/upload` | Upload a file to a session/chat; returns `session_id` + `chat_id` |
| `GET` | `/api/status/{chat_id}` | Poll processing status; includes `document_progress` (total / processed / ready / failed) |
| `POST` | `/api/process/{chat_id}` | Trigger background ingestion of all uploaded files |
| `POST` | `/api/ask` | Non-streamed answer (legacy) |
| `POST` | `/api/ask/stream` | **Streamed** answer via SSE; post-stream grounding validation events |
| `GET` | `/api/health` | Deep health check (DB, vector store, LLM status) |

### Chat Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/chats` | Create a new chat (logged-in or guest) |
| `GET` | `/api/chats` | List all chats for the current user |
| `PATCH` | `/api/chats/{chat_id}` | Rename a chat (version-guarded) |
| `DELETE` | `/api/chats/{chat_id}` | Cascade-delete a chat (messages, documents, vector data) |
| `GET` | `/api/chats/{chat_id}/messages` | Paginated message history |
| `GET` | `/api/chats/{chat_id}/documents` | List documents in a chat with status |
| `POST` | `/api/chats/{chat_id}/clear-files` | Remove all documents from a chat without deleting the chat |
| `POST` | `/api/chats/{chat_id}/ask` | Non-streamed chat query with history |
| `POST` | `/api/chats/{chat_id}/ask/stream` | **Streamed** chat query with history via SSE |

### Guest Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/guest/session` | Create a guest session UUID |
| `DELETE` | `/api/guest/session/{session_id}` | Clean up guest session data |

### Review & Evaluation Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/reviews/pending` | List queries pending human review |
| `POST` | `/api/reviews/{id}/approve` | Approve the AI-generated answer |
| `POST` | `/api/reviews/{id}/correct` | Submit a corrected answer |
| `GET` | `/api/eval/summary` | Rolling evaluation metrics (precision, recall, latency, route distribution) |
| `POST` | `/api/stress-test` | Adversarial stress tests (requires `ENABLE_STRESS_TEST=1`) |

## Requirements

- Python 3.10+
- Node.js 20+
- PostgreSQL (SQLite fallback available for local development)
- NVIDIA API Key — required for BGE-M3 embeddings and optional `nv-rerank-qa-mistral-4b:1` reranking
- OpenRouter API Key — for LLM responses
- Cerebras API Key — for high-speed post-stream validation (`ValidatorAgent`)
- Supabase Project — for Google OAuth + email/password authentication (optional; app works in guest mode without it)
- Qdrant Cloud or self-hosted — for managed vector storage (optional; defaults to local FAISS)
- Docker (optional — for containerized deployment)
- Tesseract (optional — for OCR on scanned PDFs)

## Troubleshooting

**PostgreSQL Connection Errors**

Ensure the database is running and credentials in `.env` match. The system falls back to SQLite automatically if `DATABASE_URL` is unset.

**NVIDIA API Rate Limits or Timeouts**

Increase `EMBEDDING_TIMEOUT` and `EMBEDDING_MAX_RETRIES` in `.env`. The embedding service retries with exponential backoff. For bulk ingestion, the concurrent embedding path (6 workers) may hit rate limits — reduce by setting `EMBED_WORKERS` in `IngestSession.py`.

**BM25 Index Not Loading**

If hybrid search falls back to dense-only after a build, check that `data/vector_store/bm25_index.pkl` was created. A missing file means BM25 build was skipped — usually indicates no text was extracted from documents. Check logs for chunking errors.

**Scanned PDFs Produce No Text**

Install OCR dependencies: `pip install pytesseract Pillow`. Install the Tesseract binary for your OS. IntelliDocs will automatically detect scanned pages and apply OCR.

**OOM Errors During Embedding**

For GPU models, reduce `EMBEDDING_BATCH_SIZE` or set `--device cpu`. For the NVIDIA API path, the pipeline automatically halves batch size and retries up to 3 times on OOM.

**Low Disk Space**

Run the manual cleanup utility:

```bash
python -m backend.services.cleanup_storage
```

This shows a storage breakdown by session and lets you delete old data interactively.

**Session Processing Stuck**

If a session stays in `processing` for more than 2 minutes, check `logs/` for embedding API errors. The `CascadeService` `CleanupWorker` will sweep TTL-expired guest sessions. For stuck logged-in sessions, use the `/api/health` endpoint to inspect system state or manually update the session status in the database.

**Enrichment Not Running**

Background enrichment requires `ENABLE_CHUNK_ENRICHMENT=1` and a working LLM configuration. Check that `OPENROUTER_API_KEY` or `GEMINI_API_KEY` is set. Enrichment status per chunk can be queried from the `chunks` table: `SELECT enrichment_status, COUNT(*) FROM chunks GROUP BY enrichment_status;`

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
