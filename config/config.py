"""Central configuration with environment overrides."""

import os
from pathlib import Path
from urllib.parse import urlparse, unquote
from dotenv import load_dotenv

_env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=_env_path)


def _env(key: str, default: str) -> str:
    """Read environment variable with default fallback."""
    value = os.getenv(key)
    return value if value is not None and value != "" else default


# Hugging Face token (for LLM if needed, not used for embeddings or reranking)
HF_TOKEN = _env("HF_TOKEN", "")
HF_INFERENCE_PROVIDER = _env("HF_INFERENCE_PROVIDER", "hf-inference")

# Google AI Studio configuration
GEMINI_API_KEY = _env("GEMINI_API_KEY", "")

# NVIDIA API configuration
NVIDIA_API_KEY = _env("NVIDIA_API_KEY", "")

# OpenRouter configuration
OPENROUTER_API_KEY = _env("OPENROUTER_API_KEY", "")
OPENROUTER_SITE_URL = _env("OPENROUTER_SITE_URL", "")
OPENROUTER_SITE_NAME = _env("OPENROUTER_SITE_NAME", "")

# Embeddings configuration
# Using LM Studio for local BGE-M3 embeddings
# For HuggingFace Inference, use: EMBEDDING_PROVIDER=hf-inference
# For Gemini embeddings, use: EMBEDDING_PROVIDER=gemini
# For local transformers, use: EMBEDDING_PROVIDER=local
EMBEDDING_PROVIDER = _env("EMBEDDING_PROVIDER", "nvidia")
EMBEDDING_MODEL = _env("EMBEDDING_MODEL", "baai/bge-m3")
EMBEDDING_NORMALIZE = _env("EMBEDDING_NORMALIZE", "true").lower() == "true"
EMBEDDING_TIMEOUT = float(_env("EMBEDDING_TIMEOUT", "120.0"))
EMBEDDING_MAX_RETRIES = int(_env("EMBEDDING_MAX_RETRIES", "3"))

# LM Studio configuration (for local LM Studio server)
LM_STUDIO_BASE_URL = _env("LM_STUDIO_BASE_URL", "http://127.0.0.1:1234/v1")
LM_STUDIO_API_KEY = _env("LM_STUDIO_API_KEY", "")

# Hybrid Retrieval configuration (Vespa-style M3 fusion)
RETRIEVAL_MODE = _env("RETRIEVAL_MODE", "hybrid")  # "hybrid" or "dense"
FUSION_STRATEGY = _env("FUSION_STRATEGY", "rrf")  # "rrf" or "vespa"
HYBRID_DENSE_WEIGHT = float(_env("HYBRID_DENSE_WEIGHT", "0.4"))
HYBRID_LEXICAL_WEIGHT = float(_env("HYBRID_LEXICAL_WEIGHT", "0.2"))
HYBRID_COLBERT_WEIGHT = float(_env("HYBRID_COLBERT_WEIGHT", "0.4"))
RRF_K = int(_env("RRF_K", "60"))  # RRF constant (higher = more conservative)

# Reranker configuration
# RERANKER_PROVIDER: "lm-studio" (calls LM Studio /v1/rerank) or "local" (in-process CrossEncoder)
USE_RERANKER = _env("USE_RERANKER", "true").lower() == "true"
RERANKER_PROVIDER = _env("RERANKER_PROVIDER", "lm-studio")
RERANKER_MODEL = _env("RERANKER_MODEL", "gpustack/text-embedding-bge-reranker-v2-m3")
MIN_CHUNKS_TO_RERANK = int(_env("MIN_CHUNKS_TO_RERANK", "8"))
TOP_K_AFTER_RERANK = int(_env("TOP_K_AFTER_RERANK", "5"))

# Gemini-specific embedding configuration
EMBEDDING_TASK_TYPE = _env("EMBEDDING_TASK_TYPE", "RETRIEVAL_DOCUMENT")
EMBEDDING_DIMENSION = int(_env("EMBEDDING_DIMENSION", "1024"))

# LLM configuration (now using Google Gemini)
LLM_PROVIDER = _env("LLM_PROVIDER", "openrouter")
LLM_MODEL = _env("LLM_MODEL", "openai/gpt-oss-120b:free")
LLM_TEMPERATURE = float(_env("LLM_TEMPERATURE", "0.7"))
LLM_MAX_TOKENS = int(_env("LLM_MAX_TOKENS", "1000"))

# Judge LLM configuration (for evaluation/validation)
JUDGE_PROVIDER = _env("JUDGE_PROVIDER", "cerebras")
JUDGE_MODEL = _env("JUDGE_MODEL", "llama3.1-8b")
CEREBRAS_API_KEY = _env("CEREBRAS_API_KEY", "")

# PostgreSQL Database configuration
DATABASE_URL = _env("DATABASE_URL", "")


def _parse_database_url(url: str) -> dict:
    """Parse DATABASE_URL into psycopg2-compatible parts."""
    if not url:
        return {}
    parsed = urlparse(url)
    if parsed.scheme not in {"postgres", "postgresql"}:
        return {}
    return {
        "host": parsed.hostname or "",
        "port": parsed.port or 5432,
        "dbname": (parsed.path or "/").lstrip("/") or "postgres",
        "user": unquote(parsed.username) if parsed.username else "",
        "password": unquote(parsed.password) if parsed.password else "",
    }


_db_from_url = _parse_database_url(DATABASE_URL)
POSTGRES_HOST = _env("POSTGRES_HOST", _db_from_url.get("host", "localhost"))
POSTGRES_PORT = int(_env("POSTGRES_PORT", str(_db_from_url.get("port", 5432))))
POSTGRES_DB = _env("POSTGRES_DB", _db_from_url.get("dbname", "rag_db"))
POSTGRES_USER = _env("POSTGRES_USER", _db_from_url.get("user", "postgres"))
POSTGRES_PASSWORD = _env("POSTGRES_PASSWORD", _db_from_url.get("password", ""))
POSTGRES_SSLMODE = _env("POSTGRES_SSLMODE", "require" if "supabase.co" in POSTGRES_HOST else "prefer")

QDRANT_URL = _env("QDRANT_URL", "")
QDRANT_API_KEY = _env("QDRANT_API_KEY", "")
QDRANT_COLLECTION = _env("QDRANT_COLLECTION", "chunk_embeddings")
QDRANT_DISTANCE = _env("QDRANT_DISTANCE", "cosine").lower()  # "cosine" | "dot" | "euclid"
QDRANT_TIMEOUT_SECONDS = float(_env("QDRANT_TIMEOUT_SECONDS", "10"))

# Vector backend selection
# Vector backend selection
_raw_vector_backend = _env("VECTOR_BACKEND", "auto").lower()  # "auto" | "faiss" | "qdrant"
if _raw_vector_backend == "auto":
    # Prefer qdrant for managed/remote DBs only when qdrant is configured.
    _looks_remote = POSTGRES_HOST not in {"localhost", "127.0.0.1", ""}
    VECTOR_BACKEND = "qdrant" if (_looks_remote and QDRANT_URL) else "faiss"
else:
    VECTOR_BACKEND = _raw_vector_backend if _raw_vector_backend in {"faiss", "qdrant"} else "faiss"

if VECTOR_BACKEND == "qdrant" and not QDRANT_URL:
    VECTOR_BACKEND = "faiss"

# Supabase auth/storage configuration
SUPABASE_URL = _env("SUPABASE_URL", "")
SUPABASE_ANON_KEY = _env("SUPABASE_ANON_KEY", "")
SUPABASE_SERVICE_ROLE_KEY = _env("SUPABASE_SERVICE_ROLE_KEY", "")
SUPABASE_STORAGE_BUCKET = _env("SUPABASE_STORAGE_BUCKET", "rag-documents")

# API auth behavior
_raw_auth_required = _env("AUTH_REQUIRED", "auto").lower()  # "auto" | "true" | "false"
if _raw_auth_required == "auto":
    AUTH_REQUIRED = bool(SUPABASE_URL and (SUPABASE_ANON_KEY or SUPABASE_SERVICE_ROLE_KEY))
else:
    AUTH_REQUIRED = _raw_auth_required == "true"

# CORS configuration
CORS_ALLOWED_ORIGINS = [
    origin.strip()
    for origin in _env("CORS_ALLOWED_ORIGINS", "*").split(",")
    if origin.strip()
]



# Request protection
RATE_LIMIT_REQUESTS = int(_env("RATE_LIMIT_REQUESTS", "5"))
RATE_LIMIT_WINDOW_SECONDS = int(_env("RATE_LIMIT_WINDOW_SECONDS", "60"))


# Circuit breaker
CIRCUIT_BREAKER_FAILURE_THRESHOLD = int(_env("CIRCUIT_BREAKER_FAILURE_THRESHOLD", "10"))
CIRCUIT_BREAKER_RECOVERY_SECONDS = int(_env("CIRCUIT_BREAKER_RECOVERY_SECONDS", "120"))

def postgres_connect_kwargs(connect_timeout: int = 10) -> dict:
    """Shared psycopg2 kwargs (Supabase-friendly SSL defaults included)."""
    kwargs = {
        "host": POSTGRES_HOST,
        "port": POSTGRES_PORT,
        "database": POSTGRES_DB,
        "user": POSTGRES_USER,
        "password": POSTGRES_PASSWORD,
        "connect_timeout": connect_timeout,
    }
    if POSTGRES_SSLMODE:
        kwargs["sslmode"] = POSTGRES_SSLMODE
    return kwargs
