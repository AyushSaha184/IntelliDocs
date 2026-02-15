"""Central configuration with environment overrides."""

import os
from pathlib import Path
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

# OpenRouter configuration
OPENROUTER_API_KEY = _env("OPENROUTER_API_KEY", "")
OPENROUTER_SITE_URL = _env("OPENROUTER_SITE_URL", "")
OPENROUTER_SITE_NAME = _env("OPENROUTER_SITE_NAME", "")

# Embeddings configuration
# Using LM Studio for local BGE-M3 embeddings
# For HuggingFace Inference, use: EMBEDDING_PROVIDER=hf-inference
# For Gemini embeddings, use: EMBEDDING_PROVIDER=gemini
# For local transformers, use: EMBEDDING_PROVIDER=local
EMBEDDING_PROVIDER = _env("EMBEDDING_PROVIDER", "lm-studio")
EMBEDDING_MODEL = _env("EMBEDDING_MODEL", "text-embedding-bge-m3")
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
EMBEDDING_DIMENSION = int(_env("EMBEDDING_DIMENSION", "768"))

# LLM configuration (now using Google Gemini)
LLM_PROVIDER = _env("LLM_PROVIDER", "gemini")
LLM_MODEL = _env("LLM_MODEL", "gemini-2.5-flash")
LLM_TEMPERATURE = float(_env("LLM_TEMPERATURE", "0.7"))
LLM_MAX_TOKENS = int(_env("LLM_MAX_TOKENS", "1000"))

# PostgreSQL Database configuration
POSTGRES_HOST = _env("POSTGRES_HOST", "localhost")
POSTGRES_PORT = int(_env("POSTGRES_PORT", "5432"))
POSTGRES_DB = _env("POSTGRES_DB", "rag_db")
POSTGRES_USER = _env("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD = _env("POSTGRES_PASSWORD", "")
