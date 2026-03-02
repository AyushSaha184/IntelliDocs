"""Shared singleton providers for LLM, Embedding, and Reranker instances.

Extracted from rag_service_session.py so that agents, enrichers, and services
can all import these thread-safe singletons without cross-layer imports.
"""

import threading
from typing import Optional
from src.modules.Embeddings import create_embedding_service
from src.modules.Retriever import NvidiaReranker
from src.modules.LLM import BaseLLM, create_llm
from src.utils.Logger import get_logger
from config.config import (
    LLM_PROVIDER,
    LLM_MODEL,
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS,
    HF_TOKEN,
    GEMINI_API_KEY,
    EMBEDDING_PROVIDER,
    EMBEDDING_MODEL,
    EMBEDDING_NORMALIZE,
    EMBEDDING_TASK_TYPE,
    EMBEDDING_DIMENSION,
    EMBEDDING_TIMEOUT,
    EMBEDDING_MAX_RETRIES,
    NVIDIA_API_KEY,
    USE_RERANKER,
    RERANKER_MODEL,
    MIN_CHUNKS_TO_RERANK,
    TOP_K_AFTER_RERANK,
    LM_STUDIO_BASE_URL,
    LM_STUDIO_API_KEY,
    HF_INFERENCE_PROVIDER,
)

logger = get_logger(__name__)

# ── Thread-safe singletons ──────────────────────────────────────────────

_shared_llm: Optional[BaseLLM] = None
_llm_lock = threading.Lock()

_shared_reranker = None
_reranker_lock = threading.Lock()

_shared_embedding_service = None
_embedding_lock = threading.Lock()


# ── Embedding kwargs builder ────────────────────────────────────────────

def _build_embedding_kwargs() -> dict:
    """Build embedding service kwargs based on provider."""
    kwargs = {
        "model_name": EMBEDDING_MODEL,
        "normalize_embeddings": EMBEDDING_NORMALIZE,
    }

    provider = EMBEDDING_PROVIDER.lower()

    if provider in ["hf", "hf-inference", "huggingface"]:
        kwargs.update({
            "api_key": HF_TOKEN,
            "provider": HF_INFERENCE_PROVIDER,
        })
    elif provider in ["gemini", "google", "google-gemini"]:
        kwargs.update({
            "api_key": GEMINI_API_KEY,
            "task_type": EMBEDDING_TASK_TYPE,
            "output_dimensionality": EMBEDDING_DIMENSION,
        })
    elif provider in ["nvidia", "nvidia-build", "nvidia-api"]:
        kwargs.update({
            "api_key": NVIDIA_API_KEY,
            "timeout": EMBEDDING_TIMEOUT,
            "max_retries": EMBEDDING_MAX_RETRIES,
        })
    elif provider in ["lm-studio", "lmstudio", "openai-compatible"]:
        kwargs.update({
            "base_url": LM_STUDIO_BASE_URL,
            "api_key": LM_STUDIO_API_KEY if LM_STUDIO_API_KEY else None,
            "timeout": EMBEDDING_TIMEOUT,
            "max_retries": EMBEDDING_MAX_RETRIES,
        })
    else:
        # Local models (transformers) need device
        kwargs["device"] = "cpu"

    return kwargs


# ── Public singleton accessors ──────────────────────────────────────────

def get_shared_llm() -> Optional[BaseLLM]:
    """Get or create shared LLM instance (thread-safe singleton).

    Returns:
        Shared BaseLLM instance, or None if initialization fails.
    """
    global _shared_llm

    with _llm_lock:
        if _shared_llm is not None:
            return _shared_llm

        try:
            api_key = (
                GEMINI_API_KEY
                if LLM_PROVIDER.lower() in ["gemini", "google", "google-ai"]
                else HF_TOKEN
            )

            _shared_llm = create_llm(
                provider=LLM_PROVIDER,
                model_name=LLM_MODEL,
                temperature=LLM_TEMPERATURE,
                max_tokens=LLM_MAX_TOKENS,
                api_key=api_key,
            )
            logger.info(f"Shared LLM initialized: {_shared_llm.model_name}")
        except Exception as e:
            logger.warning(f"LLM initialization failed: {e}")

        return _shared_llm


def get_shared_reranker():
    """Get or create shared NVIDIA reranker instance (thread-safe).

    Returns:
        NvidiaReranker instance, or None if reranking is disabled.
    """
    global _shared_reranker

    if not USE_RERANKER:
        return None

    if _shared_reranker is None:
        with _reranker_lock:
            if _shared_reranker is None:
                _shared_reranker = NvidiaReranker(
                    model=RERANKER_MODEL,
                    min_chunks_to_rerank=MIN_CHUNKS_TO_RERANK,
                    top_k_after_rerank=TOP_K_AFTER_RERANK,
                )
                logger.info(f"Shared reranker initialized: {RERANKER_MODEL}")

    return _shared_reranker


def get_shared_embedding_service():
    """Get or create shared embedding service instance (thread-safe singleton).

    Returns:
        Shared EmbeddingService instance.

    Raises:
        Exception: If initialization fails.
    """
    global _shared_embedding_service

    with _embedding_lock:
        if _shared_embedding_service is not None:
            return _shared_embedding_service

        try:
            _shared_embedding_service = create_embedding_service(
                model_type=EMBEDDING_PROVIDER,
                **_build_embedding_kwargs(),
            )
            logger.info("Shared EmbeddingService initialized")
        except Exception as e:
            logger.error(f"EmbeddingService initialization failed: {e}")
            raise

        return _shared_embedding_service
