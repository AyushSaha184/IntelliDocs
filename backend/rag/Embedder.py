"""Embedding helper wrapper."""

from typing import List
from src.modules.Embeddings import create_embedding_service
from config.config import (
    EMBEDDING_PROVIDER,
    EMBEDDING_MODEL,
    EMBEDDING_NORMALIZE,
    HF_TOKEN,
    HF_INFERENCE_PROVIDER,
    GEMINI_API_KEY,
    NVIDIA_API_KEY,
    EMBEDDING_TASK_TYPE,
    EMBEDDING_DIMENSION,
    LM_STUDIO_BASE_URL,
    LM_STUDIO_API_KEY,
)

_embedding_service = None


def _build_embedding_kwargs() -> dict:
    kwargs = {
        "model_name": EMBEDDING_MODEL,
        "normalize_embeddings": EMBEDDING_NORMALIZE
    }

    if EMBEDDING_PROVIDER.lower() in ["hf", "hf-inference", "huggingface"]:
        kwargs.update({
            "api_key": HF_TOKEN,
            "provider": HF_INFERENCE_PROVIDER
        })
    elif EMBEDDING_PROVIDER.lower() in ["gemini", "google", "google-gemini"]:
        kwargs.update({
            "api_key": GEMINI_API_KEY,
            "task_type": EMBEDDING_TASK_TYPE,
            "output_dimensionality": EMBEDDING_DIMENSION
        })
    elif EMBEDDING_PROVIDER.lower() in ["nvidia", "nvidia-build", "nvidia-api"]:
        kwargs.update({
            "api_key": NVIDIA_API_KEY,
        })
    elif EMBEDDING_PROVIDER.lower() in ["lm-studio", "lmstudio", "openai-compatible"]:
        kwargs.update({
            "base_url": LM_STUDIO_BASE_URL,
            "api_key": LM_STUDIO_API_KEY if LM_STUDIO_API_KEY else None
        })
    else:
        kwargs["device"] = "cpu"

    return kwargs


def embed_texts(texts: List[str]):
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = create_embedding_service(
            model_type=EMBEDDING_PROVIDER,
            **_build_embedding_kwargs()
        )
    return _embedding_service.embed_batch(texts)
