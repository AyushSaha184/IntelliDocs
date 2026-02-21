"""Vector store helper wrapper."""

from src.modules.Embeddings import create_embedding_service
from src.modules.VectorStore import FAISSVectorStore
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

_vector_store = None


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


def get_vector_store():
    global _vector_store
    if _vector_store is None:
        embedding_service = create_embedding_service(
            model_type=EMBEDDING_PROVIDER,
            **_build_embedding_kwargs()
        )
        _vector_store = FAISSVectorStore(
            dimension=embedding_service.model.dimension,
            index_type="flat"
        )
        _vector_store.load("data/vector_store")
    return _vector_store
