"""Vector store helper wrapper — NVIDIA BGE-M3 embeddings."""

from src.modules.Embeddings import create_embedding_service
from src.modules.VectorStore import FAISSVectorStore
from config.config import (
    NVIDIA_API_KEY,
    EMBEDDING_MODEL,
    EMBEDDING_NORMALIZE,
)

_vector_store = None


def get_vector_store():
    global _vector_store
    if _vector_store is None:
        embedding_service = create_embedding_service(
            model_type="nvidia",
            model_name=EMBEDDING_MODEL,
            normalize_embeddings=EMBEDDING_NORMALIZE,
            api_key=NVIDIA_API_KEY,
        )
        _vector_store = FAISSVectorStore(
            dimension=embedding_service.model.dimension,
            index_type="flat"
        )
        _vector_store.load("data/vector_store")
    return _vector_store
