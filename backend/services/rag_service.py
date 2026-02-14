"""Central RAG service (retriever + generator) with singleton initialization."""

from typing import Optional
from pathlib import Path
from src.modules.Embeddings import create_embedding_service
from src.modules.VectorStore import FAISSVectorStore
from src.modules.Retriever import RAGRetriever, LMStudioReranker
from src.modules.QueryGeneration import QueryHandler, QueryResult
from src.modules.LLM import create_llm
from src.utils.Logger import get_logger
from config.config import (
    LLM_PROVIDER,
    LLM_MODEL,
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS,
    HF_TOKEN,
    HF_INFERENCE_PROVIDER,
    GEMINI_API_KEY,
    EMBEDDING_PROVIDER,
    EMBEDDING_MODEL,
    EMBEDDING_NORMALIZE,
    EMBEDDING_TASK_TYPE,
    EMBEDDING_DIMENSION,
    USE_RERANKER,
    RERANKER_MODEL,
    MIN_CHUNKS_TO_RERANK,
    TOP_K_AFTER_RERANK,
    LM_STUDIO_BASE_URL,
)
import json

logger = get_logger(__name__)

_query_handler: Optional[QueryHandler] = None


def _create_reranker():
    """Create LM Studio reranker for local inference."""
    if not USE_RERANKER:
        return None
    
    return LMStudioReranker(
        base_url=LM_STUDIO_BASE_URL,
        model=RERANKER_MODEL,
        min_chunks_to_rerank=MIN_CHUNKS_TO_RERANK,
        top_k_after_rerank=TOP_K_AFTER_RERANK
    )


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
    else:
        kwargs["device"] = "cpu"

    return kwargs


def _load_chunks_metadata(chunks_dir: str) -> dict:
    chunks_metadata = {}
    chunks_path = Path(chunks_dir)

    if not chunks_path.exists():
        logger.warning(f"Chunks directory not found: {chunks_dir}")
        return chunks_metadata

    for chunk_file in chunks_path.glob("*_chunks.json"):
        try:
            with open(chunk_file, "r") as f:
                data = json.load(f)
                if isinstance(data, list):
                    for chunk in data:
                        chunks_metadata[chunk.get("id")] = chunk
                elif isinstance(data, dict):
                    chunks_metadata.update(data)
        except Exception as e:
            logger.error(f"Error loading {chunk_file}: {e}")

    return chunks_metadata


def get_query_handler() -> QueryHandler:
    global _query_handler

    if _query_handler is not None:
        return _query_handler

    logger.info("Initializing RAG service...")

    embedding_service = create_embedding_service(
        model_type=EMBEDDING_PROVIDER,
        **_build_embedding_kwargs()
    )

    vector_store = FAISSVectorStore(
        dimension=embedding_service.model.dimension,
        index_type="flat"
    )
    vector_store.load("data/vector_store")

    chunks_metadata = _load_chunks_metadata("data/chunks")

    # Initialize reranker if enabled
    reranker = _create_reranker()

    retriever = RAGRetriever(
        vector_store=vector_store,
        embedding_service=embedding_service,
        chunks=chunks_metadata,
        reranker=reranker,
        use_reranker=USE_RERANKER
    )

    llm = None
    try:
        # Set the correct API key based on provider
        api_key = GEMINI_API_KEY if LLM_PROVIDER.lower() in ["gemini", "google", "google-ai"] else HF_TOKEN
        
        llm = create_llm(
            provider=LLM_PROVIDER,
            model_name=LLM_MODEL,
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS,
            api_key=api_key
        )
        logger.info(f"LLM initialized: {llm.model_name}")
    except Exception as e:
        logger.warning(f"LLM initialization failed: {e}")

    _query_handler = QueryHandler(
        retriever=retriever,
        embedding_service=embedding_service,
        llm=llm,
        top_k=5
    )

    logger.info("RAG service initialized")
    return _query_handler


def ask_rag(
    question: str,
    top_k: int = 5,
    system_prompt: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None
) -> QueryResult:
    handler = get_query_handler()
    return handler.process_query_with_response(
        query=question,
        top_k=top_k,
        system_prompt=system_prompt,
        temperature=temperature if temperature is not None else LLM_TEMPERATURE,
        max_tokens=max_tokens or LLM_MAX_TOKENS
    )
