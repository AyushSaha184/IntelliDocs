"""Session-aware RAG service with isolated user sessions."""

from typing import Optional, Dict, Any
from pathlib import Path
from src.modules.Embeddings import create_embedding_service
from src.modules.VectorStore import FAISSVectorStore
from src.modules.Retriever import RAGRetriever, NvidiaReranker, BM25Retriever
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
    NVIDIA_API_KEY,
    EMBEDDING_PROVIDER,
    EMBEDDING_MODEL,
    EMBEDDING_NORMALIZE,
    EMBEDDING_TASK_TYPE,
    EMBEDDING_DIMENSION,
    EMBEDDING_TIMEOUT,
    EMBEDDING_MAX_RETRIES,
    USE_RERANKER,
    RERANKER_MODEL,
    MIN_CHUNKS_TO_RERANK,
    TOP_K_AFTER_RERANK,
    LM_STUDIO_BASE_URL,
    LM_STUDIO_API_KEY,
)
import threading

logger = get_logger(__name__)

# Thread-safe storage for session-specific query handlers
_session_handlers: Dict[str, QueryHandler] = {}
_handlers_lock = threading.Lock()

# Shared LLM instance (expensive to initialize)
_shared_llm = None
_llm_lock = threading.Lock()

# Shared reranker instance
_shared_reranker = None
_reranker_lock = threading.Lock()

# Shared embedding service instance
_shared_embedding_service = None
_embedding_lock = threading.Lock()


def _get_shared_reranker():
    """Get or create shared NVIDIA reranker instance (thread-safe)."""
    global _shared_reranker
    if not USE_RERANKER:
        return None
    
    if _shared_reranker is None:
        with _reranker_lock:
            if _shared_reranker is None:
                _shared_reranker = NvidiaReranker(
                    model=RERANKER_MODEL,
                    min_chunks_to_rerank=MIN_CHUNKS_TO_RERANK,
                    top_k_after_rerank=TOP_K_AFTER_RERANK
                )
                logger.info(f"Initialized Nvidia reranker: {RERANKER_MODEL}")
    return _shared_reranker


def _build_embedding_kwargs() -> dict:
    """Build embedding service kwargs based on provider."""
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
            "timeout": EMBEDDING_TIMEOUT,
            "max_retries": EMBEDDING_MAX_RETRIES
        })
    elif EMBEDDING_PROVIDER.lower() in ["lm-studio", "lmstudio", "openai-compatible"]:
        # LM Studio uses OpenAI-compatible API - no device parameter needed
        kwargs.update({
            "base_url": LM_STUDIO_BASE_URL,
            "api_key": LM_STUDIO_API_KEY if LM_STUDIO_API_KEY else None,
            "timeout": EMBEDDING_TIMEOUT,
            "max_retries": EMBEDDING_MAX_RETRIES
        })
    else:
        # Local models (transformers) need device
        kwargs["device"] = "cpu"

    return kwargs


def _get_shared_llm():
    """Get or create shared LLM instance (singleton)."""
    global _shared_llm
    
    with _llm_lock:
        if _shared_llm is not None:
            return _shared_llm
        
        try:
            api_key = GEMINI_API_KEY if LLM_PROVIDER.lower() in ["gemini", "google", "google-ai"] else HF_TOKEN
            
            _shared_llm = create_llm(
                provider=LLM_PROVIDER,
                model_name=LLM_MODEL,
                temperature=LLM_TEMPERATURE,
                max_tokens=LLM_MAX_TOKENS,
                api_key=api_key
            )
            logger.info(f"Shared LLM initialized: {_shared_llm.model_name}")
        except Exception as e:
            logger.warning(f"LLM initialization failed: {e}")
        
        return _shared_llm


def _get_shared_embedding_service():
    """Get or create shared embedding service instance (singleton)."""
    global _shared_embedding_service
    
    with _embedding_lock:
        if _shared_embedding_service is not None:
            return _shared_embedding_service
            
        try:
            _shared_embedding_service = create_embedding_service(
                model_type=EMBEDDING_PROVIDER,
                **_build_embedding_kwargs()
            )
            logger.info("Shared EmbeddingService initialized")
        except Exception as e:
            logger.error(f"EmbeddingService initialization failed: {e}")
            raise
            
        return _shared_embedding_service


def get_session_query_handler(
    session_id: str,
    chunks_metadata: Dict[str, Any],
    vector_store_dir: Path
) -> QueryHandler:
    """Get or create a query handler for a specific session."""
    with _handlers_lock:
        # Return existing handler if available
        if session_id in _session_handlers:
            return _session_handlers[session_id]
        
        logger.info(f"Initializing query handler for session {session_id}")
        
        # Get shared embedding service
        embedding_service = _get_shared_embedding_service()
        
        # Load session-specific vector store
        vector_store = FAISSVectorStore(
            dimension=embedding_service.model.dimension,
            index_type="flat"
        )
        
        if not vector_store_dir.exists():
            raise ValueError(f"Vector store not found for session {session_id}")
        
        vector_store.load(str(vector_store_dir))
        
        # Get shared reranker
        reranker = _get_shared_reranker()
        
        # Load session-specific BM25 sparse index
        bm25_retriever = None
        try:
            bm25 = BM25Retriever(store_path=str(vector_store_dir))
            if bm25.load():
                bm25_retriever = bm25
                logger.info(f"[{session_id[:8]}] BM25 sparse index loaded ({len(bm25.chunk_ids)} chunks)")
            else:
                logger.info(f"[{session_id[:8]}] No BM25 index found — using dense-only retrieval")
        except Exception as e:
            logger.warning(f"[{session_id[:8]}] BM25 load failed, falling back to dense-only: {e}")
        
        # Create retriever with session-specific chunks, reranker and sparse index
        retriever = RAGRetriever(
            vector_store=vector_store,
            embedding_service=embedding_service,
            chunks=chunks_metadata,
            reranker=reranker,
            use_reranker=USE_RERANKER,
            bm25_retriever=bm25_retriever
        )
        
        # Use shared LLM (thread-safe)
        llm = _get_shared_llm()
        
        # Create query handler
        handler = QueryHandler(
            retriever=retriever,
            embedding_service=embedding_service,
            llm=llm,
            top_k=5,
            session_id=session_id
        )
        
        # Cache handler for this session
        _session_handlers[session_id] = handler
        logger.info(f"Query handler initialized for session {session_id}")
        
        return handler


def clear_session_handler(session_id: str):
    """Remove query handler for a session (cleanup)."""
    with _handlers_lock:
        if session_id in _session_handlers:
            del _session_handlers[session_id]
            logger.info(f"Cleared query handler for session {session_id}")


def ask_rag_session(
    session_id: str,
    question: str,
    chunks_metadata: Dict[str, Any],
    vector_store_dir: Path,
    top_k: int = 5,
    system_prompt: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None
) -> QueryResult:
    """Process a query for a specific session."""
    handler = get_session_query_handler(session_id, chunks_metadata, vector_store_dir)
    
    return handler.process_query_with_response(
        query=question,
        top_k=top_k,
        system_prompt=system_prompt,
        temperature=temperature if temperature is not None else LLM_TEMPERATURE,
        max_tokens=max_tokens or LLM_MAX_TOKENS
    )


def get_llm_status() -> bool:
    """Check if LLM is loaded."""
    return _get_shared_llm() is not None
