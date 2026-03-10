"""Session-aware RAG service with isolated user sessions."""

from typing import Optional, Dict, Any
from pathlib import Path
from src.modules.VectorStore import FAISSVectorStore
from src.modules.QdrantStore import QdrantSessionStore
from src.modules.Retriever import RAGRetriever
from src.modules.Retriever import BM25Retriever
from src.modules.QueryGeneration import QueryHandler, QueryResult
from src.agents.Orchestrator import AgentOrchestrator
from src.utils.Logger import get_logger
from src.utils.llm_provider import (
    get_shared_llm,
    get_shared_reranker,
    get_shared_embedding_service,
)
from config.config import (
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS,
    USE_RERANKER,
    VECTOR_BACKEND,
)
import threading

logger = get_logger(__name__)

# Thread-safe storage for session-specific retrievers and handlers
_session_retrievers: Dict[str, RAGRetriever] = {}
_session_handlers: Dict[str, QueryHandler] = {}
_handlers_lock = threading.Lock()

# Shared Orchestrator instance (stateless, safe to share)
_orchestrator = AgentOrchestrator()


def _get_session_retriever(
    session_id: str,
    chunks_metadata: Dict[str, Any],
    vector_store_dir: Path,
) -> RAGRetriever:
    """Get or create a retriever for a specific session.

    Caller MUST hold _handlers_lock before entering this function.
    Both read (_session_retrievers check) and write (insert) happen here;
    the lock prevents double-construction if two threads race on the same
    new session_id.
    """
    if session_id in _session_retrievers:
        return _session_retrievers[session_id]

    logger.info(f"[session] Building retriever for session {session_id[:8]}")
    embedding_service = get_shared_embedding_service()

    bm25_retriever = BM25Retriever(store_path=str(vector_store_dir))
    bm25_loaded = bm25_retriever.load()
    if not bm25_loaded:
        logger.warning(f"[session] BM25 index not loaded for {session_id[:8]} (dense-only fallback)")

    if VECTOR_BACKEND == "qdrant":
        try:
            vector_store = QdrantSessionStore(
                session_id=session_id,
                embedding_dimension=embedding_service.model.dimension,
            )
        except Exception as e:
            logger.error(f"[session] qdrant init failed, falling back to FAISS: {e}")
            vector_store = FAISSVectorStore(
                dimension=embedding_service.model.dimension,
                index_type="flat",
            )
            if not vector_store_dir.exists():
                raise ValueError(f"Vector store not found for session {session_id}")
            vector_store.load(str(vector_store_dir))
    else:
        vector_store = FAISSVectorStore(
            dimension=embedding_service.model.dimension,
            index_type="flat",
        )

        if not vector_store_dir.exists():
            raise ValueError(f"Vector store not found for session {session_id}")

        vector_store.load(str(vector_store_dir))

    reranker = get_shared_reranker()

    retriever = RAGRetriever(
        vector_store=vector_store,
        embedding_service=embedding_service,
        chunks=chunks_metadata,
        reranker=reranker,
        use_reranker=USE_RERANKER,
        bm25_retriever=bm25_retriever if bm25_loaded else None,
    )

    _session_retrievers[session_id] = retriever
    return retriever


def get_session_query_handler(
    session_id: str,
    chunks_metadata: Dict[str, Any],
    vector_store_dir: Path
) -> QueryHandler:
    """Get or create a query handler for a specific session (legacy fallback)."""
    with _handlers_lock:
        if session_id in _session_handlers:
            return _session_handlers[session_id]

        logger.info(f"Initializing query handler for session {session_id}")

        embedding_service = get_shared_embedding_service()
        retriever = _get_session_retriever(session_id, chunks_metadata, vector_store_dir)
        llm = get_shared_llm()

        handler = QueryHandler(
            retriever=retriever,
            embedding_service=embedding_service,
            llm=llm,
            top_k=5,
            session_id=session_id
        )

        _session_handlers[session_id] = handler
        logger.info(f"Query handler initialized for session {session_id}")

        return handler


def clear_session_handler(session_id: str):
    """Remove query handler and retriever for a session (cleanup)."""
    with _handlers_lock:
        if session_id in _session_handlers:
            del _session_handlers[session_id]
        if session_id in _session_retrievers:
            del _session_retrievers[session_id]
        logger.info(f"Cleared session resources for {session_id}")


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
    """Process a query for a specific session via the Agent Orchestrator.

    Uses the multi-agent pipeline (Planner → Router → Agents → Orchestrator).
    Falls back to the legacy QueryHandler if orchestration fails.
    """
    with _handlers_lock:
        retriever = _get_session_retriever(session_id, chunks_metadata, vector_store_dir)
        embedding_service = get_shared_embedding_service()

    try:
        result = _orchestrator.run(
            query=question,
            retriever=retriever,
            embedding_service=embedding_service,
            session_id=session_id,
            top_k=top_k,
            system_prompt=system_prompt,
            temperature=temperature if temperature is not None else LLM_TEMPERATURE,
            max_tokens=max_tokens or LLM_MAX_TOKENS,
        )
        logger.info(f"[session] Query complete for session {session_id[:8]}")
        return result
    except Exception as e:
        logger.error(f"Orchestrator failed, falling back to QueryHandler: {e}")
        handler = get_session_query_handler(session_id, chunks_metadata, vector_store_dir)
        return handler.process_query_with_response(
            query=question,
            top_k=top_k,
            system_prompt=system_prompt,
            temperature=temperature if temperature is not None else LLM_TEMPERATURE,
            max_tokens=max_tokens or LLM_MAX_TOKENS,
        )


def get_llm_status() -> bool:
    """Check if LLM is loaded."""
    return get_shared_llm() is not None


def ask_rag_session_stream(
    session_id: str,
    question: str,
    chunks_metadata,
    vector_store_dir,
    top_k: int = 5,
    system_prompt=None,
    temperature=None,
    max_tokens=None,
):
    """Generator that streams orchestrator events for a session query.

    Yields dicts produced by AgentOrchestrator.run_stream().
    Each dict has: {"event": str, "data": any}
    """
    with _handlers_lock:
        retriever = _get_session_retriever(session_id, chunks_metadata, vector_store_dir)
        embedding_service = get_shared_embedding_service()

    yield from _orchestrator.run_stream(
        query=question,
        retriever=retriever,
        embedding_service=embedding_service,
        session_id=session_id,
        top_k=top_k,
        system_prompt=system_prompt,
        temperature=temperature if temperature is not None else LLM_TEMPERATURE,
        max_tokens=max_tokens or LLM_MAX_TOKENS,
    )
