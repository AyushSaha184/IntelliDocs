"""
Query Handler Module - Processes user queries through the RAG pipeline

Handles:
1. User input validation
2. Query embedding
3. Retrieval from vector store
4. Context preparation for LLM
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import re
from datetime import datetime

from src.utils.Logger import get_logger
from src.modules.LLM import BaseLLM
from src.modules.QueryCache import get_retrieval_cache, get_llm_cache

logger = get_logger(__name__)


@dataclass
class QueryResult:
    """Result from query processing."""

    query: str
    retrieved_chunks: List[str]
    metadata: List[Dict]
    query_embedding: Optional[list] = None
    retrieval_scores: Optional[List[float]] = None
    timestamp: str = None
    llm_response: Optional[str] = None
    llm_metadata: Optional[Dict] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()


class QueryHandler:
    """Handles user queries and retrieves relevant context."""

    def __init__(
        self,
        retriever,
        embedding_service,
        llm: Optional[BaseLLM] = None,
        top_k: int = 5,
        session_id: str = "default_session",
        max_query_history: int = 250,
    ):
        """Initialize query handler."""
        self.retriever = retriever
        self.embedding_service = embedding_service
        self.llm = llm
        self.top_k = top_k
        self.session_id = session_id
        self.max_query_history = max(1, int(max_query_history))
        self.query_history: List[QueryResult] = []

        logger.info(
            f"QueryHandler initialized for session {session_id} with top_k={top_k}, "
            f"LLM={'enabled' if llm else 'disabled'}"
        )

    # ------------------------------------------------------------------
    # Metadata/cache helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _source_display_name(chunk_meta: dict) -> str:
        """Return a clean document name from chunk metadata."""
        for key in ("document_name", "doc_name", "filename", "file_name", "source"):
            val = chunk_meta.get(key)
            if val:
                return str(val)
        return "Unknown"

    @staticmethod
    def _page_display(chunk_meta: dict):
        """Return page number from chunk metadata, or None if unavailable."""
        for key in ("page_number", "page", "page_num"):
            val = chunk_meta.get(key)
            if val is not None:
                return val
        return None

    @staticmethod
    def _extract_chunk_meta(raw: dict) -> dict:
        """Handle both flat metadata and nested chunk dict format."""
        if not isinstance(raw, dict):
            return {}
        if isinstance(raw.get("metadata"), dict):
            return raw["metadata"]
        return raw

    @staticmethod
    def _safe_float(value, default: float = 0.0) -> float:
        try:
            return float(value)
        except Exception:
            return default

    def _resolve_similarity(self, result) -> float:
        """Prefer retriever-provided normalized scores; fallback to distance conversion."""
        meta = result.metadata if isinstance(getattr(result, "metadata", None), dict) else {}
        for key in ("similarity_score", "score", "reranker_score", "rrf_score"):
            if key in meta and meta[key] is not None:
                return self._safe_float(meta[key], 0.0)

        dist = self._safe_float(getattr(result, "distance", 1.0), 1.0)
        if 0.0 <= dist <= 1.0:
            return max(0.0, min(1.0, 1.0 - dist))
        return 0.0

    def _retrieval_profile(self) -> str:
        """Build lightweight retrieval signature for cache namespacing."""
        backend = getattr(getattr(self.retriever, "vector_store", None), "__class__", type("x", (), {})).__name__
        has_reranker = bool(getattr(self.retriever, "reranker", None))
        has_bm25 = bool(getattr(self.retriever, "bm25_retriever", None))
        return f"{backend}|rerank={int(has_reranker)}|bm25={int(has_bm25)}"

    def _cache_session_namespace(self) -> str:
        return f"{self.session_id}::retrieval::{self._retrieval_profile()}"

    def _append_history(self, query_result: QueryResult) -> None:
        self.query_history.append(query_result)
        if len(self.query_history) > self.max_query_history:
            self.query_history = self.query_history[-self.max_query_history :]

    # ------------------------------------------------------------------
    def validate_query(self, query: str) -> Tuple[bool, str]:
        """Validate user query."""
        if not query:
            return False, "Query cannot be empty"

        if isinstance(query, str):
            query = query.strip()

        if len(query) < 3:
            return False, "Query must be at least 3 characters"

        if len(query) > 5000:
            return False, "Query exceeds maximum length of 5000 characters"

        return True, ""

    def process_query(self, query: str, top_k: Optional[int] = None) -> QueryResult:
        """Process user query and retrieve relevant chunks."""
        is_valid, error_msg = self.validate_query(query)
        if not is_valid:
            raise ValueError(f"Invalid query: {error_msg}")

        try:
            query = query.strip()
            k = top_k or self.top_k
            cache_session = self._cache_session_namespace()

            cached_result = get_retrieval_cache().get_cache(cache_session, query, k)
            if cached_result:
                query_result = QueryResult(
                    query=query,
                    retrieved_chunks=cached_result.get("retrieved_chunks", []),
                    metadata=cached_result.get("metadata", []),
                    retrieval_scores=cached_result.get("retrieval_scores", []),
                )
                self._append_history(query_result)
                return query_result

            logger.info(f"Processing query: {query[:100]}...")

            query_embedding_result = self.embedding_service.embed_text(query)
            query_embedding = query_embedding_result.embedding

            retrieval_results = self.retriever.retrieve(query, k=k)

            retrieved_chunks = []
            metadata_list = []
            scores = []

            for result in retrieval_results:
                retrieved_chunks.append(result.text)

                chunk_meta = {}
                if hasattr(self.retriever, "chunks") and self.retriever.chunks:
                    raw_chunk = self.retriever.chunks.get(result.chunk_id, {})
                    chunk_meta = self._extract_chunk_meta(raw_chunk)
                elif result.metadata:
                    chunk_meta = self._extract_chunk_meta(result.metadata)

                doc_name = self._source_display_name(chunk_meta)
                page_num = self._page_display(chunk_meta)
                similarity = self._resolve_similarity(result)
                distance = self._safe_float(getattr(result, "distance", 1.0), 1.0)

                metadata_list.append(
                    {
                        "chunk_id": result.chunk_id,
                        "document_id": result.document_id,
                        "document_name": doc_name,
                        "page_number": page_num,
                        "distance": distance,
                        "similarity_score": similarity,
                    }
                )
                scores.append(similarity)

            query_result = QueryResult(
                query=query,
                retrieved_chunks=retrieved_chunks,
                metadata=metadata_list,
                query_embedding=query_embedding.tolist()
                if hasattr(query_embedding, "tolist")
                else query_embedding,
                retrieval_scores=scores,
            )

            cache_data = {
                "retrieved_chunks": retrieved_chunks,
                "metadata": metadata_list,
                "retrieval_scores": scores,
            }
            get_retrieval_cache().set_cache(cache_session, query, k, cache_data)

            self._append_history(query_result)

            logger.info(f"Query processed successfully. Retrieved {len(retrieved_chunks)} chunks")
            return query_result

        except Exception as e:
            logger.error(f"Error processing query: {e}", exc_info=True)
            raise

    def format_context(self, query_result: QueryResult, include_scores: bool = False) -> str:
        """Format retrieved context for LLM consumption."""
        context_parts = []

        for i, (chunk, meta) in enumerate(zip(query_result.retrieved_chunks, query_result.metadata), 1):
            part = f"[Context {i}]\n{chunk}"

            if include_scores:
                score = query_result.retrieval_scores[i - 1] if query_result.retrieval_scores else 0
                part += f"\n(Relevance: {score:.2f})"

            context_parts.append(part)

        return "\n\n".join(context_parts)

    def get_query_history(self, limit: Optional[int] = None) -> List[Dict]:
        """Get query history."""
        history = self.query_history

        if limit:
            history = history[-limit:]

        return [
            {
                "query": q.query,
                "timestamp": q.timestamp,
                "num_results": len(q.retrieved_chunks),
                "avg_score": sum(q.retrieval_scores) / len(q.retrieval_scores) if q.retrieval_scores else 0,
            }
            for q in history
        ]

    def clear_history(self) -> None:
        """Clear query history."""
        self.query_history = []
        logger.info("Query history cleared")

    def generate_response(
        self,
        query_result: QueryResult,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> QueryResult:
        """Generate LLM response based on retrieved context."""
        if not self.llm:
            raise RuntimeError(
                "No LLM configured. Initialize QueryHandler with an LLM instance to generate responses."
            )

        try:
            cached_llm = get_llm_cache().get_cache(
                self.session_id, query_result.query, query_result.retrieved_chunks
            )
            if cached_llm:
                query_result.llm_response = cached_llm.get("llm_response")
                query_result.llm_metadata = cached_llm.get("llm_metadata")
                return query_result

            context = self.format_context(query_result, include_scores=False)

            prompt = self.llm.create_rag_prompt(
                query=query_result.query,
                context=context,
                system_prompt=system_prompt,
            )

            logger.info(f"Generating LLM response for query: {query_result.query[:100]}...")

            llm_response = self.llm.generate(
                prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            final_response = re.sub(r"<think>.*?</think>", "", llm_response.response, flags=re.DOTALL).strip()

            query_result.llm_response = final_response
            query_result.llm_metadata = {
                "model": llm_response.model,
                "prompt_tokens": llm_response.prompt_tokens,
                "completion_tokens": llm_response.completion_tokens,
                "total_tokens": llm_response.total_tokens,
                "metadata": llm_response.metadata,
            }

            logger.info(f"LLM response generated successfully (tokens: {llm_response.total_tokens})")

            get_llm_cache().set_cache(
                self.session_id,
                query_result.query,
                query_result.retrieved_chunks,
                {
                    "llm_response": query_result.llm_response,
                    "llm_metadata": query_result.llm_metadata,
                },
            )

            return query_result

        except Exception as e:
            logger.error(f"Error generating LLM response: {e}", exc_info=True)
            raise

    def process_query_with_response(
        self,
        query: str,
        top_k: Optional[int] = None,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> QueryResult:
        """Process query and generate LLM response in one call."""
        query_result = self.process_query(query, top_k=top_k)

        if self.llm:
            query_result = self.generate_response(
                query_result,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            )

        return query_result