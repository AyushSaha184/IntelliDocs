"""
Retriever Module

Dense retrieval with optional reranking.

Components:
    • NvidiaReranker  — NVIDIA nv-rerank-qa-mistral-4b:1 API reranker
    • RetrievalResult — Result dataclass for RAGRetriever
    • RAGRetriever    — Dense retriever with optional reranking
"""

import os
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass

try:
    import requests

    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

from src.utils.Logger import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────
# NVIDIA API Reranker (nv-rerank-qa-mistral-4b:1)
# ─────────────────────────────────────────────────────────────


class NvidiaReranker:
    """Reranker using NVIDIA's API for nv-rerank-qa-mistral-4b:1.

    Only activates when candidate count exceeds a threshold (default 8),
    then returns the top-k (default 5) reranked results.
    
    Requires NVIDIA_API_KEY environment variable.
    """

    def __init__(
        self,
        model: str = "nv-rerank-qa-mistral-4b:1",
        min_chunks_to_rerank: int = 8,
        top_k_after_rerank: int = 5,
        timeout: float = 30.0,
    ):
        """Initialize NVIDIA reranker.

        Args:
            model: Reranker model ID
            min_chunks_to_rerank: Only rerank when candidates > this
            top_k_after_rerank: Number of results after reranking
            timeout: API call timeout in seconds
        """
        if not REQUESTS_AVAILABLE:
            raise ImportError("requests not installed. Run: pip install requests")

        self.api_key = os.environ.get("NVIDIA_API_KEY")
        if not self.api_key:
            logger.warning("NVIDIA_API_KEY environment variable not set. Reranking will fail.")

        self.model = model
        self.min_chunks_to_rerank = min_chunks_to_rerank
        self.top_k_after_rerank = top_k_after_rerank
        self.timeout = timeout
        self.rerank_url = "https://ai.api.nvidia.com/v1/retrieval/nvidia/reranking"
        
        # Re-use connections for faster subsequent calls
        import requests
        self.session = requests.Session()

        logger.info(
            f"NvidiaReranker initialized: model={model}, "
            f"min_chunks={min_chunks_to_rerank}, top_k={top_k_after_rerank}"
        )

    def should_rerank(self, num_candidates: int) -> bool:
        """Check if reranking should be applied."""
        return num_candidates > self.min_chunks_to_rerank

    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: Optional[int] = None,
    ) -> List[Tuple[int, float]]:
        """Rerank documents using NVIDIA API.

        Args:
            query: Query text
            documents: List of document texts
            top_k: Override for top_k_after_rerank

        Returns:
            List of (original_index, reranker_score) sorted by score desc
        """
        if not documents:
            return []

        top_k = top_k or self.top_k_after_rerank
        
        if not self.api_key:
            logger.error("Cannot rerank: NVIDIA_API_KEY not set")
            return [(i, 0.0) for i in range(min(top_k, len(documents)))]

        try:
            # Format according to NVIDIA API spec
            payload = {
                "model": self.model,
                "query": {
                    "text": query
                },
                "passages": [
                    {"text": doc} for doc in documents
                ]
            }
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Accept": "application/json",
            }

            response = self.session.post(
                self.rerank_url,
                json=payload,
                headers=headers,
                timeout=self.timeout,
            )
            response.raise_for_status()

            result = response.json()

            # Parse NVIDIA /reranking response
            # Format is typically {"rankings": [{"index": 0, "logit": 2.5}, ...]}
            reranked = []
            for item in result.get("rankings", []):
                original_idx = item.get("index", 0)
                # 'logit' is the score returned by nv-rerank-qa-mistral-4b
                score = item.get("logit", 0.0)
                reranked.append((original_idx, score))

            # Sort by score descending
            reranked.sort(key=lambda x: x[1], reverse=True)
            return reranked[:top_k]

        except requests.exceptions.ConnectionError:
            logger.warning(
                "NVIDIA reranker API not reachable — "
                "returning results without reranking"
            )
            return [(i, 0.0) for i in range(min(top_k, len(documents)))]

        except requests.exceptions.HTTPError as e:
            logger.error(f"NVIDIA Reranker HTTP Error: {e.response.text if hasattr(e, 'response') else str(e)}")
            return [(i, 0.0) for i in range(min(top_k, len(documents)))]

        except Exception as e:
            logger.error(f"Reranker error: {e}")
            return [(i, 0.0) for i in range(min(top_k, len(documents)))]


# ─────────────────────────────────────────────────────────────
# Retriever
# ─────────────────────────────────────────────────────────────


@dataclass
class RetrievalResult:
    """Result from RAG retrieval."""

    chunk_id: str
    document_id: str
    text: str
    distance: float
    metadata: Dict[str, Any]


class RAGRetriever:
    """Dense retriever with optional NVIDIA reranking."""

    def __init__(
        self, 
        vector_store, 
        embedding_service, 
        chunks: Dict,
        reranker: Optional[Any] = None,
        use_reranker: bool = False
    ):
        """Initialize RAG retriever.
        
        Args:
            vector_store: Vector store for dense retrieval
            embedding_service: Service for generating embeddings
            chunks: Dict mapping chunk_id to chunk data
            reranker: Optional reranker (NvidiaReranker or HuggingFaceReranker)
            use_reranker: Enable reranking (requires reranker to be provided)
        """
        self.vector_store = vector_store
        self.embedding_service = embedding_service
        self.chunks = chunks
        self.reranker = reranker if use_reranker else None
        
        if use_reranker and reranker:
            logger.info(f"RAGRetriever initialized with reranking ({type(reranker).__name__})")
        else:
            logger.info("RAGRetriever initialized (legacy dense-only mode)")

    def retrieve(self, query: str, k: int = 5, force_rerank: Optional[bool] = None, filters: Optional[Dict[str, Any]] = None) -> List[RetrievalResult]:
        """Retrieve relevant chunks for a query.
        
        Args:
            query: Query text
            k: Number of results to return
            force_rerank: Override automatic rerank decision (None = auto)
            filters: Optional dictionary of metadata key-value pairs to filter by
            
        Returns:
            List of RetrievalResult objects
        """
        # Get more candidates if reranking
        retrieval_k = k * 3 if self.reranker else k
        
        query_embedding = self.embedding_service.embed_text(query)
        search_results = self.vector_store.search(
            query_vector=query_embedding.embedding, k=retrieval_k, filters=filters
        )

        results = []
        for result in search_results:
            results.append(
                RetrievalResult(
                    chunk_id=result.chunk_id,
                    document_id=result.document_id,
                    text=result.text,
                    distance=result.distance,
                    metadata=result.metadata,
                )
            )
        
        # Apply reranking if available and conditions are met
        should_rerank = force_rerank if force_rerank is not None else (
            self.reranker is not None 
            and self.reranker.should_rerank(len(results))
        )
        
        if should_rerank and self.reranker:
            doc_texts = [r.text for r in results]
            reranked = self.reranker.rerank(query, doc_texts, top_k=k)
            
            # Reorder results based on reranker scores
            reranked_results = []
            for orig_idx, reranker_score in reranked:
                if orig_idx < len(results):
                    r = results[orig_idx]
                    # Store reranker score in metadata
                    if r.metadata is None:
                        r.metadata = {}
                    r.metadata['reranker_score'] = reranker_score
                    reranked_results.append(r)
            
            logger.info(f"Reranked {len(results)} → {len(reranked_results)} results")
            return reranked_results[:k]
        
        return results[:k]

    def retrieve_with_scores(self, query: str, k: int = 5) -> List[tuple]:
        """Retrieve with similarity scores.
        
        Args:
            query: Query text
            k: Number of results
            
        Returns:
            List of (text, score) tuples
        """
        results = self.retrieve(query, k)
        scores = []
        for r in results:
            # Prefer reranker score if available
            if r.metadata and 'reranker_score' in r.metadata:
                score = r.metadata['reranker_score']
            else:
                # Convert distance to similarity score
                score = 1.0 - r.distance
            scores.append((r.text, score))
        return scores

