"""
Retriever Module

Hybrid retrieval with dense search, BM25 sparse search, and optional reranking.

Components:
    • BM25Retriever   — BM25Okapi sparse keyword retriever
    • NvidiaReranker   — NVIDIA nv-rerank-qa-mistral-4b:1 API reranker
    • RetrievalResult  — Result dataclass for RAGRetriever
    • RAGRetriever     — Hybrid retriever (dense + sparse + RRF + reranking)
"""

import os
import re
import pickle
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass

try:
    import requests

    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

try:
    from rank_bm25 import BM25Okapi
    RANK_BM25_AVAILABLE = True
except ImportError:
    RANK_BM25_AVAILABLE = False

from src.utils.Logger import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────
# BM25 Sparse Retriever
# ─────────────────────────────────────────────────────────────


class BM25Retriever:
    """Sparse retriever using BM25Okapi.
    
    Supports incremental addition of texts and saving/loading the index to disk.
    """
    
    def __init__(self, store_path: Optional[str] = None):
        """Initialize BM25 Retriever.
        
        Args:
            store_path: Directory path to save/load the BM25 index.
        """
        if not RANK_BM25_AVAILABLE:
            raise ImportError("rank-bm25 not installed. Run: pip install rank-bm25")
            
        self.store_path = store_path or "data/vector_store"
        self.index_file = os.path.join(self.store_path, "bm25_index.pkl")
        
        # State
        self.bm25: Optional[BM25Okapi] = None
        self.chunk_ids: List[str] = []
        self.corpus_tokens: List[List[str]] = []
        self.is_built = False
        
        logger.debug(f"BM25Retriever initialized. Target index: {self.index_file}")
        
    def _tokenize(self, text: str) -> List[str]:
        """Production-grade tokenizer for BM25.
        
        Lowercases, strips punctuation and special characters,
        removes common stopwords, and splits by whitespace.
        """
        if not text:
            return []
            
        stopwords = {
            "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", 
            "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", 
            "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", 
            "theirs", "themselves", "what", "which", "who", "whom", "this", "that", 
            "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", 
            "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", 
            "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", 
            "at", "by", "for", "with", "about", "against", "between", "into", "through", 
            "during", "before", "after", "above", "below", "to", "from", "up", "down", 
            "in", "out", "on", "off", "over", "under", "again", "further", "then", 
            "once", "here", "there", "when", "where", "why", "how", "all", "any", 
            "both", "each", "few", "more", "most", "other", "some", "such", "no", 
            "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", 
            "t", "can", "will", "just", "don", "should", "now"
        }
        
        text = text.lower()
        text = re.sub(r"[^\w\s]", " ", text)   # strip punctuation
        tokens = text.split()
        return [t for t in tokens if len(t) > 1 and t not in stopwords]
        
    def add_texts(self, chunk_ids: List[str], texts: List[str]):
        """Accumulate texts before building the index.
        
        Args:
            chunk_ids: List of chunk IDs corresponding to the texts
            texts: List of document/chunk texts
        """
        if len(chunk_ids) != len(texts):
            raise ValueError("chunk_ids and texts must have the same length")
            
        for chunk_id, text in zip(chunk_ids, texts):
            tokens = self._tokenize(text)
            self.corpus_tokens.append(tokens)
            self.chunk_ids.append(chunk_id)
            
        self.is_built = False
        
    def build(self):
        """Build the BM25 index from accumulated texts."""
        if not self.corpus_tokens:
            logger.warning("No texts to build BM25 index.")
            return
            
        logger.info(f"Building BM25 index for {len(self.corpus_tokens)} chunks...")
        self.bm25 = BM25Okapi(self.corpus_tokens)
        self.is_built = True
        logger.info("BM25 index built successfully.")
        
    def save(self) -> bool:
        """Save the tokenized corpus and chunk IDs to disk.
        
        Returns:
            True if saving was successful
        """
        if not self.corpus_tokens:
            logger.warning("No BM25 data to save.")
            return False
            
        os.makedirs(self.store_path, exist_ok=True)
        
        try:
            state = {
                "chunk_ids": self.chunk_ids,
                "corpus_tokens": self.corpus_tokens
            }
            with open(self.index_file, 'wb') as f:
                pickle.dump(state, f)
            logger.info(f"Saved BM25 data to {self.index_file} ({len(self.chunk_ids)} chunks)")
            return True
        except Exception as e:
            logger.error(f"Failed to save BM25 index: {e}")
            return False
            
    def load(self) -> bool:
        """Load the BM25 state from disk and build the index.
        
        Returns:
            True if loading was successful
        """
        if not os.path.exists(self.index_file):
            logger.warning(f"BM25 index file not found: {self.index_file}")
            return False
            
        try:
            with open(self.index_file, 'rb') as f:
                state = pickle.load(f)
                
            self.chunk_ids = state.get("chunk_ids", [])
            self.corpus_tokens = state.get("corpus_tokens", [])
            
            if self.corpus_tokens:
                self.build()
                logger.info(f"Loaded existing BM25 index for {len(self.chunk_ids)} chunks")
                return True
            else:
                logger.warning("Loaded BM25 state but found no tokens.")
                return False
                
        except Exception as e:
            logger.error(f"Failed to load BM25 index: {e}")
            return False
            
    def clear(self):
        """Clear the current BM25 index and state. Also removes the disk file."""
        self.bm25 = None
        self.chunk_ids = []
        self.corpus_tokens = []
        self.is_built = False
        
        if os.path.exists(self.index_file):
            try:
                os.remove(self.index_file)
                logger.info(f"Deleted old BM25 index: {self.index_file}")
            except Exception as e:
                logger.error(f"Failed to delete BM25 index file: {e}")

    def retrieve(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        """Retrieve top chunks for a query using BM25.
        
        Args:
            query: The search query
            k: Number of results to return
            
        Returns:
            List of (chunk_id, score) tuples sorted by score descending
        """
        if not self.is_built or not self.bm25:
            logger.warning("BM25 index not built. Returning empty results.")
            return []
            
        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        
        # Zip chunk_ids and scores, then sort
        results = [(self.chunk_ids[i], float(scores[i])) for i in range(len(self.chunk_ids))]
        results.sort(key=lambda x: x[1], reverse=True)
        
        # Filter zero/low scores
        filtered_results = [(c, s) for c, s in results if s > 0]
        
        return filtered_results[:k]


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
    """Hybrid retriever with dense + sparse search, RRF merging, and optional NVIDIA reranking."""

    def __init__(
        self, 
        vector_store, 
        embedding_service, 
        chunks: Dict,
        reranker: Optional[Any] = None,
        use_reranker: bool = False,
        bm25_retriever: Optional[Any] = None
    ):
        """Initialize RAG retriever.
        
        Args:
            vector_store: Vector store for dense retrieval
            embedding_service: Service for generating embeddings
            chunks: Dict mapping chunk_id to chunk data
            reranker: Optional reranker (NvidiaReranker or HuggingFaceReranker)
            use_reranker: Enable reranking (requires reranker to be provided)
            bm25_retriever: Optional sparse retriever (BM25Retriever)
        """
        self.vector_store = vector_store
        self.embedding_service = embedding_service
        self.chunks = chunks
        self.reranker = reranker if use_reranker else None
        self.bm25_retriever = bm25_retriever
        
        if use_reranker and reranker:
            logger.info(f"RAGRetriever initialized with reranking ({type(reranker).__name__})")
        else:
            logger.info("RAGRetriever initialized (legacy dense-only mode)")

        if self.bm25_retriever:
            logger.info("Hybrid Retrieval Enabled (Dense + BM25)")

    def retrieve(self, query: str, k: int = 5, force_rerank: Optional[bool] = None, filters: Optional[Dict[str, Any]] = None) -> List[RetrievalResult]:
        """Retrieve relevant chunks for a query using Hybrid Retrieval.
        
        Args:
            query: Query text
            k: Number of results to return
            force_rerank: Override automatic rerank decision (None = auto)
            filters: Optional dictionary of metadata key-value pairs to filter by
            
        Returns:
            List of RetrievalResult objects
        """
        # Get more candidates if reranking or using hybrid
        retrieval_k = k * 3 if self.reranker else k
        
        # 1. Dense Retrieval
        query_embedding = self.embedding_service.embed_text(query)
        dense_results = self.vector_store.search(
            query_vector=query_embedding.embedding, k=max(retrieval_k, 30), filters=filters
        )
        
        # 2. Sparse Retrieval (with fallback guard)
        sparse_results = []
        if self.bm25_retriever and self.bm25_retriever.is_built:
            try:
                # BM25 retrieve returns (chunk_id, bm25_score)
                sparse_results = self.bm25_retriever.retrieve(query, k=max(retrieval_k, 30))
            except Exception as e:
                logger.warning(f"BM25 sparse retrieval failed, falling back to dense-only: {e}")
                sparse_results = []
            
        # 3. Reciprocal Rank Fusion (RRF)
        rrf_scores = {}
        candidate_metadata = {}
        
        # Helper RRF function
        def _add_to_rrf(chunk_id: str, rank: int, rrf_k: int = 60):
            if chunk_id not in rrf_scores:
                rrf_scores[chunk_id] = 0.0
            rrf_scores[chunk_id] += 1.0 / (rrf_k + rank)
        
        # Add dense candidates to RRF
        for rank, res in enumerate(dense_results, 1):
            _add_to_rrf(res.chunk_id, rank)
            candidate_metadata[res.chunk_id] = res
            
        # Add sparse candidates to RRF
        for rank, (chunk_id, score) in enumerate(sparse_results, 1):
            _add_to_rrf(chunk_id, rank)
            # Find chunk text/metadata if not retrieved by dense search
            if chunk_id not in candidate_metadata:
                # First try local chunks dict (Sequential pipeline style)
                if chunk_id in self.chunks:
                    chunk_data = self.chunks[chunk_id]
                    candidate_metadata[chunk_id] = SearchResult(
                        chunk_id=chunk_id,
                        document_id=chunk_data.get('metadata', {}).get('document_id', ''),
                        text=chunk_data.get('text', ''),
                        similarity_score=float(score / 100),
                        distance=0.0,
                        metadata=chunk_data.get('metadata', {})
                    )
                # Fallback: Try vector store metadata (Parallel pipeline style)
                elif hasattr(self.vector_store, 'metadata_store'):
                    meta = self.vector_store.metadata_store.get(chunk_id)
                    if meta:
                        candidate_metadata[chunk_id] = SearchResult(
                            chunk_id=chunk_id,
                            document_id=meta.document_id,
                            text=meta.text,
                            similarity_score=float(score / 100),
                            distance=0.0,
                            metadata=meta.metadata
                        )

        # 4. Sort and select top candidates by RRF score (descending — higher = better)
        sorted_candidates = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        top_candidates = sorted_candidates[:retrieval_k]
        
        results = []
        for chunk_id, rrf_score in top_candidates:
            res = candidate_metadata.get(chunk_id)
            if res:
                results.append(
                    RetrievalResult(
                        chunk_id=res.chunk_id,
                        document_id=res.document_id,
                        text=res.text,
                        distance=res.distance, # RRF score will be injected to metadata mapping instead
                        metadata={**res.metadata, "rrf_score": rrf_score},
                    )
                )
        
        # Handle cases where hybrid produced nothing
        if not results:
            # Fallback: use dense results directly if available
            if dense_results:
                logger.info("Hybrid merge empty — falling back to dense-only results")
                results = [
                    RetrievalResult(
                        chunk_id=r.chunk_id,
                        document_id=r.document_id,
                        text=r.text,
                        distance=r.distance,
                        metadata=r.metadata,
                    ) for r in dense_results[:retrieval_k]
                ]
            elif sparse_results:
                # Edge case: FAISS empty but BM25 has results
                logger.info("Dense results empty — falling back to sparse-only results")
                for chunk_id, score in sparse_results[:retrieval_k]:
                    if chunk_id in self.chunks:
                        chunk_data = self.chunks[chunk_id]
                        results.append(RetrievalResult(
                            chunk_id=chunk_id,
                            document_id=chunk_data.get('metadata', {}).get('document_id', ''),
                            text=chunk_data.get('text', ''),
                            distance=0.0,
                            metadata=chunk_data.get('metadata', {}),
                        ))
        
        # 5. Apply reranking if available and conditions are met
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
