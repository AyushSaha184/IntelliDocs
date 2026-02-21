"""
Unified Retriever Module - BGE-M3 Hybrid + Vespa-Style Fusion + LM Studio Reranking

Implements the full BGE-M3 "Mother of all embedding models" architecture with
three retrieval representations served through a local Vespa-style index:

    M3 = Multi-linguality (100+ languages)
       + Multi-granularities (input up to 8192 tokens)
       + Multi-functionality (dense, lexical, ColBERT retrieval)

Representations (mirroring the Vespa schema):
    ┌───────────────────────────────────────────────────────────────────┐
    │ Field           │ Vespa Type                  │ Description       │
    ├───────────────────────────────────────────────────────────────────┤
    │ dense_rep       │ tensor<bfloat16>(x[1024])   │ DPR embedding     │
    │ lexical_rep     │ tensor<bfloat16>(t{})        │ Sparse weights    │
    │ colbert_rep     │ tensor<bfloat16>(t{}, x[1024])│ Multi-vector     │
    └───────────────────────────────────────────────────────────────────┘

Ranking (Vespa m3hybrid rank profile):
    Functions:
        dense   = cosine_similarity(query(q_dense), attribute(dense_rep), x)
        lexical = sum(query(q_lexical) * attribute(lexical_rep))
        max_sim = sum(reduce(sum(q_colbert * colbert_rep, x), max, t), qt) / q_len

    Fusion strategies:
        • Vespa linear: 0.4*dense + 0.2*lexical + 0.4*max_sim
        • RRF: Σ(1/(k + rank_i)) — intelligent rank-based fusion

Reranking:
    BGE-reranker-v2-m3 via LM Studio (only when candidates > 8, returns top 5)

Document types optimized for:
    Company policies, HR docs, FAQs, financial summaries,
    product docs, CSV data, website content
"""

import os
import time
import json
import hashlib
import numpy as np
from typing import List, Dict, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from pathlib import Path
from collections import defaultdict

try:
    from FlagEmbedding import BGEM3FlagModel

    BGEM3_FLAG_AVAILABLE = True
except ImportError:
    BGEM3_FLAG_AVAILABLE = False

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import faiss

    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

try:
    import requests

    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

from src.utils.Logger import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────
# Data Classes
# ─────────────────────────────────────────────────────────────


@dataclass
class M3Embedding:
    """All three BGE-M3 representations for a single text.

    Mirrors the Vespa document fields:
        dense_rep:   tensor<bfloat16>(x[1024])
        lexical_rep: tensor<bfloat16>(t{})
        colbert_rep: tensor<bfloat16>(t{}, x[1024])
    """

    dense: np.ndarray  # shape (1024,)
    lexical: Dict[int, float]  # token_id → weight
    colbert: np.ndarray  # shape (num_tokens, 1024)


@dataclass
class HybridRetrievalResult:
    """Result from hybrid retrieval pipeline"""

    chunk_id: str
    text: str
    score: float
    component_scores: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)
    reranker_score: Optional[float] = None


@dataclass
class HybridRetrieverStats:
    """Runtime statistics for monitoring"""

    queries_processed: int = 0
    avg_query_time_ms: float = 0.0
    total_query_time_ms: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    reranker_calls: int = 0
    reranker_skips: int = 0


# ─────────────────────────────────────────────────────────────
# BGE-M3 Encoder (FlagEmbedding)
# ─────────────────────────────────────────────────────────────


class BGEM3Encoder:
    """BGE-M3 multi-representation encoder using FlagEmbedding.

    Produces all three M3 representations in a single forward pass:
        dense_vecs:     Regular DPR embeddings (1024-dim)
        lexical_weights: Learned sparse token weights (like SPLADE)
        colbert_vecs:    Per-token contextualized vectors (for MaxSim)

    Usage:
        encoder = BGEM3Encoder(use_fp16=True)  # GPU with FP16
        embedding = encoder.encode("query text")
        # embedding.dense, embedding.lexical, embedding.colbert
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        use_fp16: bool = True,
        device: Optional[str] = None,
        max_length: int = 8192,
        batch_size: int = 12,
    ):
        """Initialize BGE-M3 encoder.

        Args:
            model_name: HuggingFace model ID
            use_fp16: Use FP16 for GPU (recommended for speed)
            device: 'cuda' or 'cpu' (auto-detected if None)
            max_length: Max input length (BGE-M3 supports up to 8192)
            batch_size: Batch size for encoding
        """
        if not BGEM3_FLAG_AVAILABLE:
            raise ImportError(
                "FlagEmbedding not installed. Run: pip install FlagEmbedding"
            )

        # Auto-detect device
        if device is None:
            if TORCH_AVAILABLE and torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"

        self.device = device
        self.max_length = max_length
        self.batch_size = batch_size
        self.dimension = 1024  # BGE-M3 is always 1024-dim

        # CPU should use FP32, GPU uses FP16 for speed
        # FlagEmbedding BGEM3FlagModel: use_fp16=False for CPU
        actual_fp16 = use_fp16 and device != "cpu"

        logger.info(
            f"Loading BGE-M3 encoder: device={device}, "
            f"fp16={actual_fp16}, max_length={max_length}"
        )

        self.model = BGEM3FlagModel(
            model_name, use_fp16=actual_fp16, device=device
        )

        logger.info("BGE-M3 encoder loaded successfully")

    def encode(self, text: str) -> M3Embedding:
        """Encode a single text into all three M3 representations.

        Args:
            text: Input text

        Returns:
            M3Embedding with dense, lexical, colbert fields
        """
        result = self.model.encode(
            [text],
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=True,
            max_length=self.max_length,
            batch_size=1,
        )

        return M3Embedding(
            dense=np.array(result["dense_vecs"][0], dtype=np.float32),
            lexical=dict(result["lexical_weights"][0]),
            colbert=np.array(result["colbert_vecs"][0], dtype=np.float32),
        )

    def encode_batch(self, texts: List[str]) -> List[M3Embedding]:
        """Encode multiple texts into M3 representations.

        Args:
            texts: List of input texts

        Returns:
            List of M3Embedding objects
        """
        if not texts:
            return []

        result = self.model.encode(
            texts,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=True,
            max_length=self.max_length,
            batch_size=self.batch_size,
        )

        embeddings = []
        for i in range(len(texts)):
            embeddings.append(
                M3Embedding(
                    dense=np.array(result["dense_vecs"][i], dtype=np.float32),
                    lexical=dict(result["lexical_weights"][i]),
                    colbert=np.array(
                        result["colbert_vecs"][i], dtype=np.float32
                    ),
                )
            )

        return embeddings


# ─────────────────────────────────────────────────────────────
# Vespa-Style Hybrid Index
# ─────────────────────────────────────────────────────────────


class VespaStyleIndex:
    """Local implementation of Vespa's M3 hybrid index.

    Storage mirrors the Vespa schema:
        dense_rep:   tensor<bfloat16>(x[1024])     → FAISS IndexFlatIP
        lexical_rep: tensor<bfloat16>(t{})          → Python inverted index
        colbert_rep: tensor<bfloat16>(t{}, x[1024]) → NumPy arrays in memory

    Ranking mirrors the Vespa m3hybrid rank profile:
        dense   = cosine_similarity(q_dense, dense_rep, x)
        lexical = sum(q_lexical * lexical_rep)
        max_sim = sum(reduce(sum(q_colbert * colbert_rep, x), max, t), qt) / q_len

    Fusion strategies:
        vespa: 0.4*dense + 0.2*lexical + 0.4*max_sim (linear combination)
        rrf:   Σ(1/(k + rank_i))                     (reciprocal rank fusion)
    """

    def __init__(
        self,
        dimension: int = 1024,
        store_path: Optional[str] = None,
        dense_weight: float = 0.4,
        lexical_weight: float = 0.2,
        colbert_weight: float = 0.4,
        fusion_strategy: str = "rrf",
        rrf_k: int = 60,
    ):
        """Initialize Vespa-style index.

        Args:
            dimension: Embedding dimension (1024 for BGE-M3)
            store_path: Directory to persist index data
            dense_weight: Weight for dense similarity (vespa fusion)
            lexical_weight: Weight for lexical matching (vespa fusion)
            colbert_weight: Weight for ColBERT MaxSim (vespa fusion)
            fusion_strategy: 'vespa' (linear) or 'rrf' (reciprocal rank)
            rrf_k: RRF constant (default 60, higher = more conservative)
        """
        if not FAISS_AVAILABLE:
            raise ImportError("faiss-cpu not installed. Run: pip install faiss-cpu")

        self.dimension = dimension
        self.store_path = store_path

        # Fusion config
        self.fusion_strategy = fusion_strategy.lower()
        if self.fusion_strategy not in ("vespa", "rrf"):
            raise ValueError(
                f"fusion_strategy must be 'vespa' or 'rrf', got: {fusion_strategy}"
            )

        # Vespa m3hybrid weights (for linear fusion)
        self.dense_weight = dense_weight
        self.lexical_weight = lexical_weight
        self.colbert_weight = colbert_weight

        # RRF constant
        self.rrf_k = rrf_k

        # ── Dense index (FAISS IndexFlatIP for cosine on normalized vectors) ──
        self.dense_index = faiss.IndexFlatIP(dimension)

        # ── Sparse inverted index: token_id → [(chunk_idx, weight), ...] ──
        self.inverted_index: Dict[int, List[Tuple[int, float]]] = defaultdict(list)

        # ── ColBERT storage: chunk_idx → colbert matrix ──
        self.colbert_store: Dict[int, np.ndarray] = {}

        # ── Metadata ──
        self.chunk_count = 0
        self.chunk_ids: List[str] = []  # chunk_idx → chunk_id

        logger.info(
            f"VespaStyleIndex initialized: dim={dimension}, "
            f"fusion={self.fusion_strategy}, "
            f"weights=(d={dense_weight}, l={lexical_weight}, c={colbert_weight}), "
            f"rrf_k={rrf_k}"
        )

    # ── Indexing ────────────────────────────────────────────

    def add(
        self,
        chunk_id: str,
        embedding: M3Embedding,
    ) -> int:
        """Add a single chunk with all three representations.

        Args:
            chunk_id: Unique chunk identifier
            embedding: M3Embedding with dense, lexical, colbert

        Returns:
            Internal chunk index
        """
        idx = self.chunk_count

        # Dense: normalize and add to FAISS
        dense_vec = embedding.dense.reshape(1, -1).astype(np.float32)
        norm = np.linalg.norm(dense_vec)
        if norm > 0:
            dense_vec /= norm
        self.dense_index.add(dense_vec)

        # Sparse: populate inverted index
        for token_id, weight in embedding.lexical.items():
            self.inverted_index[int(token_id)].append((idx, float(weight)))

        # ColBERT: store multi-vector
        self.colbert_store[idx] = embedding.colbert.astype(np.float32)

        # Metadata
        self.chunk_ids.append(chunk_id)
        self.chunk_count += 1

        return idx

    def add_batch(
        self,
        chunk_ids: List[str],
        embeddings: List[M3Embedding],
    ) -> List[int]:
        """Add multiple chunks at once (optimized for FAISS batch add).

        Args:
            chunk_ids: List of unique chunk identifiers
            embeddings: List of M3Embedding objects

        Returns:
            List of internal chunk indices
        """
        if len(chunk_ids) != len(embeddings):
            raise ValueError("chunk_ids and embeddings must be same length")

        if not chunk_ids:
            return []

        start_idx = self.chunk_count
        indices = []

        # ── Batch add dense vectors to FAISS ──
        dense_matrix = np.vstack(
            [e.dense.reshape(1, -1) for e in embeddings]
        ).astype(np.float32)

        # Normalize rows for cosine similarity
        norms = np.linalg.norm(dense_matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        dense_matrix /= norms
        self.dense_index.add(dense_matrix)

        # ── Add sparse and colbert per chunk ──
        for i, (chunk_id, emb) in enumerate(zip(chunk_ids, embeddings)):
            idx = start_idx + i

            # Sparse inverted index
            for token_id, weight in emb.lexical.items():
                self.inverted_index[int(token_id)].append((idx, float(weight)))

            # ColBERT multi-vector
            self.colbert_store[idx] = emb.colbert.astype(np.float32)

            # Metadata
            self.chunk_ids.append(chunk_id)
            indices.append(idx)

        self.chunk_count += len(chunk_ids)

        logger.info(
            f"Batch indexed {len(chunk_ids)} chunks "
            f"(total: {self.chunk_count})"
        )
        return indices

    # ── Search Components ──────────────────────────────────

    def _dense_search(
        self, query_dense: np.ndarray, k: int, allowed_chunk_indices: Optional[Set[int]] = None
    ) -> Dict[int, float]:
        """FAISS inner-product search (cosine on normalized vectors).

        Vespa equivalent:
            cosine_similarity(query(q_dense), attribute(dense_rep), x)
        """
        if self.chunk_count == 0:
            return {}

        query_vec = query_dense.reshape(1, -1).astype(np.float32)
        norm = np.linalg.norm(query_vec)
        if norm > 0:
            query_vec /= norm

        # Expand search if filtering
        fetch_k = min(k * 20 if allowed_chunk_indices is not None else k, self.chunk_count)
        scores, indices = self.dense_index.search(query_vec, fetch_k)

        results = {}
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:
                idx = int(idx)
                if allowed_chunk_indices is not None and idx not in allowed_chunk_indices:
                    continue
                results[idx] = float(score)
                if len(results) >= k:
                    break

        return results

    def _lexical_search(
        self, query_lexical: Dict[int, float], k: int, allowed_chunk_indices: Optional[Set[int]] = None
    ) -> Dict[int, float]:
        """Sparse dot product via inverted index.

        Vespa equivalent:
            sum(query(q_lexical) * attribute(lexical_rep))
        """
        scores: Dict[int, float] = defaultdict(float)

        for token_id, q_weight in query_lexical.items():
            token_id_int = int(token_id)
            if token_id_int in self.inverted_index:
                for chunk_idx, doc_weight in self.inverted_index[token_id_int]:
                    if allowed_chunk_indices is not None and chunk_idx not in allowed_chunk_indices:
                        continue
                    scores[chunk_idx] += q_weight * doc_weight

        # Return top-k
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return dict(sorted_scores[:k])

    def _colbert_maxsim(
        self, query_colbert: np.ndarray, candidate_indices: List[int]
    ) -> Dict[int, float]:
        """ColBERT MaxSim scoring on candidate set.

        Vespa equivalent:
            sum(
                reduce(sum(query(q_colbert) * attribute(colbert_rep), x), max, t),
                qt
            ) / query(q_len_colbert)

        For each query token, find max similarity across all doc tokens,
        then average across query tokens.
        """
        scores = {}
        q_len = query_colbert.shape[0]

        if q_len == 0:
            return scores

        # Normalize query ColBERT vectors
        q_norms = np.linalg.norm(query_colbert, axis=1, keepdims=True)
        q_norms[q_norms == 0] = 1.0
        query_normed = query_colbert / q_norms

        for idx in candidate_indices:
            if idx not in self.colbert_store:
                continue

            doc_colbert = self.colbert_store[idx]

            # Normalize doc ColBERT vectors
            d_norms = np.linalg.norm(doc_colbert, axis=1, keepdims=True)
            d_norms[d_norms == 0] = 1.0
            doc_normed = doc_colbert / d_norms

            # Similarity matrix: (q_tokens, d_tokens)
            sim_matrix = query_normed @ doc_normed.T

            # MaxSim: for each query token, take max over doc tokens
            max_sims = sim_matrix.max(axis=1)

            # Average over query tokens (divide by q_len)
            scores[idx] = float(max_sims.sum() / q_len)

        return scores

    # ── Fusion Strategies ──────────────────────────────────

    def _vespa_fusion(
        self,
        dense_scores: Dict[int, float],
        lexical_scores: Dict[int, float],
        colbert_scores: Dict[int, float],
        all_candidates: set,
    ) -> List[Tuple[int, float, Dict[str, float]]]:
        """Vespa m3hybrid linear combination.

        score = 0.4*dense + 0.2*lexical + 0.4*max_sim
        """
        combined = []
        for idx in all_candidates:
            d = dense_scores.get(idx, 0.0)
            l = lexical_scores.get(idx, 0.0)
            c = colbert_scores.get(idx, 0.0)

            final = (
                self.dense_weight * d
                + self.lexical_weight * l
                + self.colbert_weight * c
            )

            combined.append(
                (idx, final, {"dense": d, "lexical": l, "colbert": c})
            )

        combined.sort(key=lambda x: x[1], reverse=True)
        return combined

    def _rrf_fusion(
        self,
        dense_scores: Dict[int, float],
        lexical_scores: Dict[int, float],
        colbert_scores: Dict[int, float],
    ) -> List[Tuple[int, float, Dict[str, float]]]:
        """Reciprocal Rank Fusion — intelligent rank-based merging.

        RRF formula: score(d) = Σ(1 / (k + rank_i(d)))

        Advantages over linear combination:
            • No score normalization needed
            • Handles different score scales gracefully
            • More robust to outliers
            • Emphasizes consistently high-ranked items
        """
        # Build rank maps from sorted scores
        dense_ranked = sorted(dense_scores.items(), key=lambda x: x[1], reverse=True)
        lexical_ranked = sorted(
            lexical_scores.items(), key=lambda x: x[1], reverse=True
        )
        colbert_ranked = sorted(
            colbert_scores.items(), key=lambda x: x[1], reverse=True
        )

        dense_ranks = {idx: r + 1 for r, (idx, _) in enumerate(dense_ranked)}
        lexical_ranks = {idx: r + 1 for r, (idx, _) in enumerate(lexical_ranked)}
        colbert_ranks = {idx: r + 1 for r, (idx, _) in enumerate(colbert_ranked)}

        # Union of all candidates
        all_candidates = (
            set(dense_scores) | set(lexical_scores) | set(colbert_scores)
        )

        results = []
        for idx in all_candidates:
            rrf_score = 0.0

            if idx in dense_ranks:
                rrf_score += 1.0 / (self.rrf_k + dense_ranks[idx])
            if idx in lexical_ranks:
                rrf_score += 1.0 / (self.rrf_k + lexical_ranks[idx])
            if idx in colbert_ranks:
                rrf_score += 1.0 / (self.rrf_k + colbert_ranks[idx])

            component = {
                "dense": dense_scores.get(idx, 0.0),
                "lexical": lexical_scores.get(idx, 0.0),
                "colbert": colbert_scores.get(idx, 0.0),
                "dense_rank": dense_ranks.get(idx, -1),
                "lexical_rank": lexical_ranks.get(idx, -1),
                "colbert_rank": colbert_ranks.get(idx, -1),
            }

            results.append((idx, rrf_score, component))

        results.sort(key=lambda x: x[1], reverse=True)
        return results

    # ── Main Search Method ─────────────────────────────────

    def hybrid_search(
        self,
        query_embedding: M3Embedding,
        k: int = 5,
        retrieval_k: int = 50,
        allowed_chunk_indices: Optional[Set[int]] = None,
    ) -> List[Tuple[int, float, Dict[str, float]]]:
        """Hybrid search combining all three M3 representations.

        Pipeline:
            Stage 1: Dense (FAISS k=50) + Lexical (inverted k=50) → candidates
            Stage 2: ColBERT MaxSim on candidate union (expensive but precise)
            Stage 3: Fusion (vespa linear or RRF)

        Args:
            query_embedding: M3Embedding with all three representations
            k: Number of final results to return
            retrieval_k: Candidates to pull from each retriever
            allowed_chunk_indices: Restrict search space for metadata filtering

        Returns:
            List of (chunk_idx, combined_score, component_scores_dict)
        """
        # Stage 1: Gather candidates
        dense_scores = self._dense_search(query_embedding.dense, retrieval_k, allowed_chunk_indices)
        lexical_scores = self._lexical_search(query_embedding.lexical, retrieval_k, allowed_chunk_indices)

        all_candidates = set(dense_scores.keys()) | set(lexical_scores.keys())
        
        if allowed_chunk_indices is not None:
            all_candidates = all_candidates & allowed_chunk_indices
            
        if not all_candidates:
            return []

        # Stage 2: ColBERT MaxSim on candidate set only
        colbert_scores = self._colbert_maxsim(
            query_embedding.colbert, list(all_candidates)
        )

        # Stage 3: Fusion
        if self.fusion_strategy == "rrf":
            combined = self._rrf_fusion(
                dense_scores, lexical_scores, colbert_scores
            )
        else:
            combined = self._vespa_fusion(
                dense_scores, lexical_scores, colbert_scores, all_candidates
            )

        return combined[:k]

    # ── Persistence ────────────────────────────────────────

    def save(self) -> None:
        """Save index to disk."""
        if not self.store_path:
            logger.warning("No store_path set — skipping save")
            return

        os.makedirs(self.store_path, exist_ok=True)

        # FAISS dense index
        faiss.write_index(
            self.dense_index, os.path.join(self.store_path, "dense.faiss")
        )

        # Everything else as JSON
        metadata = {
            "chunk_count": self.chunk_count,
            "chunk_ids": self.chunk_ids,
            "config": {
                "dimension": self.dimension,
                "fusion_strategy": self.fusion_strategy,
                "dense_weight": self.dense_weight,
                "lexical_weight": self.lexical_weight,
                "colbert_weight": self.colbert_weight,
                "rrf_k": self.rrf_k,
            },
        }

        with open(
            os.path.join(self.store_path, "metadata.json"), "w"
        ) as f:
            json.dump(metadata, f)

        # Inverted index (convert int keys to str for JSON)
        inv_idx_serializable = {
            str(k): v for k, v in self.inverted_index.items()
        }
        with open(
            os.path.join(self.store_path, "inverted_index.json"), "w"
        ) as f:
            json.dump(inv_idx_serializable, f)

        # ColBERT store as .npz
        colbert_data = {
            str(k): v for k, v in self.colbert_store.items()
        }
        np.savez_compressed(
            os.path.join(self.store_path, "colbert_store.npz"),
            **colbert_data,
        )

        logger.info(
            f"Index saved to {self.store_path} "
            f"({self.chunk_count} chunks)"
        )

    def load(self) -> bool:
        """Load index from disk.

        Returns:
            True if loaded successfully, False otherwise
        """
        if not self.store_path:
            return False

        meta_path = os.path.join(self.store_path, "metadata.json")
        if not os.path.exists(meta_path):
            return False

        try:
            # Metadata
            with open(meta_path) as f:
                saved = json.load(f)

            self.chunk_count = saved["chunk_count"]
            self.chunk_ids = saved["chunk_ids"]

            config = saved.get("config", {})
            self.fusion_strategy = config.get(
                "fusion_strategy", self.fusion_strategy
            )
            self.dense_weight = config.get("dense_weight", self.dense_weight)
            self.lexical_weight = config.get(
                "lexical_weight", self.lexical_weight
            )
            self.colbert_weight = config.get(
                "colbert_weight", self.colbert_weight
            )
            self.rrf_k = config.get("rrf_k", self.rrf_k)

            # FAISS
            faiss_path = os.path.join(self.store_path, "dense.faiss")
            if os.path.exists(faiss_path):
                self.dense_index = faiss.read_index(faiss_path)

            # Inverted index
            inv_path = os.path.join(self.store_path, "inverted_index.json")
            if os.path.exists(inv_path):
                with open(inv_path) as f:
                    raw = json.load(f)
                self.inverted_index = defaultdict(
                    list, {int(k): v for k, v in raw.items()}
                )

            # ColBERT store
            colbert_path = os.path.join(
                self.store_path, "colbert_store.npz"
            )
            if os.path.exists(colbert_path):
                data = np.load(colbert_path, allow_pickle=False)
                self.colbert_store = {int(k): data[k] for k in data.files}

            logger.info(
                f"Index loaded from {self.store_path} "
                f"({self.chunk_count} chunks, fusion={self.fusion_strategy})"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to load index: {e}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """Return index statistics."""
        return {
            "chunk_count": self.chunk_count,
            "dimension": self.dimension,
            "fusion_strategy": self.fusion_strategy,
            "weights": {
                "dense": self.dense_weight,
                "lexical": self.lexical_weight,
                "colbert": self.colbert_weight,
            },
            "rrf_k": self.rrf_k,
            "inverted_index_tokens": len(self.inverted_index),
            "colbert_stored": len(self.colbert_store),
        }


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


class LocalCrossEncoderReranker:
    """In-process reranker using sentence-transformers CrossEncoder.

    Runs BAAI/bge-reranker-v2-m3 (or similar) locally without needing
    an external server like LM Studio. Ideal for Docker / Render deployment.

    Same interface as LMStudioReranker (should_rerank / rerank).
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        min_chunks_to_rerank: int = 8,
        top_k_after_rerank: int = 5,
        device: Optional[str] = None,
        cache_folder: Optional[str] = None,
    ):
        try:
            from sentence_transformers import CrossEncoder
        except ImportError:
            raise ImportError(
                "sentence-transformers not installed. Run: pip install sentence-transformers"
            )

        if device is None:
            device = "cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu"

        self.model_name = model_name
        self.min_chunks_to_rerank = min_chunks_to_rerank
        self.top_k_after_rerank = top_k_after_rerank
        self.device = device

        logger.info(f"Loading CrossEncoder reranker: {model_name} on {device}")
        kwargs = {"device": device, "trust_remote_code": True}
        if cache_folder:
            kwargs["cache_folder"] = cache_folder
        self._model = CrossEncoder(model_name, **kwargs)
        logger.info("CrossEncoder reranker loaded successfully")

    def should_rerank(self, num_candidates: int) -> bool:
        return num_candidates > self.min_chunks_to_rerank

    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: Optional[int] = None,
    ) -> List[Tuple[int, float]]:
        if not documents:
            return []

        top_k = top_k or self.top_k_after_rerank

        try:
            pairs = [[query, doc] for doc in documents]
            scores = self._model.predict(pairs, show_progress_bar=False)

            indexed_scores = list(enumerate(scores.tolist()))
            indexed_scores.sort(key=lambda x: x[1], reverse=True)
            return indexed_scores[:top_k]

        except Exception as e:
            logger.error(f"CrossEncoder reranker error: {e}")
            return [(i, 0.0) for i in range(min(top_k, len(documents)))]


# ─────────────────────────────────────────────────────────────
# Hybrid Retriever (Main Class)
# ─────────────────────────────────────────────────────────────


class HybridRetriever:
    """Complete hybrid retrieval system.

    Combines:
        1. BGE-M3 multi-representation encoding (dense + sparse + ColBERT)
        2. Vespa-style hybrid index with configurable fusion
        3. LM Studio reranking with BGE-reranker-v2-m3

    Pipeline:
        Query → BGE-M3 encode
              ↓
        ┌─────────────┬──────────────┬──────────────┐
        │ Dense (FAISS)│ Lexical (inv)│ ColBERT (ms) │
        │   k=50       │    k=50      │  on union     │
        └──────┬───────┴──────┬───────┴──────┬────────┘
               └──────────────┴──────────────┘
                             ↓
               Fusion (configurable):
               • Vespa: 0.4*dense + 0.2*lexical + 0.4*max_sim
               • RRF: Σ(1/(60 + rank_i))
                             ↓
               Candidates > 8?  →  BGE-reranker-v2-m3 (LM Studio)
                             ↓
                         Top 5 results
    """

    def __init__(
        self,
        chunks: Dict,
        store_path: Optional[str] = None,
        encoder: Optional[BGEM3Encoder] = None,
        index: Optional[VespaStyleIndex] = None,
        reranker: Optional[Any] = None,
        # BGE-M3 encoder settings
        device: Optional[str] = None,
        use_fp16: bool = True,
        max_length: int = 8192,
        batch_size: int = 12,
        # Fusion settings
        dense_weight: float = 0.4,
        lexical_weight: float = 0.2,
        colbert_weight: float = 0.4,
        fusion_strategy: str = "rrf",
        rrf_k: int = 60,
        # Reranker settings
        use_reranker: bool = True,
        lm_studio_base_url: str = "http://127.0.0.1:1234/v1",
        lm_studio_reranker_model: str = "gpustack/text-embedding-bge-reranker-v2-m3",
        min_chunks_to_rerank: int = 8,
        top_k_after_rerank: int = 5,
    ):
        """Initialize the hybrid retriever.

        Args:
            chunks: Dict mapping chunk_id → {text, metadata}
            store_path: Directory to persist/load the index
            encoder: Pre-built BGEM3Encoder (or auto-created)
            index: Pre-built VespaStyleIndex (or auto-created)
            reranker: Pre-built reranker (LMStudioReranker or HuggingFaceReranker)
            device: Device for BGE-M3 ('cuda' or 'cpu')
            use_fp16: Use FP16 for GPU inference
            max_length: Max input length for BGE-M3
            batch_size: Encoding batch size
            dense_weight: Vespa dense weight (0.4)
            lexical_weight: Vespa lexical weight (0.2)
            colbert_weight: Vespa ColBERT weight (0.4)
            fusion_strategy: 'rrf' (recommended) or 'vespa'
            rrf_k: RRF constant (default 60)
            use_reranker: Enable reranking
            lm_studio_base_url: LM Studio API base URL (if using LM Studio)
            lm_studio_reranker_model: Reranker model name (if using LM Studio)
            min_chunks_to_rerank: Rerank only when candidates > this
            top_k_after_rerank: Number of results after reranking
        """
        self.chunks = chunks
        self.store_path = store_path
        self.use_reranker = use_reranker
        self.stats = HybridRetrieverStats()

        # ── Encoder ──
        self.encoder = encoder or BGEM3Encoder(
            device=device,
            use_fp16=use_fp16,
            max_length=max_length,
            batch_size=batch_size,
        )

        # ── Index ──
        self.index = index or VespaStyleIndex(
            dimension=self.encoder.dimension,
            store_path=store_path,
            dense_weight=dense_weight,
            lexical_weight=lexical_weight,
            colbert_weight=colbert_weight,
            fusion_strategy=fusion_strategy,
            rrf_k=rrf_k,
        )

        # ── Reranker ──
        if use_reranker:
            self.reranker = reranker or NvidiaReranker(
                min_chunks_to_rerank=min_chunks_to_rerank,
                top_k_after_rerank=top_k_after_rerank,
            )
        else:
            self.reranker = None

        # ── Internal mapping: chunk_id → text ──
        self._chunk_texts: Dict[str, str] = {}
        self._chunk_metadata: Dict[str, Dict] = {}

        for chunk_id, chunk_data in chunks.items():
            if isinstance(chunk_data, dict):
                self._chunk_texts[chunk_id] = chunk_data.get("text", "")
                self._chunk_metadata[chunk_id] = chunk_data.get("metadata", {})
            elif hasattr(chunk_data, "text"):
                self._chunk_texts[chunk_id] = chunk_data.text
                self._chunk_metadata[chunk_id] = getattr(
                    chunk_data, "metadata", {}
                )
            else:
                self._chunk_texts[chunk_id] = str(chunk_data)
                self._chunk_metadata[chunk_id] = {}

        logger.info(
            f"HybridRetriever initialized: "
            f"{len(chunks)} chunks, "
            f"fusion={fusion_strategy}, "
            f"reranker={'on' if use_reranker else 'off'}"
        )

    # ── Index Building ─────────────────────────────────────

    def build_index(self, force_rebuild: bool = False) -> None:
        """Encode all chunks with BGE-M3 and build the hybrid index.

        Args:
            force_rebuild: Rebuild even if a saved index exists
        """
        # Try loading saved index first
        if not force_rebuild and self.index.load():
            logger.info("Loaded existing index from disk")
            return

        logger.info(
            f"Building hybrid index for {len(self._chunk_texts)} chunks..."
        )
        start = time.time()

        chunk_ids = list(self._chunk_texts.keys())
        texts = [self._chunk_texts[cid] for cid in chunk_ids]

        # Encode all chunks with BGE-M3
        logger.info("Encoding chunks with BGE-M3 (dense + sparse + ColBERT)...")
        embeddings = self.encoder.encode_batch(texts)

        # Add to index
        logger.info("Indexing all representations...")
        self.index.add_batch(chunk_ids, embeddings)

        # Persist
        self.index.save()

        elapsed = time.time() - start
        logger.info(
            f"Index built in {elapsed:.1f}s "
            f"({len(chunk_ids)} chunks, "
            f"{len(chunk_ids)/elapsed:.1f} chunks/s)"
        )

    # ── Retrieval ──────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        k: int = 5,
        retrieval_k: int = 50,
        force_rerank: Optional[bool] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[HybridRetrievalResult]:
        """Full retrieval pipeline: encode → search → fuse → (rerank).

        Args:
            query: Query text
            k: Number of final results
            retrieval_k: Candidates per retriever stage
            force_rerank: Override automatic rerank decision
            filters: Optional metadata filtering criteria

        Returns:
            List of HybridRetrievalResult objects
        """
        start = time.time()

        # 1. Encode query with BGE-M3
        query_embedding = self.encoder.encode(query)

        # 1.5 Eval filters to get subset of allowed chunk indices
        allowed_chunk_indices = None
        if filters:
            allowed_chunk_indices = set()
            for chunk_idx, chunk_id in enumerate(self.index.chunk_ids):
                meta = self._chunk_metadata.get(chunk_id, {})
                match = True
                for fk, fv in filters.items():
                    if meta.get(fk) != fv:
                        match = False
                        break
                if match:
                    allowed_chunk_indices.add(chunk_idx)
            
            # Fast-fail if filters are too restrictive
            if not allowed_chunk_indices:
                return []

        # 2. Hybrid search (dense + lexical + ColBERT → fusion)
        search_k = max(k * 3, retrieval_k) if self.reranker else k
        raw_results = self.index.hybrid_search(
            query_embedding, 
            k=search_k, 
            retrieval_k=retrieval_k,
            allowed_chunk_indices=allowed_chunk_indices
        )

        if not raw_results:
            return []

        # 3. Build result objects
        results = []
        for chunk_idx, score, components in raw_results:
            if chunk_idx < len(self.index.chunk_ids):
                chunk_id = self.index.chunk_ids[chunk_idx]
                results.append(
                    HybridRetrievalResult(
                        chunk_id=chunk_id,
                        text=self._chunk_texts.get(chunk_id, ""),
                        score=score,
                        component_scores=components,
                        metadata=self._chunk_metadata.get(chunk_id, {}),
                    )
                )

        # 4. Rerank if applicable
        should_rerank = force_rerank if force_rerank is not None else (
            self.reranker is not None
            and self.reranker.should_rerank(len(results))
        )

        if should_rerank and self.reranker:
            self.stats.reranker_calls += 1
            doc_texts = [r.text for r in results]

            reranked = self.reranker.rerank(query, doc_texts, top_k=k)

            reranked_results = []
            for orig_idx, reranker_score in reranked:
                if orig_idx < len(results):
                    r = results[orig_idx]
                    r.reranker_score = reranker_score
                    r.score = reranker_score  # Override with reranker score
                    reranked_results.append(r)

            results = reranked_results
        else:
            if self.reranker:
                self.stats.reranker_skips += 1
            results = results[:k]

        # 5. Update stats
        elapsed_ms = (time.time() - start) * 1000
        self.stats.queries_processed += 1
        self.stats.total_query_time_ms += elapsed_ms
        self.stats.avg_query_time_ms = (
            self.stats.total_query_time_ms / self.stats.queries_processed
        )

        return results

    def get_stats(self) -> Dict[str, Any]:
        """Return retriever statistics."""
        return {
            "queries_processed": self.stats.queries_processed,
            "avg_query_time_ms": round(self.stats.avg_query_time_ms, 2),
            "reranker_calls": self.stats.reranker_calls,
            "reranker_skips": self.stats.reranker_skips,
            "index": self.index.get_stats(),
        }


# ─────────────────────────────────────────────────────────────
# Legacy Simple Retriever (backwards compatibility)
# ─────────────────────────────────────────────────────────────


@dataclass
class RetrievalResult:
    """Result from simple RAG retrieval (legacy format)"""

    chunk_id: str
    document_id: str
    text: str
    distance: float
    metadata: Dict[str, Any]


class RAGRetriever:
    """Simple dense-only retriever for backwards compatibility.
    
    Now supports optional reranking for improved retrieval quality.
    For new projects with multi-vector retrieval, use HybridRetriever instead.
    """

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


# ─────────────────────────────────────────────────────────────
# Factory Functions
# ─────────────────────────────────────────────────────────────


def create_hybrid_retriever(
    chunks: Dict,
    store_path: Optional[str] = None,
    device: Optional[str] = None,
    use_fp16: bool = True,
    use_reranker: bool = True,
    fusion_strategy: str = "rrf",
    dense_weight: float = 0.4,
    lexical_weight: float = 0.2,
    colbert_weight: float = 0.4,
    rrf_k: int = 60,
    lm_studio_base_url: str = "http://127.0.0.1:1234/v1",
    lm_studio_reranker_model: str = "gpustack/text-embedding-bge-reranker-v2-m3",
    min_chunks_to_rerank: int = 8,
    top_k_after_rerank: int = 5,
) -> HybridRetriever:
    """Factory function to create a hybrid retriever.

    Args:
        chunks: Dict mapping chunk_id → {text, metadata}
        store_path: Directory to persist index
        device: 'cuda' or 'cpu' (auto-detected if None)
        use_fp16: FP16 for GPU inference
        use_reranker: Enable BGE-reranker-v2-m3
        fusion_strategy: 'rrf' (recommended) or 'vespa'
        dense_weight: Vespa dense weight (only for vespa fusion)
        lexical_weight: Vespa lexical weight (only for vespa fusion)
        colbert_weight: Vespa ColBERT weight (only for vespa fusion)
        rrf_k: RRF constant (only for rrf fusion)
        lm_studio_base_url: LM Studio API URL
        lm_studio_reranker_model: Reranker model name
        min_chunks_to_rerank: Rerank threshold
        top_k_after_rerank: Top-k after reranking

    Returns:
        Configured HybridRetriever instance
    """
    return HybridRetriever(
        chunks=chunks,
        store_path=store_path,
        device=device,
        use_fp16=use_fp16,
        use_reranker=use_reranker,
        dense_weight=dense_weight,
        lexical_weight=lexical_weight,
        colbert_weight=colbert_weight,
        fusion_strategy=fusion_strategy,
        rrf_k=rrf_k,
        lm_studio_base_url=lm_studio_base_url,
        lm_studio_reranker_model=lm_studio_reranker_model,
        min_chunks_to_rerank=min_chunks_to_rerank,
        top_k_after_rerank=top_k_after_rerank,
    )
