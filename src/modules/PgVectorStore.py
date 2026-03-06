"""PostgreSQL pgvector-backed vector store for session-scoped retrieval."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import numpy as np
import psycopg2
import psycopg2.extras

from config.config import (
    PGVECTOR_DISTANCE,
    PGVECTOR_PROBES,
    postgres_connect_kwargs,
)
from src.modules.VectorStore import SearchResult
from src.utils.Logger import get_logger

logger = get_logger(__name__)


def _vector_literal(vector: np.ndarray) -> str:
    """Return pgvector text literal format: [x1,x2,...]."""
    arr = np.asarray(vector, dtype=np.float32).reshape(-1)
    return "[" + ",".join(f"{float(x):.8f}" for x in arr.tolist()) + "]"


class PGVectorSessionStore:
    """Session-aware pgvector store with FAISS-compatible methods."""

    def __init__(self, session_id: str, embedding_dimension: int):
        self.session_id = session_id
        self.embedding_dimension = embedding_dimension
        self.distance_metric = PGVECTOR_DISTANCE
        self.probes = PGVECTOR_PROBES
        self.metadata_store: Dict[str, Any] = {}
        self._init_schema()

    def _connect(self):
        return psycopg2.connect(**postgres_connect_kwargs(connect_timeout=10))

    def _distance_sql(self) -> str:
        if self.distance_metric == "l2":
            return "embedding <-> %s::vector"
        if self.distance_metric == "inner":
            return "embedding <#> %s::vector"
        return "embedding <=> %s::vector"

    def _init_schema(self) -> None:
        conn = self._connect()
        try:
            with conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
                cur.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS chunk_embeddings (
                        chunk_id TEXT PRIMARY KEY,
                        session_id TEXT NOT NULL,
                        document_id TEXT NOT NULL,
                        text TEXT NOT NULL,
                        embedding vector({int(self.embedding_dimension)}) NOT NULL,
                        metadata_json JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    )
                    """
                )
                cur.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_chunk_embeddings_session
                    ON chunk_embeddings (session_id)
                    """
                )
                # Optional ANN index; keep best-effort to avoid blocking startup.
                try:
                    cur.execute(
                        """
                        CREATE INDEX IF NOT EXISTS idx_chunk_embeddings_embedding_cosine
                        ON chunk_embeddings USING ivfflat (embedding vector_cosine_ops)
                        WITH (lists = 100)
                        """
                    )
                except Exception as idx_err:
                    logger.warning(f"pgvector ivfflat index not created yet: {idx_err}")
            conn.commit()
        finally:
            conn.close()

    def add_vectors(
        self,
        vectors: List[np.ndarray],
        metadata: List[Dict[str, Any]],
        ids: Optional[List[str]] = None,
    ) -> List[str]:
        if len(vectors) != len(metadata):
            raise ValueError("vectors and metadata must have same length")
        if ids is None:
            ids = [m.get("chunk_id") for m in metadata]
        if not ids or any(not i for i in ids):
            raise ValueError("ids or metadata.chunk_id values are required for pgvector storage")

        rows = []
        for vec, meta, chunk_id in zip(vectors, metadata, ids):
            rows.append(
                (
                    chunk_id,
                    self.session_id,
                    meta.get("document_id", ""),
                    meta.get("text", ""),
                    _vector_literal(np.asarray(vec, dtype=np.float32)),
                    json.dumps(meta, ensure_ascii=True),
                )
            )

        conn = self._connect()
        try:
            with conn.cursor() as cur:
                psycopg2.extras.execute_batch(
                    cur,
                    """
                    INSERT INTO chunk_embeddings
                    (chunk_id, session_id, document_id, text, embedding, metadata_json)
                    VALUES (%s, %s, %s, %s, %s::vector, %s::jsonb)
                    ON CONFLICT (chunk_id) DO UPDATE SET
                        session_id = EXCLUDED.session_id,
                        document_id = EXCLUDED.document_id,
                        text = EXCLUDED.text,
                        embedding = EXCLUDED.embedding,
                        metadata_json = EXCLUDED.metadata_json
                    """,
                    rows,
                    page_size=500,
                )
            conn.commit()
        finally:
            conn.close()
        return ids

    def add_vectors_batch(
        self,
        vectors: np.ndarray,
        metadata_list: List[Dict[str, Any]],
        ids: Optional[List[str]] = None,
        batch_size: int = 10000,
    ) -> List[str]:
        """FAISS-compatible batch wrapper."""
        all_ids: List[str] = []
        total = len(vectors)
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            batch_vectors = vectors[start:end]
            batch_metadata = metadata_list[start:end]
            batch_ids = ids[start:end] if ids else None
            created = self.add_vectors(batch_vectors, batch_metadata, batch_ids)
            all_ids.extend(created)
        return all_ids

    def load(self, _path: Optional[str] = None) -> bool:
        """No-op for pgvector; data already persisted in Postgres."""
        return True

    def save(self, _path: Optional[str] = None) -> bool:
        """No-op for pgvector; writes happen on add."""
        return True

    def get_size(self) -> int:
        conn = self._connect()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT COUNT(*) FROM chunk_embeddings WHERE session_id = %s",
                    (self.session_id,),
                )
                return int(cur.fetchone()[0] or 0)
        finally:
            conn.close()

    def search(
        self,
        query_vector: np.ndarray,
        k: int = 5,
        threshold: Optional[float] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        query_literal = _vector_literal(np.asarray(query_vector, dtype=np.float32))
        fetch_k = max(k, k * 50 if filters else k)
        distance_expr = self._distance_sql()

        conn = self._connect()
        try:
            with conn.cursor() as cur:
                cur.execute("SET LOCAL ivfflat.probes = %s", (self.probes,))
                cur.execute(
                    f"""
                    SELECT chunk_id, document_id, text, metadata_json, {distance_expr} AS distance
                    FROM chunk_embeddings
                    WHERE session_id = %s
                    ORDER BY {distance_expr}
                    LIMIT %s
                    """,
                    (query_literal, self.session_id, query_literal, fetch_k),
                )
                rows = cur.fetchall()
        finally:
            conn.close()

        results: List[SearchResult] = []
        for chunk_id, document_id, text, metadata_json, distance in rows:
            if threshold is not None and distance > threshold:
                continue
            md = metadata_json or {}
            if filters:
                matched = True
                for key, expected in filters.items():
                    value = md.get(key)
                    if value is None and key == "document_id":
                        value = document_id
                    if value != expected:
                        matched = False
                        break
                if not matched:
                    continue

            distance_value = float(distance)
            similarity = float(1.0 / (1.0 + max(distance_value, 0.0)))
            result = SearchResult(
                document_id=document_id,
                chunk_id=chunk_id,
                text=text,
                similarity_score=similarity,
                distance=distance_value,
                metadata=md,
            )
            self.metadata_store[chunk_id] = result
            results.append(result)
            if len(results) >= k:
                break
        return results
