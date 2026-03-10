"""Qdrant-backed vector store for session-scoped retrieval."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from config.config import (
    QDRANT_API_KEY,
    QDRANT_COLLECTION,
    QDRANT_DISTANCE,
    QDRANT_TIMEOUT_SECONDS,
    QDRANT_URL,
)
from src.modules.VectorStore import SearchResult
from src.utils.Logger import get_logger

logger = get_logger(__name__)

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models
except Exception:
    QdrantClient = None
    models = None


def _distance_enum(distance: str):
    d = (distance or "cosine").lower()
    if d == "dot":
        return models.Distance.DOT
    if d == "euclid":
        return models.Distance.EUCLID
    return models.Distance.COSINE


def _to_point_id(chunk_id: str) -> str:
    return str(chunk_id)


def _qdrant_client() -> QdrantClient:
    if QdrantClient is None:
        raise ImportError("qdrant-client not installed. Run: pip install qdrant-client")
    if not QDRANT_URL:
        raise ValueError("QDRANT_URL is required when VECTOR_BACKEND=qdrant")

    kwargs = {
        "url": QDRANT_URL,
        "timeout": QDRANT_TIMEOUT_SECONDS,
    }
    if QDRANT_API_KEY:
        kwargs["api_key"] = QDRANT_API_KEY
    return QdrantClient(**kwargs)


class QdrantSessionStore:
    """Session-aware Qdrant store with FAISS-compatible methods."""

    def __init__(self, session_id: str, embedding_dimension: int):
        self.session_id = session_id
        self.embedding_dimension = int(embedding_dimension)
        self.collection_name = QDRANT_COLLECTION
        self.metadata_store: Dict[str, Any] = {}
        self.client = _qdrant_client()
        self._init_collection()

    def _init_collection(self) -> None:
        if models is None:
            raise ImportError("qdrant-client not installed")

        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.embedding_dimension,
                    distance=_distance_enum(QDRANT_DISTANCE),
                ),
            )

        # Create payload indexes for efficient filtering
        for field, schema in [
            ("session_id", models.PayloadSchemaType.KEYWORD),
            ("document_id", models.PayloadSchemaType.KEYWORD),
            ("chat_id", models.PayloadSchemaType.KEYWORD),
            ("user_id", models.PayloadSchemaType.KEYWORD),
            ("is_guest", models.PayloadSchemaType.BOOL),
            ("content_hash", models.PayloadSchemaType.KEYWORD),
        ]:
            try:
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field,
                    field_schema=schema,
                )
            except Exception:
                pass

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
            raise ValueError("ids or metadata.chunk_id values are required for qdrant storage")

        points: List[models.PointStruct] = []
        for vec, meta, chunk_id in zip(vectors, metadata, ids):
            payload = dict(meta or {})
            payload["session_id"] = self.session_id
            payload.setdefault("chunk_id", chunk_id)
            payload.setdefault("document_id", payload.get("document_id", ""))
            payload.setdefault("chat_id", payload.get("chat_id", ""))
            payload.setdefault("user_id", payload.get("user_id", ""))
            payload.setdefault("is_guest", payload.get("is_guest", False))
            payload.setdefault("content_hash", payload.get("content_hash", ""))
            payload.setdefault("created_at", payload.get("created_at", ""))
            payload.setdefault("text", payload.get("text", ""))

            points.append(
                models.PointStruct(
                    id=_to_point_id(chunk_id),
                    vector=np.asarray(vec, dtype=np.float32).reshape(-1).tolist(),
                    payload=payload,
                )
            )

        self.client.upsert(collection_name=self.collection_name, points=points, wait=False)
        return [str(i) for i in ids]

    def add_vectors_batch(
        self,
        vectors: np.ndarray,
        metadata_list: List[Dict[str, Any]],
        ids: Optional[List[str]] = None,
        batch_size: int = 10000,
    ) -> List[str]:
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
        return True

    def save(self, _path: Optional[str] = None) -> bool:
        return True

    def get_size(self) -> int:
        result = self.client.count(
            collection_name=self.collection_name,
            count_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="session_id",
                        match=models.MatchValue(value=self.session_id),
                    )
                ]
            ),
            exact=False,
        )
        return int(result.count or 0)

    def search(
        self,
        query_vector: np.ndarray,
        k: int = 5,
        threshold: Optional[float] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        must_filters = [
            models.FieldCondition(
                key="session_id",
                match=models.MatchValue(value=self.session_id),
            )
        ]

        if filters:
            for key, value in filters.items():
                must_filters.append(
                    models.FieldCondition(
                        key=key,
                        match=models.MatchValue(value=value),
                    )
                )

        scored = self.client.search(
            collection_name=self.collection_name,
            query_vector=np.asarray(query_vector, dtype=np.float32).reshape(-1).tolist(),
            query_filter=models.Filter(must=must_filters),
            limit=max(k, 1),
            with_payload=True,
            with_vectors=False,
        )

        results: List[SearchResult] = []
        for p in scored:
            payload = p.payload or {}
            chunk_id = str(payload.get("chunk_id") or p.id)
            document_id = str(payload.get("document_id") or "")
            text = str(payload.get("text") or "")
            similarity = float(p.score)
            distance = float(max(0.0, 1.0 - similarity))

            if threshold is not None and distance > threshold:
                continue

            result = SearchResult(
                document_id=document_id,
                chunk_id=chunk_id,
                text=text,
                similarity_score=similarity,
                distance=distance,
                metadata=dict(payload),
            )
            self.metadata_store[chunk_id] = result
            results.append(result)
        return results


def delete_session_vectors(session_id: str) -> None:
    """Delete all vectors for a session id."""
    client = _qdrant_client()
    client.delete(
        collection_name=QDRANT_COLLECTION,
        points_selector=models.FilterSelector(
            filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="session_id",
                        match=models.MatchValue(value=session_id),
                    )
                ]
            )
        ),
        wait=False,
    )


def qdrant_collection_count() -> int:
    """Return approximate total number of points in collection."""
    client = _qdrant_client()
    result = client.count(collection_name=QDRANT_COLLECTION, exact=False)
    return int(result.count or 0)


def delete_points_by_chat(chat_id: str, user_id: str = None) -> None:
    """Delete all Qdrant points for a specific chat_id, optionally scoped to user."""
    client = _qdrant_client()
    must = [
        models.FieldCondition(key="chat_id", match=models.MatchValue(value=chat_id)),
    ]
    if user_id:
        must.append(
            models.FieldCondition(key="user_id", match=models.MatchValue(value=user_id)),
        )
    client.delete(
        collection_name=QDRANT_COLLECTION,
        points_selector=models.FilterSelector(filter=models.Filter(must=must)),
        wait=False,
    )


def delete_points_by_session_batch(session_ids: List[str]) -> None:
    """Batch delete Qdrant points for multiple session_ids."""
    if not session_ids:
        return
    client = _qdrant_client()
    for sid in session_ids:
        try:
            client.delete(
                collection_name=QDRANT_COLLECTION,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="session_id",
                                match=models.MatchValue(value=sid),
                            )
                        ]
                    )
                ),
                wait=False,
            )
        except Exception as e:
            logger.warning(f"Failed to batch-delete qdrant points for session {sid}: {e}")
