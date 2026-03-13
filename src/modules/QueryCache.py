"""Redis-backed caching for Retrieval and LLM responses.

Implements two levels of caching:
1. RetrievalCache: Caches retrieved chunks based on (session_id + query + top_k + retrieval_params)
2. LLMCache: Caches LLM responses based on (session_id + query + context_hash)
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from typing import Any, Dict, List, Optional

from backend.cache.RedisCache import RedisJSONStore
from src.utils.Logger import get_logger

logger = get_logger(__name__)

REDIS_RETRIEVAL_TTL_SECONDS = int(
    os.getenv("REDIS_RETRIEVAL_TTL_SECONDS", os.getenv("REDIS_DEFAULT_TTL_SECONDS", "1800"))
)
REDIS_LLM_TTL_SECONDS = int(
    os.getenv("REDIS_LLM_TTL_SECONDS", os.getenv("REDIS_DEFAULT_TTL_SECONDS", "1800"))
)


class BaseRedisCache:
    """Base class for Redis-backed thread-safe caches."""

    def __init__(self, namespace: str, ttl_seconds: int):
        self.namespace = namespace
        self._store = RedisJSONStore(namespace=namespace, ttl_seconds=ttl_seconds)
        self._lock = threading.Lock()

    def _hash_key(self, *parts: str) -> str:
        content = ":".join(str(p) for p in parts if p is not None)
        return hashlib.sha256(content.encode("utf-8", errors="replace")).hexdigest()

    @staticmethod
    def _normalize_query(query: str) -> str:
        if not isinstance(query, str):
            return ""
        return " ".join(query.lower().strip().split())

    def close(self) -> None:
        """Compatibility no-op for cache lifecycle management."""
        return None

    def delete_older_than(self, minutes: int) -> int:
        """Best-effort deletion retained for cleanup scheduler compatibility."""
        return self._store.delete_older_than(minutes)


class RetrievalCache(BaseRedisCache):
    """Caches retrieval results for identical queries within a session."""

    def __init__(self, cache_dir: Optional[str] = None):
        super().__init__(namespace="retrieval", ttl_seconds=REDIS_RETRIEVAL_TTL_SECONDS)
        if cache_dir:
            logger.debug("RetrievalCache cache_dir ignored in Redis mode")

    def _make_key(
        self,
        session_id: str,
        query: str,
        top_k: int,
        retrieval_params: Optional[dict] = None,
    ) -> str:
        normalized_query = self._normalize_query(query)
        params_str = json.dumps(retrieval_params or {}, sort_keys=True)
        return self._hash_key(session_id, normalized_query, str(top_k), params_str)

    def get_cache(
        self,
        session_id: str,
        query: str,
        top_k: int,
        retrieval_params: Optional[dict] = None,
    ) -> Optional[Dict[str, Any]]:
        cache_key = self._make_key(session_id, query, top_k, retrieval_params)
        record = self._store.get_json(cache_key)
        if not record:
            return None

        payload = record.get("payload")
        if isinstance(payload, dict):
            logger.debug(f"Retrieval cache hit for query: '{query[:50]}...'")
            return payload
        return None

    def set_cache(
        self,
        session_id: str,
        query: str,
        top_k: int,
        result_data: Dict[str, Any],
        retrieval_params: Optional[dict] = None,
    ) -> None:
        cache_key = self._make_key(session_id, query, top_k, retrieval_params)
        record = {
            "created_at": time.time(),
            "session_id": session_id,
            "query": query,
            "top_k": int(top_k),
            "payload": result_data,
        }
        self._store.set_json(cache_key, record)


class LLMCache(BaseRedisCache):
    """Caches LLM responses for identical query/context within a session."""

    def __init__(self, cache_dir: Optional[str] = None):
        super().__init__(namespace="llm", ttl_seconds=REDIS_LLM_TTL_SECONDS)
        if cache_dir:
            logger.debug("LLMCache cache_dir ignored in Redis mode")

    def _compute_context_hash(self, chunks: List[str]) -> str:
        content = "\n---CONTEXT---\n".join(chunks)
        return self._hash_key(content)

    def get_cache(self, session_id: str, query: str, retrieved_chunks: List[str]) -> Optional[Dict[str, Any]]:
        context_hash = self._compute_context_hash(retrieved_chunks)
        normalized_query = self._normalize_query(query)
        cache_key = self._hash_key(session_id, normalized_query, context_hash)

        record = self._store.get_json(cache_key)
        if not record:
            return None

        payload = record.get("payload")
        if isinstance(payload, dict):
            logger.debug(f"LLM cache hit for query: '{query[:50]}...'")
            return payload
        return None

    def set_cache(self, session_id: str, query: str, retrieved_chunks: List[str], response_data: Dict[str, Any]) -> None:
        context_hash = self._compute_context_hash(retrieved_chunks)
        normalized_query = self._normalize_query(query)
        cache_key = self._hash_key(session_id, normalized_query, context_hash)

        record = {
            "created_at": time.time(),
            "session_id": session_id,
            "query": query,
            "context_hash": context_hash,
            "payload": response_data,
        }
        self._store.set_json(cache_key, record)


_retrieval_cache: Optional[RetrievalCache] = None
_llm_cache: Optional[LLMCache] = None
_global_cache_lock = threading.Lock()


def get_retrieval_cache() -> RetrievalCache:
    """Get or create singleton RetrievalCache."""
    global _retrieval_cache
    with _global_cache_lock:
        if _retrieval_cache is None:
            _retrieval_cache = RetrievalCache()
        return _retrieval_cache


def get_llm_cache() -> LLMCache:
    """Get or create singleton LLMCache."""
    global _llm_cache
    with _global_cache_lock:
        if _llm_cache is None:
            _llm_cache = LLMCache()
        return _llm_cache


def close_all_caches() -> None:
    """Reset singleton handles; retained for compatibility."""
    global _retrieval_cache, _llm_cache
    with _global_cache_lock:
        _retrieval_cache = None
        _llm_cache = None
