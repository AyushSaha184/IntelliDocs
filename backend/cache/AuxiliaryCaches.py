"""Auxiliary Redis caches used by agent tools.

These caches are best-effort and must never break request execution.
"""

from __future__ import annotations

import hashlib
import os
import time
from typing import Optional

from backend.cache.RedisCache import RedisJSONStore


def _hash_key(*parts: str) -> str:
    content = ":".join(str(p) for p in parts if p is not None)
    return hashlib.sha256(content.encode("utf-8", errors="replace")).hexdigest()


def _normalize_query(query: str) -> str:
    return " ".join((query or "").lower().strip().split())


def get_cached_document_summary(document_id: str) -> Optional[str]:
    ttl = int(os.getenv("REDIS_SUMMARY_TTL_SECONDS", os.getenv("REDIS_DEFAULT_TTL_SECONDS", "86400")))
    store = RedisJSONStore(namespace="doc_summary", ttl_seconds=ttl)
    key = _hash_key(str(document_id))
    data = store.get_json(key)
    if not data:
        return None
    summary = data.get("summary")
    return summary if isinstance(summary, str) and summary else None


def set_cached_document_summary(document_id: str, summary: str) -> None:
    ttl = int(os.getenv("REDIS_SUMMARY_TTL_SECONDS", os.getenv("REDIS_DEFAULT_TTL_SECONDS", "86400")))
    store = RedisJSONStore(namespace="doc_summary", ttl_seconds=ttl)
    key = _hash_key(str(document_id))
    payload = {
        "document_id": str(document_id),
        "summary": summary,
        "created_at": time.time(),
    }
    store.set_json(key, payload)


def get_cached_approved_answer(session_id: str, query: str) -> Optional[str]:
    ttl = int(os.getenv("REDIS_APPROVED_TTL_SECONDS", os.getenv("REDIS_DEFAULT_TTL_SECONDS", "2592000")))
    store = RedisJSONStore(namespace="approved_answer", ttl_seconds=ttl)
    key = _hash_key(str(session_id), _normalize_query(query))
    data = store.get_json(key)
    if not data:
        return None
    answer = data.get("answer")
    return answer if isinstance(answer, str) and answer else None


def set_cached_approved_answer(session_id: str, query: str, answer: str) -> None:
    ttl = int(os.getenv("REDIS_APPROVED_TTL_SECONDS", os.getenv("REDIS_DEFAULT_TTL_SECONDS", "2592000")))
    store = RedisJSONStore(namespace="approved_answer", ttl_seconds=ttl)
    normalized_query = _normalize_query(query)
    key = _hash_key(str(session_id), normalized_query)
    payload = {
        "session_id": str(session_id),
        "query": normalized_query,
        "answer": answer,
        "created_at": time.time(),
    }
    store.set_json(key, payload)
