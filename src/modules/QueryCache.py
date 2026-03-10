"""
Persistent SQLite-backed caching for Retrieval and LLM Responses.

Implements two levels of caching:
1. RetrievalCache: Caches retrieved chunks based on (session_id + query + top_k)
2. LLMCache: Caches LLM responses based on (session_id + query + context_hash)
"""

import os
import sqlite3
import json
import hashlib
import threading
from typing import Optional, Dict, Any, List

from src.utils.Logger import get_logger

logger = get_logger(__name__)

# Default cache directory
DEFAULT_CACHE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data",
    "cache",
)

# Optional auto-prune controls
CACHE_AUTO_PRUNE_EVERY_WRITES = int(os.getenv("CACHE_AUTO_PRUNE_EVERY_WRITES", "200"))
CACHE_TTL_MINUTES = int(os.getenv("CACHE_TTL_MINUTES", "0"))  # 0 = disabled


class BaseSQLiteCache:
    """Base class for SQLite-backed thread-safe caches."""

    def __init__(
        self,
        db_path: str,
        table_schema: str,
        table_name: str,
        index_statements: Optional[List[str]] = None,
    ):
        self.db_path = db_path
        self.table_name = table_name
        os.makedirs(os.path.dirname(db_path), exist_ok=True)

        self._lock = threading.Lock()
        self._write_count = 0
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute(table_schema)

        for stmt in index_statements or []:
            self._conn.execute(stmt)

        self._conn.commit()

    def _hash_key(self, *parts: str) -> str:
        """Create a SHA-256 hash from string parts."""
        content = ":".join(str(p) for p in parts if p is not None)
        return hashlib.sha256(content.encode("utf-8", errors="replace")).hexdigest()

    @staticmethod
    def _normalize_query(query: str) -> str:
        if not isinstance(query, str):
            return ""
        return " ".join(query.lower().strip().split())

    def close(self):
        """Close the database connection."""
        with self._lock:
            self._conn.close()

    def delete_older_than(self, minutes: int) -> int:
        """Delete cache entries older than N minutes."""
        with self._lock:
            try:
                cursor = self._conn.execute(
                    f"DELETE FROM {self.table_name} WHERE created_at < datetime('now', '-{minutes} minute')"
                )
                deleted_count = cursor.rowcount
                self._conn.commit()
                if deleted_count > 0:
                    logger.info(
                        f"Deleted {deleted_count} expired entries from {self.table_name} (older than {minutes}m)"
                    )
                return deleted_count
            except Exception as e:
                logger.error(f"Failed to delete old entries from {self.table_name}: {e}")
                return 0

    def _maybe_auto_prune(self):
        if CACHE_TTL_MINUTES <= 0 or CACHE_AUTO_PRUNE_EVERY_WRITES <= 0:
            return

        self._write_count += 1
        if self._write_count % CACHE_AUTO_PRUNE_EVERY_WRITES == 0:
            self.delete_older_than(CACHE_TTL_MINUTES)


class RetrievalCache(BaseSQLiteCache):
    """Caches retrieval results for identical queries within a session."""

    def __init__(self, cache_dir: str = DEFAULT_CACHE_DIR):
        db_path = os.path.join(cache_dir, "retrieval_cache.db")
        schema = """
            CREATE TABLE IF NOT EXISTS retrieval_cache (
                cache_key TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                query TEXT NOT NULL,
                top_k INTEGER NOT NULL,
                result_json TEXT NOT NULL,
                created_at TEXT DEFAULT (datetime('now'))
            )
        """
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_retrieval_cache_created_at ON retrieval_cache(created_at)",
            "CREATE INDEX IF NOT EXISTS idx_retrieval_cache_session_query ON retrieval_cache(session_id, query)",
        ]
        super().__init__(db_path, schema, "retrieval_cache", index_statements=indexes)
        logger.info(f"RetrievalCache initialized at {db_path}")

    def _make_key(
        self,
        session_id: str,
        query: str,
        top_k: int,
        retrieval_params: Optional[dict] = None,
    ) -> str:
        """Create a strong composite cache key."""
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
        """Retrieve cached result if available."""
        cache_key = self._make_key(session_id, query, top_k, retrieval_params)

        with self._lock:
            cursor = self._conn.execute(
                "SELECT result_json FROM retrieval_cache WHERE cache_key = ?",
                (cache_key,),
            )
            row = cursor.fetchone()

        if row:
            try:
                logger.debug(f"Retrieval cache hit for query: '{query[:50]}...'")
                return json.loads(row[0])
            except json.JSONDecodeError:
                logger.error(f"Failed to decode cached retrieval JSON for key {cache_key}")

        return None

    def set_cache(
        self,
        session_id: str,
        query: str,
        top_k: int,
        result_data: Dict[str, Any],
        retrieval_params: Optional[dict] = None,
    ):
        """Save retrieval result to cache."""
        cache_key = self._make_key(session_id, query, top_k, retrieval_params)

        try:
            payload = json.dumps(result_data)
        except (TypeError, ValueError) as e:
            logger.warning(f"Skipping retrieval cache write (non-serializable payload): {e}")
            return

        with self._lock:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO retrieval_cache
                (cache_key, session_id, query, top_k, result_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                (cache_key, session_id, query, top_k, payload),
            )
            self._conn.commit()
            self._maybe_auto_prune()


class LLMCache(BaseSQLiteCache):
    """Caches LLM responses for queries with identical context within a session."""

    def __init__(self, cache_dir: str = DEFAULT_CACHE_DIR):
        db_path = os.path.join(cache_dir, "llm_cache.db")
        schema = """
            CREATE TABLE IF NOT EXISTS llm_cache (
                cache_key TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                query TEXT NOT NULL,
                context_hash TEXT NOT NULL,
                response_json TEXT NOT NULL,
                created_at TEXT DEFAULT (datetime('now'))
            )
        """
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_llm_cache_created_at ON llm_cache(created_at)",
            "CREATE INDEX IF NOT EXISTS idx_llm_cache_session_query ON llm_cache(session_id, query)",
        ]
        super().__init__(db_path, schema, "llm_cache", index_statements=indexes)
        logger.info(f"LLMCache initialized at {db_path}")

    def _compute_context_hash(self, chunks: List[str]) -> str:
        """Compute a hash representing the exact textual context."""
        content = "\\n---CONTEXT---\\n".join(chunks)
        return self._hash_key(content)

    def get_cache(self, session_id: str, query: str, retrieved_chunks: List[str]) -> Optional[Dict[str, Any]]:
        """Retrieve cached LLM response if available."""
        context_hash = self._compute_context_hash(retrieved_chunks)
        normalized_query = self._normalize_query(query)
        cache_key = self._hash_key(session_id, normalized_query, context_hash)

        with self._lock:
            cursor = self._conn.execute(
                "SELECT response_json FROM llm_cache WHERE cache_key = ?",
                (cache_key,),
            )
            row = cursor.fetchone()

        if row:
            try:
                logger.debug(f"LLM cache hit for query: '{query[:50]}...'")
                return json.loads(row[0])
            except json.JSONDecodeError:
                logger.error(f"Failed to decode cached LLM JSON for key {cache_key}")

        return None

    def set_cache(self, session_id: str, query: str, retrieved_chunks: List[str], response_data: Dict[str, Any]):
        """Save LLM response result to cache."""
        context_hash = self._compute_context_hash(retrieved_chunks)
        normalized_query = self._normalize_query(query)
        cache_key = self._hash_key(session_id, normalized_query, context_hash)

        try:
            payload = json.dumps(response_data)
        except (TypeError, ValueError) as e:
            logger.warning(f"Skipping LLM cache write (non-serializable payload): {e}")
            return

        with self._lock:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO llm_cache
                (cache_key, session_id, query, context_hash, response_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                (cache_key, session_id, query, context_hash, payload),
            )
            self._conn.commit()
            self._maybe_auto_prune()


# Global singletons for easy sharing across handlers
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
    """Gracefully close singleton cache connections."""
    global _retrieval_cache, _llm_cache
    with _global_cache_lock:
        if _retrieval_cache is not None:
            _retrieval_cache.close()
            _retrieval_cache = None
        if _llm_cache is not None:
            _llm_cache.close()
            _llm_cache = None