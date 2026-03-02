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
from datetime import datetime
from src.utils.Logger import get_logger

logger = get_logger(__name__)

# Default cache directory
DEFAULT_CACHE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 
    "data", 
    "cache"
)


class BaseSQLiteCache:
    """Base class for SQLite-backed thread-safe caches."""
    
    def __init__(self, db_path: str, table_schema: str, table_name: str):
        self.db_path = db_path
        self.table_name = table_name
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute(table_schema)
        self._conn.commit()

    def _hash_key(self, *parts: str) -> str:
        """Create a SHA-256 hash from string parts."""
        content = ":".join(str(p) for p in parts if p is not None)
        return hashlib.sha256(content.encode('utf-8', errors='replace')).hexdigest()

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
                    logger.info(f"Deleted {deleted_count} expired entries from {self.table_name} (older than {minutes}m)")
                return deleted_count
            except Exception as e:
                logger.error(f"Failed to delete old entries from {self.table_name}: {e}")
                return 0


class RetrievalCache(BaseSQLiteCache):
    """Caches retrieval results (chunks and metrics) for exactly identical queries within a session."""
    
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
        super().__init__(db_path, schema, "retrieval_cache")
        logger.info(f"RetrievalCache initialized at {db_path}")

    def _make_key(self, session_id: str, query: str, top_k: int, retrieval_params: Optional[dict] = None) -> str:
        """Create a strong composite cache key.

        Includes: normalized_query + top_k + retrieval_params fingerprint.
        This prevents false hits when hybrid weights, filters, or top_k change.
        """
        normalized_query = " ".join(query.lower().strip().split())  # normalize whitespace/case
        params_str = json.dumps(retrieval_params or {}, sort_keys=True)
        return self._hash_key(session_id, normalized_query, str(top_k), params_str)

    def get_cache(
        self, session_id: str, query: str, top_k: int, retrieval_params: Optional[dict] = None
    ) -> Optional[Dict[str, Any]]:
        """Retrieve cached result if available."""
        cache_key = self._make_key(session_id, query, top_k, retrieval_params)

        with self._lock:
            cursor = self._conn.execute(
                "SELECT result_json FROM retrieval_cache WHERE cache_key = ?",
                (cache_key,)
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
        self, session_id: str, query: str, top_k: int, result_data: Dict[str, Any],
        retrieval_params: Optional[dict] = None
    ):
        """Save retrieval result to cache."""
        cache_key = self._make_key(session_id, query, top_k, retrieval_params)

        with self._lock:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO retrieval_cache
                (cache_key, session_id, query, top_k, result_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                (cache_key, session_id, query, top_k, json.dumps(result_data))
            )
            self._conn.commit()


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
        super().__init__(db_path, schema, "llm_cache")
        logger.info(f"LLMCache initialized at {db_path}")

    def _compute_context_hash(self, chunks: List[str]) -> str:
        """Compute a hash representing the exact textual context."""
        content = "\\n---CONTEXT---\\n".join(chunks)
        return self._hash_key(content)

    def get_cache(self, session_id: str, query: str, retrieved_chunks: List[str]) -> Optional[Dict[str, Any]]:
        """Retrieve cached LLM response if available."""
        context_hash = self._compute_context_hash(retrieved_chunks)
        cache_key = self._hash_key(session_id, query, context_hash)
        
        with self._lock:
            cursor = self._conn.execute(
                "SELECT response_json FROM llm_cache WHERE cache_key = ?", 
                (cache_key,)
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
        cache_key = self._hash_key(session_id, query, context_hash)
        
        with self._lock:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO llm_cache 
                (cache_key, session_id, query, context_hash, response_json) 
                VALUES (?, ?, ?, ?, ?)
                """,
                (cache_key, session_id, query, context_hash, json.dumps(response_data))
            )
            self._conn.commit()


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
