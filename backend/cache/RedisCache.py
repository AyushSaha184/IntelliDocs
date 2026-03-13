"""Redis cache primitives shared across cache modules.

Redis is the only cache backend. All failures degrade to cache misses.
"""

from __future__ import annotations

import base64
import json
import os
import threading
import time
from typing import Any, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from redis import Redis
else:
    Redis = Any

try:
    import redis
except Exception:  # pragma: no cover - import-time fallback
    redis = None

from src.utils.Logger import get_logger

logger = get_logger(__name__)


_DEFAULT_REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
_DEFAULT_PREFIX = os.getenv("REDIS_PREFIX", "rag")
_DEFAULT_TTL_SECONDS = int(os.getenv("REDIS_DEFAULT_TTL_SECONDS", "1800"))

_client_lock = threading.Lock()
_client: Optional[Redis] = None


def get_redis_client() -> Optional[Redis]:
    """Get shared Redis client singleton.

    Returns None if redis package is unavailable.
    """
    global _client

    if redis is None:
        logger.warning("redis package is not installed; cache will degrade to misses")
        return None

    if _client is not None:
        return _client

    with _client_lock:
        if _client is None:
            _client = redis.from_url(
                os.getenv("REDIS_URL", _DEFAULT_REDIS_URL),
                decode_responses=True,
                socket_timeout=1.5,
                socket_connect_timeout=1.5,
                health_check_interval=30,
            )
        return _client


def reset_redis_client_for_tests() -> None:
    """Reset global client singleton. Intended for tests only."""
    global _client
    with _client_lock:
        _client = None


class RedisJSONStore:
    """Thin JSON wrapper over Redis with namespace and TTL support."""

    def __init__(self, namespace: str, ttl_seconds: Optional[int] = None):
        self.namespace = namespace.strip(":")
        self.prefix = os.getenv("REDIS_PREFIX", _DEFAULT_PREFIX).strip(":")
        self.ttl_seconds = _DEFAULT_TTL_SECONDS if ttl_seconds is None else int(ttl_seconds)

    def key(self, key_hash: str) -> str:
        return f"{self.prefix}:{self.namespace}:{key_hash}"

    def get_json(self, key_hash: str) -> Optional[Dict[str, Any]]:
        try:
            client = get_redis_client()
        except Exception as exc:
            logger.warning(f"Redis client acquisition failed for {self.namespace}: {exc}")
            return None
        if client is None:
            return None

        redis_key = self.key(key_hash)
        try:
            raw = client.get(redis_key)
        except Exception as exc:
            logger.warning(f"Redis read failed for {redis_key}: {exc}")
            return None

        if not raw:
            return None

        try:
            return json.loads(raw)
        except Exception as exc:
            logger.warning(f"Redis JSON decode failed for {redis_key}: {exc}")
            return None

    def set_json(self, key_hash: str, payload: Dict[str, Any], ttl_seconds: Optional[int] = None) -> bool:
        try:
            client = get_redis_client()
        except Exception as exc:
            logger.warning(f"Redis client acquisition failed for {self.namespace}: {exc}")
            return False
        if client is None:
            return False

        redis_key = self.key(key_hash)
        ttl = self.ttl_seconds if ttl_seconds is None else int(ttl_seconds)
        try:
            serialized = json.dumps(payload, separators=(",", ":"))
        except Exception as exc:
            logger.warning(f"Skipping Redis write for {redis_key}; non-serializable payload: {exc}")
            return False

        try:
            if ttl > 0:
                client.set(redis_key, serialized, ex=ttl)
            else:
                client.set(redis_key, serialized)
            return True
        except Exception as exc:
            logger.warning(f"Redis write failed for {redis_key}: {exc}")
            return False

    def delete_older_than(self, minutes: int) -> int:
        """Delete entries with wrapper created_at older than N minutes.

        This method is best-effort and retained for backward compatibility.
        """
        try:
            client = get_redis_client()
        except Exception as exc:
            logger.warning(f"Redis client acquisition failed for prune {self.namespace}: {exc}")
            return 0
        if client is None:
            return 0

        threshold = time.time() - (minutes * 60)
        pattern = f"{self.prefix}:{self.namespace}:*"
        deleted = 0

        try:
            cursor = 0
            while True:
                cursor, keys = client.scan(cursor=cursor, match=pattern, count=200)
                for redis_key in keys:
                    try:
                        raw = client.get(redis_key)
                        if not raw:
                            continue
                        obj = json.loads(raw)
                        created_at = float(obj.get("created_at", 0.0))
                        if created_at and created_at < threshold:
                            if client.delete(redis_key):
                                deleted += 1
                    except Exception:
                        continue
                if cursor == 0:
                    break
        except Exception as exc:
            logger.warning(f"Redis prune failed for namespace {self.namespace}: {exc}")
            return 0

        return deleted


def encode_embedding_payload(vector: Any) -> Dict[str, Any]:
    """Encode float32 numpy vector as base64 with explicit metadata."""
    import numpy as np

    arr = np.asarray(vector, dtype=np.float32)
    return {
        "encoding": "base64",
        "dtype": "float32",
        "shape": list(arr.shape),
        "data": base64.b64encode(arr.tobytes()).decode("ascii"),
    }


def decode_embedding_payload(payload: Dict[str, Any]):
    """Decode embedding payload into a numpy array; returns None on error."""
    import numpy as np

    try:
        if payload.get("encoding") != "base64":
            return None
        if payload.get("dtype") != "float32":
            return None
        data = base64.b64decode(payload["data"].encode("ascii"))
        shape = tuple(int(x) for x in payload.get("shape", []))
        vec = np.frombuffer(data, dtype=np.float32)
        if shape:
            vec = vec.reshape(shape)
        return vec.copy()
    except Exception:
        return None
