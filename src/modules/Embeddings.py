"""
Embedding Module - NVIDIA API with BAAI/bge-m3

Uses NVIDIA Build API to generate 1024-dimensional BGE-M3 embeddings.
Optimized for batch processing, caching, and handling millions of embeddings.
"""

import os
import sqlite3
import threading
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
import numpy as np
import hashlib
import time
from typing import Callable, TypeVar

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

from src.utils.Logger import get_logger

logger = get_logger(__name__)

T = TypeVar('T')

EMBEDDING_DIMENSION = 1024          # BAAI/bge-m3 via NVIDIA API
NVIDIA_MODEL_NAME   = "baai/bge-m3" # Model identifier on NVIDIA Build


def retry_with_exponential_backoff(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    exponential_base: float = 2.0,
    jitter: bool = True
) -> Callable:
    """Decorator for retrying functions with exponential backoff"""
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        def wrapper(*args, **kwargs) -> T:
            delay = initial_delay
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    error_str = str(e).lower()
                    is_retriable = any(err in error_str for err in [
                        '504', 'timeout', 'gateway', '503', '502', 'connection'
                    ])
                    if not is_retriable or attempt == max_retries:
                        raise
                    wait_time = delay
                    if jitter:
                        import random
                        wait_time *= (0.5 + random.random())
                    logger.warning(
                        f"Attempt {attempt + 1}/{max_retries + 1} failed: {e}. "
                        f"Retrying in {wait_time:.2f}s..."
                    )
                    time.sleep(wait_time)
                    delay *= exponential_base

            raise last_exception
        return wrapper
    return decorator


@dataclass
class EmbeddingResult:
    """Result of an embedding operation"""
    text: str
    embedding: Optional[np.ndarray]
    dimension: int
    model_name: str


class EmbeddingModel(ABC):
    """Abstract base class for embedding models"""

    @abstractmethod
    def embed(self, text: str) -> Optional[np.ndarray]:
        pass

    @abstractmethod
    def embed_batch(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        pass


class NVIDIAEmbedding(EmbeddingModel):
    """NVIDIA Build API - BAAI/bge-m3 embedding model (1024 dimensions)

    Uses the OpenAI-compatible NVIDIA Build endpoint to generate embeddings.
    Features:
    - 1024-dimensional BGE-M3 embeddings
    - Batch processing with automatic chunking
    - Exponential backoff retry
    - Vectorized normalization
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://integrate.api.nvidia.com/v1",
        normalize_embeddings: bool = True,
        timeout: float = 120.0,
        max_retries: int = 3
    ):
        """Initialize NVIDIA BGE-M3 embedding model

        Args:
            api_key: NVIDIA API key (falls back to NVIDIA_API_KEY env var)
            base_url: NVIDIA Build API base URL
            normalize_embeddings: Normalize vectors to unit length
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts on transient errors
        """
        if not OPENAI_AVAILABLE:
            raise ImportError("openai not installed. Run: pip install openai")

        resolved_key = api_key or os.environ.get("NVIDIA_API_KEY")
        if not resolved_key:
            raise ValueError(
                "NVIDIA_API_KEY is required. Pass api_key= or set the env var."
            )

        self.normalize_embeddings = normalize_embeddings
        self.timeout = timeout
        self.max_retries = max_retries

        self.client = openai.OpenAI(
            api_key=resolved_key,
            base_url=base_url,
            timeout=timeout
        )

        logger.info(
            f"NVIDIAEmbedding initialized: model={NVIDIA_MODEL_NAME}, "
            f"endpoint={base_url}, normalize={normalize_embeddings}"
        )

    def _normalize(self, embedding: np.ndarray) -> np.ndarray:
        if not self.normalize_embeddings:
            return embedding
        norm = np.linalg.norm(embedding)
        return embedding if norm == 0 else embedding / norm

    @retry_with_exponential_backoff(max_retries=3, initial_delay=1.0)
    def embed(self, text: str) -> Optional[np.ndarray]:
        """Embed a single text"""
        if not text or not text.strip():
            return None
        response = self.client.embeddings.create(
            model=NVIDIA_MODEL_NAME,
            input=text,
            encoding_format="float"
        )
        if response.data:
            arr = np.array(response.data[0].embedding, dtype=np.float32)
            return self._normalize(arr)
        return None

    @retry_with_exponential_backoff(max_retries=3, initial_delay=1.0)
    def embed_batch(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Embed multiple texts in batches

        Args:
            texts: List of texts to embed
            batch_size: Texts per API call

        Returns:
            (N, 1024) float32 numpy array; zero rows for empty/missing texts
        """
        if not texts:
            return np.zeros((0, EMBEDDING_DIMENSION), dtype=np.float32)

        # Filter empty texts, track original positions
        valid_texts: List[str] = []
        valid_indices: List[int] = []
        for idx, text in enumerate(texts):
            if text and text.strip():
                valid_texts.append(text)
                valid_indices.append(idx)

        if not valid_texts:
            return np.zeros((len(texts), EMBEDDING_DIMENSION), dtype=np.float32)

        # Truncate oversized texts (~4 chars per token, 7500 token safety margin)
        MAX_CHARS = 7500 * 4
        truncated = []
        for text in valid_texts:
            if len(text) > MAX_CHARS:
                logger.warning(
                    f"Truncating text ({len(text)} chars) to {MAX_CHARS} chars"
                )
                truncated.append(text[:MAX_CHARS])
            else:
                truncated.append(text)

        all_embeddings: List[np.ndarray] = []
        for i in range(0, len(truncated), batch_size):
            batch = truncated[i:i + batch_size]
            response = self.client.embeddings.create(
                model=NVIDIA_MODEL_NAME,
                input=batch,
                encoding_format="float"
            )
            for item in response.data:
                arr = np.array(item.embedding, dtype=np.float32)
                if self.normalize_embeddings:
                    norm = np.linalg.norm(arr)
                    if norm > 0:
                        arr /= norm
                all_embeddings.append(arr)

        # Scatter into result array
        result = np.zeros((len(texts), EMBEDDING_DIMENSION), dtype=np.float32)
        for i, idx in enumerate(valid_indices):
            result[idx] = all_embeddings[i]
        return result

    @property
    def dimension(self) -> int:
        return EMBEDDING_DIMENSION

    @property
    def model_name(self) -> str:
        return NVIDIA_MODEL_NAME


class _EmbeddingDiskCache:
    """SQLite-backed persistent embedding cache.

    Stores {text_hash -> embedding_vector} on disk so embeddings survive
    process restarts. Avoids re-embedding identical text across rebuilds.

    Schema:
        cache_key  TEXT PRIMARY KEY  -- SHA-256 hash of (model_name + text)
        embedding  BLOB              -- numpy array serialized as bytes
        dimension  INTEGER           -- embedding dimension for validation
        created_at TEXT              -- ISO timestamp
    """

    def __init__(self, cache_dir: str, model_name: str):
        os.makedirs(cache_dir, exist_ok=True)
        db_path = os.path.join(cache_dir, "embedding_cache.db")
        self._model_name = model_name
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._lock = threading.Lock()
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS embeddings (
                cache_key  TEXT PRIMARY KEY,
                embedding  BLOB NOT NULL,
                dimension  INTEGER NOT NULL,
                model_name TEXT NOT NULL,
                created_at TEXT DEFAULT (datetime('now'))
            )
        """)
        self._conn.commit()
        logger.info(f"Disk embedding cache opened at {db_path}")

    def load_all(self) -> Dict[str, np.ndarray]:
        """Load all cached embeddings for the current model into memory."""
        cache: Dict[str, np.ndarray] = {}
        with self._lock:
            cursor = self._conn.execute(
                "SELECT cache_key, embedding, dimension FROM embeddings WHERE model_name = ?",
                (self._model_name,)
            )
            for key, blob, dim in cursor:
                vec = np.frombuffer(blob, dtype=np.float32).copy()
                if len(vec) == dim:
                    cache[key] = vec
        logger.info(f"Loaded {len(cache)} cached embeddings from disk")
        return cache

    def put_batch(self, items: List[tuple]):
        """Persist multiple (cache_key, embedding) pairs in one transaction."""
        if not items:
            return
        with self._lock:
            self._conn.executemany(
                "INSERT OR IGNORE INTO embeddings "
                "(cache_key, embedding, dimension, model_name) VALUES (?, ?, ?, ?)",
                [
                    (key, emb.astype(np.float32).tobytes(), len(emb), self._model_name)
                    for key, emb in items
                ]
            )
            self._conn.commit()

    def count(self) -> int:
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*) FROM embeddings WHERE model_name = ?",
                (self._model_name,)
            ).fetchone()
            return row[0] if row else 0

    def db_size_mb(self) -> float:
        with self._lock:
            page_count = self._conn.execute("PRAGMA page_count").fetchone()[0]
            page_size  = self._conn.execute("PRAGMA page_size").fetchone()[0]
            return (page_count * page_size) / (1024 * 1024)

    def close(self):
        with self._lock:
            self._conn.close()


class EmbeddingService:
    """Service for generating and caching NVIDIA BGE-M3 embeddings.

    Features:
    - In-memory cache for fast repeated lookups
    - SQLite disk cache for persistence across restarts
    - Deduplication: identical texts embedded only once per batch
    """

    def __init__(
        self,
        model: EmbeddingModel,
        cache_dir: Optional[str] = None,
        use_cache: bool = True,
        max_cache_size: int = 100000
    ):
        self.model = model
        self.use_cache = use_cache
        self.max_cache_size = max_cache_size
        self._embedding_cache: Dict[str, np.ndarray] = {}
        self._cache_lock = threading.Lock()
        self._cache_hits = 0
        self._cache_misses = 0
        self._disk_cache: Optional[_EmbeddingDiskCache] = None

        if use_cache and cache_dir:
            try:
                self._disk_cache = _EmbeddingDiskCache(cache_dir, model.model_name)
                self._embedding_cache = self._disk_cache.load_all()
                logger.info(
                    f"EmbeddingService ready: {model.model_name}, "
                    f"disk cache preloaded {len(self._embedding_cache)} entries"
                )
            except Exception as e:
                logger.warning(f"Disk cache init failed, using memory-only: {e}")
        else:
            logger.info(f"EmbeddingService ready: {model.model_name} (memory cache only)")

    def _cache_key(self, text: str) -> str:
        content = f"{self.model.model_name}:{text}"
        return hashlib.sha256(content.encode('utf-8', errors='replace')).hexdigest()

    def embed_text(self, text: str) -> EmbeddingResult:
        """Embed a single text with caching"""
        cache_key = self._cache_key(text) if self.use_cache else None

        with self._cache_lock:
            if cache_key and cache_key in self._embedding_cache:
                self._cache_hits += 1
                return EmbeddingResult(
                    text=text,
                    embedding=self._embedding_cache[cache_key],
                    dimension=self.model.dimension,
                    model_name=self.model.model_name
                )
            if self.use_cache:
                self._cache_misses += 1

        embedding = self.model.embed(text)

        if self.use_cache and embedding is not None:
            with self._cache_lock:
                if len(self._embedding_cache) < self.max_cache_size:
                    self._embedding_cache[cache_key] = embedding
            if self._disk_cache:
                self._disk_cache.put_batch([(cache_key, embedding)])

        return EmbeddingResult(
            text=text,
            embedding=embedding,
            dimension=self.model.dimension,
            model_name=self.model.model_name
        )

    def embed_batch(self, texts: List[str], batch_size: int = 64) -> List[EmbeddingResult]:
        """Embed multiple texts with deduplication and caching.

        - Cache-hit texts never reach the API
        - Duplicate texts within the batch are embedded once
        - New embeddings persisted to disk in a single transaction
        """
        if not texts:
            return []

        dim        = self.model.dimension
        model_name = self.model.model_name

        embeddings: List[Optional[np.ndarray]] = [None] * len(texts)
        uncached_unique: Dict[str, int] = {}
        text_to_key:    Dict[str, str]  = {}

        with self._cache_lock:
            for idx, text in enumerate(texts):
                if not text or not text.strip():
                    continue
                if self.use_cache:
                    ck = self._cache_key(text)
                    text_to_key[text] = ck
                    if ck in self._embedding_cache:
                        embeddings[idx] = self._embedding_cache[ck]
                        self._cache_hits += 1
                        continue
                    self._cache_misses += 1
                if text not in uncached_unique:
                    uncached_unique[text] = idx

        new_entries: List[tuple] = []
        text_to_emb: Dict[str, np.ndarray] = {}

        if uncached_unique:
            unique_list = list(uncached_unique.keys())
            logger.info(
                f"Embedding {len(unique_list)} new texts "
                f"({len(texts) - len(unique_list)} cache hits)"
            )
            unique_embs = self.model.embed_batch(unique_list, batch_size=batch_size)

            for i, text in enumerate(unique_list):
                emb = unique_embs[i]
                text_to_emb[text] = emb
                if self.use_cache and emb is not None:
                    ck = text_to_key.get(text, self._cache_key(text))
                    with self._cache_lock:
                        if len(self._embedding_cache) < self.max_cache_size:
                            self._embedding_cache[ck] = emb
                    new_entries.append((ck, emb))
        else:
            logger.info(f"All {len(texts)} texts served from cache (100% hit rate)")

        if new_entries and self._disk_cache:
            self._disk_cache.put_batch(new_entries)

        results = []
        for idx, text in enumerate(texts):
            emb = embeddings[idx] if embeddings[idx] is not None else text_to_emb.get(text)
            results.append(EmbeddingResult(
                text=text,
                embedding=emb,
                dimension=dim,
                model_name=model_name
            ))
        return results

    def get_statistics(self) -> Dict[str, Any]:
        """Get embedding service statistics"""
        total = self._cache_hits + self._cache_misses
        stats = {
            "model_name":           self.model.model_name,
            "embedding_dimension":  self.model.dimension,
            "cache_hits":           self._cache_hits,
            "cache_misses":         self._cache_misses,
            "cache_hit_ratio":      self._cache_hits / total if total > 0 else 0,
            "memory_cache_entries": len(self._embedding_cache),
            "cache_enabled":        self.use_cache,
            "disk_cache_enabled":   self._disk_cache is not None,
        }
        if self._disk_cache:
            stats["disk_cache_entries"] = self._disk_cache.count()
            stats["disk_cache_size_mb"] = round(self._disk_cache.db_size_mb(), 2)
        return stats


# Default cache directory
DEFAULT_CACHE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data", "cache"
)


def create_embedding_service(
    api_key: Optional[str] = None,
    use_cache: bool = True,
    max_cache_size: int = 100000,
    cache_dir: Optional[str] = None,
    **kwargs
) -> EmbeddingService:
    """Create an EmbeddingService backed by NVIDIA BAAI/bge-m3

    Args:
        api_key: NVIDIA API key (falls back to NVIDIA_API_KEY env var)
        use_cache: Enable in-memory + disk embedding cache
        max_cache_size: Maximum in-memory cache entries
        cache_dir: Directory for persistent SQLite cache (default: data/cache/)
        **kwargs: Extra args forwarded to NVIDIAEmbedding
                  (e.g. normalize_embeddings=False, timeout=60.0)

    Returns:
        EmbeddingService instance
    """
    # Strip legacy kwargs that belonged to the old multi-model factory
    for _key in ("model_type", "model_name", "device", "use_fp16",
                 "normalize", "max_length", "provider", "task_type",
                 "output_dimensionality", "base_url"):
        kwargs.pop(_key, None)
    model = NVIDIAEmbedding(api_key=api_key, **kwargs)

    resolved_cache_dir = cache_dir or os.environ.get("EMBEDDING_CACHE_DIR", DEFAULT_CACHE_DIR)

    return EmbeddingService(
        model,
        cache_dir=resolved_cache_dir,
        use_cache=use_cache,
        max_cache_size=max_cache_size
    )
