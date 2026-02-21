"""
Embedding Module - Modular, swappable embedding interface with Enterprise Scale support

Supports any embedding model through a clean abstraction layer.
Optimized for batch processing, caching, and handling millions of embeddings.
"""

import os
import sqlite3
import threading
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, Generator
from dataclasses import dataclass
import numpy as np
import logging
import hashlib
import time
from typing import Callable, TypeVar
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from huggingface_hub import InferenceClient
    HUGGINGFACE_AVAILABLE = True
except ImportError:
    HUGGINGFACE_AVAILABLE = False

try:
    from google import genai
    from google.genai import types as genai_types
    GOOGLE_GENAI_AVAILABLE = True
except ImportError:
    GOOGLE_GENAI_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

from src.utils.Logger import get_logger

logger = get_logger(__name__)

# Type variable for retry decorator
T = TypeVar('T')

# NVIDIA API configuration
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


def retry_with_exponential_backoff(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    exponential_base: float = 2.0,
    jitter: bool = True
) -> Callable:
    """Decorator for retrying functions with exponential backoff
    
    Args:
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay between retries in seconds
        exponential_base: Base for exponential backoff
        jitter: Add random jitter to prevent thundering herd
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        def wrapper(*args, **kwargs) -> T:
            delay = initial_delay
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    
                    # Check if it's a retriable error (timeout, gateway errors)
                    error_str = str(e).lower()
                    is_retriable = any(err in error_str for err in [
                        '504', 'timeout', 'gateway', '503', '502', 'connection'
                    ])
                    
                    if not is_retriable or attempt == max_retries:
                        raise
                    
                    # Calculate delay with optional jitter
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


EMBEDDING_DIMENSIONS = {
    "BAAI/bge-m3": 1024,
    "BAAI/bge-large-en": 1024,
    "sentence-transformers/all-MiniLM-L6-v2": 384,
    "gemini-embedding-001": 768,  # Default dimension for Gemini
    "text-embedding-004": 768,      # Another Gemini model option
    "text-embedding-bge-m3": 1024,   # LM Studio BGE-M3
    "nvidia/nv-embeddings-v1": 1024,  # NVIDIA Build API
    "nvidia/nv-embed-v1": 1024,       # NVIDIA Build API alternative
    "baai/bge-m3": 1024               # NVIDIA Build BGE-M3
}


def _get_registered_dimension(model_name: str) -> int:
    dimension = EMBEDDING_DIMENSIONS.get(model_name)
    if dimension is None:
        raise ValueError(
            f"Embedding dimension for '{model_name}' is not registered. "
            "Add it to EMBEDDING_DIMENSIONS in src/modules/embeddings.py."
        )
    return dimension


@dataclass
class EmbeddingResult:
    """Result of embedding operation"""
    text: str
    embedding: Optional[np.ndarray]
    dimension: int
    model_name: str


class EmbeddingModel(ABC):
    """Abstract base class for embedding models
    
    Allows easy swapping of different embedding implementations.
    """
    
    @abstractmethod
    def embed(self, text: str) -> np.ndarray:
        """Embed a single text
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector as numpy array
        """
        pass
    
    @abstractmethod
    def embed_batch(self, texts: List[str], batch_size: int = 32) -> List[np.ndarray]:
        """Embed multiple texts efficiently in batches
        
        Args:
            texts: List of texts to embed
            batch_size: Batch size for processing
            
        Returns:
            List of embedding vectors
        """
        pass
    
    @property
    @abstractmethod
    def dimension(self) -> int:
        """Get embedding dimension"""
        pass
    
    @property
    @abstractmethod
    def model_name(self) -> str:
        """Get model name"""
        pass


class BGEm3Embedding(EmbeddingModel):
    """BAAI/bge-m3 Embedding Model
    
    Multi-lingual, multi-functionality embedding model.
    Optimized for both dense retrieval and semantic search.
    
    Features:
    - Multi-lingual support (100+ languages)
    - 1024-dimensional embeddings
    - Fast inference with optimizations
    """
    
    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        device: Optional[str] = None,
        max_length: int = 8192,
        normalize_embeddings: bool = True,
        use_fp16: bool = False
    ):
        """Initialize BGE-M3 embedding model
        
        Args:
            model_name: Model identifier (default: BAAI/bge-m3)
            device: Device to use ('cpu', 'cuda', 'cuda:0', etc.)
            max_length: Maximum input length
            normalize_embeddings: Whether to normalize embeddings to unit vectors
            use_fp16: Use half precision for faster inference
        """
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError("sentence-transformers not installed. Run: pip install sentence-transformers")
        
        # Auto-detect device if not specified
        if device is None:
            device = "cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu"
        
        self.model_name_str = model_name
        self.device = device
        self.max_length = max_length
        self.normalize_embeddings = normalize_embeddings
        self.use_fp16 = use_fp16
        
        logger.info(f"Loading {model_name} on {device}")
        
        # Load model with optimizations
        self.model = SentenceTransformer(
            model_name,
            device=device,
            trust_remote_code=True
        )
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Enable FP16 for GPU inference (huge speedup)
        if use_fp16 and TORCH_AVAILABLE and device.startswith('cuda'):
            try:
                self.model.half()  # Convert model to FP16
                logger.info("FP16 enabled for GPU inference")
            except Exception as e:
                logger.warning(f"Could not enable FP16: {e}")
                use_fp16 = False
        
        # Enable inference mode for faster inference and lower memory
        if TORCH_AVAILABLE:
            self._inference_mode = torch.inference_mode()
            self._inference_mode.__enter__()
        
        logger.info(f"Model loaded. Embedding dimension: {self.dimension}")
    
    def embed(self, text: str) -> Optional[np.ndarray]:
        """Embed a single text
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector or None if text is empty
        """
        if not text or not text.strip():
            return None
        
        # Model handles truncation internally at max_length
        embedding = self.model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings
        )
        
        return embedding.astype(np.float32)
    
    def embed_batch(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Embed multiple texts efficiently as numpy array
        
        Args:
            texts: List of texts to embed
            batch_size: Batch size for processing
            
        Returns:
            Numpy array of embedding vectors
        """
        if not texts:
            return np.array([], dtype=np.float32).reshape(0, self.dimension)
        
        # Filter empty texts and create index mapping
        valid_texts = []
        valid_indices = []
        for idx, text in enumerate(texts):
            if text and text.strip():
                valid_texts.append(text)
                valid_indices.append(idx)
        
        if not valid_texts:
            return np.zeros((len(texts), self.dimension), dtype=np.float32)
        
        # Deduplicate to reduce compute
        unique_texts = list(dict.fromkeys(valid_texts))  # Preserve order
        embeddings_dict = {}
        
        if len(unique_texts) > 0:
            # Encode unique texts (model handles truncation internally)
            unique_embeddings = self.model.encode(
                unique_texts,
                batch_size=batch_size,
                convert_to_numpy=True,
                normalize_embeddings=self.normalize_embeddings,
                show_progress_bar=False
            )
            
            # Create mapping from unique text to embedding
            for text, emb in zip(unique_texts, unique_embeddings):
                embeddings_dict[text] = emb.astype(np.float32)
        
        # Build result array with all texts (empty as zeros)
        result = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for idx, text in zip(valid_indices, valid_texts):
            result[idx] = embeddings_dict[text]
        
        return result
    
    @property
    def dimension(self) -> int:
        """Get embedding dimension (1024 for BGE-M3)"""
        return _get_registered_dimension(self.model_name_str)
    
    @property
    def model_name(self) -> str:
        """Get model name"""
        return self.model_name_str


class HFInferenceEmbedding(EmbeddingModel):
    """Hugging Face Inference API embedding model with direct HTTP requests.

    Uses direct HTTP POST requests to HuggingFace router for feature extraction.
    Includes exponential backoff retry for handling gateway timeouts.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        api_key: Optional[str] = None,
        provider: str = "hf-inference",
        normalize_embeddings: bool = True,
        timeout: float = 120.0,
        max_retries: int = 3
    ):
        if not REQUESTS_AVAILABLE:
            raise ImportError("requests not installed. Run: pip install requests")
        
        if not api_key:
            raise ValueError("HuggingFace API token (HF_TOKEN) is required")

        self.model_name_str = model_name
        self.normalize_embeddings = normalize_embeddings
        self.timeout = timeout
        self.max_retries = max_retries
        self.api_key = api_key
        
        # HuggingFace router endpoint for feature extraction
        self.api_url = f"https://router.huggingface.co/hf-inference/models/{model_name}/pipeline/feature-extraction"
        
        logger.info(f"HF Inference (HTTP) initialized for model: {model_name} (timeout: {timeout}s, retries: {max_retries})")

    def _normalize(self, embedding: np.ndarray) -> np.ndarray:
        if not self.normalize_embeddings:
            return embedding

        norm = np.linalg.norm(embedding)
        if norm == 0:
            return embedding
        return embedding / norm

    @retry_with_exponential_backoff(max_retries=3, initial_delay=2.0)
    def embed(self, text: str) -> Optional[np.ndarray]:
        if not text or not text.strip():
            return None

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {"inputs": text}
        
        response = requests.post(
            self.api_url,
            json=payload,
            headers=headers,
            timeout=self.timeout
        )
        response.raise_for_status()
        
        embedding = response.json()
        embedding_array = np.array(embedding, dtype=np.float32)
        embedding_array = self._normalize(embedding_array)

        return embedding_array

    @retry_with_exponential_backoff(max_retries=3, initial_delay=2.0)
    def embed_batch(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        if not texts:
            dimension = self.dimension
            return np.array([], dtype=np.float32).reshape(0, dimension)

        valid_texts = []
        valid_indices = []
        for idx, text in enumerate(texts):
            if text and text.strip():
                valid_texts.append(text)
                valid_indices.append(idx)

        if not valid_texts:
            return np.zeros((len(texts), self.dimension), dtype=np.float32)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {"inputs": valid_texts}
        
        response = requests.post(
            self.api_url,
            json=payload,
            headers=headers,
            timeout=self.timeout
        )
        response.raise_for_status()
        
        embeddings_raw = response.json()

        if isinstance(embeddings_raw, list) and embeddings_raw and not isinstance(embeddings_raw[0], list):
            embeddings_raw = [embeddings_raw]

        embeddings_array = np.array(embeddings_raw, dtype=np.float32)
        if self.normalize_embeddings:
            norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            embeddings_array = embeddings_array / norms

        result = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for i, idx in enumerate(valid_indices):
            result[idx] = embeddings_array[i]

        return result

    @property
    def dimension(self) -> int:
        return _get_registered_dimension(self.model_name_str)

    @property
    def model_name(self) -> str:
        return self.model_name_str


class LMStudioEmbedding(EmbeddingModel):
    """LM Studio OpenAI-Compatible Embedding Model
    
    Uses LM Studio's local API server with OpenAI-compatible endpoints.
    Works with any embedding model hosted in LM Studio.
    
    Features:
    - Local inference (no API costs)
    - OpenAI-compatible API format
    - Batch processing support
    - Automatic retry with exponential backoff
    """
    
    def __init__(
        self,
        model_name: str = "text-embedding-bge-m3",
        base_url: str = "http://127.0.0.1:1234/v1",
        api_key: Optional[str] = None,
        normalize_embeddings: bool = True,
        timeout: float = 120.0,
        max_retries: int = 3
    ):
        """Initialize LM Studio embedding model
        
        Args:
            model_name: Model identifier (e.g., text-embedding-bge-m3)
            base_url: Base URL for LM Studio API (default: http://127.0.0.1:1234/v1)
            api_key: Optional API key (not required for local LM Studio)
            normalize_embeddings: Whether to normalize embeddings to unit vectors
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
        """
        if not REQUESTS_AVAILABLE:
            raise ImportError("requests not installed. Run: pip install requests")
        
        self.model_name_str = model_name
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.normalize_embeddings = normalize_embeddings
        self.timeout = timeout
        self.max_retries = max_retries
        
        # Build embeddings endpoint
        self.embeddings_url = f"{self.base_url}/embeddings"
        
        # Connection pooling - reuses TCP connections (saves ~50ms per request)
        self._session = requests.Session()
        if self.api_key:
            self._session.headers.update({"Authorization": f"Bearer {self.api_key}"})
        self._session.headers.update({"Content-Type": "application/json"})
        
        # Concurrent batch settings (optimized for LM Studio performance)
        self._max_concurrent_batches = 6  # Parallel sub-batch requests (sweet spot for most systems)
        self._optimal_sub_batch_size = 64  # Texts per sub-batch (tuned for LM Studio)
        
        logger.info(
            f"LM Studio embedding initialized: model={model_name}, "
            f"endpoint={self.embeddings_url}, "
            f"concurrent_batches={self._max_concurrent_batches}"
        )
    
    def _get_headers(self) -> Dict[str, str]:
        """Get request headers (used for non-session requests)"""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"        
        return headers
    
    def _normalize(self, embedding: np.ndarray) -> np.ndarray:
        """Normalize embedding to unit vector"""
        if not self.normalize_embeddings:
            return embedding
        
        norm = np.linalg.norm(embedding)
        if norm == 0:
            return embedding
        return embedding / norm
    
    @retry_with_exponential_backoff(max_retries=3, initial_delay=1.0)
    def embed(self, text: str) -> Optional[np.ndarray]:
        """Embed a single text
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector or None if text is empty
        """
        if not text or not text.strip():
            return None
        
        try:
            payload = {
                "model": self.model_name_str,
                "input": text
            }
            
            # Use session for connection pooling
            response = self._session.post(
                self.embeddings_url,
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            result = response.json()
            
            if "data" in result and len(result["data"]) > 0:
                embedding_values = result["data"][0]["embedding"]
                embedding_array = np.array(embedding_values, dtype=np.float32)
                embedding_array = self._normalize(embedding_array)
                return embedding_array
            else:
                logger.error(f"Unexpected response format: {result}")
                return None
                
        except requests.exceptions.RequestException as e:
            logger.error(f"LM Studio API request failed: {e}")
            raise
        except Exception as e:
            logger.error(f"LM Studio embedding error: {e}")
            raise
    
    def _embed_sub_batch(self, texts: List[str]) -> np.ndarray:
        """Embed a sub-batch of texts (internal, used for concurrent batching)
        
        Args:
            texts: List of texts to embed (should be pre-filtered)
            
        Returns:
            Numpy array of embeddings for this sub-batch
        """
        payload = {
            "model": self.model_name_str,
            "input": texts
        }
        
        response = self._session.post(
            self.embeddings_url,
            json=payload,
            timeout=self.timeout
        )
        response.raise_for_status()
        result = response.json()
        
        if "data" not in result:
            raise ValueError("Invalid response format from LM Studio API")
        
        # Sort by index and extract embeddings directly into pre-allocated array
        data_items = sorted(result["data"], key=lambda x: x.get("index", 0))
        batch_embeddings = np.array(
            [item["embedding"] for item in data_items],
            dtype=np.float32
        )
        
        # Vectorized normalization (much faster than per-vector)
        if self.normalize_embeddings:
            norms = np.linalg.norm(batch_embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            batch_embeddings /= norms
        
        return batch_embeddings
    
    @retry_with_exponential_backoff(max_retries=3, initial_delay=1.0)
    def embed_batch(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        """Embed multiple texts with concurrent sub-batching for maximum throughput
        
        Splits texts into sub-batches and sends them concurrently to LM Studio.
        This exploits the fact that LM Studio can handle multiple requests in parallel,
        giving ~2-4x speedup over a single large request.
        
        Args:
            texts: List of texts to embed
            batch_size: Sub-batch size (default: 64, tuned for LM Studio)
            
        Returns:
            Numpy array of embedding vectors
        """
        if not texts:
            return np.array([], dtype=np.float32).reshape(0, self.dimension)
        
        # Filter empty texts and create index mapping
        valid_texts = []
        valid_indices = []
        for idx, text in enumerate(texts):
            if text and text.strip():
                valid_texts.append(text)
                valid_indices.append(idx)
        
        if not valid_texts:
            return np.zeros((len(texts), self.dimension), dtype=np.float32)
        
        try:
            sub_batch_size = min(batch_size, self._optimal_sub_batch_size)
            
            # For small batches, send directly (avoid thread overhead)
            if len(valid_texts) <= sub_batch_size:
                batch_embeddings = self._embed_sub_batch(valid_texts)
                
                # Pre-allocate result and scatter
                result_array = np.zeros((len(texts), self.dimension), dtype=np.float32)
                for i, idx in enumerate(valid_indices):
                    result_array[idx] = batch_embeddings[i]
                return result_array
            
            # Split into sub-batches for concurrent processing
            sub_batches = [
                valid_texts[i:i + sub_batch_size]
                for i in range(0, len(valid_texts), sub_batch_size)
            ]
            
            logger.info(
                f"Concurrent embedding: {len(valid_texts)} texts in "
                f"{len(sub_batches)} sub-batches of ~{sub_batch_size} "
                f"({self._max_concurrent_batches} concurrent workers)"
            )
            
            # Process sub-batches concurrently
            all_embeddings = [None] * len(sub_batches)
            
            with ThreadPoolExecutor(max_workers=self._max_concurrent_batches) as executor:
                future_to_idx = {
                    executor.submit(self._embed_sub_batch, batch): i
                    for i, batch in enumerate(sub_batches)
                }
                
                for future in as_completed(future_to_idx):
                    batch_idx = future_to_idx[future]
                    all_embeddings[batch_idx] = future.result()
            
            # Concatenate all sub-batch results
            combined_embeddings = np.vstack(all_embeddings)
            
            # Pre-allocate result and scatter into correct positions
            result_array = np.zeros((len(texts), self.dimension), dtype=np.float32)
            for i, idx in enumerate(valid_indices):
                result_array[idx] = combined_embeddings[i]
            
            return result_array
            
        except requests.exceptions.RequestException as e:
            logger.error(f"LM Studio API batch request failed: {e}")
            raise
        except Exception as e:
            logger.error(f"LM Studio batch embedding error: {e}")
            raise
    
    def close(self):
        """Close the session and release connections"""
        if self._session:
            self._session.close()
    
    @property
    def dimension(self) -> int:
        """Get embedding dimension"""
        return _get_registered_dimension(self.model_name_str)
    
    @property
    def model_name(self) -> str:
        """Get model name"""
        return self.model_name_str


class GeminiEmbedding(EmbeddingModel):
    """Google Gemini Embedding Model
    
    Uses the Gemini API to generate text embeddings optimized for various tasks.
    Supports different task types for improved performance.
    
    Features:
    - Task-specific optimization (SEMANTIC_SIMILARITY, RETRIEVAL_DOCUMENT, etc.)
    - Configurable output dimensions (768, 1536, 3072)
    - Batch processing support
    - Automatic normalization for smaller dimensions
    """
    
    def __init__(
        self,
        model_name: str = "gemini-embedding-001",
        api_key: Optional[str] = None,
        task_type: str = "RETRIEVAL_DOCUMENT",
        output_dimensionality: int = 768,
        normalize_embeddings: bool = True,
        timeout: float = 120.0,
        max_retries: int = 3
    ):
        """Initialize Gemini embedding model
        
        Args:
            model_name: Model identifier (default: gemini-embedding-001)
            api_key: Google AI API key
            task_type: Task type for optimization (SEMANTIC_SIMILARITY, RETRIEVAL_DOCUMENT, 
                      RETRIEVAL_QUERY, CLASSIFICATION, CLUSTERING, etc.)
            output_dimensionality: Output dimension (768, 1536, or 3072)
            normalize_embeddings: Whether to normalize embeddings to unit vectors
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
        """
        if not GOOGLE_GENAI_AVAILABLE:
            raise ImportError("google-genai not installed. Run: pip install google-genai")
        
        if not api_key:
            raise ValueError("GEMINI_API_KEY is required for Gemini embeddings")
        
        self.model_name_str = model_name
        self.task_type = task_type
        self.output_dimensionality = output_dimensionality
        self.normalize_embeddings = normalize_embeddings
        self.timeout = timeout
        self.max_retries = max_retries
        
        # Initialize Gemini client
        self.client = genai.Client(api_key=api_key)
        
        # Update dimension registry if custom dimension is used
        if output_dimensionality not in [768, 1536, 3072]:
            logger.warning(
                f"Unusual output dimension {output_dimensionality}. "
                "Recommended: 768, 1536, or 3072"
            )
        
        logger.info(
            f"Gemini embedding initialized: model={model_name}, "
            f"dimension={output_dimensionality}, task_type={task_type}"
        )
    
    def _normalize(self, embedding: np.ndarray) -> np.ndarray:
        """Normalize embedding to unit vector"""
        if not self.normalize_embeddings:
            return embedding
        
        norm = np.linalg.norm(embedding)
        if norm == 0:
            return embedding
        return embedding / norm
    
    @retry_with_exponential_backoff(max_retries=3, initial_delay=2.0)
    def embed(self, text: str) -> Optional[np.ndarray]:
        """Embed a single text
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector or None if text is empty
        """
        if not text or not text.strip():
            return None
        
        try:
            # Create embedding config
            config = genai_types.EmbedContentConfig(
                task_type=self.task_type,
                output_dimensionality=self.output_dimensionality
            )
            
            # Generate embedding
            result = self.client.models.embed_content(
                model=self.model_name_str,
                contents=text,
                config=config
            )
            
            # Extract embedding values
            embedding_values = result.embeddings[0].values
            embedding_array = np.array(embedding_values, dtype=np.float32)
            
            # Normalize if needed (especially for dimensions < 3072)
            if self.output_dimensionality < 3072:
                embedding_array = self._normalize(embedding_array)
            
            return embedding_array
            
        except Exception as e:
            logger.error(f"Gemini embedding error: {e}")
            raise
    
    @retry_with_exponential_backoff(max_retries=3, initial_delay=2.0)
    def embed_batch(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Embed multiple texts efficiently in batches
        
        Args:
            texts: List of texts to embed
            batch_size: Maximum batch size for processing (adjusted for payload limits)
            
        Returns:
            Numpy array of embedding vectors
        """
        if not texts:
            return np.array([], dtype=np.float32).reshape(0, self.dimension)
        
        # Filter empty texts and create index mapping
        valid_texts = []
        valid_indices = []
        for idx, text in enumerate(texts):
            if text and text.strip():
                valid_texts.append(text)
                valid_indices.append(idx)
        
        if not valid_texts:
            return np.zeros((len(texts), self.dimension), dtype=np.float32)
        
        try:
            # Create embedding config
            config = genai_types.EmbedContentConfig(
                task_type=self.task_type,
                output_dimensionality=self.output_dimensionality
            )
            
            # Gemini has a ~40MB payload limit
            # Estimate payload size and split into smaller batches if needed
            max_payload_mb = 35  # Conservative limit (API limit is ~40MB)
            max_payload_bytes = max_payload_mb * 1024 * 1024
            
            # Calculate approximate payload size
            total_chars = sum(len(text.encode('utf-8')) for text in valid_texts)
            
            # If payload is too large, process in smaller batches
            if total_chars > max_payload_bytes:
                # Calculate safe batch size based on average text size
                avg_text_size = total_chars / len(valid_texts)
                safe_batch_size = max(1, int(max_payload_bytes / avg_text_size * 0.8))
                
                logger.info(
                    f"Large batch detected ({total_chars / 1024 / 1024:.2f}MB). "
                    f"Splitting into batches of {safe_batch_size}"
                )
                
                embeddings_list = []
                for i in range(0, len(valid_texts), safe_batch_size):
                    batch = valid_texts[i:i+safe_batch_size]
                    
                    # Process this sub-batch
                    result = self.client.models.embed_content(
                        model=self.model_name_str,
                        contents=batch,
                        config=config
                    )
                    
                    # Extract embeddings from this batch
                    for embedding_obj in result.embeddings:
                        embedding_array = np.array(embedding_obj.values, dtype=np.float32)
                        
                        # Normalize if needed
                        if self.output_dimensionality < 3072:
                            embedding_array = self._normalize(embedding_array)
                        
                        embeddings_list.append(embedding_array)
            else:
                # Process all texts at once if payload is small enough
                result = self.client.models.embed_content(
                    model=self.model_name_str,
                    contents=valid_texts,
                    config=config
                )
                
                # Extract embeddings
                embeddings_list = []
                for embedding_obj in result.embeddings:
                    embedding_array = np.array(embedding_obj.values, dtype=np.float32)
                    
                    # Normalize if needed
                    if self.output_dimensionality < 3072:
                        embedding_array = self._normalize(embedding_array)
                    
                    embeddings_list.append(embedding_array)
            
            # Build result array with all texts (empty as zeros)
            result_array = np.zeros((len(texts), self.dimension), dtype=np.float32)
            for i, idx in enumerate(valid_indices):
                result_array[idx] = embeddings_list[i]
            
            return result_array
            
        except Exception as e:
            logger.error(f"Gemini batch embedding error: {e}")
            raise
    
    @property
    def dimension(self) -> int:
        """Get embedding dimension"""
        return self.output_dimensionality
    
    @property
    def model_name(self) -> str:
        """Get model name"""
        return self.model_name_str


class NVIDIAEmbedding(EmbeddingModel):
    """NVIDIA Build API Embedding Model
    
    Uses NVIDIA's hosted API for embeddings with BGE-M3 and other models.
    Provides high-performance embeddings without local GPU requirements.
    
    Features:
    - NVIDIA-optimized embedding models
    - OpenAI-compatible API format
    - Batch processing support
    - Automatic retry with exponential backoff
    """
    
    def __init__(
        self,
        model_name: str = "nvidia/nv-embeddings-v1",
        api_key: Optional[str] = None,
        base_url: str = "https://integrate.api.nvidia.com/v1",
        normalize_embeddings: bool = True,
        timeout: float = 120.0,
        max_retries: int = 3
    ):
        """Initialize NVIDIA Build embedding model
        
        Args:
            model_name: NVIDIA model identifier (e.g., nvidia/nv-embeddings-v1)
            api_key: NVIDIA API key (required)
            base_url: NVIDIA API base URL
            normalize_embeddings: Whether to normalize embeddings to unit vectors
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
        """
        if not OPENAI_AVAILABLE:
            raise ImportError("openai not installed. Run: pip install openai")
            
        if not api_key:
            raise ValueError("NVIDIA_API_KEY is required for NVIDIA embeddings")
        
        self.model_name_str = model_name
        self.normalize_embeddings = normalize_embeddings
        self.timeout = timeout
        self.max_retries = max_retries
        
        # Initialize OpenAI client configured for NVIDIA
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        
        logger.info(
            f"NVIDIA embedding initialized: model={model_name}, "
            f"endpoint={base_url}, timeout={timeout}s"
        )
    
    def _normalize(self, embedding: np.ndarray) -> np.ndarray:
        """Normalize embedding to unit vector"""
        if not self.normalize_embeddings:
            return embedding
        
        norm = np.linalg.norm(embedding)
        if norm == 0:
            return embedding
        return embedding / norm
    
    @retry_with_exponential_backoff(max_retries=3, initial_delay=1.0)
    def embed(self, text: str) -> Optional[np.ndarray]:
        """Embed a single text
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector or None if text is empty
        """
        if not text or not text.strip():
            return None
        
        try:
            response = self.client.embeddings.create(
                model=self.model_name_str,
                input=text,
                encoding_format="float"
            )
            
            if response.data and len(response.data) > 0:
                embedding_values = response.data[0].embedding
                embedding_array = np.array(embedding_values, dtype=np.float32)
                embedding_array = self._normalize(embedding_array)
                return embedding_array
            else:
                logger.error(f"Unexpected NVIDIA API response format: {response}")
                return None
                
        except Exception as e:
            logger.error(f"NVIDIA API embedding error: {e}")
            raise
    
    @retry_with_exponential_backoff(max_retries=3, initial_delay=1.0)
    def embed_batch(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Embed multiple texts efficiently in batches
        
        Args:
            texts: List of texts to embed
            batch_size: Batch size for processing
            
        Returns:
            Numpy array of embedding vectors
        """
        if not texts:
            return np.array([], dtype=np.float32).reshape(0, self.dimension)
        
        # Filter empty texts and create index mapping
        valid_texts = []
        valid_indices = []
        for idx, text in enumerate(texts):
            if text and text.strip():
                valid_texts.append(text)
                valid_indices.append(idx)
        
        if not valid_texts:
            return np.zeros((len(texts), self.dimension), dtype=np.float32)
        
        try:
            # SAFETY: Truncate texts exceeding NVIDIA's per-text token limit (8192)
            # Use character-based estimate: ~4 chars per token for English text
            MAX_TOKENS = 7500  # Leave margin below 8192 limit
            MAX_CHARS = MAX_TOKENS * 4  # ~30,000 chars
            
            truncated_texts = []
            for text in valid_texts:
                if len(text) > MAX_CHARS:
                    logger.warning(
                        f"Truncating oversized text ({len(text)} chars, ~{len(text)//4} tokens) "
                        f"to {MAX_CHARS} chars for NVIDIA embedding API"
                    )
                    truncated_texts.append(text[:MAX_CHARS])
                else:
                    truncated_texts.append(text)
            valid_texts = truncated_texts
            
            # Process in batches if needed
            all_embeddings = []
            
            for i in range(0, len(valid_texts), batch_size):
                batch_texts = valid_texts[i:i+batch_size]
                
                response = self.client.embeddings.create(
                    model=self.model_name_str,
                    input=batch_texts,
                    encoding_format="float"
                )
                
                # Extract embeddings from this batch
                batch_embeddings = []
                for item in response.data:
                    embedding_array = np.array(item.embedding, dtype=np.float32)
                    if self.normalize_embeddings:
                        embedding_array = self._normalize(embedding_array)
                    batch_embeddings.append(embedding_array)
                
                all_embeddings.extend(batch_embeddings)
            
            # Build result array with all texts (empty as zeros)
            result_array = np.zeros((len(texts), self.dimension), dtype=np.float32)
            for i, idx in enumerate(valid_indices):
                result_array[idx] = all_embeddings[i]
            
            return result_array
            
        except Exception as e:
            logger.error(f"NVIDIA batch embedding error: {e}")
            raise
    
    @property
    def dimension(self) -> int:
        """Get embedding dimension"""
        return _get_registered_dimension(self.model_name_str)
    
    @property
    def model_name(self) -> str:
        """Get model name"""
        return self.model_name_str


class _EmbeddingDiskCache:
    """SQLite-backed persistent embedding cache.
    
    Stores {text_hash -> embedding_vector} on disk so embeddings survive
    process restarts. Avoids re-embedding identical text across rebuilds
    and queries.
    
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
        self._conn.execute("PRAGMA journal_mode=WAL")  # concurrent reads
        self._conn.execute("PRAGMA synchronous=NORMAL")  # faster writes
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
        cache = {}
        with self._lock:
            cursor = self._conn.execute(
                "SELECT cache_key, embedding, dimension FROM embeddings WHERE model_name = ?",
                (self._model_name,)
            )
            for row in cursor:
                key, blob, dim = row
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
                "INSERT OR IGNORE INTO embeddings (cache_key, embedding, dimension, model_name) VALUES (?, ?, ?, ?)",
                [
                    (key, emb.astype(np.float32).tobytes(), len(emb), self._model_name)
                    for key, emb in items
                ]
            )
            self._conn.commit()

    def count(self) -> int:
        """Return total cached entries for this model."""
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*) FROM embeddings WHERE model_name = ?",
                (self._model_name,)
            ).fetchone()
            return row[0] if row else 0

    def db_size_mb(self) -> float:
        """Return the SQLite file size in MB."""
        with self._lock:
            row = self._conn.execute("PRAGMA page_count").fetchone()
            page_count = row[0] if row else 0
            row = self._conn.execute("PRAGMA page_size").fetchone()
            page_size = row[0] if row else 4096
            return (page_count * page_size) / (1024 * 1024)

    def close(self):
        """Close the database connection."""
        with self._lock:
            self._conn.close()


class EmbeddingService:
    """Service for generating and managing embeddings
    
    Abstracts away model details and provides a clean API.
    Features:
    - In-memory LRU cache for fast repeated lookups
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
        """Initialize embedding service
        
        Args:
            model: EmbeddingModel instance to use
            cache_dir: Directory for persistent SQLite cache (auto-created)
            use_cache: Whether to use caching
            max_cache_size: Maximum in-memory cache entries
        """
        self.model = model
        self.cache_dir = cache_dir
        self.use_cache = use_cache
        self.max_cache_size = max_cache_size
        self._embedding_cache: Dict[str, np.ndarray] = {}
        self._cache_lock = threading.Lock()
        self._cache_hits = 0
        self._cache_misses = 0
        self._disk_cache: Optional[_EmbeddingDiskCache] = None
        
        # Initialize persistent disk cache
        if use_cache and cache_dir:
            try:
                self._disk_cache = _EmbeddingDiskCache(cache_dir, model.model_name)
                # Preload disk cache into memory for fast lookups
                self._embedding_cache = self._disk_cache.load_all()
                logger.info(
                    f"EmbeddingService initialized with {model.model_name}, "
                    f"disk cache: {len(self._embedding_cache)} entries preloaded"
                )
            except Exception as e:
                logger.warning(f"Failed to initialize disk cache, using memory-only: {e}")
                self._disk_cache = None
        else:
            logger.info(f"EmbeddingService initialized with {model.model_name}")
    
    def _cache_key(self, text: str) -> str:
        """Generate cache key using SHA-256 for disk persistence safety.
        
        Combines model name + text to avoid cross-model collisions.
        """
        content = f"{self.model.model_name}:{text}"
        return hashlib.sha256(content.encode('utf-8', errors='replace')).hexdigest()
    
    def embed_text(self, text: str) -> EmbeddingResult:
        """Embed a single text with caching
        
        Args:
            text: Text to embed
            
        Returns:
            EmbeddingResult with embedding vector
        """
        cache_key = self._cache_key(text) if self.use_cache else None
        
        # Check cache
        with self._cache_lock:
            if cache_key and cache_key in self._embedding_cache:
                self._cache_hits += 1
                embedding = self._embedding_cache[cache_key]
                found_in_cache = True
            else:
                if self.use_cache:
                    self._cache_misses += 1
                found_in_cache = False

        if not found_in_cache:
            embedding = self.model.embed(text)
            
            # Cache in memory + disk
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
        """Embed multiple texts efficiently with deduplication and caching
        
        Optimizations:
        - Fast hash-based cache lookup (avoids redundant API calls)
        - Deduplication: identical texts are embedded only once
        - Only uncached unique texts are sent to the model
        - New embeddings persisted to disk cache in one transaction
        - Results are scattered back to correct positions
        
        Args:
            texts: List of texts to embed
            batch_size: Batch size for processing
            
        Returns:
            List of EmbeddingResult objects
        """
        if not texts:
            return []
        
        dim = self.model.dimension
        model_name = self.model.model_name
        
        # Phase 1: Fast cache lookup + deduplication
        embeddings = [None] * len(texts)
        uncached_unique_texts = {}  # text -> first index (deduplication)
        text_to_cache_key = {}  # text -> cache_key (avoid recomputing)
        
        with self._cache_lock:
            for idx, text in enumerate(texts):
                if not text or not text.strip():
                    continue
                
                if self.use_cache:
                    cache_key = self._cache_key(text)
                    text_to_cache_key[text] = cache_key
                    
                    if cache_key in self._embedding_cache:
                        embeddings[idx] = self._embedding_cache[cache_key]
                        self._cache_hits += 1
                        continue
                    
                    self._cache_misses += 1
                
                # Track unique uncached texts (deduplication)
                if text not in uncached_unique_texts:
                    uncached_unique_texts[text] = idx
        
        # Phase 2: Embed only unique uncached texts
        new_cache_entries = []  # collect for batch disk write
        if uncached_unique_texts:
            unique_texts = list(uncached_unique_texts.keys())
            
            if unique_texts:
                logger.info(
                    f"Embedding {len(unique_texts)} new texts "
                    f"({len(texts) - len(unique_texts)} cache hits)"
                )
            
            unique_embeddings = self.model.embed_batch(unique_texts, batch_size=batch_size)
            
            # Build text -> embedding mapping from unique results
            text_to_embedding = {}
            for i, text in enumerate(unique_texts):
                emb = unique_embeddings[i]
                text_to_embedding[text] = emb
                
                # Cache in memory
                if self.use_cache and emb is not None:
                    ck = text_to_cache_key.get(text, self._cache_key(text))
                    with self._cache_lock:
                        if len(self._embedding_cache) < self.max_cache_size:
                            self._embedding_cache[ck] = emb
                    new_cache_entries.append((ck, emb))
        else:
            text_to_embedding = {}
            if texts:
                logger.info(f"All {len(texts)} texts found in cache (100% hit rate)")
        
        # Persist new embeddings to disk in one transaction
        if new_cache_entries and self._disk_cache:
            self._disk_cache.put_batch(new_cache_entries)
        
        # Phase 3: Scatter results back to all positions
        results = []
        for idx, text in enumerate(texts):
            if embeddings[idx] is not None:
                embedding = embeddings[idx]
            elif text in text_to_embedding:
                embedding = text_to_embedding[text]
            else:
                embedding = None
            
            results.append(EmbeddingResult(
                text=text,
                embedding=embedding,
                dimension=dim,
                model_name=model_name
            ))
        
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get embedding service statistics"""
        total_cache = self._cache_hits + self._cache_misses
        hit_ratio = self._cache_hits / total_cache if total_cache > 0 else 0
        
        stats = {
            "model_name": self.model.model_name,
            "embedding_dimension": self.model.dimension,
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "cache_hit_ratio": hit_ratio,
            "memory_cache_entries": len(self._embedding_cache),
            "cache_enabled": self.use_cache,
            "disk_cache_enabled": self._disk_cache is not None,
        }
        
        if self._disk_cache:
            stats["disk_cache_entries"] = self._disk_cache.count()
            stats["disk_cache_size_mb"] = round(self._disk_cache.db_size_mb(), 2)
        
        return stats


def create_embedding_model(
    model_type: str = "bge-m3",
    **kwargs
) -> EmbeddingModel:
    """Factory function to create embedding models
    
    Args:
        model_type: Type of embedding model ('bge-m3', 'lm-studio', 'gemini', etc.)
        **kwargs: Additional arguments to pass to model constructor
        
    Returns:
        EmbeddingModel instance
    """
    model_type_normalized = model_type.lower()

    if model_type_normalized in ["bge-m3", "baai/bge-m3", "sentence-transformers", "local"]:
        return BGEm3Embedding(**kwargs)

    if model_type_normalized in ["hf", "hf-inference", "huggingface"]:
        return HFInferenceEmbedding(**kwargs)
    
    if model_type_normalized in ["lm-studio", "lmstudio", "openai-compatible"]:
        return LMStudioEmbedding(**kwargs)
    
    if model_type_normalized in ["gemini", "google", "google-gemini"]:
        return GeminiEmbedding(**kwargs)
    
    if model_type_normalized in ["nvidia", "nvidia-build", "nvidia-api"]:
        return NVIDIAEmbedding(**kwargs)

    raise ValueError(f"Unknown embedding model type: {model_type}")


# Default cache directory
DEFAULT_CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "cache")


def create_embedding_service(
    model_type: str = "bge-m3",
    use_cache: bool = True,
    max_cache_size: int = 100000,
    cache_dir: Optional[str] = None,
    **kwargs
) -> EmbeddingService:
    """Factory function to create embedding service
    
    Args:
        model_type: Type of embedding model
        use_cache: Whether to enable caching
        max_cache_size: Maximum in-memory cache entries
        cache_dir: Directory for persistent cache (default: data/cache/)
        **kwargs: Additional arguments for model
        
    Returns:
        EmbeddingService instance with disk-backed persistent cache
    """
    model = create_embedding_model(model_type, **kwargs)
    
    # Use provided cache_dir, env var, or default
    if cache_dir is None:
        cache_dir = os.environ.get("EMBEDDING_CACHE_DIR", DEFAULT_CACHE_DIR)
    
    return EmbeddingService(
        model,
        cache_dir=cache_dir,
        use_cache=use_cache,
        max_cache_size=max_cache_size
    )
