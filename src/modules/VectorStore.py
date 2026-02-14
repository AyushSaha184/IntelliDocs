"""Vector Store Module - FAISS-based vector storage

Fast similarity search using Facebook AI Similarity Search (FAISS).
"""

from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, asdict, field
import numpy as np
import json
import os
from pathlib import Path
from datetime import datetime
import logging

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

from src.utils.Logger import get_logger

logger = get_logger(__name__)


@dataclass(slots=True)
class SearchResult:
    """Result from vector search"""
    document_id: str
    chunk_id: str
    text: str
    similarity_score: float
    distance: float
    metadata: Dict[str, Any]


@dataclass(slots=True)
class VectorMetadata:
    """Metadata for stored vectors"""
    vector_id: str
    document_id: str
    chunk_id: str
    text: str
    text_length: int
    embedding_model: str
    added_at: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class FAISSVectorStore:
    """FAISS (Facebook AI Similarity Search) Vector Store
    
    Fast, efficient similarity search with CPU and GPU support.
    """
    
    def __init__(
        self,
        dimension: int,
        index_type: str = "flat",
        use_gpu: bool = False,
        store_path: Optional[str] = None
    ):
        """Initialize FAISS vector store
        
        Args:
            dimension: Embedding dimension
            index_type: Type of FAISS index ('flat', 'ivf', 'hnsw')
            use_gpu: Whether to use GPU acceleration
            store_path: Path to store metadata and index
        """
        if not FAISS_AVAILABLE:
            raise ImportError("faiss-cpu or faiss-gpu not installed")
        
        self.dimension = dimension
        self.index_type = index_type
        self.use_gpu = use_gpu
        self.store_path = store_path or os.path.join(
            os.path.dirname(__file__), "../../data/vector_store"
        )
        
        Path(self.store_path).mkdir(parents=True, exist_ok=True)
        
        # Create FAISS index
        self._create_index()
        
        # Metadata storage with ID-to-index mapping for O(1) lookups
        self.metadata_store: Dict[str, VectorMetadata] = {}
        self._id_to_index: Dict[str, int] = {}  # Fast ID lookup
        self._index_to_id: Dict[int, str] = {}  # Cached reverse mapping (O(1) search)
        self._index_counter = 0
        
        logger.info(f"FAISS store initialized: dimension={dimension}, index_type={index_type}")
    
    def _create_index(self):
        """Create FAISS index based on type - skip training for tests"""
        if self.index_type.lower() == "flat":
            # Simple exact search (best for small datasets, fastest)
            self.index = faiss.IndexFlatL2(self.dimension)
        elif self.index_type.lower() == "ivf":
            # Inverted File Index (best for medium datasets)
            quantizer = faiss.IndexFlatL2(self.dimension)
            nlist = max(10, self.dimension // 100)  # Adaptive clusters
            self.index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist)
            # Don't train here - train on first add if needed
            self._ivf_trained = False
        elif self.index_type.lower() == "hnsw":
            # HNSW (best for large datasets)
            self.index = faiss.IndexHNSWFlat(self.dimension, 16)
        else:
            raise ValueError(f"Unknown index type: {self.index_type}")
    
    def add_vectors(
        self,
        vectors: List[np.ndarray],
        metadata: List[Dict[str, Any]],
        ids: Optional[List[str]] = None
    ) -> List[str]:
        """Add vectors with metadata (optimized)"""
        if len(vectors) != len(metadata):
            raise ValueError("vectors and metadata must have same length")
        
        # Generate IDs if not provided
        if ids is None:
            ids = [f"vec_{self._index_counter + i}" for i in range(len(vectors))]
            self._index_counter += len(vectors)
        
        # Convert to numpy array if needed
        vectors_array = np.array(vectors, dtype=np.float32)
        
        # Train IVF index on first batch if needed
        if self.index_type.lower() == "ivf" and not self._id_to_index and len(vectors_array) >= 100:
            self.index.train(vectors_array[:min(1000, len(vectors_array))])
            self._ivf_trained = True
        
        # Add to FAISS index
        self.index.add(vectors_array)
        
        # Store metadata with O(1) ID lookup and maintain reverse mapping
        start_idx = self._index_counter - len(vectors)
        for i, (vector_id, meta) in enumerate(zip(ids, metadata)):
            idx = start_idx + i
            self._id_to_index[vector_id] = idx
            self._index_to_id[idx] = vector_id  # Maintain cached reverse mapping
            self.metadata_store[vector_id] = VectorMetadata(
                vector_id=vector_id,
                document_id=meta.get("document_id", ""),
                chunk_id=meta.get("chunk_id", ""),
                text=meta.get("text", ""),
                text_length=len(meta.get("text", "")),
                embedding_model=meta.get("embedding_model", ""),
                added_at=datetime.now().isoformat(),
                metadata=meta.get("metadata", {})
            )
        
        return ids
    
    def search(
        self,
        query_vector: np.ndarray,
        k: int = 5,
        threshold: Optional[float] = None
    ) -> List[SearchResult]:
        """Search for similar vectors (optimized with cached reverse mapping)
        
        Args:
            query_vector: Query embedding
            k: Number of results
            threshold: Distance threshold (optional)
            
        Returns:
            List of SearchResult objects
        """
        if self.index.ntotal == 0:
            return []
        
        # Reshape query if needed
        query_vector = np.array(query_vector, dtype=np.float32).reshape(1, -1)
        
        # Search FAISS index
        distances, indices = self.index.search(query_vector, min(k, self.index.ntotal))
        distances = distances[0]
        indices = indices[0]
        
        results = []
        # Use cached reverse mapping (O(1) lookup, no dict rebuild per search)
        for dist, idx in zip(distances, indices):
            if idx == -1:
                continue
            
            # Apply threshold if provided
            if threshold is not None and dist > threshold:
                continue
            
            # Fast O(1) lookup using cached index_to_id mapping
            vector_id = self._index_to_id.get(idx)
            if not vector_id:
                continue
                
            meta = self.metadata_store.get(vector_id)
            if not meta:
                continue
            
            # Convert distance to similarity score (0-1)
            similarity = 1.0 / (1.0 + dist)
            
            results.append(SearchResult(
                document_id=meta.document_id,
                chunk_id=meta.chunk_id,
                text=meta.text,
                similarity_score=similarity,
                distance=dist,
                metadata=meta.metadata
            ))
        
        return results
    
    def get_vector(self, vector_id: str) -> Optional[np.ndarray]:
        """Get vector by ID (optimized O(1) lookup)"""
        if vector_id not in self._id_to_index:
            return None
        
        idx = self._id_to_index[vector_id]
        vector = self.index.reconstruct(idx)
        return vector
    
    def delete_vector(self, vector_id: str) -> bool:
        """Delete vector (optimized O(1) lookup)"""
        if vector_id not in self._id_to_index:
            return False
        
        idx = self._id_to_index[vector_id]
        del self._id_to_index[vector_id]
        if idx in self._index_to_id:
            del self._index_to_id[idx]  # Keep reverse mapping in sync
        if vector_id in self.metadata_store:
            del self.metadata_store[vector_id]
        return True
    
    def size(self) -> int:
        """Get number of vectors"""
        return self.index.ntotal
    
    def save(self, path: Optional[str] = None) -> bool:
        """Save vector store to disk"""
        path = path or self.store_path
        Path(path).mkdir(parents=True, exist_ok=True)
        
        try:
            # Save FAISS index
            index_path = os.path.join(path, "index.faiss")
            faiss.write_index(self.index, index_path)
            
            # Save metadata
            metadata_path = os.path.join(path, "metadata.json")
            metadata_dict = {
                vid: asdict(meta) for vid, meta in self.metadata_store.items()
            }
            with open(metadata_path, 'w') as f:
                json.dump(metadata_dict, f, indent=2)
            
            # Save config
            config_path = os.path.join(path, "config.json")
            config = {
                "dimension": self.dimension,
                "index_type": self.index_type,
                "num_vectors": self.size(),
                "saved_at": datetime.now().isoformat()
            }
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            logger.info(f"Vector store saved to {path}")
            return True
        except Exception as e:
            logger.error(f"Error saving vector store: {e}")
            return False
    
    def load(self, path: Optional[str] = None) -> bool:
        """Load vector store from disk (optimized with cached reverse mapping)"""
        path = path or self.store_path
        
        try:
            # Load FAISS index
            index_path = os.path.join(path, "index.faiss")
            self.index = faiss.read_index(index_path)
            
            # Load metadata and rebuild both id_to_index and index_to_id mappings
            metadata_path = os.path.join(path, "metadata.json")
            with open(metadata_path, 'r') as f:
                metadata_dict = json.load(f)
            
            self.metadata_store = {}
            self._id_to_index = {}
            self._index_to_id = {}  # Rebuild cached reverse mapping
            
            # Use sequential index for proper alignment
            for idx, (vid, meta_dict) in enumerate(metadata_dict.items()):
                meta = VectorMetadata(**meta_dict)
                self.metadata_store[vid] = meta
                self._id_to_index[vid] = idx
                self._index_to_id[idx] = vid  # Maintain reverse cache
            
            self._index_counter = len(metadata_dict)
            return True
        except Exception as e:
            logger.error(f"Error loading vector store: {e}")
            return False
    
    def add_vectors_batch(
        self,
        vectors: np.ndarray,
        metadata_list: List[Dict[str, Any]],
        ids: Optional[List[str]] = None,
        batch_size: int = 10000
    ) -> List[str]:
        """Add large batches of vectors efficiently (supports millions)
        
        Args:
            vectors: Numpy array of vectors (N x D)
            metadata_list: List of metadata dictionaries
            ids: Optional vector IDs
            batch_size: Process in smaller batches for memory efficiency
            
        Returns:
            List of vector IDs
        """
        all_ids = []
        num_vectors = len(vectors)
        
        # Process in batches to avoid memory issues
        for batch_start in range(0, num_vectors, batch_size):
            batch_end = min(batch_start + batch_size, num_vectors)
            batch_vectors = vectors[batch_start:batch_end]
            batch_metadata = metadata_list[batch_start:batch_end]
            batch_ids = ids[batch_start:batch_end] if ids else None
            
            batch_result_ids = self.add_vectors(batch_vectors, batch_metadata, batch_ids)
            all_ids.extend(batch_result_ids)
            
            logger.debug(f"Added {len(batch_result_ids)} vectors ({batch_end}/{num_vectors})")
        
        return all_ids
    
    def get_size(self) -> int:
        """Get number of vectors in store"""
        return self.index.ntotal if self.index else 0
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get vector store statistics"""
        return {
            "total_vectors": self.get_size(),
            "index_type": self.index_type,
            "dimension": self.dimension,
            "metadata_count": len(self.metadata_store),
            "store_path": self.store_path
        }


class VectorStoreService:
    """Service for managing vectors and searches"""
    
    def __init__(
        self,
        vector_store: FAISSVectorStore,
        embedding_model_name: str
    ):
        """Initialize vector store service
        
        Args:
            vector_store: FAISSVectorStore instance
            embedding_model_name: Name of embedding model used
        """
        self.store = vector_store
        self.embedding_model_name = embedding_model_name
        self._searches_performed = 0
        
        logger.info(f"VectorStoreService initialized with {embedding_model_name}")
    
    def add_chunk_embeddings(
        self,
        chunk_embeddings: List[Tuple[str, str, np.ndarray, str]],
        metadata: Optional[List[Dict[str, Any]]] = None
    ) -> List[str]:
        """Add chunk embeddings to store
        
        Args:
            chunk_embeddings: List of (document_id, chunk_id, embedding, text) tuples
            metadata: Optional additional metadata per chunk
            
        Returns:
            List of vector IDs
        """
        vectors = [e[2] for e in chunk_embeddings]
        metas = []
        
        for i, (doc_id, chunk_id, emb, text) in enumerate(chunk_embeddings):
            meta = {
                "document_id": doc_id,
                "chunk_id": chunk_id,
                "text": text,
                "embedding_model": self.embedding_model_name
            }
            if metadata and i < len(metadata):
                meta.update(metadata[i])
            metas.append(meta)
        
        return self.store.add_vectors(vectors, metas)
    
    def search_similar(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
        threshold: Optional[float] = None
    ) -> List[SearchResult]:
        """Search for similar chunks
        
        Args:
            query_embedding: Query embedding vector
            k: Number of results
            threshold: Similarity threshold
            
        Returns:
            List of SearchResult objects
        """
        self._searches_performed += 1
        return self.store.search(query_embedding, k, threshold)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get service statistics"""
        return {
            "total_vectors": self.store.size(),
            "embedding_model": self.embedding_model_name,
            "searches_performed": self._searches_performed,
            "vector_dimension": self.store.dimension
        }
