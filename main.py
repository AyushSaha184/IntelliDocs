"""
Complete RAG System - Build & Query - Enterprise Scale

Unified interface for the complete RAG system with support for millions of documents:
1. BUILD MODE: Load documents -> chunk -> embed -> index (with streaming and batching)
2. QUERY MODE: Interactive query interface
3. API MODE: Start REST API server
4. TEST MODE: Run test queries

Usage:
    python main.py --build              # Build vector store
    python main.py --query              # Interactive queries
    python main.py --api                # Start API server
    python main.py --test "query text"  # Test a query
"""
from typing import Dict, List, Optional, Generator
import sys
import os
from pathlib import Path
import argparse
from datetime import datetime
import time
import numpy as np
import json
import psycopg2
import shutil
import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(__file__))

from src.modules.Loader import DocumentLoader, DocumentMetadata, load_documents
from src.modules.Chunking import TextChunker, TextChunk, ChunkingStrategy, create_chunker
from src.modules.Embeddings import create_embedding_service
from src.modules.VectorStore import FAISSVectorStore
from src.modules.Retriever import RAGRetriever, LMStudioReranker
from src.modules.QueryGeneration import QueryHandler
from src.modules.LLM import create_llm, BaseLLM
from src.modules.ParallelPipeline import ParallelRAGPipeline, GPUConfig
from src.utils.Logger import get_logger
from config.config import (
    LLM_PROVIDER,
    LLM_MODEL,
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS,
    HF_TOKEN,
    HF_INFERENCE_PROVIDER,
    GEMINI_API_KEY,
    OPENROUTER_API_KEY,
    OPENROUTER_SITE_URL,
    OPENROUTER_SITE_NAME,
    EMBEDDING_PROVIDER,
    EMBEDDING_MODEL,
    EMBEDDING_NORMALIZE,
    EMBEDDING_TIMEOUT,
    EMBEDDING_MAX_RETRIES,
    POSTGRES_HOST,
    POSTGRES_PORT,
    POSTGRES_DB,
    POSTGRES_USER,
    POSTGRES_PASSWORD,
    EMBEDDING_TASK_TYPE,
    EMBEDDING_DIMENSION,
    LM_STUDIO_BASE_URL,
    LM_STUDIO_API_KEY,
    USE_RERANKER,
    RERANKER_MODEL,
    MIN_CHUNKS_TO_RERANK,
    TOP_K_AFTER_RERANK,
)

logger = get_logger(__name__)

# Configuration for scalability
BATCH_SIZE_DOCS = 1000
BATCH_SIZE_CHUNKS = 5000
BATCH_SIZE_EMBEDDINGS = 32  # Reduced from 128 for LM Studio stability
VECTOR_BATCH_SIZE = 10000

def clear_postgres_tables():
    """Clear all PostgreSQL tables with timeout protection and proper lock handling"""
    try:
        logger.info("  [INFO] Connecting to PostgreSQL...")
        conn = psycopg2.connect(
            host=POSTGRES_HOST,
            port=POSTGRES_PORT,
            database=POSTGRES_DB,
            user=POSTGRES_USER,
            password=POSTGRES_PASSWORD,
            connect_timeout=10,
            options='-c statement_timeout=10000'  # 10 second timeout
        )
        conn.autocommit = True  # Important: autocommit to avoid locks
        cursor = conn.cursor()
        
        # Terminate other connections if needed
        logger.debug("  [INFO] Terminating other connections...")
        cursor.execute("""
            SELECT pg_terminate_backend(pid) 
            FROM pg_stat_activity 
            WHERE datname = %s AND pid <> pg_backend_pid()
        """, (POSTGRES_DB,))
        
        # Clear tables in order (respecting foreign keys)
        tables = ['chunks', 'hash_index', 'processing_log', 'documents']
        total_deleted = 0
        for table in tables:
            cursor.execute(f"DELETE FROM {table}")
            count = cursor.rowcount
            total_deleted += count
            if count > 0:
                logger.debug(f"    Deleted {count} rows from {table}")
        
        cursor.close()
        conn.close()
        logger.info(f"  [OK] Cleared PostgreSQL tables ({total_deleted} total rows deleted)")
        return True
        
    except psycopg2.extensions.QueryCanceledError as e:
        logger.error(f"  [ERROR] PostgreSQL timeout (tables locked?): {e}")
        logger.warning(f"  [HINT] Try: python clear_db.py to manually clear tables")
        return False
    except psycopg2.Error as e:
        logger.error(f"  [ERROR] PostgreSQL error: {e}")
        return False
    except Exception as e:
        logger.error(f"  [ERROR] Unexpected error clearing database: {e}")
        return False


# Global state for API
pipeline_global: Optional['RAGPipeline'] = None
query_handler_global: Optional['QueryHandler'] = None


def _build_embedding_kwargs(device: str, model_name: Optional[str] = None) -> Dict[str, object]:
    kwargs: Dict[str, object] = {
        "model_name": model_name or EMBEDDING_MODEL,
        "normalize_embeddings": EMBEDDING_NORMALIZE
    }

    if EMBEDDING_PROVIDER.lower() in ["hf", "hf-inference", "huggingface"]:
        kwargs.update({
            "api_key": HF_TOKEN,
            "provider": HF_INFERENCE_PROVIDER,
            "timeout": EMBEDDING_TIMEOUT,
            "max_retries": EMBEDDING_MAX_RETRIES
        })
    elif EMBEDDING_PROVIDER.lower() in ["gemini", "google", "google-gemini"]:
        kwargs.update({
            "api_key": GEMINI_API_KEY,
            "task_type": EMBEDDING_TASK_TYPE,
            "output_dimensionality": EMBEDDING_DIMENSION,
            "timeout": EMBEDDING_TIMEOUT,
            "max_retries": EMBEDDING_MAX_RETRIES
        })
    elif EMBEDDING_PROVIDER.lower() in ["lm-studio", "lmstudio", "openai-compatible"]:
        kwargs.update({
            "base_url": LM_STUDIO_BASE_URL,
            "api_key": LM_STUDIO_API_KEY if LM_STUDIO_API_KEY else None,
            "timeout": EMBEDDING_TIMEOUT,
            "max_retries": EMBEDDING_MAX_RETRIES
        })
    else:
        kwargs["device"] = device

    return kwargs


def _create_reranker():
    """Create LM Studio reranker for local inference."""
    if not USE_RERANKER:
        return None
    
    return LMStudioReranker(
        base_url=LM_STUDIO_BASE_URL,
        model=RERANKER_MODEL,
        min_chunks_to_rerank=MIN_CHUNKS_TO_RERANK,
        top_k_after_rerank=TOP_K_AFTER_RERANK
    )


class RAGPipeline:
    """
    Enterprise-scale end-to-end RAG pipeline for millions of documents.
    
    Optimizations:
    - Streaming document loading with generator-based processing
    - Batch chunking with database persistence
    - Batch embedding with caching and deduplication
    - Incremental vector store indexing
    - Resume capability for interrupted processing
    """
    
    def __init__(
        self,
        documents_dir: Optional[str] = None,
        chunks_dir: Optional[str] = None,
        vector_store_dir: Optional[str] = None,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        strategy: str = "fixed_size",
        encoding_name: str = "cl100k_base",
        embedding_model: str = EMBEDDING_MODEL,
        vector_store_type: str = "flat",
        device: str = "cpu",
        use_db: bool = True,
        batch_size: int = BATCH_SIZE_DOCS
    ):
        """Initialize scalable RAG pipeline with all components.
        
        Args:
            documents_dir: Path to documents directory
            chunks_dir: Path to save chunk metadata
            vector_store_dir: Path to save vector store
            chunk_size: Target chunk size in tokens
            chunk_overlap: Token overlap between chunks
            strategy: Chunking strategy
            encoding_name: Tokenizer encoding name
            embedding_model: Embedding model ID to use
            vector_store_type: Type of vector store (flat, ivf, hnsw)
            device: Device for embeddings (cpu, cuda)
            use_db: Use database backends for scalability
            batch_size: Documents to process before checkpoint
        """
        self.documents_dir = documents_dir or "data/documents"
        self.chunks_dir = chunks_dir or "data/chunks"
        self.vector_store_dir = vector_store_dir or "data/vector_store"
        
        # Initialize components with database support
        self.document_loader = DocumentLoader(
            documents_dir=self.documents_dir,
            use_db=use_db,
            batch_size=batch_size
        )
        
        self.text_chunker = TextChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            default_strategy=ChunkingStrategy(strategy),
            encoding_name=encoding_name,
            min_chunk_size=50,
            chunks_dir=self.chunks_dir,
            use_db=use_db
        )
        
        # Initialize embedding service with caching
        self.embedding_service = create_embedding_service(
            model_type=EMBEDDING_PROVIDER,
            use_cache=True,
            max_cache_size=10000,
            **_build_embedding_kwargs(device, embedding_model)
        )
        
        # Initialize vector store with batch support
        self.vector_store = FAISSVectorStore(
            dimension=self.embedding_service.model.dimension,
            index_type=vector_store_type,
            store_path=self.vector_store_dir
        )
        
        # Initialize reranker if enabled
        self.reranker = _create_reranker()
        
        # Initialize retriever with optional reranking
        self.retriever = RAGRetriever(
            vector_store=self.vector_store,
            embedding_service=self.embedding_service,
            chunks={},
            reranker=self.reranker,
            use_reranker=USE_RERANKER
        )
        
        self.processing_stats = {
            "step_1_load_documents": 0,
            "step_2_chunk_documents": 0,
            "step_3_generate_embeddings": 0,
            "step_4_build_vector_store": 0,
            "total_documents": 0,
            "total_chunks": 0,
            "total_vectors": 0,
            "total_time_seconds": 0
        }
        
        logger.info("Enterprise RAG Pipeline initialized with DB backends")
    
    def step_1_load_documents_stream(self) -> Generator[DocumentMetadata, None, None]:
        """Step 1: Stream documents from directory (memory efficient for millions)
        
        Yields:
            DocumentMetadata objects one at a time
        """
        step_start = time.time()
        logger.info("STEP 1: Streaming Documents")
        
        loaded_count = 0
        for doc in self.document_loader.load_all_documents(store_content=True):
            if doc:
                loaded_count += 1
                yield doc
                
                # Log progress
                if loaded_count % (BATCH_SIZE_DOCS * 5) == 0:
                    logger.info(f"Streamed {loaded_count} documents")
        
        step_time = time.time() - step_start
        self.processing_stats["step_1_load_documents"] = step_time
        self.processing_stats["total_documents"] = loaded_count
        logger.info(f"Step 1: Streamed {loaded_count} documents in {step_time:.2f}s")
    
    def step_2_chunk_documents_batch(self, documents: List[DocumentMetadata]) -> List[TextChunk]:
        """Step 2: Chunk documents in batches with database persistence
        
        Args:
            documents: List of DocumentMetadata objects
            
        Returns:
            List of TextChunk objects (from recent batch)
        """
        if not documents:
            return []
        
        chunks_batch = []
        
        for doc in documents:
            # Get document content from database
            if self.document_loader.use_db:
                content = self.document_loader.get_document_content(doc.id)
                if not content:
                    continue
            else:
                content = doc.content if hasattr(doc, 'content') else ""
            
            # Stream chunks from document
            for chunk in self.text_chunker.chunk_document_stream(doc.id, content, doc.name):
                chunks_batch.append(chunk)
                
                # Save batch when reaching threshold
                if len(chunks_batch) >= BATCH_SIZE_CHUNKS:
                    self.text_chunker.save_chunks_batch(chunks_batch)
                    logger.debug(f"Saved {len(chunks_batch)} chunks to database")
                    chunks_batch = []
        
        # Save remaining chunks
        if chunks_batch:
            self.text_chunker.save_chunks_batch(chunks_batch)
        
        return chunks_batch
    
    def step_3_generate_embeddings_batch(self, chunks: List[TextChunk], save_to_store: bool = True, save_chunks_to_db: bool = False) -> Dict[str, np.ndarray]:
        """Step 3: Generate embeddings for chunks in batches
        
        Args:
            chunks: List of TextChunk objects
            save_to_store: Save embeddings to vector store immediately
            save_chunks_to_db: Save chunks to PostgreSQL (disabled by default for speed)
            
        Returns:
            Dictionary mapping chunk_id to embedding
        """
        if not chunks:
            return {}
        
        step_start = time.time()
        logger.info(f"STEP 3: Generating Embeddings for {len(chunks)} chunks")
        
        embeddings_dict = {}
        vectors = []
        metadata_list = []
        chunk_ids = []
        
        # Batch embed chunks
        chunk_texts = [chunk.text for chunk in chunks]
        embedding_results = self.embedding_service.embed_batch(
            chunk_texts,
            batch_size=BATCH_SIZE_EMBEDDINGS
        )
        
        for chunk, result in zip(chunks, embedding_results):
            if result.embedding is not None:
                embeddings_dict[chunk.id] = result.embedding
                vectors.append(result.embedding)
                chunk_ids.append(chunk.id)
                
                metadata_list.append({
                    "chunk_id": chunk.id,
                    "document_id": chunk.metadata.document_id,
                    "text": chunk.text,
                    "embedding_model": result.model_name,
                    "metadata": {
                        "chunk_index": chunk.metadata.chunk_index,
                        "token_count": chunk.metadata.token_count
                    }
                })
        
        # Save chunks to database only if requested (disabled by default for build performance)
        if save_chunks_to_db and self.text_chunker.use_db and chunks:
            self.text_chunker.save_chunks_batch(chunks)
        
        # Save to vector store if requested
        if save_to_store and vectors:
            vectors_array = np.array(vectors, dtype=np.float32)
            self.vector_store.add_vectors_batch(
                vectors_array,
                metadata_list,
                chunk_ids,
                batch_size=VECTOR_BATCH_SIZE
            )
            logger.info(f"Added {len(vectors)} vectors to store")
            self.processing_stats["total_vectors"] += len(vectors)
        
        step_time = time.time() - step_start
        self.processing_stats["step_3_generate_embeddings"] += step_time
        logger.info(f"Step 3: Generated {len(embeddings_dict)} embeddings in {step_time:.2f}s")
        
        return embeddings_dict
    
    def build_pipeline_scalable(self) -> bool:
        """Build RAG pipeline with streaming for millions of documents"""
        pipeline_start = time.time()
        logger.info("="*80)
        logger.info("BUILDING ENTERPRISE RAG PIPELINE - SCALABLE MODE")
        logger.info("="*80)
        
        try:
            docs_processed = 0
            chunks_batch = []
            
            # Stream and process documents in batches
            for doc in self.step_1_load_documents_stream():
                # Get content from database
                if self.document_loader.use_db:
                    content = self.document_loader.get_document_content(doc.id)
                    if not content:
                        logger.warning(f"No content for {doc.id}")
                        continue
                else:
                    content = doc.content if hasattr(doc, 'content') else ""
                
                # Chunk document
                for chunk in self.text_chunker.chunk_document_stream(doc.id, content, doc.name):
                    chunks_batch.append(chunk)
                    
                    # Process batch when reaching threshold
                    # Note: save_chunks_to_db defaults to False for build performance
                    # Vector store already contains all chunk data, so PostgreSQL storage
                    # during build is redundant and slows down processing by 2-3x
                    if len(chunks_batch) >= BATCH_SIZE_CHUNKS:
                        self.step_3_generate_embeddings_batch(chunks_batch, save_to_store=True)
                        chunks_batch = []
                
                docs_processed += 1
                if docs_processed % (BATCH_SIZE_DOCS * 10) == 0:
                    logger.info(f"Progress: {docs_processed} documents processed")
            
            # Process remaining chunks
            if chunks_batch:
                self.step_3_generate_embeddings_batch(chunks_batch, save_to_store=True)
            
            # Save vector store
            self.vector_store.save()
            logger.info(f"Vector store saved with {self.vector_store.get_size()} vectors")
            
            pipeline_time = time.time() - pipeline_start
            self.processing_stats["total_time_seconds"] = pipeline_time
            
            logger.info("="*80)
            logger.info("PIPELINE BUILD COMPLETE")
            logger.info(self._format_stats())
            logger.info("="*80)
            
            return True
            
        except Exception as e:
            logger.error(f"Error building pipeline: {e}")
            return False
    
    def _format_stats(self) -> str:
        """Format processing statistics for display"""
        stats = self.processing_stats
        docs_per_sec = (
            stats["total_documents"] / stats["total_time_seconds"]
            if stats["total_time_seconds"] > 0
            else 0
        )
        return f"""
        Documents: {stats['total_documents']}
        Chunks: {self.text_chunker.get_chunk_count()}
        Vectors: {stats['total_vectors']}
        Time: {stats['total_time_seconds']:.2f}s
        Docs/sec: {docs_per_sec:.2f}
        """
    
    def step_3_generate_embeddings(self) -> Dict[str, np.ndarray]:
        """Step 3: Generate embeddings for all chunks."""
        step_start = time.time()
        logger.info("STEP 3: Generating Embeddings")
        
        if not self.all_chunks:
            logger.warning("No chunks to embed")
            return {}
        
        embeddings_dict = {}
        total_chunks = sum(len(chunks) for chunks in self.all_chunks.values())
        
        # Collect all chunk texts
        chunk_texts = []
        chunk_ids = []
        for doc_id, chunks in self.all_chunks.items():
            for chunk in chunks:
                chunk_texts.append(chunk.text)
                chunk_ids.append(chunk.id)
        
        # Batch embed all texts
        embedding_results = self.embedding_service.embed_batch(
            chunk_texts,
            batch_size=32
        )
        
        # Store embeddings
        for chunk_id, result in zip(chunk_ids, embedding_results):
            embeddings_dict[chunk_id] = result.embedding
        
        step_time = time.time() - step_start
        self.processing_stats["step_3_generate_embeddings"] = step_time
        
        logger.info(f"Step 3: Generated {total_chunks} embeddings in {step_time:.2f}s")
        return embeddings_dict
    
    def step_4_build_vector_store(self, embeddings_dict: Dict[str, np.ndarray]) -> None:
        """Step 4: Build vector store from embeddings."""
        step_start = time.time()
        logger.info("STEP 4: Building Vector Store")
        
        if not embeddings_dict:
            logger.warning("No embeddings to store")
            return
        
        # Prepare vectors and metadata
        vectors = []
        ids = []
        metadata = []
        
        for doc_id, chunks in self.all_chunks.items():
            for chunk in chunks:
                if chunk.id in embeddings_dict:
                    vectors.append(embeddings_dict[chunk.id])
                    ids.append(chunk.id)
                    
                    # Store chunk metadata
                    metadata.append({
                        "document_id": doc_id,
                        "chunk_id": chunk.id,
                        "text": chunk.text,
                        "token_count": chunk.metadata.token_count,
                        "char_count": chunk.metadata.char_count,
                        "chunk_index": chunk.metadata.chunk_index
                    })
        
        # Add vectors to store
        if vectors:
            vectors_array = np.array(vectors, dtype=np.float32)
            self.vector_store.add_vectors(
                vectors=vectors_array,
                ids=ids,
                metadata=metadata
            )
        
        # Save vector store
        os.makedirs(self.vector_store_dir, exist_ok=True)
        self.vector_store.save(self.vector_store_dir)
        
        step_time = time.time() - step_start
        self.processing_stats["step_4_build_vector_store"] = step_time
        self.processing_stats["total_vectors"] = len(vectors)
        
        logger.info(f"Step 4: Built vector store with {len(vectors)} vectors in {step_time:.2f}s")
    
    def run_complete_pipeline(self) -> bool:
        """Execute complete RAG pipeline from start to finish."""
        return self.build_pipeline_scalable()


def load_chunks_metadata(chunks_dir: str) -> dict:
    """Load chunk metadata from files"""
    chunks_metadata = {}
    chunks_path = Path(chunks_dir)
    
    if not chunks_path.exists():
        logger.warning(f"Chunks directory not found: {chunks_dir}")
        return chunks_metadata
    
    for chunk_file in chunks_path.glob("*_chunks.json"):
        try:
            with open(chunk_file, 'r') as f:
                data = json.load(f)
                if isinstance(data, list):
                    for chunk in data:
                        chunks_metadata[chunk.get('id')] = chunk
                elif isinstance(data, dict):
                    chunks_metadata.update(data)
        except Exception as e:
            logger.error(f"Error loading {chunk_file}: {e}")
    
    return chunks_metadata


def run_interactive_query(args):
    """Run interactive query mode"""
    print("\n" + "="*80)
    print("RAG INTERACTIVE QUERY MODE")
    print("="*80)
    print("\nLoading RAG components...")
    
    try:
        # Initialize embedding service first
        embedding_service = create_embedding_service(
            model_type=EMBEDDING_PROVIDER,
            **_build_embedding_kwargs(args.device)
        )
        logger.info("Embedding service initialized")
        
        # Load vector store
        vector_store = FAISSVectorStore(
            dimension=embedding_service.model.dimension,
            index_type="flat"
        )
        vector_store.load(args.vector_store_dir)
        logger.info(f"Loaded vector store from {args.vector_store_dir}")
        
        # Load chunks metadata
        chunks = load_chunks_metadata(args.chunks_dir)
        logger.info(f"Loaded {len(chunks)} chunks")
        
        # Initialize reranker if enabled
        reranker = _create_reranker()
        
        # Initialize retriever with optional reranking
        retriever = RAGRetriever(
            vector_store=vector_store,
            embedding_service=embedding_service,
            chunks=chunks,
            reranker=reranker,
            use_reranker=USE_RERANKER
        )
        
        # Initialize LLM (OpenRouter, Google Gemini, or HuggingFace)
        llm = None
        try:
            if LLM_PROVIDER.lower() in ["openrouter", "open-router"]:
                provider_display = "OpenRouter"
                api_key = OPENROUTER_API_KEY
            elif LLM_PROVIDER.lower() in ["gemini", "google", "google-ai"]:
                provider_display = "Google Gemini"
                api_key = GEMINI_API_KEY
            else:
                provider_display = "Hugging Face Inference"
                api_key = HF_TOKEN
                
            print(f"Initializing LLM: {args.llm_model} (via {provider_display})...")
            
            # Add OpenRouter-specific parameters if applicable
            llm_kwargs = {}
            if LLM_PROVIDER.lower() in ["openrouter", "open-router"]:
                llm_kwargs.update({
                    "site_url": OPENROUTER_SITE_URL,
                    "site_name": OPENROUTER_SITE_NAME
                })
            
            llm = create_llm(
                provider=LLM_PROVIDER,
                model_name=args.llm_model,
                temperature=args.llm_temperature,
                max_tokens=LLM_MAX_TOKENS,
                api_key=api_key,
                **llm_kwargs
            )
            print(f"[OK] LLM initialized: {llm.model_name}\n")
        except Exception as e:
            logger.warning(f"Failed to initialize LLM: {e}")
            print(f"[WARNING] Could not initialize LLM - {e}\n")
        
        # Initialize query handler
        query_handler = QueryHandler(
            retriever=retriever,
            embedding_service=embedding_service,
            llm=llm,
            top_k=args.top_k
        )
        
        print("\n" + "="*80)
        print("Commands:")
        print("  - Type your question and press Enter")
        print("  - 'history' - Show query history")
        print("  - 'clear' - Clear history")
        print("  - 'exit' or 'quit' - Exit")
        print("="*80 + "\n")
        
        # Interactive loop
        while True:
            try:
                user_input = input("Query> ").strip()
                
                if user_input.lower() in ['exit', 'quit']:
                    print("\nGoodbye!")
                    break
                    
                elif user_input.lower() == 'history':
                    history = query_handler.get_query_history()
                    if not history:
                        print("\nNo query history.")
                    else:
                        print("\n" + "="*80)
                        print("QUERY HISTORY")
                        print("="*80)
                        for i, entry in enumerate(history, 1):
                            print(f"\n{i}. {entry['query'][:60]}")
                            print(f"   Results: {entry['num_results']} | Avg Score: {entry['avg_score']:.2%}")
                    continue
                    
                elif user_input.lower() == 'clear':
                    query_handler.clear_history()
                    print("\nHistory cleared.")
                    continue
                    
                elif not user_input:
                    continue
                
                # Process query with LLM response if available
                if llm:
                    result = query_handler.process_query_with_response(
                        user_input,
                        top_k=args.top_k
                    )
                    
                    print("\n" + "="*80)
                    print("AI RESPONSE")
                    print("="*80)
                    print(result.llm_response)
                    print("\n" + "="*80)
                    print("SOURCES")
                    print("="*80)
                    
                    for i, (chunk, meta, score) in enumerate(
                        zip(result.retrieved_chunks, result.metadata, result.retrieval_scores), 1
                    ):
                        print(f"\n[{i}] Relevance: {score:.2%}")
                        print(f"Document: {meta['document_id']} | Chunk: {meta['chunk_id']}")
                        display_text = chunk[:200] + "..." if len(chunk) > 200 else chunk
                        print(display_text)
                else:
                    # Retrieval-only mode
                    result = query_handler.process_query(user_input, top_k=args.top_k)
                    
                    print("\n" + "-"*80)
                    print(f"Query: {result.query}")
                    print(f"Retrieved: {len(result.retrieved_chunks)} chunks")
                    print("-"*80)
                    
                    for i, (chunk, meta, score) in enumerate(
                        zip(result.retrieved_chunks, result.metadata, result.retrieval_scores), 1
                    ):
                        print(f"\n[{i}] Relevance: {score:.2%}")
                        print(f"Document: {meta['document_id']} | Chunk: {meta['chunk_id']}")
                        print("-"*40)
                        display_text = chunk[:300] + "..." if len(chunk) > 300 else chunk
                        print(display_text)
                
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                logger.error(f"Query error: {e}", exc_info=True)
                print(f"\nError: {e}")
                
    except Exception as e:
        logger.error(f"Failed to initialize query mode: {e}", exc_info=True)
        print(f"\nError: {e}")
        print("Tip: Run 'python main.py --build' first to create the vector store")
        sys.exit(1)



def ingest_single_file(pipeline: 'RAGPipeline', file_path: str) -> bool:
    """Ingest a single file into the RAG system using the existing pipeline components"""
    try:
        logger.info(f"Ingesting file: {file_path}")
        logger.info(f"Pipeline type: {type(pipeline)}")
        
        # 1. Load Document
        # Handle both pipeline types
        if hasattr(pipeline, 'document_loader'):
            doc = pipeline.document_loader._load_document(file_path, store_content=True)
        else:
            # Parallel pipeline might not expose document_loader publicly strictly same way?
            # Actually ParallelRAGPipeline creates loader locally in workers.
            # But let's assume we are using standard RAGPipeline for now as per startup.
            # If parallel, we might need a different approach or just instantiate a loader.
            from src.modules.Loader import DocumentLoader
            loader = DocumentLoader(documents_dir=os.path.dirname(file_path), use_db=False)
            doc, _ = loader._load_document_with_unstructured(file_path)
            # Create DocumentMetadata if needed
            # For now, let's Stick to RAGPipeline logic which is what is used.
            logger.error(f"Pipeline {type(pipeline)} does not have document_loader")
            return False

        if not doc:
            logger.error(f"Failed to load document: {file_path}")
            return False
            
        # 2. Chunk Document
        if hasattr(pipeline, 'step_2_chunk_documents_batch'):
            chunks = pipeline.step_2_chunk_documents_batch([doc])
        elif hasattr(pipeline, 'chunker_config'):
             # fallback for parallel pipeline if it doesn't expose step_2
             # But parallel pipeline is for batch processing.
             # For single file upload, we should use the components directly if possible.
             # Let's assume RAGPipeline for API mode.
             logger.warning("Using fallback chunking for non-standard pipeline")
             from src.modules.Chunking import TextChunker, ChunkingStrategy
             chunker = TextChunker(chunk_size=512)
             chunks = chunker.chunk_document(doc)
        else:
             logger.error("Pipeline missing chunking capability")
             return False

        if not chunks:
            logger.warning(f"No chunks generated for {file_path}")
            return False
            
        # 3. Embed & Index
        if hasattr(pipeline, 'step_3_generate_embeddings_batch'):
            pipeline.step_3_generate_embeddings_batch(chunks, save_to_store=True)
        else:
            logger.error("Pipeline missing embedding capability")
            return False
        
        # 4. Save Vector Store to Disk
        if hasattr(pipeline, 'vector_store'):
            pipeline.vector_store.save(pipeline.vector_store_dir)
        
        logger.info(f"Successfully ingested {file_path}")
        return True
    except Exception as e:
        logger.error(f"Error ingesting file {file_path}: {e}", exc_info=True)
        return False


def run_api_server(args):
    """Run API server mode"""
    print("\n" + "="*80)
    print("RAG API SERVER MODE")
    print("="*80)
    print(f"\nFrontend available at http://localhost:{args.api_port}")
    print(f"API docs at http://localhost:{args.api_port}/docs")
    print("Press Ctrl+C to stop\n")
    
    try:
        # Create FastAPI app
        app = FastAPI(
            title="RAG Query API",
            description="REST API for RAG queries and file uploads",
            version="1.0.0"
        )

        # CORS Middleware
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Allow all origins for dev
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # Mount frontend static files
        frontend_path = os.path.join(os.path.dirname(__file__), "frontend/static")
        if os.path.exists(frontend_path):
            app.mount("/static", StaticFiles(directory=frontend_path), name="static")
            logger.info(f"Serving static files from {frontend_path}")
        else:
            logger.warning(f"Frontend static directory not found at {frontend_path}")

        # Global RAG Pipeline variables are defined at module level

        
        # Pydantic models
        class QueryRequest(BaseModel):
            query: str = Field(..., min_length=1)  # Allow shorter queries
            top_k: Optional[int] = Field(5, ge=1, le=100)
        
        # Chat-like response model
        class ChatResponse(BaseModel):
            answer: str
            sources: Optional[List[Dict]] = None

        @app.on_event("startup")
        async def startup():
            global pipeline_global, query_handler_global
            logger.info("Initializing RAG Pipeline...")
            
            # Initialize the full pipeline (Loader, Chunker, Embedder, Store)
            # We reuse the arguments passed to main.py
            pipeline_global = RAGPipeline(
                documents_dir=args.documents_dir,
                chunks_dir=args.chunks_dir,
                vector_store_dir=args.vector_store_dir,
                chunk_size=args.chunk_size,
                chunk_overlap=args.chunk_overlap,
                strategy=args.strategy,
                encoding_name=args.encoding,
                embedding_model=EMBEDDING_MODEL,
                vector_store_type=args.vector_store_type,
                device=args.device,
                use_db=True, # Assuming DB is available or fallback handles it
                batch_size=BATCH_SIZE_DOCS
            )

            # Load existing vector store index if available
            try:
                pipeline_global.vector_store.load(args.vector_store_dir)
                logger.info("Loaded existing vector store.")
            except Exception as e:
                logger.warning(f"Could not load vector store (might be new): {e}")

            # Load existing chunks metadata for the Retriever
            # (RAGPipeline initializes retriever with empty chunks dict)
            try:
                loaded_chunks = load_chunks_metadata(args.chunks_dir)
                pipeline_global.retriever.chunks = loaded_chunks
                logger.info(f"Loaded {len(loaded_chunks)} existing chunks metadata.")
            except Exception as e:
                logger.warning(f"Could not load chunks metadata: {e}")

            # Initialize QueryHandler
            # We assume LLM init logic inside RAGPipeline or Create one here
            # RAGPipeline doesn't seem to hold an LLM instance in self?
            # Let's create LLM here as before.
            llm = None
            try:
                if LLM_PROVIDER.lower() in ["openrouter", "open-router"]:
                    api_key = OPENROUTER_API_KEY
                elif LLM_PROVIDER.lower() in ["gemini", "google", "google-ai"]:
                    api_key = GEMINI_API_KEY
                else:
                    api_key = HF_TOKEN
                
                llm_kwargs = {}
                if LLM_PROVIDER.lower() in ["openrouter", "open-router"]:
                    llm_kwargs.update({
                        "site_url": OPENROUTER_SITE_URL,
                        "site_name": OPENROUTER_SITE_NAME
                    })
                
                llm = create_llm(
                    provider=LLM_PROVIDER,
                    model_name=args.llm_model,
                    temperature=args.llm_temperature,
                    max_tokens=LLM_MAX_TOKENS,
                    api_key=api_key,
                    **llm_kwargs
                )
                logger.info(f"LLM initialized: {llm.model_name}")
            except Exception as e:
                logger.warning(f"Failed to initialize LLM: {e}")

            query_handler_global = QueryHandler(
                pipeline_global.retriever, 
                pipeline_global.embedding_service, 
                llm, 
                top_k=args.top_k
            )

            logger.info("RAG API Ready.")
        
        @app.get("/health")
        async def health():
            ready = query_handler_global is not None and pipeline_global is not None
            return {
                "status": "ok", 
                "ready": ready,
                "vector_store_size": pipeline_global.vector_store.get_size() if pipeline_global else 0
            }

        @app.post("/api/upload")
        async def upload_document(file: UploadFile = File(...), background_tasks: BackgroundTasks = None):
            if not pipeline_global:
                raise HTTPException(503, "System not ready")
            
            # Ensure upload directory exists
            upload_dir = args.documents_dir
            os.makedirs(upload_dir, exist_ok=True)
            
            file_location = os.path.join(upload_dir, file.filename)
            try:
                with open(file_location, "wb+") as file_object:
                    shutil.copyfileobj(file.file, file_object)
            except Exception as e:
                raise HTTPException(500, f"Failed to save file: {e}")
            
            # Start ingestion (Blocking for MVP to ensure 'Processing...' finishes with result)
            # In a real system, use BackgroundTasks. For this UI, user expects "Ready" logic.
            # We'll run it synchronously here so the UI 'success' signal implies it's ready to query.
            success = ingest_single_file(pipeline_global, file_location)
            
            if not success:
                 # Clean up if failed?
                 # os.remove(file_location) 
                 raise HTTPException(500, "Failed to ingest document")
            
            # Update retriever chunks metadata so new queries find it immediately
            # (Ingestion updates DB/Disk, but Retriever has in-memory chunks dict)
            # We should reload or incrementally update.
            # Lazy approach: Reload all metadata (might be slow if millions, but fine for now)
            try:
                pipeline_global.retriever.chunks = load_chunks_metadata(args.chunks_dir)
            except:
                pass

            return {
                "filename": file.filename, 
                "status": "success", 
                "message": "File uploaded and processed"
            }

        @app.post("/api/ask")
        async def ask_question(request: QueryRequest):
            if not query_handler_global:
                raise HTTPException(503, "System not ready")
            try:
                # query_handler_global has .llm if matched
                response = query_handler_global.process_query_with_response(request.query, top_k=request.top_k)
                
                # Format sources
                sources = []
                if response.retrieved_chunks:
                    for i, (chunk, meta, score) in enumerate(zip(response.retrieved_chunks, response.metadata or [{}], response.retrieval_scores or [])):
                        sources.append({
                            "text": chunk,
                            "score": float(score),
                            "metadata": meta
                        })

                return {
                    "answer": response.llm_response or "I found some information but couldn't generate an answer.",
                    "sources": sources
                }
            except Exception as e:
                logger.error(f"Error processing query: {e}", exc_info=True)
                raise HTTPException(500, str(e))

        # Frontend Route
        @app.get("/")
        async def read_index():
            index_path = os.path.join(frontend_path, "index.html")
            if os.path.exists(index_path):
                return FileResponse(index_path)
            return {"error": "Frontend not found"}

        # Start Server
        uvicorn.run(app, host="0.0.0.0", port=args.api_port, log_level="info")

    except ImportError as e:
        print(f"Import Error: {e}")
        print("Please run: pip install fastapi uvicorn python-multipart")
        sys.exit(1)
    except Exception as e:
        logger.error(f"API Server crashed: {e}", exc_info=True)
        sys.exit(1)


def run_test_query(args):
    """Run a test query"""
    print("\n" + "="*80)
    print("RAG TEST QUERY MODE")
    print("="*80)
    
    try:
        # Initialize components
        print("Initializing...")
        embedding_service = create_embedding_service(
            model_type=EMBEDDING_PROVIDER,
            **_build_embedding_kwargs(args.device)
        )
        vector_store = FAISSVectorStore(
            dimension=embedding_service.model.dimension,
            index_type="flat"
        )
        vector_store.load(args.vector_store_dir)
        chunks = load_chunks_metadata(args.chunks_dir)
        
        # Initialize reranker if enabled
        reranker = _create_reranker()
        
        retriever = RAGRetriever(vector_store, embedding_service, chunks, reranker=reranker, use_reranker=USE_RERANKER)
        
        # Initialize LLM (OpenRouter, Google Gemini, or HuggingFace)
        llm = None
        try:
            if LLM_PROVIDER.lower() in ["openrouter", "open-router"]:
                api_key = OPENROUTER_API_KEY
            elif LLM_PROVIDER.lower() in ["gemini", "google", "google-ai"]:
                api_key = GEMINI_API_KEY
            else:
                api_key = HF_TOKEN
            
            # Add OpenRouter-specific parameters if applicable
            llm_kwargs = {}
            if LLM_PROVIDER.lower() in ["openrouter", "open-router"]:
                llm_kwargs.update({
                    "site_url": OPENROUTER_SITE_URL,
                    "site_name": OPENROUTER_SITE_NAME
                })
            
            llm = create_llm(
                provider=LLM_PROVIDER,
                model_name=args.llm_model,
                temperature=args.llm_temperature,
                max_tokens=LLM_MAX_TOKENS,
                api_key=api_key,
                **llm_kwargs
            )
            print(f"[OK] LLM: {llm.model_name}\n")
        except Exception as e:
            print(f"[WARNING] LLM init failed: {e}\n")
        
        query_handler = QueryHandler(retriever, embedding_service, llm, top_k=args.top_k)
        
        # Process query
        print(f"\nQuery: {args.test_query}\n")
        
        if llm:
            result = query_handler.process_query_with_response(args.test_query, top_k=args.top_k)
            print("="*80)
            print("AI RESPONSE")
            print("="*80)
            print(result.llm_response)
            print("\n" + "="*80)
            print("SOURCES")
            print("="*80)
            for i, (chunk, score) in enumerate(zip(result.retrieved_chunks, result.retrieval_scores), 1):
                print(f"[{i}] Score: {score:.2%}")
                print(f"    {chunk[:150]}...\n")
        else:
            result = query_handler.process_query(args.test_query, top_k=args.top_k)
            print(f"Retrieved {len(result.retrieved_chunks)} chunks:\n")
            for i, (chunk, score) in enumerate(zip(result.retrieved_chunks, result.retrieval_scores), 1):
                print(f"[{i}] Score: {score:.2%}")
                print(f"    {chunk[:150]}...\n")
        
    except Exception as e:
        logger.error(f"Test query failed: {e}", exc_info=True)
        print(f"Error: {e}")
        sys.exit(1)


def count_documents(documents_dir: str) -> int:
    """Count total number of documents in directory
    
    Args:
        documents_dir: Path to documents directory
        
    Returns:
        Number of document files found
    """
    doc_path = Path(documents_dir)
    if not doc_path.exists():
        return 0
    
    # Count common document file types
    extensions = ['*.txt', '*.pdf', '*.docx', '*.csv', '*.json', '*.html', '*.md']
    count = 0
    for ext in extensions:
        count += len(list(doc_path.rglob(ext)))
    
    return count


def main():
    """Main entry point - unified RAG system"""
    parser = argparse.ArgumentParser(
        description="Complete RAG System - Build & Query",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
  --build                Build vector store (auto-detects parallel vs sequential based on file count)
  --build --parallel     Force parallel pipeline (recommended for 50+ documents)
  --build --sequential   Force sequential pipeline (better for <50 documents)
  --build --incremental  Incremental build (keep existing data, add new documents only)
  --query                Interactive query interface
  --api                  Start REST API server
  --test "query"         Test a single query

Examples:
  python main.py --build                         # Auto-detect best pipeline (clears existing data)
  python main.py --build --parallel              # Force parallel mode (clears existing data)
  python main.py --build --incremental           # Add new documents without clearing
  python main.py --query
  python main.py --query --top-k 10 --device cuda
  python main.py --api --api-port 8000
  python main.py --test "What is machine learning?"
  
Performance:
  Sequential: ~8 docs/sec, 40% CPU, 40% GPU (best for <50 docs)
  Parallel:   ~60+ docs/sec, 95% CPU, 98% GPU (best for 50+ docs, 8x speedup!)
  Auto-detect: Automatically chooses based on document count (threshold: 50)
  
Note: By default, --build clears existing databases for a fresh rebuild.
      Use --incremental to keep existing data and only add new documents.
        """
    )
    
    # Mode selection
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--build", action="store_true", help="Build vector store")
    mode_group.add_argument("--query", action="store_true", help="Interactive query mode")
    mode_group.add_argument("--api", action="store_true", help="Start API server")
    mode_group.add_argument("--test-query", type=str, metavar="QUERY", help="Test a single query")
    
    # Common arguments
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Device (default: cpu)")
    parser.add_argument("--top-k", type=int, default=5, help="Results per query (default: 5)")
    
    # LLM arguments (Hugging Face Inference)
    parser.add_argument("--llm-model", type=str, default=LLM_MODEL,
                       help="Hugging Face model ID (default from LLM_MODEL)")
    parser.add_argument("--llm-temperature", type=float, default=LLM_TEMPERATURE,
                       help="LLM temperature (default: 0.7)")
    
    # Build mode arguments
    parser.add_argument("--incremental", action="store_true", help="Incremental build (keep existing databases, add new documents only)")
    pipeline_group = parser.add_mutually_exclusive_group()
    pipeline_group.add_argument("--parallel", action="store_true", help="Force parallel pipeline (recommended for 50+ documents)")
    pipeline_group.add_argument("--sequential", action="store_true", help="Force sequential pipeline (better for <50 documents)")
    parser.add_argument("--documents-dir", type=str, default="data/documents", help="Documents directory")
    parser.add_argument("--chunks-dir", type=str, default="data/chunks", help="Chunks directory")
    parser.add_argument("--vector-store-dir", type=str, default="data/vector_store", help="Vector store directory")
    parser.add_argument("--chunk-size", type=int, default=512, help="Chunk size (default: 512)")
    parser.add_argument("--chunk-overlap", type=int, default=50, help="Chunk overlap (default: 50)")
    parser.add_argument("--strategy", type=str, default="fixed_size", 
                       choices=["fixed_size", "semantic", "sliding_window", "sentence", "paragraph"],
                       help="Chunking strategy")
    parser.add_argument("--encoding", type=str, default="cl100k_base", help="Tokenizer encoding")
    parser.add_argument("--vector-store-type", type=str, default="flat", 
                       choices=["flat", "ivf", "hnsw"], help="Vector store type")
    parser.add_argument("--loader-threads", type=int, default=8, help="Number of loader threads for parallel mode (default: 8)")
    parser.add_argument("--chunker-processes", type=int, default=None, help="Number of chunker processes for parallel mode (default: cpu_count)")
    
    # API mode arguments
    parser.add_argument("--api-port", type=int, default=8000, help="API port (default: 8000)")
    
    args = parser.parse_args()
    
    # Determine mode
    if args.build:
        # Clear databases by default (unless --incremental is specified)
        if not args.incremental:
            logger.info("Clearing existing databases (use --incremental to skip)...")
            try:
                # Clear PostgreSQL tables with proper lock handling
                if not clear_postgres_tables():
                    logger.warning("  [WARNING] PostgreSQL clearing failed - continuing with build")
                    logger.info("  [INFO] You may need to manually clear tables: python clear_db.py")
                
                # Clear vector store files
                vector_store_path = Path(args.vector_store_dir)
                if vector_store_path.exists():
                    for file in vector_store_path.glob("*"):
                        if file.is_file():
                            file.unlink()
                    logger.info("  [OK] Cleared vector_store")
                
                # Clear log files (skip files in use by current logger)
                logs_path = Path("logs")
                if logs_path.exists():
                    log_count = 0
                    skipped_count = 0
                    for log_file in logs_path.glob("*.log"):
                        if log_file.is_file():
                            try:
                                log_file.unlink()
                                log_count += 1
                            except (PermissionError, OSError):
                                # Skip files in use (current log file)
                                skipped_count += 1
                    if log_count > 0:
                        logger.info(f"  [OK] Cleared {log_count} log file(s)")
                    if skipped_count > 0:
                        logger.info(f"  [INFO] Skipped {skipped_count} log file(s) in use")
            except Exception as e:
                logger.error(f"Error clearing databases: {e}")
        else:
            logger.info("Incremental build: Keeping existing databases")
        
        # Build mode - run pipeline
        # Auto-detect pipeline mode if not explicitly set
        use_parallel = args.parallel
        
        if not args.parallel and not args.sequential:
            # Auto-detect based on document count
            doc_count = count_documents(args.documents_dir)
            threshold = 10  # Switch to parallel for 20+ documents
            use_parallel = doc_count >= threshold
            
            logger.info(f"Auto-detecting pipeline mode: {doc_count} documents found")
            logger.info(f"Threshold: {threshold} documents")
            if use_parallel:
                logger.info(f"[OK] Using PARALLEL pipeline (document count >= {threshold})")
            else:
                logger.info(f"[OK] Using SEQUENTIAL pipeline (document count < {threshold})")
        elif args.sequential:
            use_parallel = False
            logger.info("Using SEQUENTIAL pipeline (forced by --sequential flag)")
        else:
            logger.info("Using PARALLEL pipeline (forced by --parallel flag)")
        
        if use_parallel:
            # Use high-performance parallel pipeline
            logger.info("PARALLEL PIPELINE mode (optimized CPU/GPU distribution)")
            
            # Detect GPU configuration
            gpu_config = GPUConfig.detect()
            
            pipeline = ParallelRAGPipeline(
                documents_dir=args.documents_dir,
                vector_store_dir=args.vector_store_dir,
                chunk_size=args.chunk_size,
                chunk_overlap=args.chunk_overlap,
                strategy=args.strategy,
                encoding_name=args.encoding,
                embedding_model=EMBEDDING_MODEL,
                embedding_provider=EMBEDDING_PROVIDER,
                num_loader_threads=args.loader_threads,
                num_chunker_processes=args.chunker_processes,
                gpu_config=gpu_config,
                model_name=EMBEDDING_MODEL,
                normalize_embeddings=EMBEDDING_NORMALIZE
            )
            success = pipeline.build()
        else:
            # Use traditional sequential pipeline
            logger.info("Using SEQUENTIAL PIPELINE mode")
            pipeline = RAGPipeline(
                documents_dir=args.documents_dir,
                chunks_dir=args.chunks_dir,
                vector_store_dir=args.vector_store_dir,
                chunk_size=args.chunk_size,
                chunk_overlap=args.chunk_overlap,
                strategy=args.strategy,
                encoding_name=args.encoding,
                vector_store_type=args.vector_store_type,
                device=args.device
            )
            success = pipeline.run_complete_pipeline()
        
        sys.exit(0 if success else 1)
        
    elif args.query:
        # Query mode - interactive
        run_interactive_query(args)
        
    elif args.api:
        # API mode - start server
        run_api_server(args)
        
    elif args.test_query:
        # Test mode - single query
        run_test_query(args)
        
    else:
        # Default: show help and run build
        print("No mode specified. Running BUILD mode by default.")
        print("Use --query for interactive queries, --api for API server")
        print("="*80 + "\n")
        
        # Clear databases by default (unless --incremental is specified)
        if not args.incremental:
            logger.info("Clearing existing databases (use --incremental to skip)...")
            try:
                # Clear PostgreSQL tables with proper lock handling
                if not clear_postgres_tables():
                    logger.warning("  [WARNING] PostgreSQL clearing failed - continuing with build")
                    logger.info("  [INFO] You may need to manually clear tables: python clear_db.py")
                
                # Clear vector store files
                vector_store_path = Path(args.vector_store_dir)
                if vector_store_path.exists():
                    for file in vector_store_path.glob("*"):
                        if file.is_file():
                            file.unlink()
                    logger.info("  [OK] Cleared vector_store")
                
                # Clear log files (skip files in use by current logger)
                logs_path = Path("logs")
                if logs_path.exists():
                    log_count = 0
                    skipped_count = 0
                    for log_file in logs_path.glob("*.log"):
                        if log_file.is_file():
                            try:
                                log_file.unlink()
                                log_count += 1
                            except (PermissionError, OSError):
                                # Skip files in use (current log file)
                                skipped_count += 1
                    if log_count > 0:
                        logger.info(f"  [OK] Cleared {log_count} log file(s)")
                    if skipped_count > 0:
                        logger.info(f"  [INFO] Skipped {skipped_count} log file(s) in use")
            except Exception as e:
                logger.error(f"Error clearing databases: {e}")
        
        pipeline = RAGPipeline(
            documents_dir=args.documents_dir,
            chunks_dir=args.chunks_dir,
            vector_store_dir=args.vector_store_dir,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            strategy=args.strategy,
            encoding_name=args.encoding,
            vector_store_type=args.vector_store_type,
            device=args.device
        )
        success = pipeline.run_complete_pipeline()
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()