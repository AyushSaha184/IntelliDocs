"""
RAG Query API - FastAPI interface for the RAG system

Provides REST endpoints for:
- Query submission
- Retrieval and context formatting
- Query history management
"""

import sys
import os
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from datetime import datetime
import json
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.modules.Embeddings import create_embedding_service
from src.modules.VectorStore import FAISSVectorStore
from src.modules.Retriever import RAGRetriever
from src.modules.QueryGeneration import QueryHandler
from src.modules.LLM import create_llm
from src.utils.Logger import get_logger
from config.config import (
    LLM_PROVIDER,
    LLM_MODEL,
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS,
    HF_TOKEN,
    HF_INFERENCE_PROVIDER,
    GEMINI_API_KEY,
    EMBEDDING_PROVIDER,
    EMBEDDING_MODEL,
    EMBEDDING_NORMALIZE,
    EMBEDDING_TASK_TYPE,
    EMBEDDING_DIMENSION,
)

logger = get_logger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="RAG Query API",
    description="REST API for Retrieval-Augmented Generation queries",
    version="1.0.0"
)

# Global state - will be initialized on startup
query_handler: Optional[QueryHandler] = None


def _build_embedding_kwargs() -> dict:
    kwargs = {
        "model_name": EMBEDDING_MODEL,
        "normalize_embeddings": EMBEDDING_NORMALIZE
    }

    if EMBEDDING_PROVIDER.lower() in ["hf", "hf-inference", "huggingface"]:
        kwargs.update({
            "api_key": HF_TOKEN,
            "provider": HF_INFERENCE_PROVIDER
        })
    elif EMBEDDING_PROVIDER.lower() in ["gemini", "google", "google-gemini"]:
        kwargs.update({
            "api_key": GEMINI_API_KEY,
            "task_type": EMBEDDING_TASK_TYPE,
            "output_dimensionality": EMBEDDING_DIMENSION
        })
    else:
        kwargs["device"] = "cpu"

    return kwargs


# Pydantic models
class QueryRequest(BaseModel):
    """Query request model"""
    query: str = Field(..., min_length=3, max_length=5000, description="Query text")
    top_k: Optional[int] = Field(5, ge=1, le=100, description="Number of results to retrieve")


class AskRequest(BaseModel):
    """Ask request model for retrieval + generation"""
    query: str = Field(..., min_length=3, max_length=5000, description="Query text")
    top_k: Optional[int] = Field(5, ge=1, le=100, description="Number of results to retrieve")
    system_prompt: Optional[str] = Field(None, max_length=4000, description="Optional system prompt")
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0, description="LLM temperature override")
    max_tokens: Optional[int] = Field(None, ge=1, le=4000, description="LLM max tokens override")


class RetrievalMetadata(BaseModel):
    """Metadata for a retrieved chunk"""
    chunk_id: str
    document_id: str
    distance: float
    similarity_score: float


class QueryResponse(BaseModel):
    """Query response model"""
    query: str
    timestamp: str
    retrieved_chunks: List[str]
    metadata: List[RetrievalMetadata]
    retrieval_scores: List[float]
    formatted_context: str


class AskResponse(BaseModel):
    """Ask response model"""
    query: str
    timestamp: str
    retrieved_chunks: List[str]
    metadata: List[RetrievalMetadata]
    retrieval_scores: List[float]
    formatted_context: str
    llm_response: Optional[str] = None
    llm_metadata: Optional[dict] = None


class HistoryEntry(BaseModel):
    """Query history entry"""
    query: str
    timestamp: str
    num_results: int
    avg_score: float


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    ready: bool
    timestamp: str


@app.on_event("startup")
async def startup_event():
    """Initialize RAG components on startup"""
    global query_handler
    
    try:
        logger.info("Initializing RAG Query API...")
        
        # Initialize embedding service first to get dimension
        embedding_service = create_embedding_service(
            model_type=EMBEDDING_PROVIDER,
            **_build_embedding_kwargs()
        )
        logger.info("Embedding service initialized")
        
        # Load vector store with correct dimension
        vector_store = FAISSVectorStore(
            dimension=embedding_service.model.dimension,
            index_type="flat"
        )
        vector_store.load("data/vector_store")
        logger.info("Vector store loaded")
        
        # Load chunks metadata
        chunks_metadata = _load_chunks_metadata("data/chunks")
        logger.info(f"Loaded {len(chunks_metadata)} chunks")
        
        # Initialize retriever
        retriever = RAGRetriever(
            vector_store=vector_store,
            embedding_service=embedding_service,
            chunks=chunks_metadata
        )
        
        # Initialize LLM (Google Gemini or HuggingFace)
        llm = None
        try:
            # Set the correct API key based on provider
            api_key = GEMINI_API_KEY if LLM_PROVIDER.lower() in ["gemini", "google", "google-ai"] else HF_TOKEN
            
            llm = create_llm(
                provider=LLM_PROVIDER,
                model_name=LLM_MODEL,
                temperature=LLM_TEMPERATURE,
                max_tokens=LLM_MAX_TOKENS,
                api_key=api_key
            )
            logger.info(f"LLM initialized: {llm.model_name}")
        except Exception as e:
            logger.warning(f"LLM initialization failed: {e}")

        # Initialize query handler
        query_handler = QueryHandler(
            retriever=retriever,
            embedding_service=embedding_service,
            llm=llm,
            top_k=5
        )
        
        logger.info("RAG Query API initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize RAG Query API: {e}", exc_info=True)
        raise


def _load_chunks_metadata(chunks_dir: str) -> dict:
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


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="operational" if query_handler else "not_ready",
        ready=query_handler is not None,
        timestamp=datetime.now().isoformat()
    )


@app.post("/query", response_model=QueryResponse)
async def submit_query(request: QueryRequest):
    """Submit a query and retrieve relevant context
    
    Args:
        request: QueryRequest containing query text and top_k
        
    Returns:
        QueryResponse with retrieved chunks and metadata
        
    Raises:
        HTTPException: If query processing fails
    """
    if not query_handler:
        raise HTTPException(
            status_code=503,
            detail="RAG system not initialized"
        )
    
    try:
        # Process query
        query_result = query_handler.process_query(
            query=request.query,
            top_k=request.top_k
        )
        
        # Format context for LLM
        formatted_context = query_handler.format_context(
            query_result,
            include_scores=True
        )
        
        # Build response
        response_metadata = [
            RetrievalMetadata(**meta) for meta in query_result.metadata
        ]
        
        return QueryResponse(
            query=query_result.query,
            timestamp=query_result.timestamp,
            retrieved_chunks=query_result.retrieved_chunks,
            metadata=response_metadata,
            retrieval_scores=query_result.retrieval_scores,
            formatted_context=formatted_context
        )
        
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Query processing failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Query processing failed: {str(e)}"
        )


@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest):
    """Ask a question with retrieval + generation

    Flow: frontend -> /ask -> FastAPI -> retriever -> generator
    """
    if not query_handler:
        raise HTTPException(
            status_code=503,
            detail="RAG system not initialized"
        )

    if not query_handler.llm:
        raise HTTPException(
            status_code=503,
            detail="LLM not initialized for generation"
        )

    try:
        result = query_handler.process_query_with_response(
            query=request.query,
            top_k=request.top_k,
            system_prompt=request.system_prompt,
            temperature=request.temperature,
            max_tokens=request.max_tokens or LLM_MAX_TOKENS
        )

        formatted_context = query_handler.format_context(
            result,
            include_scores=True
        )

        response_metadata = [
            RetrievalMetadata(**meta) for meta in result.metadata
        ]

        return AskResponse(
            query=result.query,
            timestamp=result.timestamp,
            retrieved_chunks=result.retrieved_chunks,
            metadata=response_metadata,
            retrieval_scores=result.retrieval_scores,
            formatted_context=formatted_context,
            llm_response=result.llm_response,
            llm_metadata=result.llm_metadata
        )

    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Ask processing failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Ask processing failed: {str(e)}"
        )


@app.get("/query/history", response_model=List[HistoryEntry])
async def get_query_history(limit: Optional[int] = Query(None, ge=1, le=1000)):
    """Get query history
    
    Args:
        limit: Maximum number of historical queries to return
        
    Returns:
        List of HistoryEntry objects
    """
    if not query_handler:
        raise HTTPException(
            status_code=503,
            detail="RAG system not initialized"
        )
    
    history = query_handler.get_query_history(limit=limit)
    return [HistoryEntry(**entry) for entry in history]


@app.post("/query/history/clear")
async def clear_query_history():
    """Clear query history
    
    Returns:
        Success message
    """
    if not query_handler:
        raise HTTPException(
            status_code=503,
            detail="RAG system not initialized"
        )
    
    query_handler.clear_history()
    return {
        "status": "success",
        "message": "Query history cleared",
        "timestamp": datetime.now().isoformat()
    }


@app.get("/query/batch", response_model=List[QueryResponse])
async def batch_query(
    queries: str = Query(..., description="Comma-separated list of queries"),
    top_k: int = Query(5, ge=1, le=100)
):
    """Process multiple queries in batch
    
    Args:
        queries: Comma-separated query strings
        top_k: Number of results per query
        
    Returns:
        List of QueryResponse objects
    """
    if not query_handler:
        raise HTTPException(
            status_code=503,
            detail="RAG system not initialized"
        )
    
    query_list = [q.strip() for q in queries.split(",") if q.strip()]
    
    if not query_list:
        raise HTTPException(
            status_code=400,
            detail="No valid queries provided"
        )
    
    if len(query_list) > 10:
        raise HTTPException(
            status_code=400,
            detail="Maximum 10 queries per batch request"
        )
    
    responses = []
    
    for query_text in query_list:
        try:
            query_result = query_handler.process_query(
                query=query_text,
                top_k=top_k
            )
            
            formatted_context = query_handler.format_context(
                query_result,
                include_scores=True
            )
            
            response_metadata = [
                RetrievalMetadata(**meta) for meta in query_result.metadata
            ]
            
            responses.append(QueryResponse(
                query=query_result.query,
                timestamp=query_result.timestamp,
                retrieved_chunks=query_result.retrieved_chunks,
                metadata=response_metadata,
                retrieval_scores=query_result.retrieval_scores,
                formatted_context=formatted_context
            ))
            
        except Exception as e:
            logger.error(f"Batch query processing failed for '{query_text}': {e}")
            # Skip failed queries in batch
            continue
    
    return responses


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
