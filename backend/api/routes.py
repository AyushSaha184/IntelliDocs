"""FastAPI routes for session-aware RAG backend."""

from typing import Optional, List
from pathlib import Path
from fastapi import APIRouter, HTTPException, UploadFile, File, Depends, BackgroundTasks
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session as DBSession
from backend.database import get_db
from backend.services.session_service import get_session_manager
from backend.services.rag_service_session import ask_rag_session, get_llm_status, clear_session_handler
from backend.rag.IngestSession import ingest_documents_session
from src.modules.QueryGeneration import QueryResult
import asyncio

router = APIRouter()


# Request/Response Models
class Query(BaseModel):
    session_id: str = Field(..., description="Session ID from upload")
    question: str = Field(..., min_length=3, max_length=5000)
    top_k: Optional[int] = Field(5, ge=1, le=100)
    system_prompt: Optional[str] = Field(None, max_length=4000)
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(None, ge=1, le=4000)


class AskResponse(BaseModel):
    answer: str
    query: str
    timestamp: str
    retrieved_chunks: List[str]
    retrieval_scores: List[float]
    metadata: List[dict]
    formatted_context: str
    session_id: str


class UploadResponse(BaseModel):
    session_id: str
    status: str
    filename: str
    message: str


class SessionStatusResponse(BaseModel):
    session_id: str
    status: str
    filename: str
    created_at: str
    last_accessed: str
    chunks_count: int
    error_message: Optional[str] = None


# Background task helper
def process_document_async(session_id: str, session_manager):
    """Background task to process uploaded document."""
    from backend.database.models import SessionLocal
    
    db = SessionLocal()
    try:
        documents_dir = session_manager.get_documents_dir(session_id)
        chunks_dir = session_manager.get_chunks_dir(session_id)
        vector_store_dir = session_manager.get_vector_store_dir(session_id)
        
        # Run ingestion
        chunks_count = ingest_documents_session(
            session_id=session_id,
            documents_dir=documents_dir,
            chunks_dir=chunks_dir,
            vector_store_dir=vector_store_dir
        )
        
        # Update status to ready
        session_manager.update_session_status(
            session_id=session_id,
            status="ready",
            db=db,
            chunks_count=chunks_count
        )
        
    except Exception as e:
        # Update status to error
        session_manager.update_session_status(
            session_id=session_id,
            status="error",
            db=db,
            error_message=str(e)
        )
        raise
    finally:
        db.close()


# API Routes
@router.post("/upload", response_model=UploadResponse)
async def upload(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    db: DBSession = Depends(get_db)
):
    """
    Upload a document and create an isolated session.
    Returns session_id for subsequent requests.
    """
    try:
        session_manager = get_session_manager()
        
        # Check available disk space before accepting upload
        import shutil
        stat = shutil.disk_usage("data")
        available_gb = stat.free / (1024**3)
        
        if available_gb < 4.0:  # Less than 4GB available
            raise HTTPException(
                status_code=507, 
                detail=f"Insufficient storage space. Only {available_gb:.2f}GB available. Cannot accept upload."
            )
        
        # Read file
        contents = await file.read()
        file_size = len(contents)
        
        # Validate file size (20MB limit)
        if file_size > 20 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="File too large (max 20MB)")
        
        # Create session
        session_id = session_manager.create_session(file.filename, file_size, db)
        
        # Save file to session-isolated directory (prevents cross-user mixup)
        documents_dir = session_manager.get_documents_dir(session_id)
        dest_path = documents_dir / file.filename
        with open(dest_path, "wb") as f:
            f.write(contents)
        
        # NOTE: We do NOT copy to central data/documents/ to ensure strict
        # session isolation. Each user's files stay in their own session directory.
        # The main CLI pipeline (python main.py build) uses data/documents/ separately.
        
        # Process in background
        background_tasks.add_task(
            process_document_async,
            session_id=session_id,
            session_manager=session_manager
        )
        
        return UploadResponse(
            session_id=session_id,
            status="processing",
            filename=file.filename,
            message="Document uploaded successfully. Processing in background."
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@router.get("/status/{session_id}", response_model=SessionStatusResponse)
def get_status(session_id: str, db: DBSession = Depends(get_db)):
    """Get the status of a session."""
    session_manager = get_session_manager()
    session = session_manager.get_session(session_id, db)
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return SessionStatusResponse(
        session_id=session.session_id,
        status=session.status,
        filename=session.filename,
        created_at=session.created_at.isoformat(),
        last_accessed=session.last_accessed.isoformat(),
        chunks_count=session.chunks_count,
        error_message=session.error_message
    )


@router.post("/ask", response_model=AskResponse)
def ask(query: Query, db: DBSession = Depends(get_db)):
    """Ask a question about an uploaded document."""
    try:
        session_manager = get_session_manager()
        
        # Validate session
        session = session_manager.get_session(query.session_id, db)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        if session.status != "ready":
            raise HTTPException(
                status_code=400, 
                detail=f"Session not ready. Current status: {session.status}"
            )
        
        # Check LLM
        if not get_llm_status():
            raise HTTPException(status_code=503, detail="LLM not initialized")
        
        # Load session-specific data
        chunks_metadata = session_manager.load_chunks_metadata(query.session_id)
        vector_store_dir = session_manager.get_vector_store_dir(query.session_id)
        
        # Process query
        result: QueryResult = ask_rag_session(
            session_id=query.session_id,
            question=query.question,
            chunks_metadata=chunks_metadata,
            vector_store_dir=vector_store_dir,
            top_k=query.top_k,
            system_prompt=query.system_prompt,
            temperature=query.temperature,
            max_tokens=query.max_tokens
        )
        
        formatted_context = "\n\n".join(result.retrieved_chunks)
        return AskResponse(
            answer=result.llm_response or "",
            query=result.query,
            timestamp=result.timestamp,
            retrieved_chunks=result.retrieved_chunks,
            retrieval_scores=result.retrieval_scores or [],
            metadata=result.metadata,
            formatted_context=formatted_context,
            session_id=query.session_id
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ask failed: {str(e)}")


@router.get("/health")
def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "llm_loaded": get_llm_status()
    }
