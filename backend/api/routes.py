"""FastAPI routes for session-aware RAG backend."""

import threading
from typing import Optional, List
from pathlib import Path
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Depends, BackgroundTasks
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session as DBSession
from backend.database import get_db
from backend.services.session_service import get_session_manager, DATA_DIR
from backend.services.rag_service_session import ask_rag_session, get_llm_status, clear_session_handler
from backend.rag.IngestSession import ingest_documents_session
from src.modules.QueryGeneration import QueryResult
from src.utils.Logger import get_logger
import asyncio

logger = get_logger(__name__)
router = APIRouter()

# Concurrency control: max 5 sessions processing at once (backpressure)
_processing_semaphore = threading.Semaphore(5)

# Supported file extensions for upload validation
SUPPORTED_EXTENSIONS = {
    ".pdf", ".csv", ".tsv", ".xlsx", ".xls",
    ".json", ".yaml", ".yml", ".md", ".html", ".htm",
    ".py", ".js", ".java", ".cpp", ".c", ".cs", ".go",
    ".rs", ".ts", ".jsx", ".tsx", ".txt", ".docx",
    ".ipynb",
}


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
    """Background task to process uploaded document.
    
    Uses a semaphore for backpressure (max 5 concurrent processing sessions).
    Retries the full ingestion up to 2 times on transient failures.
    """
    from backend.database.models import get_session_local
    
    acquired = _processing_semaphore.acquire(timeout=300)  # Wait up to 5 min
    if not acquired:
        # Could not acquire — too many concurrent sessions
        db = get_session_local()()
        try:
            session_manager.update_session_status(
                session_id=session_id,
                status="error",
                db=db,
                error_message="Server busy — too many documents being processed. Please try again later."
            )
        finally:
            db.close()
        return
    
    db = get_session_local()()
    try:
        documents_dir = session_manager.get_documents_dir(session_id)
        chunks_dir = session_manager.get_chunks_dir(session_id)
        vector_store_dir = session_manager.get_vector_store_dir(session_id)
        
        # Retry ingestion up to 2 times on transient errors
        max_attempts = 2
        last_error = None
        
        for attempt in range(max_attempts):
            try:
                chunks_count = ingest_documents_session(
                    session_id=session_id,
                    documents_dir=documents_dir,
                    chunks_dir=chunks_dir,
                    vector_store_dir=vector_store_dir
                )
                
                # Success — update status
                session_manager.update_session_status(
                    session_id=session_id,
                    status="ready",
                    db=db,
                    chunks_count=chunks_count
                )
                logger.info(f"[{session_id[:8]}] Processing complete: {chunks_count} chunks")
                return
                
            except Exception as e:
                last_error = e
                if attempt < max_attempts - 1:
                    logger.warning(
                        f"[{session_id[:8]}] Ingestion attempt {attempt + 1} failed, retrying: {e}"
                    )
                    import time
                    time.sleep(2)  # Brief pause before retry
        
        # All attempts failed
        logger.error(f"[{session_id[:8]}] Processing failed after {max_attempts} attempts: {last_error}")
        session_manager.update_session_status(
            session_id=session_id,
            status="error",
            db=db,
            error_message=str(last_error)
        )
        
    except Exception as e:
        logger.error(f"[{session_id[:8]}] Unexpected error: {e}")
        session_manager.update_session_status(
            session_id=session_id,
            status="error",
            db=db,
            error_message=str(e)
        )
    finally:
        db.close()
        _processing_semaphore.release()


# API Routes
@router.post("/upload", response_model=UploadResponse)
async def upload(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    session_id: Optional[str] = Form(None),
    db: DBSession = Depends(get_db)
):
    """
    Upload a document to a session.
    If session_id is provided, adds file to existing session.
    Otherwise, creates a new isolated session.
    Returns session_id for subsequent requests.
    """
    try:
        session_manager = get_session_manager()
        
        # Check available disk space before accepting upload
        import shutil
        stat = shutil.disk_usage(str(DATA_DIR))
        available_gb = stat.free / (1024**3)
        
        if available_gb < 4.0:  # Less than 4GB available
            raise HTTPException(
                status_code=507, 
                detail=f"Insufficient storage space. Only {available_gb:.2f}GB available. Cannot accept upload."
            )
        
        # Validate file extension before reading
        file_ext = Path(file.filename).suffix.lower() if file.filename else ""
        if file_ext not in SUPPORTED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type '{file_ext}'. "
                       f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
            )
        
        # Read file
        contents = await file.read()
        file_size = len(contents)
        
        # Validate file size (20MB limit)
        if file_size > 20 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="File too large (max 20MB)")
        
        # Use existing session or create new one
        if session_id:
            # Verify session exists
            session = session_manager.get_session(session_id, db)
            if not session:
                raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
            # Reset to processing status when adding new files
            session_manager.update_session_status(session_id, "processing", db)
        else:
            # Create new session
            session_id = session_manager.create_session(file.filename, file_size, db)
        
        # Save file to session-isolated directory (prevents cross-user mixup)
        documents_dir = session_manager.get_documents_dir(session_id)
        dest_path = documents_dir / file.filename
        
        # Check if file already exists
        if dest_path.exists():
            raise HTTPException(status_code=400, detail=f"File {file.filename} already exists in this session")
        
        with open(dest_path, "wb") as f:
            f.write(contents)
        
        # NOTE: We do NOT copy to central data/documents/ to ensure strict
        # session isolation. Each user's files stay in their own session directory.
        # The main CLI pipeline (python main.py build) uses data/documents/ separately.
        
        # Don't process yet - user will click "Process" after uploading all files.
        # Set status to "uploaded" so the frontend knows files are ready.
        session_manager.update_session_status(session_id, "uploaded", db)
        
        return UploadResponse(
            session_id=session_id,
            status="uploaded",
            filename=file.filename,
            message="Document uploaded successfully. Click Process when ready."
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


@router.post("/process/{session_id}")
async def process_session(
    session_id: str,
    background_tasks: BackgroundTasks,
    db: DBSession = Depends(get_db)
):
    """
    Trigger processing of all uploaded documents in a session.
    Called when user clicks 'Process' after uploading all their files.
    """
    session_manager = get_session_manager()
    session = session_manager.get_session(session_id, db)
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    if session.status == "processing":
        return {"session_id": session_id, "status": "processing", "message": "Already processing"}
    
    if session.status == "ready":
        return {"session_id": session_id, "status": "ready", "message": "Already processed"}
    
    # Set status to processing
    session_manager.update_session_status(session_id, "processing", db)
    
    # Kick off background ingestion of ALL files in session
    background_tasks.add_task(
        process_document_async,
        session_id=session_id,
        session_manager=session_manager
    )
    
    return {"session_id": session_id, "status": "processing", "message": "Processing started for all uploaded files"}


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
def health(db: DBSession = Depends(get_db)):
    """Deep health check endpoint validating system components."""
    status_report = {
        "status": "ok",
        "llm_loaded": get_llm_status(),
        "database": "unknown",
        "system_stats": {}
    }

    # 1. Database Check
    try:
        from sqlalchemy import text
        db.execute(text("SELECT 1"))
        status_report["database"] = "connected"
    except Exception as e:
        logger.error(f"Healthcheck failed Database verification: {e}")
        status_report["database"] = "disconnected"
        status_report["status"] = "degraded"

    # 2. Vector Store File Check
    try:
        from config.config import DATA_DIR
        vstore_dir = Path(DATA_DIR) / "vector_store"
        if vstore_dir.exists():
            faiss_files = list(vstore_dir.glob("**/index.faiss"))
            status_report["system_stats"]["total_faiss_indexes"] = len(faiss_files)
        else:
            status_report["system_stats"]["total_faiss_indexes"] = 0
    except Exception as e:
        logger.error(f"Healthcheck failed FAISS verification: {e}")
        status_report["status"] = "degraded"

    return status_report
