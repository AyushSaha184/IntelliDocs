"""FastAPI routes for session-aware RAG backend."""

import threading
from typing import Optional, List
from pathlib import Path
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Depends, BackgroundTasks
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session as DBSession
from backend.database import get_db
from backend.auth.supabase_auth import get_current_user, AuthUser
from backend.services.session_service import get_session_manager, DATA_DIR
from backend.services.rag_service_session import ask_rag_session, get_llm_status, clear_session_handler
from backend.rag.IngestSession import ingest_documents_session
from src.modules.QueryGeneration import QueryResult
from src.utils.Logger import get_logger
from config.config import VECTOR_BACKEND
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
        documents_dir = session_manager.prepare_documents_for_processing(session_id)
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
    db: DBSession = Depends(get_db),
    current_user: AuthUser = Depends(get_current_user),
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
            session = session_manager.get_session(session_id, db, user_id=current_user.id)
            if not session:
                raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
            # Reset to processing status when adding new files
            session_manager.update_session_status(session_id, "processing", db)
        else:
            # Create new session
            session_id = session_manager.create_session(
                file.filename, file_size, db, user_id=current_user.id
            )
        
        # Check if file already exists
        if session_manager.document_exists(session_id, file.filename):
            raise HTTPException(status_code=400, detail=f"File {file.filename} already exists in this session")

        # Save file via configured storage backend (local or Supabase Storage)
        session_manager.save_uploaded_file(session_id, file.filename, contents)
        
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
def get_status(
    session_id: str,
    db: DBSession = Depends(get_db),
    current_user: AuthUser = Depends(get_current_user),
):
    """Get the status of a session."""
    session_manager = get_session_manager()
    session = session_manager.get_session(session_id, db, user_id=current_user.id)
    
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
    db: DBSession = Depends(get_db),
    current_user: AuthUser = Depends(get_current_user),
):
    """
    Trigger processing of all uploaded documents in a session.
    Called when user clicks 'Process' after uploading all their files.
    """
    session_manager = get_session_manager()
    session = session_manager.get_session(session_id, db, user_id=current_user.id)
    
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
def ask(
    query: Query,
    db: DBSession = Depends(get_db),
    current_user: AuthUser = Depends(get_current_user),
):
    """Ask a question about an uploaded document."""
    try:
        session_manager = get_session_manager()
        
        # Validate session
        session = session_manager.get_session(query.session_id, db, user_id=current_user.id)
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


@router.post("/ask/stream")
async def ask_stream(
    query: Query,
    db: DBSession = Depends(get_db),
    current_user: AuthUser = Depends(get_current_user),
):
    """Stream answer tokens via Server-Sent Events (SSE).

    Returns a text/event-stream response. Each SSE event is a JSON object:
      data: {"event": "chunk",    "data": "<token>"}
      data: {"event": "metadata", "data": {sources, retrieval_scores, metadata}}
      data: {"event": "success",  "data": {grounded, confidence}}
      data: {"event": "warning",  "data": {grounded, confidence, message}}
      data: {"event": "error",    "data": "<error message>"}
    """
    from fastapi.responses import StreamingResponse as _SR
    import json as _json
    from backend.services.rag_service_session import ask_rag_session_stream

    session_manager = get_session_manager()
    session = session_manager.get_session(query.session_id, db, user_id=current_user.id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    if session.status != "ready":
        raise HTTPException(status_code=400, detail=f"Session not ready: {session.status}")
    if not get_llm_status():
        raise HTTPException(status_code=503, detail="LLM not initialized")

    chunks_metadata = session_manager.load_chunks_metadata(query.session_id)
    vector_store_dir = session_manager.get_vector_store_dir(query.session_id)

    def _event_generator():
        try:
            for event_dict in ask_rag_session_stream(
                session_id=query.session_id,
                question=query.question,
                chunks_metadata=chunks_metadata,
                vector_store_dir=vector_store_dir,
                top_k=query.top_k,
                system_prompt=query.system_prompt,
                temperature=query.temperature,
                max_tokens=query.max_tokens,
            ):
                yield f"data: {_json.dumps(event_dict)}\n\n"
        except Exception as e:
            logger.error(f"[stream] Unhandled error: {e}", exc_info=True)
            yield f"data: {_json.dumps({'event': 'error', 'data': str(e)})}\n\n"

    return _SR(_event_generator(), media_type="text/event-stream")




@router.get("/health")
def health(db: DBSession = Depends(get_db)):
    """Deep health check endpoint validating system components."""
    status_report = {
        "status": "ok",
        "llm_loaded": get_llm_status(),
        "database": "unknown",
        "vector_backend": VECTOR_BACKEND,
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
        if VECTOR_BACKEND == "qdrant":
            from src.modules.QdrantStore import qdrant_collection_count
            status_report["system_stats"]["total_qdrant_points"] = int(qdrant_collection_count() or 0)
        else:
            from backend.services.session_service import DATA_DIR
            vstore_dir = Path(DATA_DIR) / "vector_store"
            if vstore_dir.exists():
                faiss_files = list(vstore_dir.glob("**/index.faiss"))
                status_report["system_stats"]["total_faiss_indexes"] = len(faiss_files)
            else:
                status_report["system_stats"]["total_faiss_indexes"] = 0
    except Exception as e:
        logger.error(f"Healthcheck failed vector backend verification: {e}")
        status_report["status"] = "degraded"

    return status_report


# ── Phase 4: Human Review Endpoints ────────────────────────────────────

@router.get("/review/pending")
def get_pending_reviews(
    limit: int = 50,
    current_user: AuthUser = Depends(get_current_user),
):
    """Get queries pending human review."""
    from src.agents.HumanValidation import ReviewManager
    manager = ReviewManager()
    return {"reviews": manager.get_pending_reviews(limit=limit)}


@router.post("/review/{review_id}/approve")
def approve_review(
    review_id: str,
    current_user: AuthUser = Depends(get_current_user),
):
    """Approve a pending review."""
    from src.agents.HumanValidation import ReviewManager
    manager = ReviewManager()
    success = manager.approve_review(review_id)
    if not success:
        raise HTTPException(status_code=404, detail="Review not found or already processed")
    return {"status": "approved", "review_id": review_id}


class CorrectionRequest(BaseModel):
    corrected_answer: str


@router.post("/review/{review_id}/correct")
def correct_review(
    review_id: str,
    body: CorrectionRequest,
    current_user: AuthUser = Depends(get_current_user),
):
    """Correct a pending review with a human-provided answer."""
    from src.agents.HumanValidation import ReviewManager
    manager = ReviewManager()
    success = manager.correct_review(review_id, body.corrected_answer)
    if not success:
        raise HTTPException(status_code=404, detail="Review not found or already processed")
    return {"status": "corrected", "review_id": review_id}


# ── Phase 5: Evaluation Endpoints ──────────────────────────────────────

@router.get("/eval/summary")
def get_eval_summary(
    last_n: Optional[int] = None,
    current_user: AuthUser = Depends(get_current_user),
):
    """Get rolling metrics summary."""
    from src.evaluation.Metrics import get_metrics_collector
    collector = get_metrics_collector()
    return collector.get_summary(last_n=last_n)


# ── Phase 6: Stress Test Endpoint ──────────────────────────────────────

@router.post("/stress-test")
def run_stress_test(
    session_id: str,
    current_user: AuthUser = Depends(get_current_user),
):
    """Run adversarial stress tests — gated behind ENABLE_STRESS_TEST env flag."""
    import os
    if not os.getenv("ENABLE_STRESS_TEST", "false").lower() == "true":
        raise HTTPException(
            status_code=403,
            detail="Stress tests disabled. Set ENABLE_STRESS_TEST=true to enable."
        )

    from tests.stress.adversarial_tests import run_stress_tests
    report = run_stress_tests(session_id)
    return report


# ---------------------------------------------------------------------------
# Merged Chat Routes (previously backend/api/chat_routes.py)
# ---------------------------------------------------------------------------
"""API routes for chats, messages, guest sessions, and chat-aware upload/query."""


import json
import uuid
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session as DBSession

from backend.auth.supabase_auth import AuthUser, get_current_user
from backend.database import get_db
from backend.services.chat_service import (
    LimitError,
    add_message,
    check_and_register_document,
    compute_content_hash,
    create_chat,
    get_chat,
    get_chat_doc_count,
    get_chat_documents,
    get_messages,
    get_user_chats,
    rename_chat,
    update_document_status,
    verify_chat_access,
)
from backend.services.cascade_service import cleanup_guest_session, delete_chat_cascade
from src.utils.Logger import get_logger

logger = get_logger(__name__)
chat_router = APIRouter()

SUPPORTED_EXTENSIONS = {
    ".pdf", ".csv", ".tsv", ".xlsx", ".xls",
    ".json", ".yaml", ".yml", ".md", ".html", ".htm",
    ".py", ".js", ".java", ".cpp", ".c", ".cs", ".go",
    ".rs", ".ts", ".jsx", ".tsx", ".txt", ".docx", ".ipynb",
}


def _error_response(status_code: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(status_code=status_code, content={"error_code": code, "detail": message})


def _is_guest(user: AuthUser) -> bool:
    return user.id == "anonymous"


def _verify_access(db: DBSession, chat_id: str, current_user: AuthUser, session_id: Optional[str] = None):
    if _is_guest(current_user):
        verify_chat_access(db, chat_id, session_id=session_id)
    else:
        verify_chat_access(db, chat_id, user_id=current_user.id)


def _build_recent_history(
    db: DBSession,
    chat_id: str,
    fallback_history: Optional[List[Dict[str, str]]] = None,
) -> List[Dict[str, str]]:
    """Use client-provided history when present, otherwise derive from stored messages."""
    if fallback_history:
        clean: List[Dict[str, str]] = []
        for item in fallback_history[-6:]:
            if not isinstance(item, dict):
                continue
            role = str(item.get("role", "")).strip().lower()
            content = str(item.get("content", "")).strip()
            if role in {"user", "assistant", "system"} and content:
                clean.append({"role": role, "content": content})
        return clean

    msgs = get_messages(db, chat_id, limit=200, offset=0)
    return [
        {"role": m.role, "content": m.content}
        for m in msgs[-6:]
        if m.role in {"user", "assistant", "system"} and (m.content or "").strip()
    ]


class CreateChatRequest(BaseModel):
    title: str = Field("New Chat", max_length=500)
    session_id: Optional[str] = Field(None, description="Guest session ID")


class RenameChatRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=500)
    version: int = Field(..., ge=1)


class ChatResponse(BaseModel):
    id: str
    title: str
    is_guest: bool
    status: str
    version: int
    doc_count: int = 0
    created_at: str
    updated_at: str


class MessageResponse(BaseModel):
    id: str
    role: str
    content: str
    metadata_json: str = "{}"
    created_at: str


class SendMessageRequest(BaseModel):
    role: str = Field("user", pattern="^(user|assistant|system)$")
    content: str = Field(..., min_length=1, max_length=50000)
    metadata_json: str = Field("{}", max_length=10000)


class ChatQueryRequest(BaseModel):
    chat_id: str
    session_id: Optional[str] = None
    question: str = Field(..., min_length=3, max_length=5000)
    top_k: Optional[int] = Field(5, ge=1, le=100)
    system_prompt: Optional[str] = Field(None, max_length=4000)
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(None, ge=1, le=4000)
    chat_history: Optional[List[Dict[str, str]]] = None


@chat_router.post("/guest/session")
def create_guest_session():
    return {"session_id": str(uuid.uuid4())}


@chat_router.delete("/guest/session/{session_id}")
def delete_guest_session(session_id: str, db: DBSession = Depends(get_db)):
    try:
        cleanup_guest_session(db, session_id)
        return {"status": "cleaned"}
    except Exception as e:
        logger.error(f"Guest cleanup error: {e}")
        return {"status": "partial", "detail": str(e)}


@chat_router.post("/chats", response_model=ChatResponse)
def create_new_chat(
    body: CreateChatRequest,
    db: DBSession = Depends(get_db),
    current_user: AuthUser = Depends(get_current_user),
):
    is_guest = _is_guest(current_user)
    chat = create_chat(
        db,
        user_id=None if is_guest else current_user.id,
        session_id=body.session_id if is_guest else None,
        title=body.title,
        is_guest=is_guest,
    )
    db.commit()
    return ChatResponse(
        id=chat.id,
        title=chat.title,
        is_guest=chat.is_guest,
        status=chat.status,
        version=chat.version,
        doc_count=0,
        created_at=chat.created_at.isoformat(),
        updated_at=chat.updated_at.isoformat(),
    )


@chat_router.get("/chats")
def list_chats(
    limit: int = 50,
    offset: int = 0,
    db: DBSession = Depends(get_db),
    current_user: AuthUser = Depends(get_current_user),
):
    if _is_guest(current_user):
        return {"chats": []}

    chats = get_user_chats(db, current_user.id, limit=limit, offset=offset)
    return {
        "chats": [
            ChatResponse(
                id=c.id,
                title=c.title,
                is_guest=c.is_guest,
                status=c.status,
                version=c.version,
                doc_count=get_chat_doc_count(db, c.id),
                created_at=c.created_at.isoformat(),
                updated_at=c.updated_at.isoformat(),
            )
            for c in chats
        ]
    }


@chat_router.patch("/chats/{chat_id}", response_model=ChatResponse)
def rename_existing_chat(
    chat_id: str,
    body: RenameChatRequest,
    db: DBSession = Depends(get_db),
    current_user: AuthUser = Depends(get_current_user),
):
    if _is_guest(current_user):
        return _error_response(403, "FORBIDDEN_CHAT_ACCESS", "Guests cannot rename chats")

    try:
        verify_chat_access(db, chat_id, user_id=current_user.id)
        chat = rename_chat(db, chat_id, body.title, body.version)
        db.commit()
        return ChatResponse(
            id=chat.id,
            title=chat.title,
            is_guest=chat.is_guest,
            status=chat.status,
            version=chat.version,
            doc_count=get_chat_doc_count(db, chat.id),
            created_at=chat.created_at.isoformat(),
            updated_at=chat.updated_at.isoformat(),
        )
    except LimitError as e:
        code_map = {"STALE_VERSION": 409, "CHAT_NOT_FOUND": 404, "FORBIDDEN_CHAT_ACCESS": 403}
        return _error_response(code_map.get(e.code, 400), e.code, e.message)


@chat_router.delete("/chats/{chat_id}")
def delete_existing_chat(
    chat_id: str,
    session_id: Optional[str] = None,
    db: DBSession = Depends(get_db),
    current_user: AuthUser = Depends(get_current_user),
):
    try:
        chat = get_chat(db, chat_id)
        if not chat:
            return {"status": "deleted"}

        _verify_access(db, chat_id, current_user, session_id=session_id)
        ok = delete_chat_cascade(db, chat_id, user_id=current_user.id if not _is_guest(current_user) else None)
        return {"status": "deleted" if ok else "deleting"}
    except LimitError as e:
        code_map = {"CHAT_NOT_FOUND": 404, "FORBIDDEN_CHAT_ACCESS": 403}
        return _error_response(code_map.get(e.code, 400), e.code, e.message)
    except Exception as e:
        logger.error(f"Chat delete error: {e}")
        return _error_response(500, "DELETE_FAILED", str(e))


@chat_router.get("/chats/{chat_id}/messages")
def list_messages(
    chat_id: str,
    limit: int = 200,
    offset: int = 0,
    session_id: Optional[str] = None,
    db: DBSession = Depends(get_db),
    current_user: AuthUser = Depends(get_current_user),
):
    try:
        _verify_access(db, chat_id, current_user, session_id=session_id)
    except LimitError as e:
        return _error_response(403, e.code, e.message)

    msgs = get_messages(db, chat_id, limit=limit, offset=offset)
    return {
        "messages": [
            MessageResponse(
                id=m.id,
                role=m.role,
                content=m.content,
                metadata_json=m.metadata_json or "{}",
                created_at=m.created_at.isoformat(),
            )
            for m in msgs
        ]
    }


@chat_router.post("/chats/{chat_id}/messages", response_model=MessageResponse)
def post_message(
    chat_id: str,
    body: SendMessageRequest,
    session_id: Optional[str] = None,
    db: DBSession = Depends(get_db),
    current_user: AuthUser = Depends(get_current_user),
):
    try:
        _verify_access(db, chat_id, current_user, session_id=session_id)
    except LimitError as e:
        return _error_response(403, e.code, e.message)

    msg = add_message(db, chat_id, body.role, body.content, body.metadata_json)
    db.commit()
    return MessageResponse(
        id=msg.id,
        role=msg.role,
        content=msg.content,
        metadata_json=msg.metadata_json or "{}",
        created_at=msg.created_at.isoformat(),
    )


@chat_router.post("/upload")
async def upload_with_chat(
    file: UploadFile = File(...),
    chat_id: Optional[str] = Form(None),
    session_id: Optional[str] = Form(None),
    db: DBSession = Depends(get_db),
    current_user: AuthUser = Depends(get_current_user),
):
    from backend.services.session_service import get_session_manager

    is_guest = _is_guest(current_user)

    file_ext = Path(file.filename).suffix.lower() if file.filename else ""
    if file_ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported file type '{file_ext}'.")

    contents = await file.read()
    file_size = len(contents)
    if file_size > 20 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large (max 20MB)")

    content_hash = compute_content_hash(contents)

    if chat_id:
        try:
            _verify_access(db, chat_id, current_user, session_id=session_id)
        except LimitError as e:
            code_map = {"CHAT_NOT_FOUND": 404, "FORBIDDEN_CHAT_ACCESS": 403}
            return _error_response(code_map.get(e.code, 400), e.code, e.message)
    else:
        chat = create_chat(
            db,
            user_id=None if is_guest else current_user.id,
            session_id=session_id if is_guest else None,
            title=file.filename or "New Chat",
            is_guest=is_guest,
        )
        chat_id = chat.id
        db.flush()

    try:
        doc = check_and_register_document(
            db,
            chat_id=chat_id,
            user_id=None if is_guest else current_user.id,
            session_id=session_id if is_guest else None,
            is_guest=is_guest,
            filename=file.filename,
            file_size=file_size,
            content_hash=content_hash,
        )
    except LimitError as e:
        db.rollback()
        return _error_response(429 if "LIMIT" in e.code else 409, e.code, e.message)

    try:
        session_manager = get_session_manager()
        effective_session = chat_id
        session_dir = session_manager.BASE_STORAGE_DIR / effective_session
        (session_dir / "documents").mkdir(parents=True, exist_ok=True)
        (session_dir / "chunks").mkdir(parents=True, exist_ok=True)
        (session_dir / "vector_store").mkdir(parents=True, exist_ok=True)

        session_manager._storage.save_document(effective_session, file.filename, contents)
        update_document_status(db, doc.id, "pending")
        db.commit()
    except FileExistsError:
        db.rollback()
        return _error_response(409, "DUPLICATE_DOCUMENT", f"File {file.filename} already exists in this chat")
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

    return {
        "chat_id": chat_id,
        "document_id": doc.id,
        "session_id": effective_session,
        "status": "uploaded",
        "filename": file.filename,
        "content_hash": content_hash,
        "message": "Document uploaded. Click Process when ready.",
    }


@chat_router.post("/ask")
def ask_with_chat(
    query: ChatQueryRequest,
    db: DBSession = Depends(get_db),
    current_user: AuthUser = Depends(get_current_user),
):
    from backend.services.rag_service_session import ask_rag_session, get_llm_status
    from backend.services.session_service import get_session_manager

    try:
        _verify_access(db, query.chat_id, current_user, session_id=query.session_id)
    except LimitError as e:
        return _error_response(403, e.code, e.message)

    chat = get_chat(db, query.chat_id)
    if not chat or chat.status != "active":
        return _error_response(404, "CHAT_NOT_FOUND", "Chat not found or not ready")

    if not get_llm_status():
        raise HTTPException(status_code=503, detail="LLM not initialized")

    effective_session = query.chat_id
    session_manager = get_session_manager()
    chunks_metadata = session_manager.load_chunks_metadata(effective_session)
    vector_store_dir = session_manager.get_vector_store_dir(effective_session)
    chat_history_payload = _build_recent_history(db, query.chat_id, query.chat_history)

    add_message(db, query.chat_id, "user", query.question)

    result = ask_rag_session(
        session_id=effective_session,
        question=query.question,
        chunks_metadata=chunks_metadata,
        vector_store_dir=vector_store_dir,
        top_k=query.top_k,
        system_prompt=query.system_prompt,
        temperature=query.temperature,
        max_tokens=query.max_tokens,
        chat_history=chat_history_payload,
    )

    answer = result.llm_response or ""
    meta = json.dumps({
        "sources": result.metadata[:5] if result.metadata else [],
        "scores": (result.retrieval_scores or [])[:5],
    })
    add_message(db, query.chat_id, "assistant", answer, metadata_json=meta)
    db.commit()

    return {
        "answer": answer,
        "query": result.query,
        "timestamp": result.timestamp,
        "retrieved_chunks": result.retrieved_chunks,
        "retrieval_scores": result.retrieval_scores or [],
        "metadata": result.metadata,
        "chat_id": query.chat_id,
    }


@chat_router.post("/ask/stream")
async def ask_stream_with_chat(
    query: ChatQueryRequest,
    db: DBSession = Depends(get_db),
    current_user: AuthUser = Depends(get_current_user),
):
    from backend.services.rag_service_session import ask_rag_session_stream, get_llm_status
    from backend.services.session_service import get_session_manager

    try:
        _verify_access(db, query.chat_id, current_user, session_id=query.session_id)
    except LimitError as e:
        return _error_response(403, e.code, e.message)

    chat = get_chat(db, query.chat_id)
    if not chat or chat.status != "active":
        return _error_response(404, "CHAT_NOT_FOUND", "Chat not found or not ready")

    if not get_llm_status():
        raise HTTPException(status_code=503, detail="LLM not initialized")

    effective_session = query.chat_id
    session_manager = get_session_manager()
    chunks_metadata = session_manager.load_chunks_metadata(effective_session)
    vector_store_dir = session_manager.get_vector_store_dir(effective_session)
    chat_history_payload = _build_recent_history(db, query.chat_id, query.chat_history)

    add_message(db, query.chat_id, "user", query.question)
    db.commit()

    def _event_generator():
        collected_answer = []
        try:
            for event_dict in ask_rag_session_stream(
                session_id=effective_session,
                question=query.question,
                chunks_metadata=chunks_metadata,
                vector_store_dir=vector_store_dir,
                top_k=query.top_k,
                system_prompt=query.system_prompt,
                temperature=query.temperature,
                max_tokens=query.max_tokens,
                chat_history=chat_history_payload,
            ):
                if event_dict.get("event") == "chunk":
                    collected_answer.append(event_dict.get("data", ""))
                yield f"data: {json.dumps(event_dict)}\\n\\n"
        except Exception as e:
            logger.error(f"[stream] Error: {e}", exc_info=True)
            yield f"data: {json.dumps({'event': 'error', 'data': str(e)})}\\n\\n"

        try:
            from backend.database.models import get_session_local

            save_db = get_session_local()()
            try:
                full_answer = "".join(collected_answer)
                add_message(save_db, query.chat_id, "assistant", full_answer)
                save_db.commit()
            finally:
                save_db.close()
        except Exception as e:
            logger.warning(f"Failed to save streamed answer to chat history: {e}")

    return StreamingResponse(_event_generator(), media_type="text/event-stream")


@chat_router.post("/process/{chat_id}")
async def process_chat_documents(
    chat_id: str,
    background_tasks: BackgroundTasks,
    session_id: Optional[str] = None,
    db: DBSession = Depends(get_db),
    current_user: AuthUser = Depends(get_current_user),
):
    from backend.database.models import Session as LegacySession
    from backend.services.session_service import get_session_manager

    try:
        _verify_access(db, chat_id, current_user, session_id=session_id)
    except LimitError as e:
        return _error_response(403, e.code, e.message)

    session_manager = get_session_manager()
    legacy = db.query(LegacySession).filter(LegacySession.session_id == chat_id).first()
    if not legacy:
        docs = get_chat_documents(db, chat_id)
        first_file = docs[0].filename if docs else "documents"
        total_size = sum(d.file_size for d in docs)
        legacy = LegacySession(
            session_id=chat_id,
            user_id=current_user.id if not _is_guest(current_user) else None,
            filename=first_file,
            file_size=total_size,
            status="uploaded",
        )
        db.add(legacy)
        db.commit()

    if legacy.status == "processing":
        return {"chat_id": chat_id, "status": "processing", "message": "Already processing"}
    if legacy.status == "ready":
        return {"chat_id": chat_id, "status": "ready", "message": "Already processed"}

    session_manager.update_session_status(chat_id, "processing", db)
    background_tasks.add_task(process_document_async, session_id=chat_id, session_manager=session_manager)
    return {"chat_id": chat_id, "status": "processing", "message": "Processing started"}


@chat_router.get("/status/{chat_id}")
def get_chat_status(
    chat_id: str,
    session_id: Optional[str] = None,
    db: DBSession = Depends(get_db),
    current_user: AuthUser = Depends(get_current_user),
):
    from backend.database.models import Session as LegacySession

    try:
        _verify_access(db, chat_id, current_user, session_id=session_id)
    except LimitError as e:
        return _error_response(403, e.code, e.message)

    legacy = db.query(LegacySession).filter(LegacySession.session_id == chat_id).first()
    if not legacy:
        return _error_response(404, "CHAT_NOT_FOUND", "No processing session found for this chat")

    return {
        "chat_id": chat_id,
        "session_id": chat_id,
        "status": legacy.status,
        "filename": legacy.filename,
        "created_at": legacy.created_at.isoformat() if legacy.created_at else "",
        "last_accessed": legacy.last_accessed.isoformat() if legacy.last_accessed else "",
        "chunks_count": legacy.chunks_count,
        "error_message": legacy.error_message,
    }
