"""FastAPI routes for anonymous single-session RAG backend."""

import json
import shutil
import threading
import uuid
from typing import Optional, List, Dict
from pathlib import Path
from datetime import datetime
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Depends, BackgroundTasks
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session as DBSession
from backend.database import get_db
from backend.services.session_service import get_session_manager, DATA_DIR
from backend.services.rag_service_session import (
    ask_rag_session,
    get_llm_status,
    clear_session_handler,
    get_session_chat_history,
    append_session_chat_history,
)
from backend.services.chat_service import (
    LimitError,
    check_and_register_document,
    compute_content_hash,
    get_session_documents,
    update_document_status,
)
from backend.services.cascade_service import delete_session_data
from backend.rag.IngestSession import ingest_documents_session
from src.utils.Logger import get_logger
from config.config import VECTOR_BACKEND

logger = get_logger(__name__)
router = APIRouter()

# Concurrency control: max 5 sessions processing at once
_processing_semaphore = threading.Semaphore(5)

SUPPORTED_EXTENSIONS = {
    ".pdf", ".csv", ".tsv", ".xlsx", ".xls",
    ".json", ".yaml", ".yml", ".md", ".html", ".htm",
    ".py", ".js", ".java", ".cpp", ".c", ".cs", ".go",
    ".rs", ".ts", ".jsx", ".tsx", ".txt", ".docx",
    ".ipynb",
}


# ── Request / Response models ────────────────────────────────────────────

class AskRequest(BaseModel):
    session_id: str
    question: str = Field(..., min_length=3, max_length=5000)
    top_k: Optional[int] = Field(5, ge=1, le=100)
    system_prompt: Optional[str] = Field(None, max_length=4000)
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(None, ge=1, le=4000)
    chat_history: Optional[List[Dict[str, str]]] = None


class DocumentResponse(BaseModel):
    id: str
    filename: str
    file_size: int
    status: str
    created_at: str


def _error_response(status_code: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(status_code=status_code, content={"error_code": code, "detail": message})


def _build_recent_history(
    history: Optional[List[Dict[str, str]]] = None,
) -> List[Dict[str, str]]:
    """Sanitize client-provided chat history."""
    if not history:
        return []
    clean: List[Dict[str, str]] = []
    for item in history[-6:]:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role", "")).strip().lower()
        content = str(item.get("content", "")).strip()
        if role in {"user", "assistant", "system"} and content:
            clean.append({"role": role, "content": content})
    return clean


def _merge_history(
    session_id: str,
    client_history: Optional[List[Dict[str, str]]] = None,
) -> List[Dict[str, str]]:
    """Merge server-side session history with client-provided history."""
    server_history = _build_recent_history(get_session_chat_history(session_id, limit=6))
    clean_client = _build_recent_history(client_history)
    combined = server_history + clean_client
    if len(combined) > 8:
        combined = combined[-8:]
    return combined


# ── Background processing ────────────────────────────────────────────────

def process_document_async(session_id: str, session_manager):
    """Background task to ingest all uploaded documents for a session."""
    from backend.database.models import get_session_local

    acquired = _processing_semaphore.acquire(timeout=300)
    if not acquired:
        db = get_session_local()()
        try:
            session_manager.update_session_status(
                session_id=session_id, status="error", db=db,
                error_message="Server busy — too many documents being processed. Please try again later.",
            )
        finally:
            db.close()
        return

    db = get_session_local()()
    try:
        documents_dir = session_manager.prepare_documents_for_processing(session_id)
        chunks_dir = session_manager.get_chunks_dir(session_id)
        vector_store_dir = session_manager.get_vector_store_dir(session_id)

        max_attempts = 2
        last_error = None

        for attempt in range(max_attempts):
            try:
                chunks_count = ingest_documents_session(
                    session_id=session_id,
                    documents_dir=documents_dir,
                    chunks_dir=chunks_dir,
                    vector_store_dir=vector_store_dir,
                )

                docs = get_session_documents(db, session_id)
                for d in docs:
                    if d.status == "pending":
                        update_document_status(db, d.id, "ready")

                session_manager.update_session_status(
                    session_id=session_id, status="ready", db=db, chunks_count=chunks_count,
                )
                logger.info(f"[{session_id[:8]}] Processing complete: {chunks_count} chunks")
                return
            except Exception as e:
                last_error = e
                if attempt < max_attempts - 1:
                    logger.warning(f"[{session_id[:8]}] Attempt {attempt + 1} failed, retrying: {e}")
                    import time
                    time.sleep(2)

        logger.error(f"[{session_id[:8]}] Processing failed after {max_attempts} attempts: {last_error}")
        docs = get_session_documents(db, session_id)
        for d in docs:
            if d.status == "pending":
                update_document_status(db, d.id, "failed")
        session_manager.update_session_status(
            session_id=session_id, status="error", db=db, error_message=str(last_error),
        )
    except Exception as e:
        logger.error(f"[{session_id[:8]}] Unexpected error: {e}")
        try:
            docs = get_session_documents(db, session_id)
            for d in docs:
                if d.status == "pending":
                    update_document_status(db, d.id, "failed")
            session_manager.update_session_status(
                session_id=session_id, status="error", db=db, error_message=str(e),
            )
        except Exception:
            pass
    finally:
        db.close()
        _processing_semaphore.release()


# ── Endpoints ────────────────────────────────────────────────────────────

@router.post("/session")
def create_session():
    """Create a new anonymous session. Returns a session_id."""
    return {"session_id": str(uuid.uuid4())}


@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    session_id: Optional[str] = Form(None),
    db: DBSession = Depends(get_db),
):
    """Upload a document to a session (max 15 per session)."""
    session_manager = get_session_manager()

    file_ext = Path(file.filename).suffix.lower() if file.filename else ""
    if file_ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported file type '{file_ext}'.")

    contents = await file.read()
    file_size = len(contents)
    if file_size > 20 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large (max 20MB)")

    content_hash = compute_content_hash(contents)

    # Create session dirs if needed
    if not session_id:
        session_id = str(uuid.uuid4())

    session_dir = session_manager.BASE_STORAGE_DIR / session_id
    (session_dir / "documents").mkdir(parents=True, exist_ok=True)
    (session_dir / "chunks").mkdir(parents=True, exist_ok=True)
    (session_dir / "vector_store").mkdir(parents=True, exist_ok=True)

    try:
        doc = check_and_register_document(
            db,
            session_id=session_id,
            filename=file.filename,
            file_size=file_size,
            content_hash=content_hash,
        )
    except LimitError as e:
        db.rollback()
        return _error_response(429, e.code, e.message)

    try:
        session_manager._storage.save_document(session_id, file.filename, contents)
        update_document_status(db, doc.id, "pending")
        db.commit()
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

    return {
        "session_id": session_id,
        "document_id": doc.id,
        "status": "uploaded",
        "filename": file.filename,
        "message": "Document uploaded. Click Process when ready.",
    }


@router.post("/process/{session_id}")
async def process_session(
    session_id: str,
    background_tasks: BackgroundTasks,
    db: DBSession = Depends(get_db),
):
    """Trigger processing of all uploaded documents in a session."""
    from backend.database.models import Session as SessionModel

    session_manager = get_session_manager()

    # Ensure a processing row exists
    legacy = db.query(SessionModel).filter(SessionModel.session_id == session_id).first()
    if not legacy:
        docs = get_session_documents(db, session_id)
        if not docs:
            raise HTTPException(status_code=404, detail="No documents found for this session")
        first_file = docs[0].filename
        total_size = sum(d.file_size for d in docs)
        legacy = SessionModel(
            session_id=session_id,
            filename=first_file,
            file_size=total_size,
            status="uploaded",
        )
        db.add(legacy)
        db.commit()

    if legacy.status == "processing":
        return {"session_id": session_id, "status": "processing", "message": "Already processing"}
    if legacy.status == "ready":
        return {"session_id": session_id, "status": "ready", "message": "Already processed"}

    session_manager.update_session_status(session_id, "processing", db)
    background_tasks.add_task(process_document_async, session_id=session_id, session_manager=session_manager)
    return {"session_id": session_id, "status": "processing", "message": "Processing started"}


@router.get("/status/{session_id}")
def get_status(
    session_id: str,
    db: DBSession = Depends(get_db),
):
    """Get processing status for a session."""
    from backend.database.models import Session as SessionModel

    legacy = db.query(SessionModel).filter(SessionModel.session_id == session_id).first()
    if not legacy:
        raise HTTPException(status_code=404, detail="Session not found")

    docs = get_session_documents(db, session_id)
    total_documents = len(docs)
    ready_documents = sum(1 for d in docs if d.status == "ready")
    failed_documents = sum(1 for d in docs if d.status == "failed")
    pending_documents = sum(1 for d in docs if d.status == "pending")

    return {
        "session_id": session_id,
        "status": legacy.status,
        "filename": legacy.filename,
        "created_at": legacy.created_at.isoformat() if legacy.created_at else "",
        "last_accessed": legacy.last_accessed.isoformat() if legacy.last_accessed else "",
        "chunks_count": legacy.chunks_count,
        "error_message": legacy.error_message,
        "document_progress": {
            "total": total_documents,
            "processed": ready_documents + failed_documents,
            "ready": ready_documents,
            "failed": failed_documents,
            "pending": pending_documents,
        },
    }


@router.get("/documents/{session_id}")
def list_documents(
    session_id: str,
    db: DBSession = Depends(get_db),
):
    """List documents for a session."""
    docs = get_session_documents(db, session_id)
    docs_sorted = sorted(docs, key=lambda d: d.created_at)
    return {
        "documents": [
            DocumentResponse(
                id=d.id,
                filename=d.filename,
                file_size=d.file_size or 0,
                status=d.status,
                created_at=d.created_at.isoformat(),
            )
            for d in docs_sorted
        ]
    }


@router.post("/ask")
def ask(
    query: AskRequest,
    db: DBSession = Depends(get_db),
):
    """Ask a question about processed documents."""
    from backend.services.rag_service_session import ask_rag_session

    session_manager = get_session_manager()
    session = session_manager.get_session(query.session_id, db)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    if session.status != "ready":
        raise HTTPException(status_code=400, detail=f"Session not ready: {session.status}")
    if not get_llm_status():
        raise HTTPException(status_code=503, detail="LLM not initialized")

    chunks_metadata = session_manager.load_chunks_metadata(query.session_id)
    vector_store_dir = session_manager.get_vector_store_dir(query.session_id)
    chat_history = _merge_history(query.session_id, query.chat_history)

    result = ask_rag_session(
        session_id=query.session_id,
        question=query.question,
        chunks_metadata=chunks_metadata,
        vector_store_dir=vector_store_dir,
        top_k=query.top_k,
        system_prompt=query.system_prompt,
        temperature=query.temperature,
        max_tokens=query.max_tokens,
        chat_history=chat_history,
    )

    append_session_chat_history(
        query.session_id,
        query.question,
        result.llm_response or "",
    )

    return {
        "answer": result.llm_response or "",
        "query": result.query,
        "timestamp": result.timestamp,
        "retrieved_chunks": result.retrieved_chunks,
        "retrieval_scores": result.retrieval_scores or [],
        "metadata": result.metadata,
        "session_id": query.session_id,
    }


@router.post("/ask/stream")
async def ask_stream(
    query: AskRequest,
    db: DBSession = Depends(get_db),
):
    """Stream answer tokens via Server-Sent Events."""
    from backend.services.rag_service_session import ask_rag_session_stream

    session_manager = get_session_manager()
    session = session_manager.get_session(query.session_id, db)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    if session.status != "ready":
        raise HTTPException(status_code=400, detail=f"Session not ready: {session.status}")
    if not get_llm_status():
        raise HTTPException(status_code=503, detail="LLM not initialized")

    chunks_metadata = session_manager.load_chunks_metadata(query.session_id)
    vector_store_dir = session_manager.get_vector_store_dir(query.session_id)
    chat_history = _merge_history(query.session_id, query.chat_history)

    def _event_generator():
        assistant_parts: List[str] = []
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
                chat_history=chat_history,
            ):
                if event_dict.get("event") == "chunk":
                    assistant_parts.append(str(event_dict.get("data", "")))
                if event_dict.get("event") == "success":
                    append_session_chat_history(
                        query.session_id,
                        query.question,
                        "".join(assistant_parts),
                    )
                yield f"data: {json.dumps(event_dict)}\n\n"
        except Exception as e:
            logger.error(f"[stream] Error: {e}", exc_info=True)
            yield f"data: {json.dumps({'event': 'error', 'data': str(e)})}\n\n"

    return StreamingResponse(_event_generator(), media_type="text/event-stream")


@router.delete("/session/{session_id}")
def delete_session(
    session_id: str,
    db: DBSession = Depends(get_db),
):
    """Delete a session and all its data (documents, chunks, embeddings)."""
    clear_session_handler(session_id)
    ok = delete_session_data(db, session_id)
    return {"status": "deleted" if ok else "partial", "session_id": session_id}


@router.get("/health")
def health(db: DBSession = Depends(get_db)):
    """Health check endpoint."""
    status_report = {
        "status": "ok",
        "llm_loaded": get_llm_status(),
        "database": "unknown",
        "vector_backend": VECTOR_BACKEND,
    }

    try:
        from sqlalchemy import text
        db.execute(text("SELECT 1"))
        status_report["database"] = "connected"
    except Exception:
        status_report["database"] = "disconnected"
        status_report["status"] = "degraded"

    return status_report
