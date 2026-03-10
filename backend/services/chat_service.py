"""Chat and document lifecycle service with concurrency controls.

Handles CRUD for chats, messages, document tracking, quota enforcement,
deduplication, and cascading deletes with retry logic.
"""

from __future__ import annotations

import hashlib
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from sqlalchemy import func, and_
from sqlalchemy.orm import Session as DBSession

from backend.database.models import Chat, ChatMessage, Document, CleanupJob
from src.utils.Logger import get_logger

logger = get_logger(__name__)

# ── Quota constants ──────────────────────────────────────────────────────
GUEST_DOC_LIMIT = 3
PER_CHAT_DOC_LIMIT = 15
ACCOUNT_DOC_LIMIT = 40

ACTIVE_DOC_STATUSES = ("pending", "ready", "failed")


# ── Error codes ──────────────────────────────────────────────────────────
class LimitError(Exception):
    def __init__(self, code: str, message: str):
        self.code = code
        self.message = message
        super().__init__(message)


# ── Helpers ──────────────────────────────────────────────────────────────

def compute_content_hash(content: bytes) -> str:
    """SHA-256 hash of file content."""
    return hashlib.sha256(content).hexdigest()


def _new_id() -> str:
    return str(uuid.uuid4())


# ── Chat CRUD ────────────────────────────────────────────────────────────

def create_chat(
    db: DBSession,
    *,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    title: str = "New Chat",
    is_guest: bool = False,
) -> Chat:
    chat = Chat(
        id=_new_id(),
        user_id=user_id,
        session_id=session_id,
        title=title,
        is_guest=is_guest,
        status="active",
        version=1,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )
    db.add(chat)
    db.flush()
    return chat


def get_chat(db: DBSession, chat_id: str) -> Optional[Chat]:
    return db.query(Chat).filter(
        Chat.id == chat_id, Chat.status != "deleted"
    ).first()


def get_user_chats(
    db: DBSession, user_id: str, *, limit: int = 50, offset: int = 0
) -> List[Chat]:
    return (
        db.query(Chat)
        .filter(Chat.user_id == user_id, Chat.status == "active")
        .order_by(Chat.updated_at.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )


def rename_chat(
    db: DBSession, chat_id: str, new_title: str, expected_version: int
) -> Chat:
    chat = db.query(Chat).filter(
        Chat.id == chat_id, Chat.status == "active"
    ).with_for_update().first()
    if not chat:
        raise LimitError("CHAT_NOT_FOUND", "Chat not found")
    if chat.version != expected_version:
        raise LimitError("STALE_VERSION", "Chat was modified by another request. Refresh and try again.")
    chat.title = new_title
    chat.version += 1
    chat.updated_at = datetime.utcnow()
    db.flush()
    return chat


def mark_chat_deleting(db: DBSession, chat_id: str) -> Optional[Chat]:
    """Mark chat as deleting (idempotent). Returns chat or None."""
    chat = db.query(Chat).filter(
        Chat.id == chat_id, Chat.status.in_(("active", "deleting"))
    ).with_for_update().first()
    if not chat:
        return None
    if chat.status == "active":
        chat.status = "deleting"
        chat.updated_at = datetime.utcnow()
        db.flush()
    return chat


def mark_chat_deleted(db: DBSession, chat_id: str) -> None:
    chat = db.query(Chat).filter(Chat.id == chat_id).first()
    if chat:
        chat.status = "deleted"
        chat.updated_at = datetime.utcnow()
        db.flush()


# ── Ownership checks ────────────────────────────────────────────────────

def verify_chat_access(
    db: DBSession,
    chat_id: str,
    *,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
) -> Chat:
    """Verify the caller owns this chat. Raises on mismatch."""
    chat = get_chat(db, chat_id)
    if not chat:
        raise LimitError("CHAT_NOT_FOUND", "Chat not found")
    if chat.status == "deleting":
        raise LimitError("CHAT_NOT_FOUND", "Chat is being deleted")
    if chat.is_guest:
        if chat.session_id != session_id:
            raise LimitError("FORBIDDEN_CHAT_ACCESS", "Access denied")
    else:
        if chat.user_id != user_id:
            raise LimitError("FORBIDDEN_CHAT_ACCESS", "Access denied")
    return chat


# ── Messages ─────────────────────────────────────────────────────────────

def add_message(
    db: DBSession,
    chat_id: str,
    role: str,
    content: str,
    metadata_json: str = "{}",
) -> ChatMessage:
    msg = ChatMessage(
        id=_new_id(),
        chat_id=chat_id,
        role=role,
        content=content,
        metadata_json=metadata_json,
        created_at=datetime.utcnow(),
    )
    db.add(msg)
    # Update chat's updated_at
    chat = db.query(Chat).filter(Chat.id == chat_id).first()
    if chat:
        chat.updated_at = datetime.utcnow()
    db.flush()
    return msg


def get_messages(
    db: DBSession, chat_id: str, *, limit: int = 200, offset: int = 0
) -> List[ChatMessage]:
    return (
        db.query(ChatMessage)
        .filter(ChatMessage.chat_id == chat_id)
        .order_by(ChatMessage.created_at.asc())
        .offset(offset)
        .limit(limit)
        .all()
    )


# ── Document quota enforcement (transactional) ──────────────────────────

def check_and_register_document(
    db: DBSession,
    *,
    chat_id: str,
    user_id: Optional[str],
    session_id: Optional[str],
    is_guest: bool,
    filename: str,
    file_size: int,
    content_hash: str,
) -> Document:
    """
    Atomically check quotas and create a document record.
    Must be called within a transaction. Uses SELECT ... FOR UPDATE on the chat row
    to serialize concurrent uploads to the same chat.
    """
    # Lock the chat row to serialize concurrent uploads
    chat = db.query(Chat).filter(
        Chat.id == chat_id, Chat.status == "active"
    ).with_for_update().first()
    if not chat:
        raise LimitError("CHAT_NOT_FOUND", "Chat not found or is being deleted")

    # Count docs in this chat (active statuses only)
    chat_doc_count = db.query(func.count(Document.id)).filter(
        Document.chat_id == chat_id,
        Document.status.in_(ACTIVE_DOC_STATUSES),
    ).scalar() or 0

    if is_guest:
        # Guest: max 3 per session (session == chat for guests)
        if chat_doc_count >= GUEST_DOC_LIMIT:
            raise LimitError(
                "GUEST_LIMIT_REACHED",
                f"Guest upload limit reached ({GUEST_DOC_LIMIT}). Sign in for higher limits."
            )
    else:
        # Logged-in: per-chat limit
        if chat_doc_count >= PER_CHAT_DOC_LIMIT:
            raise LimitError(
                "PER_CHAT_LIMIT_REACHED",
                f"Chat document limit reached ({PER_CHAT_DOC_LIMIT}). Create a new chat to upload more."
            )
        # Logged-in: global account limit
        account_doc_count = db.query(func.count(Document.id)).filter(
            Document.user_id == user_id,
            Document.status.in_(ACTIVE_DOC_STATUSES),
        ).scalar() or 0
        if account_doc_count >= ACCOUNT_DOC_LIMIT:
            raise LimitError(
                "ACCOUNT_LIMIT_REACHED",
                f"Account limit reached ({ACCOUNT_DOC_LIMIT}). Delete older chats/documents to upload new files."
            )

    # Deduplication: check content_hash for this user
    if not is_guest and user_id and content_hash:
        existing = db.query(Document).filter(
            Document.user_id == user_id,
            Document.content_hash == content_hash,
            Document.status.in_(ACTIVE_DOC_STATUSES),
        ).first()
        if existing:
            # If same doc already in this chat, reject
            if existing.chat_id == chat_id:
                raise LimitError(
                    "DUPLICATE_DOCUMENT",
                    f"This document is already uploaded in this chat."
                )
            # If in another chat, allow (user may want it in multiple contexts)

    doc = Document(
        id=_new_id(),
        chat_id=chat_id,
        user_id=user_id,
        session_id=session_id,
        filename=filename,
        file_size=file_size,
        is_guest=is_guest,
        content_hash=content_hash,
        status="pending",
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )
    db.add(doc)
    db.flush()
    return doc


def update_document_status(
    db: DBSession, doc_id: str, status: str, error_message: Optional[str] = None
) -> None:
    doc = db.query(Document).filter(Document.id == doc_id).first()
    if doc:
        doc.status = status
        doc.updated_at = datetime.utcnow()
        db.flush()


def get_chat_documents(db: DBSession, chat_id: str) -> List[Document]:
    return db.query(Document).filter(
        Document.chat_id == chat_id,
        Document.status.in_(ACTIVE_DOC_STATUSES),
    ).all()


def get_chat_doc_count(db: DBSession, chat_id: str) -> int:
    return db.query(func.count(Document.id)).filter(
        Document.chat_id == chat_id,
        Document.status.in_(ACTIVE_DOC_STATUSES),
    ).scalar() or 0


def get_user_doc_count(db: DBSession, user_id: str) -> int:
    return db.query(func.count(Document.id)).filter(
        Document.user_id == user_id,
        Document.status.in_(ACTIVE_DOC_STATUSES),
    ).scalar() or 0


# ── Cleanup jobs ─────────────────────────────────────────────────────────

def create_cleanup_job(
    db: DBSession,
    job_type: str,
    target_id: str,
) -> CleanupJob:
    job = CleanupJob(
        id=_new_id(),
        job_type=job_type,
        target_id=target_id,
        status="pending",
        attempts=0,
        max_attempts=5,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )
    db.add(job)
    db.flush()
    return job


def get_pending_cleanup_jobs(db: DBSession, limit: int = 20) -> List[CleanupJob]:
    return (
        db.query(CleanupJob)
        .filter(
            CleanupJob.status.in_(("pending", "failed")),
            CleanupJob.attempts < CleanupJob.max_attempts,
        )
        .order_by(CleanupJob.created_at.asc())
        .limit(limit)
        .all()
    )


def complete_cleanup_job(db: DBSession, job_id: str) -> None:
    job = db.query(CleanupJob).filter(CleanupJob.id == job_id).first()
    if job:
        job.status = "completed"
        job.updated_at = datetime.utcnow()
        db.flush()


def fail_cleanup_job(db: DBSession, job_id: str, error: str) -> None:
    job = db.query(CleanupJob).filter(CleanupJob.id == job_id).first()
    if job:
        job.attempts += 1
        job.error_message = error
        job.status = "failed" if job.attempts < job.max_attempts else "completed"
        job.updated_at = datetime.utcnow()
        db.flush()


# ── Guest cleanup helpers ────────────────────────────────────────────────

def get_guest_chats_by_session(db: DBSession, session_id: str) -> List[Chat]:
    return db.query(Chat).filter(
        Chat.session_id == session_id,
        Chat.is_guest == True,
        Chat.status != "deleted",
    ).all()


def get_stale_guest_chats(db: DBSession, max_age_minutes: int = 120) -> List[Chat]:
    """Find guest chats older than max_age_minutes."""
    from datetime import timedelta
    cutoff = datetime.utcnow() - timedelta(minutes=max_age_minutes)
    return db.query(Chat).filter(
        Chat.is_guest == True,
        Chat.status == "active",
        Chat.updated_at < cutoff,
    ).all()

