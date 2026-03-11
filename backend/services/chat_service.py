"""Document registration service with per-session quota enforcement.

Simplified: no chat/message CRUD, no user/guest distinction,
just session-scoped document tracking with a 15-doc limit.
"""

from __future__ import annotations

import hashlib
import uuid
from datetime import datetime
from typing import List, Optional

from sqlalchemy import func
from sqlalchemy.orm import Session as DBSession

from backend.database.models import Document
from src.utils.Logger import get_logger

logger = get_logger(__name__)

# ── Quota constants ──────────────────────────────────────────────────────
SESSION_DOC_LIMIT = 15

ACTIVE_DOC_STATUSES = ("pending", "ready", "failed")


class LimitError(Exception):
    def __init__(self, code: str, message: str):
        self.code = code
        self.message = message
        super().__init__(message)


def compute_content_hash(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _new_id() -> str:
    return str(uuid.uuid4())


# ── Document registration ───────────────────────────────────────────────

def check_and_register_document(
    db: DBSession,
    *,
    session_id: str,
    filename: str,
    file_size: int,
    content_hash: str,
) -> Document:
    """Atomically check session quota and create a document record."""
    doc_count = db.query(func.count(Document.id)).filter(
        Document.session_id == session_id,
        Document.status.in_(ACTIVE_DOC_STATUSES),
    ).scalar() or 0

    if doc_count >= SESSION_DOC_LIMIT:
        raise LimitError(
            "SESSION_LIMIT_REACHED",
            f"Session document limit reached ({SESSION_DOC_LIMIT}).",
        )

    doc = Document(
        id=_new_id(),
        session_id=session_id,
        filename=filename,
        file_size=file_size,
        content_hash=content_hash,
        status="pending",
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )
    db.add(doc)
    db.flush()
    return doc


def update_document_status(
    db: DBSession, doc_id: str, status: str
) -> None:
    doc = db.query(Document).filter(Document.id == doc_id).first()
    if doc:
        doc.status = status
        doc.updated_at = datetime.utcnow()
        db.flush()


def get_session_documents(db: DBSession, session_id: str) -> List[Document]:
    return db.query(Document).filter(
        Document.session_id == session_id,
        Document.status.in_(ACTIVE_DOC_STATUSES),
    ).all()


def get_session_doc_count(db: DBSession, session_id: str) -> int:
    return db.query(func.count(Document.id)).filter(
        Document.session_id == session_id,
        Document.status.in_(ACTIVE_DOC_STATUSES),
    ).scalar() or 0

