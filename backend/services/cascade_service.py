"""Session cleanup service.

Handles full deletion of session data: DB rows, local files, vector store.
"""

from __future__ import annotations

import shutil
from typing import Optional

from backend.database.models import Document, Session
from src.utils.Logger import get_logger

logger = get_logger(__name__)


def delete_session_data(db, session_id: str) -> bool:
    """
    Full cleanup for a session:
    1. Delete Qdrant/FAISS vectors
    2. Delete local session files
    3. Delete DB rows (documents, session)
    Returns True on success.
    """
    from backend.services.session_service import get_session_manager

    errors = []

    # 1) Delete vector store points
    try:
        _delete_vectors_for_session(session_id)
    except Exception as e:
        logger.warning(f"Vector delete failed for session {session_id}: {e}")
        errors.append(str(e))

    # 2) Delete local session directory
    try:
        session_manager = get_session_manager()
        session_manager._storage.delete_session(session_id)
        session_dir = session_manager._get_session_dir(session_id)
        if session_dir.exists():
            shutil.rmtree(session_dir, ignore_errors=True)
    except Exception as e:
        logger.warning(f"Storage delete failed for session {session_id}: {e}")
        errors.append(str(e))

    # 3) Delete DB rows
    try:
        docs = db.query(Document).filter(Document.session_id == session_id).all()
        for doc in docs:
            db.delete(doc)
        session_row = db.query(Session).filter(Session.session_id == session_id).first()
        if session_row:
            db.delete(session_row)
        db.commit()
    except Exception as e:
        logger.error(f"DB delete failed for session {session_id}: {e}")
        db.rollback()
        errors.append(str(e))

    if errors:
        logger.warning(f"Session {session_id} partially deleted: {errors}")
        return False

    logger.info(f"Session {session_id} fully deleted")
    return True


def _delete_vectors_for_session(session_id: str) -> None:
    """Delete vector store data for a session."""
    from config.config import VECTOR_BACKEND
    if VECTOR_BACKEND != "qdrant":
        return

    try:
        from src.modules.QdrantStore import delete_session_vectors
        delete_session_vectors(session_id)
    except ImportError:
        pass

