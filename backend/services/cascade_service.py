"""Cascade delete and cleanup worker for chats, documents, storage, and Qdrant.

Handles:
- Full chat cascade (DB rows + Supabase storage + Qdrant points)
- Guest session cleanup
- Retry queue processing for partial failures
- Stale guest TTL sweeper
"""

from __future__ import annotations

import threading
import time
from datetime import datetime
from typing import Optional

from backend.database.models import get_session_local
from backend.services.chat_service import (
    CleanupJob,
    create_cleanup_job,
    complete_cleanup_job,
    fail_cleanup_job,
    get_pending_cleanup_jobs,
    get_stale_guest_chats,
    mark_chat_deleted,
    mark_chat_deleting,
    get_chat_documents,
)
from backend.database.models import Chat, Document, ChatMessage
from src.utils.Logger import get_logger

logger = get_logger(__name__)

GUEST_TTL_MINUTES = 120  # 2 hours


def delete_chat_cascade(db, chat_id: str, user_id: Optional[str] = None) -> bool:
    """
    Full cascade delete for a chat:
    1. Mark chat as deleting
    2. Delete Qdrant points
    3. Delete Supabase Storage files
    4. Delete DB rows (documents, messages, chat)
    5. Delete local session files
    
    On partial failure, queues a cleanup job for retry.
    Returns True if fully cleaned, False if retry queued.
    """
    from backend.services.session_service import get_session_manager

    chat = mark_chat_deleting(db, chat_id)
    if not chat:
        return True  # Already deleted or doesn't exist
    db.commit()

    errors = []

    # 1) Delete Qdrant points
    try:
        _delete_qdrant_for_chat(chat_id, user_id=user_id, session_id=chat.session_id)
    except Exception as e:
        logger.warning(f"Qdrant delete failed for chat {chat_id}: {e}")
        errors.append(f"qdrant: {e}")

    # 2) Delete Supabase Storage files
    try:
        session_manager = get_session_manager()
        # Documents are stored under session paths
        if chat.session_id:
            session_manager._storage.delete_session(chat.session_id)
    except Exception as e:
        logger.warning(f"Storage delete failed for chat {chat_id}: {e}")
        errors.append(f"storage: {e}")

    # 3) Delete local session directory
    try:
        if chat.session_id:
            session_manager = get_session_manager()
            session_dir = session_manager._get_session_dir(chat.session_id)
            if session_dir.exists():
                import shutil
                shutil.rmtree(session_dir, ignore_errors=True)
    except Exception as e:
        logger.warning(f"Local dir delete failed for chat {chat_id}: {e}")
        errors.append(f"local: {e}")

    # 4) Mark all documents as deleted
    try:
        docs = db.query(Document).filter(Document.chat_id == chat_id).all()
        for doc in docs:
            doc.status = "deleted"
        db.flush()
    except Exception as e:
        errors.append(f"doc_status: {e}")

    # 5) Mark chat as deleted
    try:
        mark_chat_deleted(db, chat_id)
        db.commit()
    except Exception as e:
        logger.error(f"Failed to mark chat deleted: {e}")
        db.rollback()
        errors.append(f"chat_mark: {e}")

    if errors:
        # Queue retry
        try:
            create_cleanup_job(db, "chat_delete", chat_id)
            db.commit()
        except Exception:
            db.rollback()
        logger.warning(f"Chat {chat_id} partially deleted, retry queued: {errors}")
        return False

    logger.info(f"Chat {chat_id} fully cascade deleted")
    return True


def cleanup_guest_session(db, session_id: str) -> bool:
    """Hard-delete all guest data for a session."""
    from backend.database.models import Chat

    guest_chats = db.query(Chat).filter(
        Chat.session_id == session_id,
        Chat.is_guest == True,
        Chat.status != "deleted",
    ).all()

    all_ok = True
    for chat in guest_chats:
        ok = delete_chat_cascade(db, chat.id)
        if not ok:
            all_ok = False

    return all_ok


def _delete_qdrant_for_chat(
    chat_id: str,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
) -> None:
    """Delete Qdrant points filtered by chat ownership."""
    from config.config import VECTOR_BACKEND
    if VECTOR_BACKEND != "qdrant":
        return

    from src.modules.QdrantStore import _qdrant_client, QDRANT_COLLECTION
    try:
        from qdrant_client.http import models
    except ImportError:
        return

    client = _qdrant_client()
    must_filters = [
        models.FieldCondition(
            key="chat_id",
            match=models.MatchValue(value=chat_id),
        )
    ]
    if user_id:
        must_filters.append(
            models.FieldCondition(
                key="user_id",
                match=models.MatchValue(value=user_id),
            )
        )
    elif session_id:
        must_filters.append(
            models.FieldCondition(
                key="session_id",
                match=models.MatchValue(value=session_id),
            )
        )

    client.delete(
        collection_name=QDRANT_COLLECTION,
        points_selector=models.FilterSelector(
            filter=models.Filter(must=must_filters)
        ),
        wait=False,
    )


def delete_qdrant_by_session(session_id: str) -> None:
    """Delete all Qdrant points for a guest session_id."""
    from config.config import VECTOR_BACKEND
    if VECTOR_BACKEND != "qdrant":
        return

    from src.modules.QdrantStore import delete_session_vectors
    delete_session_vectors(session_id)


# ── Background cleanup worker ───────────────────────────────────────────

class CleanupWorker:
    """Background thread that processes cleanup jobs and TTL-expires guest data."""

    def __init__(self, interval_seconds: int = 60):
        self.interval_seconds = interval_seconds
        self._stop_event = threading.Event()
        self._thread = None

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="cleanup-worker")
        self._thread.start()
        logger.info(f"CleanupWorker started (interval={self.interval_seconds}s)")

    def stop(self):
        if not self._thread:
            return
        self._stop_event.set()
        self._thread.join(timeout=10)
        logger.info("CleanupWorker stopped")

    def _run(self):
        while not self._stop_event.is_set():
            try:
                self._process_jobs()
            except Exception as e:
                logger.error(f"CleanupWorker error (jobs): {e}")
            try:
                self._ttl_sweep()
            except Exception as e:
                logger.error(f"CleanupWorker error (ttl): {e}")
            self._stop_event.wait(self.interval_seconds)

    def _process_jobs(self):
        db = get_session_local()()
        try:
            jobs = get_pending_cleanup_jobs(db, limit=10)
            for job in jobs:
                try:
                    job.status = "running"
                    job.updated_at = datetime.utcnow()
                    db.commit()

                    if job.job_type == "chat_delete":
                        delete_chat_cascade(db, job.target_id)
                    elif job.job_type == "guest_cleanup":
                        cleanup_guest_session(db, job.target_id)

                    complete_cleanup_job(db, job.id)
                    db.commit()
                except Exception as e:
                    db.rollback()
                    fail_cleanup_job(db, job.id, str(e))
                    db.commit()
                    logger.warning(f"Cleanup job {job.id} failed (attempt {job.attempts}): {e}")
        finally:
            db.close()

    def _ttl_sweep(self):
        """Delete stale guest chats past TTL."""
        db = get_session_local()()
        try:
            stale = get_stale_guest_chats(db, max_age_minutes=GUEST_TTL_MINUTES)
            for chat in stale:
                logger.info(f"TTL sweep: deleting guest chat {chat.id}")
                delete_chat_cascade(db, chat.id)
            if stale:
                logger.info(f"TTL sweep cleaned {len(stale)} stale guest chats")
        finally:
            db.close()


# Global instance
_cleanup_worker: Optional[CleanupWorker] = None


def start_cleanup_worker(interval_seconds: int = 60):
    global _cleanup_worker
    if _cleanup_worker is None:
        _cleanup_worker = CleanupWorker(interval_seconds)
    _cleanup_worker.start()


def stop_cleanup_worker():
    global _cleanup_worker
    if _cleanup_worker:
        _cleanup_worker.stop()

