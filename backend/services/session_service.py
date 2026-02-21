"""Session management service for multi-user RAG system."""

import uuid
import shutil
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from sqlalchemy.orm import Session as DBSession
from backend.database.models import Session, get_db
from src.utils.Logger import get_logger
import threading
import json
import os

logger = get_logger(__name__)

# Get absolute project root (Enterprise-ai-assistant directory)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"


class SessionManager:
    """Manages user sessions with isolated storage.
    
    Each session gets fully isolated directories:
      data/sessions/{session_id}/documents/   ← Only this user's files
      data/sessions/{session_id}/chunks/
      data/sessions/{session_id}/vector_store/
    
    Documents are also copied to data/documents/ for the main CLI pipeline.
    
    Auto-cleanup rules:
      - Delete after 2 hours from creation (max session duration)
      - OR delete after 1 hour of inactivity (no queries)
    """
    
    BASE_STORAGE_DIR = DATA_DIR / "sessions"
    CENTRAL_DOCUMENTS_DIR = DATA_DIR / "documents"  # For main CLI pipeline
    MAX_SESSION_DURATION = timedelta(hours=2)   # Delete after 2 hours from creation
    INACTIVITY_TIMEOUT = timedelta(hours=1)     # Delete after 1 hour of no activity
    
    def __init__(self):
        logger.info(f"SessionManager initializing with base dir: {self.BASE_STORAGE_DIR}")
        self.BASE_STORAGE_DIR.mkdir(parents=True, exist_ok=True)
        self.CENTRAL_DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
    
    def create_session(self, filename: str, file_size: int, db: DBSession) -> str:
        """Create a new isolated session with its own document storage."""
        session_id = str(uuid.uuid4())
        
        # Create fully isolated session directories
        session_dir = self._get_session_dir(session_id)
        session_dir.mkdir(parents=True, exist_ok=True)
        (session_dir / "documents").mkdir(exist_ok=True)  # Session-isolated documents
        (session_dir / "chunks").mkdir(exist_ok=True)
        (session_dir / "vector_store").mkdir(exist_ok=True)
        
        # Create database record
        session = Session(
            session_id=session_id,
            filename=filename,
            file_size=file_size,
            status="processing",
            created_at=datetime.utcnow(),
            last_accessed=datetime.utcnow()
        )
        db.add(session)
        db.commit()
        
        logger.info(f"Created session {session_id} for file {filename}")
        return session_id
    
    def get_session(self, session_id: str, db: DBSession) -> Optional[Session]:
        """Retrieve session from database."""
        session = db.query(Session).filter(Session.session_id == session_id).first()
        if session:
            # Update last accessed time
            session.last_accessed = datetime.utcnow()
            db.commit()
        return session
    
    def update_session_status(
        self, 
        session_id: str, 
        status: str, 
        db: DBSession,
        error_message: Optional[str] = None,
        chunks_count: Optional[int] = None
    ):
        """Update session status."""
        session = db.query(Session).filter(Session.session_id == session_id).first()
        if session:
            session.status = status
            session.last_accessed = datetime.utcnow()
            if error_message:
                session.error_message = error_message
            if chunks_count is not None:
                session.chunks_count = chunks_count
            db.commit()
            logger.info(f"Updated session {session_id} status to {status}")
    
    def _get_session_dir(self, session_id: str) -> Path:
        """Get the session directory path."""
        return self.BASE_STORAGE_DIR / session_id
    
    def get_documents_dir(self, session_id: str) -> Path:
        """Get session-isolated documents directory (prevents cross-user mixup)."""
        return self._get_session_dir(session_id) / "documents"
    
    def get_central_documents_dir(self) -> Path:
        """Get central documents directory for main CLI pipeline."""
        return self.CENTRAL_DOCUMENTS_DIR
    
    def copy_to_central(self, session_id: str, filename: str):
        """Copy a session document to the central data/documents/ for the main pipeline.
        
        This keeps session isolation intact while also making docs available
        to the main CLI pipeline (python main.py build).
        """
        import shutil as sh
        session_doc = self.get_documents_dir(session_id) / filename
        if session_doc.exists():
            dest = self.CENTRAL_DOCUMENTS_DIR / filename
            # Avoid overwriting: add session prefix if file already exists
            if dest.exists():
                stem = dest.stem
                suffix = dest.suffix
                dest = self.CENTRAL_DOCUMENTS_DIR / f"{stem}_{session_id[:8]}{suffix}"
            sh.copy2(str(session_doc), str(dest))
            logger.info(f"Copied {filename} to central documents (for main pipeline)")
    
    def get_chunks_dir(self, session_id: str) -> Path:
        """Get session chunks directory."""
        return self._get_session_dir(session_id) / "chunks"
    
    def get_vector_store_dir(self, session_id: str) -> Path:
        """Get session vector store directory."""
        return self._get_session_dir(session_id) / "vector_store"
    
    def get_session_files(self, session_id: str) -> list:
        """List all files in a session's documents directory.
        
        Returns:
            List of dicts with 'name', 'size', and 'path' keys
        """
        docs_dir = self.get_documents_dir(session_id)
        if not docs_dir.exists():
            return []
        
        files = []
        for f in docs_dir.iterdir():
            if f.is_file():
                files.append({
                    "name": f.name,
                    "size": f.stat().st_size,
                    "path": str(f),
                })
        return files
    
    def save_chunks_metadata(self, session_id: str, metadata: Dict[str, Any]):
        """Save chunk metadata for a session."""
        chunks_dir = self.get_chunks_dir(session_id)
        chunks_dir.mkdir(parents=True, exist_ok=True)
        
        metadata_file = chunks_dir / "chunks_metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Saved metadata for session {session_id}")
    
    def load_chunks_metadata(self, session_id: str) -> Dict[str, Any]:
        """Load chunk metadata for a session."""
        metadata_file = self.get_chunks_dir(session_id) / "chunks_metadata.json"
        
        if not metadata_file.exists():
            logger.warning(f"No metadata found for session {session_id}")
            return {}
        
        with open(metadata_file, "r") as f:
            return json.load(f)
    
    def delete_session(self, session_id: str, db: DBSession):
        """Delete session and all associated data."""
        with self._lock:
            # Delete from database
            session = db.query(Session).filter(Session.session_id == session_id).first()
            if session:
                db.delete(session)
                db.commit()
            
            # Delete storage
            session_dir = self._get_session_dir(session_id)
            if session_dir.exists():
                shutil.rmtree(session_dir)
                logger.info(f"Deleted session storage for {session_id}")
    
    def cleanup_inactive_sessions(self, db: DBSession):
        """Clean up sessions based on two conditions:
        1. Sessions older than 15 minutes (max duration from creation)
        2. Sessions inactive for more than 30 minutes (no queries)
        """
        now = datetime.utcnow()
        max_age_cutoff = now - self.MAX_SESSION_DURATION
        inactivity_cutoff = now - self.INACTIVITY_TIMEOUT
        
        # Find sessions that meet either condition
        expired_sessions = db.query(Session).filter(
            # Either: Created more than 15 min ago
            (Session.created_at < max_age_cutoff) |
            # Or: No activity for 30 min
            (Session.last_accessed < inactivity_cutoff)
        ).all()
        
        for session in expired_sessions:
            try:
                age_minutes = (now - session.created_at).total_seconds() / 60
                inactive_minutes = (now - session.last_accessed).total_seconds() / 60
                
                reason = ""
                if session.created_at < max_age_cutoff:
                    reason = f"max duration exceeded ({age_minutes:.1f} min old)"
                else:
                    reason = f"inactivity timeout ({inactive_minutes:.1f} min idle)"
                
                logger.info(f"Cleaning up session {session.session_id}: {reason}")
                self.delete_session(session.session_id, db)
            except Exception as e:
                logger.error(f"Error cleaning up session {session.session_id}: {e}")
        
        if expired_sessions:
            logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")


# Global session manager instance
_session_manager: Optional[SessionManager] = None


def get_session_manager() -> SessionManager:
    """Get the global session manager instance."""
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionManager()
    return _session_manager
