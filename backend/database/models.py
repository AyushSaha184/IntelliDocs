"""Database models for session management."""

from datetime import datetime
from typing import Optional
from sqlalchemy import Column, String, DateTime, Integer, Text, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from project root
_env_path = Path(__file__).resolve().parent.parent.parent / ".env"
load_dotenv(dotenv_path=_env_path)

Base = declarative_base()


class Session(Base):
    """User session model for RAG document processing."""
    
    __tablename__ = "sessions"
    
    session_id = Column(String(36), primary_key=True, index=True)
    user_id = Column(String(64), nullable=True, index=True)
    filename = Column(String(255), nullable=False)
    file_size = Column(Integer, nullable=False)
    status = Column(String(50), default="processing")  # processing, ready, error
    created_at = Column(DateTime, default=datetime.utcnow)
    last_accessed = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    error_message = Column(Text, nullable=True)
    chunks_count = Column(Integer, default=0)
    
    def __repr__(self):
        return f"<Session(session_id={self.session_id}, status={self.status})>"


# Database setup - PostgreSQL (lazy initialization)
_engine = None
_SessionLocal = None


def _is_usable_db_url(url: str) -> bool:
    """Check if a DATABASE_URL is actually usable (not a localhost URL on a remote server)."""
    if not url:
        return False
    # On Render, a localhost PostgreSQL URL from .env won't work
    is_remote = os.getenv("RENDER") or os.getenv("PORT")
    if is_remote and ("localhost" in url or "127.0.0.1" in url):
        return False
    return True


def _get_sqlite_url() -> str:
    """Get SQLite fallback URL."""
    data_dir = Path(__file__).resolve().parent.parent.parent / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{data_dir}/sessions.db"


def get_engine():
    """Get or create database engine."""
    global _engine
    if _engine is None:
        raw_url = os.getenv("DATABASE_URL", "")
        print(f"[DB] DATABASE_URL from env: {'set (' + raw_url[:30] + '...)' if raw_url else 'NOT SET'}")

        if not _is_usable_db_url(raw_url):
            DB_URL = _get_sqlite_url()
            print(f"[DB] Falling back to SQLite: {DB_URL}")
        else:
            DB_URL = raw_url

        # Handle Render.com postgres:// URL (should be postgresql://)
        if DB_URL.startswith("postgres://"):
            DB_URL = DB_URL.replace("postgres://", "postgresql://", 1)

        print(f"[DB] Using database: {DB_URL[:50]}...")
        _engine = create_engine(DB_URL, connect_args={"check_same_thread": False} if "sqlite" in DB_URL else {})
    return _engine


def get_session_local():
    """Get or create SessionLocal."""
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=get_engine())
    return _SessionLocal


def init_db():
    """Initialize database tables."""
    Base.metadata.create_all(bind=get_engine())


def get_db():
    """Dependency for database sessions."""
    session_local = get_session_local()
    db = session_local()
    try:
        yield db
    finally:
        db.close()
