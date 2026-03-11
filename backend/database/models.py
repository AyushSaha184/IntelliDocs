"""Database models and session/engine setup."""

from datetime import datetime
from sqlalchemy import (
    Column,
    String,
    DateTime,
    Integer,
    Text,
    Index,
    text,
    create_engine,
)
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
    """Anonymous session model for RAG document processing."""

    __tablename__ = "sessions"

    session_id = Column(String(36), primary_key=True, index=True)
    filename = Column(String(255), nullable=False)
    file_size = Column(Integer, nullable=False)
    status = Column(String(50), default="processing")  # processing, ready, error
    created_at = Column(DateTime, default=datetime.utcnow)
    last_accessed = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    error_message = Column(Text, nullable=True)
    chunks_count = Column(Integer, default=0)

    def __repr__(self):
        return f"<Session(session_id={self.session_id}, status={self.status})>"


class Document(Base):
    __tablename__ = "documents"

    id = Column(String(36), primary_key=True)
    session_id = Column(String(36), nullable=False, index=True)
    filename = Column(String(500), nullable=False)
    file_size = Column(Integer, default=0)
    content_hash = Column(String(64), nullable=True)
    status = Column(String(20), default="pending", nullable=False)  # pending|ready|failed
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    def __repr__(self):
        return f"<Document(id={self.id}, filename={self.filename}, status={self.status})>"


# Database setup - PostgreSQL (lazy initialization)
_engine = None
_SessionLocal = None


def _is_usable_db_url(url: str) -> bool:
    """Check if a DATABASE_URL is actually usable (not localhost on remote hosts)."""
    if not url:
        return False
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
            db_url = _get_sqlite_url()
            print(f"[DB] Falling back to SQLite: {db_url}")
        else:
            db_url = raw_url

        if db_url.startswith("postgres://"):
            db_url = db_url.replace("postgres://", "postgresql://", 1)

        print(f"[DB] Using database: {db_url[:50]}...")
        _engine = create_engine(db_url, connect_args={"check_same_thread": False} if "sqlite" in db_url else {})
    return _engine


def get_session_local():
    """Get or create SessionLocal."""
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=get_engine())
    return _SessionLocal


def init_db():
    """Initialize database tables."""
    engine = get_engine()
    _migrate_schema(engine)
    Base.metadata.create_all(bind=engine)


def _migrate_schema(engine):
    """Rename legacy tables that conflict with the new simplified schema."""
    with engine.begin() as conn:
        # Rename old tables so create_all builds fresh ones
        for old_table in ("chats", "chat_messages", "cleanup_jobs"):
            try:
                result = conn.execute(text(
                    f"SELECT 1 FROM information_schema.tables WHERE table_name='{old_table}'"
                ))
                if result.fetchone():
                    conn.execute(text(f"ALTER TABLE {old_table} RENAME TO {old_table}_legacy"))
                    print(f"[DB] Migration: renamed {old_table} → {old_table}_legacy")
            except Exception as exc:
                print(f"[DB] Migration warning ({old_table}): {exc}")

        # Rename documents table if it has legacy columns (chat_id, user_id)
        try:
            result = conn.execute(text(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name='documents' AND column_name='chat_id'"
            ))
            if result.fetchone():
                conn.execute(text("ALTER TABLE documents RENAME TO documents_legacy_v2"))
                print("[DB] Migration: renamed old documents table → documents_legacy_v2")
        except Exception as exc:
            print(f"[DB] Migration warning (documents): {exc}")


def get_db():
    """Dependency for database sessions."""
    session_local = get_session_local()
    db = session_local()
    try:
        yield db
    finally:
        db.close()
