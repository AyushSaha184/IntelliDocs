"""Database models and session/engine setup."""

from datetime import datetime
from sqlalchemy import (
    Column,
    String,
    DateTime,
    Integer,
    Text,
    Boolean,
    ForeignKey,
    Index,
    text,
    create_engine,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
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


class Chat(Base):
    __tablename__ = "chats"

    id = Column(String(36), primary_key=True)
    user_id = Column(String(64), nullable=True, index=True)
    session_id = Column(String(36), nullable=True, index=True)
    title = Column(String(500), nullable=False, default="New Chat")
    is_guest = Column(Boolean, default=False, nullable=False)
    status = Column(String(20), default="active", nullable=False)  # active|deleting|deleted
    version = Column(Integer, default=1, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    messages = relationship("ChatMessage", back_populates="chat", cascade="all, delete-orphan")
    documents = relationship("Document", back_populates="chat", cascade="all, delete-orphan")

    __table_args__ = (
        Index("ix_chats_user_updated", "user_id", "updated_at"),
    )

    def __repr__(self):
        return f"<Chat(id={self.id}, title={self.title}, status={self.status})>"


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id = Column(String(36), primary_key=True)
    chat_id = Column(String(36), ForeignKey("chats.id", ondelete="CASCADE"), nullable=False)
    role = Column(String(20), nullable=False)  # user|assistant|system
    content = Column(Text, nullable=False)
    metadata_json = Column(Text, default="{}")  # JSON string
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    chat = relationship("Chat", back_populates="messages")

    __table_args__ = (
        Index("ix_chat_messages_chat_created", "chat_id", "created_at"),
    )

    def __repr__(self):
        return f"<ChatMessage(id={self.id}, role={self.role})>"


class Document(Base):
    __tablename__ = "documents"

    id = Column(String(36), primary_key=True)
    chat_id = Column(String(36), ForeignKey("chats.id", ondelete="CASCADE"), nullable=False)
    user_id = Column(String(64), nullable=True)
    session_id = Column(String(36), nullable=True)
    filename = Column(String(500), nullable=False)
    file_size = Column(Integer, default=0)
    is_guest = Column(Boolean, default=False, nullable=False)
    content_hash = Column(String(64), nullable=True)
    status = Column(String(20), default="pending", nullable=False)  # pending|ready|failed|deleting|deleted
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    chat = relationship("Chat", back_populates="documents")

    __table_args__ = (
        Index("ix_documents_chat_id", "chat_id"),
        Index("ix_documents_user_id", "user_id"),
        Index("ix_documents_user_hash", "user_id", "content_hash"),
    )

    def __repr__(self):
        return f"<Document(id={self.id}, filename={self.filename}, status={self.status})>"


class CleanupJob(Base):
    """Tracks retryable cleanup tasks for cascading deletes."""

    __tablename__ = "cleanup_jobs"

    id = Column(String(36), primary_key=True)
    job_type = Column(String(50), nullable=False)  # chat_delete|guest_cleanup|orphan_sweep
    target_id = Column(String(36), nullable=False)  # chat_id or session_id
    status = Column(String(20), default="pending", nullable=False)  # pending|running|completed|failed
    attempts = Column(Integer, default=0, nullable=False)
    max_attempts = Column(Integer, default=5, nullable=False)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    def __repr__(self):
        return f"<CleanupJob(id={self.id}, type={self.job_type}, status={self.status})>"


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
    Base.metadata.create_all(bind=engine)
    _drop_document_dedupe_indexes(engine)


def _drop_document_dedupe_indexes(engine):
    """Remove legacy dedupe indexes so same document can be uploaded multiple times."""
    statements = [
        "DROP INDEX IF EXISTS uq_documents_active_user_hash",
        "DROP INDEX IF EXISTS uq_documents_active_session_hash",
    ]

    with engine.begin() as conn:
        for stmt in statements:
            try:
                conn.execute(text(stmt))
            except Exception as exc:
                print(f"[DB] Warning: could not drop dedupe index: {exc}")


def get_db():
    """Dependency for database sessions."""
    session_local = get_session_local()
    db = session_local()
    try:
        yield db
    finally:
        db.close()
