"""Database models for chats, messages, and document tracking.

These models map to Supabase/Postgres tables that must be created manually.
See DELIVERABLES.md for the required DB schema checklist.
"""

from datetime import datetime
from typing import Optional
from sqlalchemy import (
    Column, String, DateTime, Integer, Text, Boolean, ForeignKey,
    Index, text as sa_text,
)
from sqlalchemy.orm import relationship
from backend.database.models import Base


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
