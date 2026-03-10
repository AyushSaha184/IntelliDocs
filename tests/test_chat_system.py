"""Tests for chat service limits and access control."""

import hashlib
import threading
import uuid

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from backend.database.models import Base
from backend.database.chat_models import Chat, Document
from backend.services.chat_service import (
    ACCOUNT_DOC_LIMIT,
    GUEST_DOC_LIMIT,
    PER_CHAT_DOC_LIMIT,
    LimitError,
    check_and_register_document,
    compute_content_hash,
    create_chat,
    get_chat,
    get_messages,
    get_user_chats,
    mark_chat_deleting,
    rename_chat,
    verify_chat_access,
    add_message,
)


_engine = create_engine(
    "sqlite://",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
_Session = sessionmaker(bind=_engine)
Base.metadata.create_all(bind=_engine)


def _uid() -> str:
    return str(uuid.uuid4())


@pytest.fixture()
def db():
    session = _Session()
    yield session
    session.rollback()
    session.close()


def _register_doc(db, chat_id: str, user_id: str, session_id: str, i: int, is_guest: bool = False):
    payload = f"content-{uuid.uuid4()}".encode()
    return check_and_register_document(
        db,
        chat_id=chat_id,
        user_id=user_id,
        session_id=session_id,
        is_guest=is_guest,
        filename=f"file_{i}.pdf",
        file_size=len(payload),
        content_hash=hashlib.sha256(payload).hexdigest(),
    )


def test_guest_limit_3_docs(db):
    uid = _uid()
    sid = _uid()
    chat = create_chat(db, user_id=uid, session_id=sid, is_guest=True)
    for i in range(GUEST_DOC_LIMIT):
        _register_doc(db, chat.id, uid, sid, i, is_guest=True)

    with pytest.raises(LimitError) as e:
        _register_doc(db, chat.id, uid, sid, 999, is_guest=True)
    assert e.value.code == "GUEST_LIMIT_REACHED"


def test_per_chat_limit_15_docs(db):
    uid = _uid()
    sid = _uid()
    chat = create_chat(db, user_id=uid, session_id=sid)
    for i in range(PER_CHAT_DOC_LIMIT):
        _register_doc(db, chat.id, uid, sid, i)

    with pytest.raises(LimitError) as e:
        _register_doc(db, chat.id, uid, sid, 999)
    assert e.value.code == "PER_CHAT_LIMIT_REACHED"


def test_account_limit_40_docs(db):
    uid = _uid()
    sid = _uid()

    created = 0
    while created < ACCOUNT_DOC_LIMIT:
        chat = create_chat(db, user_id=uid, session_id=sid)
        batch = min(PER_CHAT_DOC_LIMIT, ACCOUNT_DOC_LIMIT - created)
        for i in range(batch):
            _register_doc(db, chat.id, uid, sid, created + i)
        created += batch

    overflow_chat = create_chat(db, user_id=uid, session_id=sid)
    with pytest.raises(LimitError) as e:
        _register_doc(db, overflow_chat.id, uid, sid, 10000)
    assert e.value.code == "ACCOUNT_LIMIT_REACHED"


def test_dedup_same_chat_rejected(db):
    uid = _uid()
    sid = _uid()
    chat = create_chat(db, user_id=uid, session_id=sid)
    h = hashlib.sha256(b"same-content").hexdigest()

    check_and_register_document(
        db,
        chat_id=chat.id,
        user_id=uid,
        session_id=sid,
        is_guest=False,
        filename="a.pdf",
        file_size=10,
        content_hash=h,
    )

    with pytest.raises(LimitError) as e:
        check_and_register_document(
            db,
            chat_id=chat.id,
            user_id=uid,
            session_id=sid,
            is_guest=False,
            filename="b.pdf",
            file_size=10,
            content_hash=h,
        )
    assert e.value.code == "DUPLICATE_DOCUMENT"


def test_access_and_rename(db):
    owner = _uid()
    other = _uid()
    sid = _uid()
    chat = create_chat(db, user_id=owner, session_id=sid, title="old")

    with pytest.raises(LimitError):
        verify_chat_access(db, chat.id, user_id=other)

    verify_chat_access(db, chat.id, user_id=owner)
    updated = rename_chat(db, chat.id, "new", chat.version)
    assert updated.title == "new"


def test_mark_deleting_and_messages(db):
    uid = _uid()
    sid = _uid()
    chat = create_chat(db, user_id=uid, session_id=sid)
    add_message(db, chat.id, "user", "hello")
    add_message(db, chat.id, "assistant", "world")

    assert len(get_messages(db, chat.id)) == 2
    mark_chat_deleting(db, chat.id)
    db.refresh(chat)
    assert chat.status == "deleting"


def test_concurrent_guest_last_slot(db):
    uid = _uid()
    sid = _uid()
    chat = create_chat(db, user_id=uid, session_id=sid, is_guest=True)
    for i in range(GUEST_DOC_LIMIT - 1):
        _register_doc(db, chat.id, uid, sid, i, is_guest=True)
    db.commit()

    results = {"ok": 0, "limit": 0}
    lock = threading.Lock()

    def worker(i: int):
        s = _Session()
        try:
            _register_doc(s, chat.id, uid, sid, i, is_guest=True)
            s.commit()
            with lock:
                results["ok"] += 1
        except LimitError:
            s.rollback()
            with lock:
                results["limit"] += 1
        finally:
            s.close()

    t1 = threading.Thread(target=worker, args=(101,))
    t2 = threading.Thread(target=worker, args=(102,))
    t1.start(); t2.start(); t1.join(); t2.join()

    total = db.query(Document).filter(Document.chat_id == chat.id).count()
    assert total <= GUEST_DOC_LIMIT


def test_compute_hash_is_deterministic():
    assert compute_content_hash(b"x") == compute_content_hash(b"x")


def test_chat_crud_basics(db):
    uid = _uid()
    sid = _uid()
    a = create_chat(db, user_id=uid, session_id=sid, title="A")
    create_chat(db, user_id=uid, session_id=sid, title="B")

    assert get_chat(db, a.id).title == "A"
    assert len(get_user_chats(db, uid)) >= 2