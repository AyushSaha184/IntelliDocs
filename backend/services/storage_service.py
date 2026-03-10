"""Session document storage backends (local filesystem or Supabase Storage)."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional

from src.utils.Logger import get_logger

logger = get_logger(__name__)


class BaseSessionStorage:
    def save_document(self, session_id: str, filename: str, content: bytes) -> None:
        raise NotImplementedError

    def document_exists(self, session_id: str, filename: str) -> bool:
        raise NotImplementedError

    def list_documents(self, session_id: str) -> List[Dict]:
        raise NotImplementedError

    def materialize_documents(self, session_id: str, target_dir: Path) -> int:
        raise NotImplementedError

    def delete_session(self, session_id: str) -> None:
        raise NotImplementedError


class LocalSessionStorage(BaseSessionStorage):
    def __init__(self, base_storage_dir: Path):
        self.base_storage_dir = Path(base_storage_dir)
        self.base_storage_dir.mkdir(parents=True, exist_ok=True)

    def _documents_dir(self, session_id: str) -> Path:
        d = self.base_storage_dir / session_id / "documents"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _next_available_name(self, docs_dir: Path, filename: str) -> str:
        path = docs_dir / filename
        if not path.exists():
            return filename
        stem = path.stem
        suffix = path.suffix
        counter = 2
        while True:
            candidate = f"{stem} ({counter}){suffix}"
            if not (docs_dir / candidate).exists():
                return candidate
            counter += 1

    def save_document(self, session_id: str, filename: str, content: bytes) -> None:
        docs_dir = self._documents_dir(session_id)
        path = docs_dir / self._next_available_name(docs_dir, filename)
        with open(path, "wb") as f:
            f.write(content)

    def document_exists(self, session_id: str, filename: str) -> bool:
        return (self._documents_dir(session_id) / filename).exists()

    def list_documents(self, session_id: str) -> List[Dict]:
        docs_dir = self._documents_dir(session_id)
        files = []
        for f in docs_dir.iterdir():
            if f.is_file():
                files.append({"name": f.name, "size": f.stat().st_size, "path": str(f)})
        return files

    def materialize_documents(self, session_id: str, target_dir: Path) -> int:
        docs_dir = self._documents_dir(session_id)
        target_dir.mkdir(parents=True, exist_ok=True)
        count = 0
        for f in docs_dir.iterdir():
            if f.is_file():
                dest = target_dir / f.name
                if f.resolve() != dest.resolve():
                    shutil.copy2(str(f), str(dest))
                count += 1
        return count

    def delete_session(self, session_id: str) -> None:
        # Local session directories are deleted by SessionManager.
        return


class SupabaseSessionStorage(BaseSessionStorage):
    def __init__(self, supabase_url: str, service_role_key: str, bucket: str):
        self.bucket = bucket
        self._client = None
        try:
            from supabase import create_client
            self._client = create_client(supabase_url, service_role_key)
        except Exception as e:
            logger.error(f"Failed to initialize Supabase storage client: {e}")
            raise

    def _prefix(self, session_id: str) -> str:
        return f"sessions/{session_id}/documents"

    def _object_path(self, session_id: str, filename: str) -> str:
        return f"{self._prefix(session_id)}/{filename}"

    def _next_available_name(self, session_id: str, filename: str) -> str:
        if not self.document_exists(session_id, filename):
            return filename
        p = Path(filename)
        stem = p.stem
        suffix = p.suffix
        counter = 2
        while True:
            candidate = f"{stem} ({counter}){suffix}"
            if not self.document_exists(session_id, candidate):
                return candidate
            counter += 1

    def save_document(self, session_id: str, filename: str, content: bytes) -> None:
        object_path = self._object_path(session_id, self._next_available_name(session_id, filename))

        self._client.storage.from_(self.bucket).upload(
            path=object_path,
            file=content,
            file_options={"upsert": "false", "content-type": "application/octet-stream"},
        )

    def document_exists(self, session_id: str, filename: str) -> bool:
        docs = self.list_documents(session_id)
        return any(d.get("name") == filename for d in docs)

    def list_documents(self, session_id: str) -> List[Dict]:
        prefix = self._prefix(session_id)
        items = self._client.storage.from_(self.bucket).list(path=prefix) or []
        files: List[Dict] = []
        for it in items:
            name = it.get("name")
            if not name:
                continue
            meta = it.get("metadata") or {}
            size = meta.get("size") or it.get("size") or 0
            files.append(
                {
                    "name": name,
                    "size": int(size) if str(size).isdigit() else 0,
                    "path": self._object_path(session_id, name),
                }
            )
        return files

    def materialize_documents(self, session_id: str, target_dir: Path) -> int:
        target_dir.mkdir(parents=True, exist_ok=True)
        # Clean stale local files before download.
        for p in target_dir.iterdir():
            if p.is_file():
                p.unlink()

        docs = self.list_documents(session_id)
        count = 0
        for d in docs:
            name = d["name"]
            object_path = self._object_path(session_id, name)
            content = self._client.storage.from_(self.bucket).download(object_path)
            with open(target_dir / name, "wb") as f:
                f.write(content)
            count += 1
        return count

    def delete_session(self, session_id: str) -> None:
        docs = self.list_documents(session_id)
        if not docs:
            return
        paths = [self._object_path(session_id, d["name"]) for d in docs if d.get("name")]
        if paths:
            self._client.storage.from_(self.bucket).remove(paths)


def create_session_storage(base_storage_dir: Path) -> BaseSessionStorage:
    backend = os.getenv("STORAGE_BACKEND", "local").lower()
    if backend == "supabase":
        supabase_url = os.getenv("SUPABASE_URL", "").strip()
        service_role_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip()
        bucket = os.getenv("SUPABASE_STORAGE_BUCKET", "rag-documents").strip()
        if not supabase_url or not service_role_key:
            logger.warning(
                "STORAGE_BACKEND=supabase but SUPABASE_URL/SUPABASE_SERVICE_ROLE_KEY missing; falling back to local"
            )
            return LocalSessionStorage(base_storage_dir)
        logger.info(f"Using Supabase Storage backend (bucket={bucket})")
        return SupabaseSessionStorage(supabase_url, service_role_key, bucket)

    logger.info("Using local filesystem storage backend")
    return LocalSessionStorage(base_storage_dir)
