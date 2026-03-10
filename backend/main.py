"""FastAPI entrypoint for RAG backend."""

from collections import defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path
import threading
import time

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from backend.api.routes import router
from backend.database import init_db
from backend.services.cleanup_scheduler import start_cleanup_scheduler, stop_cleanup_scheduler
from config.config import (
    CORS_ALLOWED_ORIGINS,
    RATE_LIMIT_REQUESTS,
    RATE_LIMIT_WINDOW_SECONDS,
)

# Load environment variables from .env file
load_dotenv()


class InMemoryRateLimiter:
    """Simple fixed-window rate limiter keyed by client IP."""

    def __init__(self, max_requests: int, window_seconds: int):
        self.max_requests = max(1, int(max_requests))
        self.window_seconds = max(1, int(window_seconds))
        self._events = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, key: str) -> bool:
        now = time.time()
        cutoff = now - self.window_seconds
        with self._lock:
            q = self._events[key]
            while q and q[0] < cutoff:
                q.popleft()
            if len(q) >= self.max_requests:
                return False
            q.append(now)
            return True


_limiter = InMemoryRateLimiter(RATE_LIMIT_REQUESTS, RATE_LIMIT_WINDOW_SECONDS)


def _client_ip(request: Request) -> str:
    # Respect proxy header first, then socket address.
    xff = request.headers.get("x-forwarded-for", "")
    if xff:
        return xff.split(",")[0].strip()
    if request.client and request.client.host:
        return request.client.host
    return "unknown"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    init_db()
    start_cleanup_scheduler(interval_minutes=10)
    yield
    stop_cleanup_scheduler()


app = FastAPI(title="RAG Backend", lifespan=lifespan)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOWED_ORIGINS if CORS_ALLOWED_ORIGINS else ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    # Skip preflight and health endpoint from throttling.
    if request.method == "OPTIONS" or request.url.path == "/api/health":
        return await call_next(request)

    ip = _client_ip(request)
    if not _limiter.allow(ip):
        return JSONResponse(
            status_code=429,
            content={
                "detail": "Rate limit exceeded",
                "limit": RATE_LIMIT_REQUESTS,
                "window_seconds": RATE_LIMIT_WINDOW_SECONDS,
            },
        )
    return await call_next(request)


# API routes under /api prefix
app.include_router(router, prefix="/api")

# Serve static frontend files
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend" / "static"
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

    @app.get("/{full_path:path}")
    async def serve_frontend(full_path: str):
        """Serve the SPA index.html for all non-API routes."""
        file_path = FRONTEND_DIR / full_path
        if file_path.exists() and file_path.is_file():
            return FileResponse(file_path)
        return FileResponse(FRONTEND_DIR / "index.html")
