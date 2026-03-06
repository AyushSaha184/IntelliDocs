"""FastAPI entrypoint for RAG backend."""

from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from backend.api.routes import router
from backend.database import init_db
from backend.services.cleanup_scheduler import start_cleanup_scheduler, stop_cleanup_scheduler
from config.config import CORS_ALLOWED_ORIGINS


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    # Startup
    init_db()  # Initialize database tables
    start_cleanup_scheduler(interval_minutes=10)  # Check every 10 min, delete sessions 15min+ old OR 30min+ idle
    yield
    # Shutdown
    stop_cleanup_scheduler()


app = FastAPI(title="RAG Backend", lifespan=lifespan)

# CORS – allow local dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOWED_ORIGINS if CORS_ALLOWED_ORIGINS else ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

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
