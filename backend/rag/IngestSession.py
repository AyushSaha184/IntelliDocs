"""Session-aware ingestion service."""

from pathlib import Path
from typing import Optional
from src.utils.Logger import get_logger
from main import RAGPipeline

logger = get_logger(__name__)


def ingest_documents_session(
    session_id: str,
    documents_dir: Path,
    chunks_dir: Path,
    vector_store_dir: Path,
    device: str = "cpu"
) -> int:
    """
    Ingest documents for a specific session.
    
    Returns:
        Number of chunks created
    """
    logger.info(f"Starting ingestion for session {session_id}")
    
    # Ensure directories exist
    documents_dir.mkdir(parents=True, exist_ok=True)
    chunks_dir.mkdir(parents=True, exist_ok=True)
    vector_store_dir.mkdir(parents=True, exist_ok=True)
    
    # Run pipeline for this session
    pipeline = RAGPipeline(
        documents_dir=str(documents_dir),
        chunks_dir=str(chunks_dir),
        vector_store_dir=str(vector_store_dir),
        device=device
    )
    
    pipeline.build_vector_store()
    
    # Count chunks
    chunks_count = 0
    for chunk_file in chunks_dir.glob("*_chunks.json"):
        import json
        with open(chunk_file, "r") as f:
            data = json.load(f)
            if isinstance(data, list):
                chunks_count += len(data)
            elif isinstance(data, dict):
                chunks_count += len(data)
    
    logger.info(f"Ingestion completed for session {session_id}: {chunks_count} chunks")
    return chunks_count
