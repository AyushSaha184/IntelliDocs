"""Ingestion placeholder wired to existing pipeline modules."""

from src.utils.Logger import get_logger
from main import RAGPipeline

logger = get_logger(__name__)


def ingest_documents(
    documents_dir: str = "data/documents",
    chunks_dir: str = "data/chunks",
    vector_store_dir: str = "data/vector_store",
    device: str = "cpu"
) -> None:
    pipeline = RAGPipeline(
        documents_dir=documents_dir,
        chunks_dir=chunks_dir,
        vector_store_dir=vector_store_dir,
        device=device
    )
    pipeline.run_complete_pipeline()
    logger.info("Ingestion completed")
