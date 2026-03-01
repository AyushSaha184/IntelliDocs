"""Retriever wrapper for backend architecture."""

from src.modules.QueryGeneration import QueryResult
from backend.services.rag_service import get_query_handler


def retrieve_chunks(question: str, top_k: int = 5) -> QueryResult:
    handler = get_query_handler()
    return handler.process_query(query=question, top_k=top_k)
