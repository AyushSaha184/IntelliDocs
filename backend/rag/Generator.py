"""Generator wrapper for backend architecture."""

from typing import Optional
from src.modules.QueryGeneration import QueryResult
from backend.services.rag_service import get_query_handler


def generate_answer(
    question: str,
    query_result: QueryResult,
    system_prompt: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None
) -> QueryResult:
    handler = get_query_handler()
    return handler.generate_response(
        query_result=query_result,
        system_prompt=system_prompt,
        temperature=temperature,
        max_tokens=max_tokens
    )
