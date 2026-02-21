"""
Query Handler Module - Processes user queries through the RAG pipeline

Handles:
1. User input validation
2. Query embedding
3. Retrieval from vector store
4. Context preparation for LLM
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import json
from datetime import datetime
from src.utils.Logger import get_logger
from src.modules.LLM import BaseLLM, LLMResponse
from src.modules.QueryCache import get_retrieval_cache, get_llm_cache

logger = get_logger(__name__)


@dataclass
class QueryResult:
    """Result from query processing"""
    query: str
    retrieved_chunks: List[str]
    metadata: List[Dict]
    query_embedding: Optional[list] = None
    retrieval_scores: Optional[List[float]] = None
    timestamp: str = None
    llm_response: Optional[str] = None
    llm_metadata: Optional[Dict] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()


class QueryHandler:
    """Handles user queries and retrieves relevant context"""
    
    def __init__(self, retriever, embedding_service, llm: Optional[BaseLLM] = None, top_k: int = 5, session_id: str = "default_session"):
        """Initialize query handler
        
        Args:
            retriever: RAGRetriever instance for retrieving chunks
            embedding_service: Embedding service for encoding queries
            llm: Optional LLM instance for response generation
            top_k: Number of top results to retrieve
            session_id: Identifier for user session to isolate caches
        """
        self.retriever = retriever
        self.embedding_service = embedding_service
        self.llm = llm
        self.top_k = top_k
        self.session_id = session_id
        self.query_history: List[QueryResult] = []
        logger.info(f"QueryHandler initialized for session {session_id} with top_k={top_k}, LLM={'enabled' if llm else 'disabled'}")
    
    def validate_query(self, query: str) -> Tuple[bool, str]:
        """Validate user query
        
        Args:
            query: User input query
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not query:
            return False, "Query cannot be empty"
        
        if isinstance(query, str):
            query = query.strip()
        
        if len(query) < 3:
            return False, "Query must be at least 3 characters"
        
        if len(query) > 5000:
            return False, "Query exceeds maximum length of 5000 characters"
        
        return True, ""
    
    def process_query(self, query: str, top_k: Optional[int] = None) -> QueryResult:
        """Process user query and retrieve relevant chunks
        
        Args:
            query: User input query
            top_k: Override default top_k for this query
            
        Returns:
            QueryResult object with retrieved chunks and metadata
            
        Raises:
            ValueError: If query validation fails
        """
        is_valid, error_msg = self.validate_query(query)
        if not is_valid:
            raise ValueError(f"Invalid query: {error_msg}")
        
        try:
            query = query.strip()
            k = top_k or self.top_k
            
            # 1. Check Retrieval Cache first
            cached_result = get_retrieval_cache().get_cache(self.session_id, query, k)
            if cached_result:
                query_result = QueryResult(
                    query=query,
                    retrieved_chunks=cached_result.get("retrieved_chunks", []),
                    metadata=cached_result.get("metadata", []),
                    retrieval_scores=cached_result.get("retrieval_scores", [])
                )
                self.query_history.append(query_result)
                return query_result

            logger.info(f"Processing query: {query[:100]}...")
            
            # Embed the query
            query_embedding_result = self.embedding_service.embed_text(query)
            query_embedding = query_embedding_result.embedding
            
            # Retrieve relevant chunks
            retrieval_results = self.retriever.retrieve(query, k=k)
            
            # Extract chunks and metadata
            retrieved_chunks = []
            metadata_list = []
            scores = []
            
            for result in retrieval_results:
                retrieved_chunks.append(result.text)
                
                # Extract metadata, prioritizing the retriever's chunk dictionary if available
                chunk_meta = {}
                if hasattr(self.retriever, 'chunks') and self.retriever.chunks:
                    chunk_meta = self.retriever.chunks.get(result.chunk_id, {})
                elif result.metadata:
                    chunk_meta = result.metadata

                doc_name = chunk_meta.get("document_name", "Unknown Document")
                page_num = chunk_meta.get("page_number", "Unknown Page")
                
                metadata_list.append({
                    "chunk_id": result.chunk_id,
                    "document_id": result.document_id,
                    "document_name": doc_name,
                    "page_number": page_num,
                    "distance": float(result.distance),
                    "similarity_score": float(1.0 - result.distance)
                })
                scores.append(1.0 - result.distance)
            
            # Create result object
            query_result = QueryResult(
                query=query,
                retrieved_chunks=retrieved_chunks,
                metadata=metadata_list,
                query_embedding=query_embedding.tolist() if hasattr(query_embedding, 'tolist') else query_embedding,
                retrieval_scores=scores
            )
            
            # Store in cache (stripping embedding to save space)
            cache_data = {
                "retrieved_chunks": retrieved_chunks,
                "metadata": metadata_list,
                "retrieval_scores": scores
            }
            get_retrieval_cache().set_cache(self.session_id, query, k, cache_data)
            
            # Store in history
            self.query_history.append(query_result)
            
            logger.info(f"Query processed successfully. Retrieved {len(retrieved_chunks)} chunks")
            
            return query_result
            
        except Exception as e:
            logger.error(f"Error processing query: {e}", exc_info=True)
            raise
    
    def format_context(self, query_result: QueryResult, include_scores: bool = False) -> str:
        """Format retrieved context for LLM consumption
        
        Args:
            query_result: QueryResult object
            include_scores: Whether to include similarity scores
            
        Returns:
            Formatted context string
        """
        context_parts = []
        
        for i, (chunk, meta) in enumerate(zip(query_result.retrieved_chunks, query_result.metadata), 1):
            part = f"[Context {i}]\n{chunk}"
            
            if include_scores:
                score = query_result.retrieval_scores[i - 1] if query_result.retrieval_scores else 0
                part += f"\n(Relevance: {score:.2f})"
            
            context_parts.append(part)
        
        return "\n\n".join(context_parts)
    
    def get_query_history(self, limit: Optional[int] = None) -> List[Dict]:
        """Get query history
        
        Args:
            limit: Maximum number of queries to return
            
        Returns:
            List of query history as dictionaries
        """
        history = self.query_history
        
        if limit:
            history = history[-limit:]
        
        return [
            {
                "query": q.query,
                "timestamp": q.timestamp,
                "num_results": len(q.retrieved_chunks),
                "avg_score": sum(q.retrieval_scores) / len(q.retrieval_scores) if q.retrieval_scores else 0
            }
            for q in history
        ]
    
    def clear_history(self) -> None:
        """Clear query history"""
        self.query_history = []
        logger.info("Query history cleared")
    
    def generate_response(
        self,
        query_result: QueryResult,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> QueryResult:
        """Generate LLM response based on retrieved context
        
        Args:
            query_result: QueryResult with retrieved chunks
            system_prompt: Optional custom system prompt
            temperature: Optional temperature override
            max_tokens: Optional max tokens override
            
        Returns:
            Updated QueryResult with LLM response
            
        Raises:
            RuntimeError: If no LLM is configured
        """
        if not self.llm:
            raise RuntimeError(
                "No LLM configured. Initialize QueryHandler with an LLM instance to generate responses."
            )
        
        try:
            # 1. Check LLM Cache first
            cached_llm = get_llm_cache().get_cache(self.session_id, query_result.query, query_result.retrieved_chunks)
            if cached_llm:
                query_result.llm_response = cached_llm.get("llm_response")
                query_result.llm_metadata = cached_llm.get("llm_metadata")
                return query_result

            # Format context from retrieved chunks
            context = self.format_context(query_result, include_scores=False)
            
            # Create RAG prompt
            prompt = self.llm.create_rag_prompt(
                query=query_result.query,
                context=context,
                system_prompt=system_prompt
            )
            
            logger.info(f"Generating LLM response for query: {query_result.query[:100]}...")
            
            # Generate response
            llm_response = self.llm.generate(
                prompt,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            import re
            # Strip reasoning blocks typically generated by logic-heavy models like DeepSeek
            final_response = re.sub(r'<think>.*?</think>', '', llm_response.response, flags=re.DOTALL).strip()
            
            # Update query result with LLM response
            query_result.llm_response = final_response
            query_result.llm_metadata = {
                "model": llm_response.model,
                "prompt_tokens": llm_response.prompt_tokens,
                "completion_tokens": llm_response.completion_tokens,
                "total_tokens": llm_response.total_tokens,
                "metadata": llm_response.metadata
            }
            
            logger.info(f"LLM response generated successfully (tokens: {llm_response.total_tokens})")
            
            # Store in LLM Cache
            get_llm_cache().set_cache(
                self.session_id, 
                query_result.query, 
                query_result.retrieved_chunks, 
                {
                    "llm_response": query_result.llm_response,
                    "llm_metadata": query_result.llm_metadata
                }
            )
            
            return query_result
            
        except Exception as e:
            logger.error(f"Error generating LLM response: {e}", exc_info=True)
            raise
    
    def process_query_with_response(
        self,
        query: str,
        top_k: Optional[int] = None,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> QueryResult:
        """Process query and generate LLM response in one call
        
        Args:
            query: User input query
            top_k: Override default top_k
            system_prompt: Optional custom system prompt
            temperature: Optional temperature override
            max_tokens: Optional max tokens override
            
        Returns:
            QueryResult with retrieved chunks and LLM response
        """
        # First retrieve relevant chunks
        query_result = self.process_query(query, top_k=top_k)
        
        # Generate LLM response if LLM is configured
        if self.llm:
            query_result = self.generate_response(
                query_result,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens
            )
        
        return query_result
