"""LLM Module - OpenRouter Integration for RAG."""

from typing import Optional, Dict, List
from dataclasses import dataclass
from abc import ABC, abstractmethod
import os
import requests
import json
from src.utils.Logger import get_logger

try:
    from huggingface_hub import InferenceClient
    HUGGINGFACE_AVAILABLE = True
except ImportError:
    HUGGINGFACE_AVAILABLE = False

logger = get_logger(__name__)


@dataclass
class LLMResponse:
    """Response from LLM generation"""
    response: str
    model: str
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    metadata: Optional[Dict] = None


class BaseLLM(ABC):
    """Base class for LLM implementations"""
    
    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name
        self.config = kwargs
        logger.info(f"Initialized {self.__class__.__name__} with model: {model_name}")
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate response from prompt"""
        pass
    
    def create_rag_prompt(
        self,
        query: str,
        context: str,
        system_prompt: Optional[str] = None
    ) -> str:
        """Create RAG prompt with query and context
        
        Args:
            query: User query
            context: Retrieved context from documents
            system_prompt: Optional custom system prompt
            
        Returns:
            Formatted prompt string
        """
        default_system = (
            "You are a helpful AI assistant. Answer the user's question based on the provided context. "
            "If the context doesn't contain relevant information, say so honestly. "
            "Be concise and accurate. Do NOT cite sources inline or use bracketed citations like [Context 1]. "
            "The system will automatically attach the source documents to your response."
        )
        
        system = system_prompt or default_system
        
        prompt = f"""{system}

Context:
{context}

User Question: {query}

Answer:"""
        
        return prompt


class OpenRouterLLM(BaseLLM):
    """OpenRouter API LLM implementation (OpenAI-compatible)."""

    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = 1000,
        site_url: Optional[str] = None,
        site_name: Optional[str] = None,
        **kwargs
    ):
        super().__init__(model_name, **kwargs)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        
        # Get API key from parameter or environment
        self.api_key = api_key or os.getenv('OPENROUTER_API_KEY')
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY is required. Set it in .env or pass as parameter.")
        
        # Optional ranking metadata
        self.site_url = site_url or os.getenv('OPENROUTER_SITE_URL', '')
        self.site_name = site_name or os.getenv('OPENROUTER_SITE_NAME', '')
        
        logger.info(f"OpenRouter client initialized with model: {model_name}")
        logger.info(f"Using API key (length: {len(self.api_key)})")

    def generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate response using OpenRouter API (OpenAI-compatible)."""
        try:
            temp = temperature if temperature is not None else self.temperature
            tokens = max_tokens if max_tokens is not None else self.max_tokens

            # Prepare headers
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            
            # Add optional ranking headers
            if self.site_url:
                headers["HTTP-Referer"] = self.site_url
            if self.site_name:
                headers["X-Title"] = self.site_name

            # Prepare request payload (OpenAI-compatible format)
            payload = {
                "model": self.model_name,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": temp,
                "max_tokens": tokens,
                "reasoning": {"enabled": True},
            }

            # Make API request
            response = requests.post(
                url=self.base_url,
                headers=headers,
                data=json.dumps(payload),
                timeout=60
            )
            response.raise_for_status()
            
            # Parse response
            result = response.json()
            
            # Extract content
            content = result.get('choices', [{}])[0].get('message', {}).get('content', '')
            
            # Extract token usage
            usage = result.get('usage', {})
            prompt_tokens = usage.get('prompt_tokens')
            completion_tokens = usage.get('completion_tokens')
            total_tokens = usage.get('total_tokens')
            
            return LLMResponse(
                response=content,
                model=self.model_name,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                metadata={
                    "finish_reason": result.get('choices', [{}])[0].get('finish_reason'),
                    "provider": "openrouter"
                }
            )

        except requests.exceptions.RequestException as e:
            logger.error(f"OpenRouter API request error: {e}", exc_info=True)
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Response content: {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"OpenRouter API error: {e}", exc_info=True)
            raise


def create_llm(
    provider: str = "openrouter",
    model_name: Optional[str] = None,
    **kwargs
) -> BaseLLM:
    """Factory function to create LLM instances."""
    model_name = model_name or os.getenv("LLM_MODEL", "nvidia/nemotron-3-nano-30b-a3b:free")
    api_key = kwargs.pop("api_key", None) or os.getenv("OPENROUTER_API_KEY")
    
    if not api_key:
        raise ValueError("No OPENROUTER_API_KEY found in environment or kwargs. Set it in .env file.")

    return OpenRouterLLM(
        model_name=model_name,
        api_key=api_key,
        **kwargs
    )


# Convenience function for RAG response generation
def generate_rag_response(
    llm: BaseLLM,
    query: str,
    context: str,
    system_prompt: Optional[str] = None,
    **kwargs
) -> LLMResponse:
    """Generate RAG response using LLM
    
    Args:
        llm: LLM instance
        query: User query
        context: Retrieved context
        system_prompt: Optional custom system prompt
        **kwargs: Additional generation parameters
        
    Returns:
        LLMResponse object
    """
    prompt = llm.create_rag_prompt(query, context, system_prompt)
    return llm.generate(prompt, **kwargs)
