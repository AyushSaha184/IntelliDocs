"""LLM Module - OpenRouter Integration for RAG."""

from typing import Optional, Dict, List
from dataclasses import dataclass
from abc import ABC, abstractmethod
import os
import requests
import json
from src.utils.Logger import get_logger
from src.utils.CircuitBreaker import CircuitBreaker, CircuitBreakerOpenError
from config.config import CIRCUIT_BREAKER_FAILURE_THRESHOLD, CIRCUIT_BREAKER_RECOVERY_SECONDS

try:
    from huggingface_hub import InferenceClient
    HUGGINGFACE_AVAILABLE = True
except ImportError:
    HUGGINGFACE_AVAILABLE = False

try:
    from cerebras.cloud.sdk import Cerebras
    CEREBRAS_AVAILABLE = True
except ImportError:
    CEREBRAS_AVAILABLE = False

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
        
        self._breaker = CircuitBreaker(
            name="openrouter_llm",
            failure_threshold=CIRCUIT_BREAKER_FAILURE_THRESHOLD,
            recovery_seconds=CIRCUIT_BREAKER_RECOVERY_SECONDS,
        )

        logger.info(f"OpenRouter client initialized with model: {model_name}")
        logger.info(f"Using API key (length: {len(self.api_key)})")

    def generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        messages: Optional[List[Dict]] = None,
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

            # Use provided messages or construct single prompt
            req_messages = messages if messages is not None else [{"role": "user", "content": prompt}]

            # Prepare request payload (OpenAI-compatible format)
            payload = {
                "model": self.model_name,
                "messages": req_messages,
                "temperature": temp,
                "max_tokens": tokens,
                "reasoning": {"enabled": True},
            }

            # Make API request
            response = self._breaker.call(
                lambda: requests.post(
                    url=self.base_url,
                    headers=headers,
                    data=json.dumps(payload),
                    timeout=60
                )
            )
            response.raise_for_status()
            
            # Parse response
            result = response.json()
            
            # Extract content and reasoning details
            message_data = result.get('choices', [{}])[0].get('message', {})
            content = message_data.get('content', '')
            reasoning_details = message_data.get('reasoning_details', None)
            
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
                    "provider": "openrouter",
                    "reasoning_details": reasoning_details
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

    def create_rag_prompt_messages(
        self,
        query: str,
        context: str,
        system_prompt: Optional[str] = None,
    ) -> List[Dict]:
        """Build a messages list with KV-cache-friendly structure.

        The static system prompt is placed first with a cache_control hint so
        OpenRouter/Anthropic can cache it across repeated queries. Dynamic
        content (the RAG context + user question) is at the end to maximise
        prefix-cache reuse.
        """
        sys_text = system_prompt or (
            "You are a helpful AI assistant. Answer the user's question based ONLY on the provided context."
        )

        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": sys_text,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {query}",
            },
        ]
        return messages

    def generate_stream(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        messages: Optional[List[Dict]] = None,
        **kwargs
    ):
        """Stream response tokens from OpenRouter API via SSE.

        Yields:
            str: Each token delta as it arrives.

        Raises:
            requests.exceptions.RequestException: On network errors.
        """
        import json as _json

        temp = temperature if temperature is not None else self.temperature
        tokens = max_tokens if max_tokens is not None else self.max_tokens

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.site_url:
            headers["HTTP-Referer"] = self.site_url
        if self.site_name:
            headers["X-Title"] = self.site_name

        req_messages = messages if messages is not None else [{"role": "user", "content": prompt}]

        payload = {
            "model": self.model_name,
            "messages": req_messages,
            "temperature": temp,
            "max_tokens": tokens,
            "stream": True,
        }

        with requests.post(
            url=self.base_url,
            headers=headers,
            data=_json.dumps(payload),
            stream=True,
            timeout=120,
        ) as response:
            response.raise_for_status()
            for raw_line in response.iter_lines():
                if not raw_line:
                    continue
                line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
                if line.startswith("data:"):
                    data_str = line[len("data:"):].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        data = _json.loads(data_str)
                        delta = data.get("choices", [{}])[0].get("delta", {})
                        token = delta.get("content")
                        if token:
                            yield token
                    except (_json.JSONDecodeError, KeyError, IndexError):
                        continue


class CerebrasLLM(BaseLLM):
    """Cerebras Cloud SDK LLM implementation."""

    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = 1000,
        **kwargs
    ):
        super().__init__(model_name, **kwargs)
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        self.api_key = api_key or os.getenv('CEREBRAS_API_KEY')
        if not self.api_key:
            raise ValueError("CEREBRAS_API_KEY is required. Set it in .env or pass as parameter.")
            
        if not CEREBRAS_AVAILABLE:
            raise ImportError("cerebras_cloud_sdk is not installed. Run `pip install cerebras_cloud_sdk`")
            
        self.client = Cerebras(api_key=self.api_key)
        self._breaker = CircuitBreaker(
            name="cerebras_llm",
            failure_threshold=CIRCUIT_BREAKER_FAILURE_THRESHOLD,
            recovery_seconds=CIRCUIT_BREAKER_RECOVERY_SECONDS,
        )
        logger.info(f"Cerebras client initialized with model: {model_name}")

    def generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate response using Cerebras API."""
        try:
            temp = temperature if temperature is not None else self.temperature
            tokens = max_tokens if max_tokens is not None else self.max_tokens

            response = self._breaker.call(
                lambda: self.client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model=self.model_name,
                    temperature=temp,
                    max_completion_tokens=tokens,
                )
            )

            content = response.choices[0].message.content
            
            # Extract token usage if available
            prompt_tokens = response.usage.prompt_tokens if hasattr(response, 'usage') and response.usage else None
            completion_tokens = response.usage.completion_tokens if hasattr(response, 'usage') and response.usage else None
            total_tokens = response.usage.total_tokens if hasattr(response, 'usage') and response.usage else None
            
            finish_reason = None
            if hasattr(response.choices[0], 'finish_reason'):
                finish_reason = response.choices[0].finish_reason
                
            return LLMResponse(
                response=content,
                model=self.model_name,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                metadata={
                    "finish_reason": finish_reason,
                    "provider": "cerebras"
                }
            )

        except Exception as e:
            logger.error(f"Cerebras API error: {e}", exc_info=True)
            raise


def create_llm(
    provider: str = "openrouter",
    model_name: Optional[str] = None,
    **kwargs
) -> BaseLLM:
    """Factory function to create LLM instances."""
    provider = provider.lower()
    
    if provider == "cerebras":
        model_name = model_name or os.getenv("JUDGE_MODEL", "llama3.1-8b")
        api_key = kwargs.pop("api_key", None) or os.getenv("CEREBRAS_API_KEY")
        if not api_key:
            raise ValueError("No CEREBRAS_API_KEY found in environment or kwargs. Set it in .env file.")
        return CerebrasLLM(model_name=model_name, api_key=api_key, **kwargs)
        
    model_name = model_name or os.getenv("LLM_MODEL", "openai/gpt-oss-120b:free")
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
