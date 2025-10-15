"""Utility functions for LLM Judge implementations"""

import litellm
import json
import inspect
import functools
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from typing import Callable, TypeVar

T = TypeVar('T')


def validate_api_configuration(model: str, **litellm_kwargs):
    """
    Validate that the API configuration is working by making a minimal test call
    
    Args:
        model: LiteLLM model name to test
        **litellm_kwargs: Additional arguments passed to LiteLLM
    
    Raises:
        ValueError: If API key is invalid or missing
        ConnectionError: If there are network/endpoint issues
    """
    try:
        # Make a minimal test call to validate the configuration
        test_messages = [
            {"role": "system", "content": "You are a test assistant."},
            {"role": "user", "content": "Say 'test' and nothing else."}
        ]
        
        response = litellm.completion(
            model=model,
            messages=test_messages,
            max_tokens=5,
            temperature=0.0,
            **litellm_kwargs
        )
        
        # If we get here, the API configuration is working
        if not response or not response.choices:
            raise ValueError(f"Invalid response from {model} - API configuration may be incorrect")
            
    except Exception as e:
        # Parse different types of authentication/configuration errors
        error_msg = str(e).lower()
        
        if "authentication" in error_msg or "api key" in error_msg or "unauthorized" in error_msg:
            raise ValueError(
                f"API key authentication failed for model '{model}'. "
                f"Please check your API key is valid and has the correct permissions. "
                f"Error: {str(e)}"
            ) from e
        elif "not found" in error_msg or "model" in error_msg:
            raise ValueError(
                f"Model '{model}' not found or not accessible with current API key. "
                f"Please check the model name and your API key permissions. "
                f"Error: {str(e)}"
            ) from e
        elif "rate limit" in error_msg or "quota" in error_msg:
            raise ValueError(
                f"Rate limit or quota exceeded for model '{model}'. "
                f"Please check your API usage limits. "
                f"Error: {str(e)}"
            ) from e
        elif "connection" in error_msg or "network" in error_msg or "timeout" in error_msg:
            raise ConnectionError(
                f"Failed to connect to API endpoint for model '{model}'. "
                f"Please check your internet connection and base_url if using custom endpoint. "
                f"Error: {str(e)}"
            ) from e
        else:
            # Generic configuration error
            raise ValueError(
                f"Failed to initialize judge with model '{model}'. "
                f"Please check your API configuration, model name, and API key. "
                f"Error: {str(e)}"
            ) from e


def parse_json_response(response_text: str) -> dict:
    """Extract JSON dict from LLM response"""
    try:
        return json.loads(response_text.strip())
    except json.JSONDecodeError:
        # Try to extract JSON from text
        import re
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        raise ValueError(f"No valid JSON found in response: {response_text[:100]}...")


def extract_message_content(message: dict) -> str:
    """
    Extract both content and tool calls from a message
    
    Args:
        message: OpenAI format message dict
        
    Returns:
        Formatted string with content and tool calls
    """
    parts = []
    
    # Add regular content
    if message.get("content"):
        parts.append(message["content"])
    
    # Add tool calls if present
    if message.get("tool_calls"):
        tool_calls_text = []
        for tool_call in message["tool_calls"]:
            func_name = tool_call["function"]["name"]
            func_args = tool_call["function"]["arguments"]
            tool_calls_text.append(f"Tool call: {func_name}({func_args})")
        parts.append("\n".join(tool_calls_text))
    
    return "\n".join(parts) if parts else ""


def with_retry(
    max_attempts: int = 3,
    min_wait: float = 0.1,
    max_wait: float = 10.0,
    multiplier: float = 1.0,
    retry_exceptions: tuple = (Exception,)
):
    """
    Decorator to add retry logic to async and sync functions
    
    Args:
        max_attempts: Maximum number of retry attempts
        min_wait: Minimum wait time between retries (seconds)
        max_wait: Maximum wait time between retries (seconds)
        multiplier: Exponential backoff multiplier
        retry_exceptions: Tuple of exception types to retry on
        
    Returns:
        Decorated function with retry logic
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        retry_config = retry(
            stop=stop_after_attempt(max_attempts),
            wait=wait_exponential(multiplier=multiplier, min=min_wait, max=max_wait),
            retry=retry_if_exception_type(retry_exceptions),
            reraise=True
        )
        
        if inspect.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                async_retry_func = retry_config(func)
                return await async_retry_func(*args, **kwargs)
            return async_wrapper
        else:
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                sync_retry_func = retry_config(func)
                return sync_retry_func(*args, **kwargs)
            return sync_wrapper
    
    return decorator


async def call_with_retry(fn: Callable[..., T], *args, **kwargs) -> T:
    """
    Wrapper function to add retry logic to any async function call
    
    Args:
        fn: The async function to call with retry
        *args: Positional arguments to pass to the function
        **kwargs: Keyword arguments to pass to the function
        
    Returns:
        Result of the function call
    """
    retry_fn = retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=0.1, max=10),
        retry=retry_if_exception_type(Exception),
        reraise=True
    )(fn)
    return await retry_fn(*args, **kwargs)