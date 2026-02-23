"""Utility functions for LLM Judge implementations"""

from collections import OrderedDict
import litellm
import json
import inspect
import functools
import logging
from threading import Lock
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from typing import Any, Callable, Optional, TypeVar

T = TypeVar('T')
_LOGGER = logging.getLogger(__name__)

_RESPONSE_FORMAT_UNSUPPORTED_CACHE_MAXSIZE = 256
_RESPONSE_FORMAT_UNSUPPORTED_CACHE: OrderedDict[tuple[str, str], bool] = OrderedDict()
_RESPONSE_FORMAT_UNSUPPORTED_CACHE_LOCK = Lock()

_RESPONSE_FORMAT_FALLBACK_COUNTER_MAX = 99999
_RESPONSE_FORMAT_FALLBACK_COUNT = 0
_RESPONSE_FORMAT_FALLBACK_COUNT_LOCK = Lock()


def _response_format_cache_key(*, model: str, base_url: Optional[str]) -> tuple[str, str]:
    return model.strip().lower(), (base_url or "").strip().lower()


def is_response_format_cached_as_unsupported(*, model: str, base_url: Optional[str]) -> bool:
    key = _response_format_cache_key(model=model, base_url=base_url)
    with _RESPONSE_FORMAT_UNSUPPORTED_CACHE_LOCK:
        if key not in _RESPONSE_FORMAT_UNSUPPORTED_CACHE:
            return False
        _RESPONSE_FORMAT_UNSUPPORTED_CACHE.move_to_end(key)
        return _RESPONSE_FORMAT_UNSUPPORTED_CACHE[key]


def mark_response_format_as_unsupported(*, model: str, base_url: Optional[str]) -> None:
    key = _response_format_cache_key(model=model, base_url=base_url)
    with _RESPONSE_FORMAT_UNSUPPORTED_CACHE_LOCK:
        _RESPONSE_FORMAT_UNSUPPORTED_CACHE[key] = True
        _RESPONSE_FORMAT_UNSUPPORTED_CACHE.move_to_end(key)
        while len(_RESPONSE_FORMAT_UNSUPPORTED_CACHE) > _RESPONSE_FORMAT_UNSUPPORTED_CACHE_MAXSIZE:
            _RESPONSE_FORMAT_UNSUPPORTED_CACHE.popitem(last=False)


def clear_response_format_unsupported_cache() -> None:
    with _RESPONSE_FORMAT_UNSUPPORTED_CACHE_LOCK:
        _RESPONSE_FORMAT_UNSUPPORTED_CACHE.clear()


def increment_response_format_fallback_counter() -> None:
    global _RESPONSE_FORMAT_FALLBACK_COUNT
    with _RESPONSE_FORMAT_FALLBACK_COUNT_LOCK:
        _RESPONSE_FORMAT_FALLBACK_COUNT = min(
            _RESPONSE_FORMAT_FALLBACK_COUNT + 1,
            _RESPONSE_FORMAT_FALLBACK_COUNTER_MAX,
        )


def get_response_format_fallback_counter() -> int:
    with _RESPONSE_FORMAT_FALLBACK_COUNT_LOCK:
        return _RESPONSE_FORMAT_FALLBACK_COUNT


def get_response_format_fallback_counter_display() -> str:
    count = get_response_format_fallback_counter()
    if count >= _RESPONSE_FORMAT_FALLBACK_COUNTER_MAX:
        return f"{_RESPONSE_FORMAT_FALLBACK_COUNTER_MAX}+"
    return str(count)


def get_response_format_unsupported_cache_size() -> int:
    with _RESPONSE_FORMAT_UNSUPPORTED_CACHE_LOCK:
        return len(_RESPONSE_FORMAT_UNSUPPORTED_CACHE)


def get_structured_output_fallback_stats() -> dict[str, int | str]:
    return {
        "fallback_count": get_response_format_fallback_counter(),
        "fallback_count_display": get_response_format_fallback_counter_display(),
        "unsupported_cache_size": get_response_format_unsupported_cache_size(),
        "unsupported_cache_maxsize": _RESPONSE_FORMAT_UNSUPPORTED_CACHE_MAXSIZE,
    }


def log_structured_output_fallback_stats() -> None:
    stats = get_structured_output_fallback_stats()
    _LOGGER.info(
        "llm_judge structured_output fallback_count=%s fallback_count_display=%s "
        "unsupported_cache_size=%s unsupported_cache_maxsize=%s",
        stats["fallback_count"],
        stats["fallback_count_display"],
        stats["unsupported_cache_size"],
        stats["unsupported_cache_maxsize"],
    )


def reset_response_format_fallback_state() -> None:
    global _RESPONSE_FORMAT_FALLBACK_COUNT
    clear_response_format_unsupported_cache()
    with _RESPONSE_FORMAT_FALLBACK_COUNT_LOCK:
        _RESPONSE_FORMAT_FALLBACK_COUNT = 0


def normalize_structured_output_mode(mode: str) -> str:
    normalized = mode.strip().lower()
    if normalized not in {"auto", "strict", "off"}:
        raise ValueError(f"Invalid structured_output_mode '{mode}'. Must be one of: auto, strict, off")
    return normalized


def should_attempt_structured_output(*, structured_output_mode: str, model: str, base_url: Optional[str]) -> bool:
    if structured_output_mode == "off":
        return False
    if structured_output_mode == "auto" and is_response_format_cached_as_unsupported(model=model, base_url=base_url):
        increment_response_format_fallback_counter()
        return False
    return True


def handle_structured_output_fallback(*, exc: Exception, structured_output_mode: str,
                                      model: str, base_url: Optional[str]) -> bool:
    if structured_output_mode != "auto":
        return False
    if not is_response_format_unsupported_error(exc):
        return False
    mark_response_format_as_unsupported(model=model, base_url=base_url)
    increment_response_format_fallback_counter()
    return True


def is_response_format_unsupported_error(exc: Exception) -> bool:
    message = str(exc).lower()
    if "response_format" not in message and "json_schema" not in message:
        return False
    unsupported_markers = (
        "not supported", "unsupported", "unknown",
        "invalid parameter", "extra inputs are not permitted",
    )
    return any(marker in message for marker in unsupported_markers)


def build_pointwise_response_format() -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "pointwise_judge_response",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "reasoning": {"type": "string"},
                    "score": {"type": "number", "minimum": 0.0, "maximum": 10.0},
                },
                "required": ["reasoning", "score"],
            },
        },
    }


def build_groupwise_response_format(*, num_responses: int, top_n: int) -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "groupwise_judge_response",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "reasoning": {"type": "string"},
                    "selected_indices": {
                        "type": "array",
                        "items": {"type": "integer", "minimum": 0, "maximum": num_responses - 1},
                        "minItems": top_n,
                        "maxItems": top_n,
                        "uniqueItems": True,
                    },
                },
                "required": ["reasoning", "selected_indices"],
            },
        },
    }


def validate_pointwise_judge_result(result: dict[str, Any]) -> tuple[float, str]:
    if not isinstance(result.get("reasoning"), str):
        raise ValueError("pointwise judge response must include string 'reasoning'")
    score = result.get("score")
    if not isinstance(score, (int, float)):
        raise ValueError("pointwise judge response must include numeric 'score'")
    score_value = float(score)
    if score_value < 0.0 or score_value > 10.0:
        raise ValueError("pointwise judge response 'score' must be between 0 and 10")
    return score_value, result["reasoning"]


def validate_groupwise_judge_result(result: dict[str, Any], *, num_responses: int,
                                    top_n: int) -> tuple[list[int], str]:
    if not isinstance(result.get("reasoning"), str):
        raise ValueError("groupwise judge response must include string 'reasoning'")
    selected_indices = result.get("selected_indices")
    if not isinstance(selected_indices, list):
        raise ValueError("groupwise judge response must include list 'selected_indices'")
    if len(selected_indices) != top_n:
        raise ValueError(f"groupwise judge response 'selected_indices' must contain exactly {top_n} indices")
    if len(set(selected_indices)) != len(selected_indices):
        raise ValueError("groupwise judge response 'selected_indices' must be unique")
    normalized_indices: list[int] = []
    for index in selected_indices:
        if not isinstance(index, int):
            raise ValueError("groupwise judge response 'selected_indices' must contain integers")
        if index < 0 or index >= num_responses:
            raise ValueError("groupwise judge response 'selected_indices' contains out-of-range index")
        normalized_indices.append(index)
    return normalized_indices, result["reasoning"]


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