"""
Base model class for NovaEval.

This module defines the abstract base class for all model implementations.
"""

import functools
import logging
import os
from abc import ABC, abstractmethod
from typing import Any, Callable, Optional, Union

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Create module-level logger
logger = logging.getLogger(__name__)

NOVEUM_TRACE_AVAILABLE = False


# Fallback implementation that mimics trace_llm decorator signature but doesn't use tracing params
# The noqa: ARG001 directives are justified here because this function maintains interface compatibility
# with the real trace_llm decorator while providing a no-op implementation when tracing is unavailable
def _trace_llm_noop(
    func: Optional[Callable] = None,
    *,
    name: Optional[str] = None,  # noqa: ARG001
    provider: Optional[str] = None,  # noqa: ARG001
    capture_prompts: bool = True,  # noqa: ARG001
    capture_completions: bool = True,  # noqa: ARG001
    capture_tokens: bool = True,  # noqa: ARG001
    estimate_costs: bool = True,  # noqa: ARG001
    redact_pii: bool = False,  # noqa: ARG001
    metadata: Optional[dict[str, Any]] = None,  # noqa: ARG001
    tags: Optional[dict[str, str]] = None,  # noqa: ARG001
    **kwargs: Any,  # noqa: ARG001
) -> Any:
    # runtime no-op that behaves like the real decorator factory
    if func is None:
        # Called as @trace_llm(...) - return a decorator
        def deco(f: Callable) -> Callable:
            return functools.wraps(f)(lambda *args, **kwargs: f(*args, **kwargs))

        return deco
    # Called as @trace_llm - return the function unchanged
    return func


try:
    from noveum_trace import trace_llm  # types come from stub

    NOVEUM_TRACE_AVAILABLE = True
except ImportError:
    trace_llm = _trace_llm_noop  # type: ignore[assignment]


class BaseModel(ABC):
    """
    Abstract base class for all model implementations.

    This class defines the interface that all models must implement.
    """

    def __init__(
        self,
        name: str,
        model_name: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs: Any,
    ):
        """
        Initialize the model.

        Args:
            name: Human-readable name for this model instance
            model_name: Specific model identifier (e.g., "gpt-4", "claude-3-opus")
            api_key: API key for authentication
            base_url: Base URL for API requests
            **kwargs: Additional model-specific parameters
        """
        self.name = name
        self.model_name = model_name
        self.api_key = api_key
        self.base_url = base_url
        self.kwargs = kwargs

        # Statistics tracking
        self.total_requests = 0
        self.total_tokens = 0
        self.total_cost = 0.0
        self.errors: list[str] = []

        if os.getenv("NOVEUM_API_KEY"):
            # if ENABLE_TRACING is set to true or unset, we trace, we stop tracing only if set to False
            # for starting tracing, we do this - get the variables from the env
            enable_tracing = os.getenv("ENABLE_TRACING", "true").lower()
            if enable_tracing != "false":
                try:
                    import noveum_trace

                    noveum_trace.init(
                        api_key=os.getenv("NOVEUM_API_KEY"),
                        project=os.getenv("NOVEUM_PROJECT", "example-project"),
                        environment=os.getenv("NOVEUM_ENVIRONMENT", "development"),
                    )
                    logger.info("Noveum tracing initialized successfully")
                except ImportError:
                    logger.warning("noveum_trace not available, tracing disabled")
                except Exception as e:
                    logger.error(f"Failed to initialize Noveum tracing: {e}")

    @abstractmethod
    @trace_llm
    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stop: Optional[Union[str, list[str]]] = None,
        **kwargs: Any,
    ) -> str:
        """
        Generate text from the model.

        Args:
            prompt: Input prompt for the model
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0.0 to 1.0)
            stop: Stop sequences for generation
            **kwargs: Additional generation parameters

        Returns:
            Generated text response
        """
        pass

    @abstractmethod
    def generate_batch(
        self,
        prompts: list[str],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stop: Optional[Union[str, list[str]]] = None,
        **kwargs: Any,
    ) -> list[str]:
        """
        Generate text for multiple prompts in batch.

        Args:
            prompts: List of input prompts
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0.0 to 1.0)
            stop: Stop sequences for generation
            **kwargs: Additional generation parameters

        Returns:
            List of generated text responses
        """
        pass

    def get_info(self) -> dict[str, Any]:
        """
        Get information about the model.

        Returns:
            Dictionary containing model metadata
        """
        return {
            "name": self.name,
            "model_name": self.model_name,
            "type": self.__class__.__name__,
            "provider": self.get_provider(),
            "total_requests": self.total_requests,
            "total_tokens": self.total_tokens,
            "total_cost": self.total_cost,
            "error_count": len(self.errors),
        }

    @abstractmethod
    def get_provider(self) -> str:
        """
        Get the provider name for this model.

        Returns:
            Provider name (e.g., "openai", "anthropic")
        """
        pass

    def validate_connection(self) -> bool:
        """
        Validate that the model can be accessed.

        Returns:
            True if connection is valid, False otherwise
        """
        try:
            # Try a simple generation to test connectivity
            response = self.generate("Hello", max_tokens=1)
            return response is not None
        except Exception as e:
            self.errors.append(f"Connection validation failed: {e}")
            return False

    def estimate_cost(self, prompt: str, response: str = "") -> float:
        """
        Estimate the cost for a generation request.

        Args:
            prompt: Input prompt
            response: Generated response

        Returns:
            Estimated cost in USD
        """
        # Default implementation returns 0
        # Subclasses should implement provider-specific cost calculation
        return 0.0

    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text.

        Args:
            text: Text to count tokens for

        Returns:
            Number of tokens
        """
        # Simple approximation: 1 token ≈ 4 characters
        # Subclasses should implement more accurate token counting
        return len(text) // 4

    def _track_request(
        self, prompt: str, response: str, tokens_used: int = 0, cost: float = 0.0
    ) -> None:
        """
        Track request statistics.

        Args:
            prompt: Input prompt
            response: Generated response
            tokens_used: Number of tokens used
            cost: Cost of the request
        """
        self.total_requests += 1
        self.total_tokens += tokens_used
        self.total_cost += cost

    def _retry_with_exponential_backoff(
        self, func: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> Any:
        """
        Retry a function with exponential backoff.

        Args:
            func: Function to retry
            *args: Arguments to pass to the function
            **kwargs: Keyword arguments to pass to the function

        Returns:
            Result of the function call

        Raises:
            Exception: If all retries are exhausted and the last error is not 429
        """
        last_exception = None
        current_timeout = getattr(self, "timeout", self.kwargs.get("timeout", 60.0))
        max_retries = getattr(self, "max_retries", self.kwargs.get("max_retries", 3))

        for attempt in range(max_retries + 1):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e

                # Check if it's a 429 error (rate limit)
                # Try multiple ways to get the status code
                status_code = self._extract_status_code(e)

                if status_code == 429:
                    if attempt < max_retries:
                        logger.warning(
                            "Rate limit hit (429) on attempt %d/%d. Retrying in %.2f seconds...",
                            attempt + 1,
                            max_retries + 1,
                            current_timeout,
                        )
                        import time

                        time.sleep(current_timeout)
                        current_timeout *= 1.25  # Exponential backoff factor
                        continue
                    else:
                        # All retries exhausted for 429 error
                        logger.error(
                            "Rate limit error (429) persisted after %d attempts. Giving up on request.",
                            max_retries + 1,
                        )
                        return ""  # Return empty string instead of raising
                else:
                    # For non-429 errors, re-raise immediately
                    raise

        # This should never be reached, but just in case
        if last_exception is not None:
            raise last_exception
        else:
            raise RuntimeError("Unexpected error in retry logic")

    def _extract_status_code(self, exception: Exception) -> Optional[int]:
        """
        Extract status code from various exception types.

        This method handles exceptions from different model providers:
        - OpenAI: Uses 'status_code' attribute
        - Anthropic: Uses 'status_code' attribute
        - Google GenAI: Uses 'code' attribute
        - Azure OpenAI: Uses 'status_code' attribute

        Args:
            exception: The exception to extract status code from

        Returns:
            Status code if found, None otherwise
        """
        # Method 1: Direct attribute (works for OpenAI, Anthropic RateLimitError)
        status_code = getattr(exception, "status_code", None)
        if status_code is not None:
            return status_code

        # Method 2: Google GenAI uses 'code' attribute instead of 'status_code'
        code = getattr(exception, "code", None)
        if code is not None:
            return code

        # Method 3: Nested in response object
        if hasattr(exception, "response"):
            response = exception.response
            if hasattr(response, "status_code"):
                return getattr(response, "status_code", None)
            if hasattr(response, "status"):
                return getattr(response, "status", None)
            if hasattr(response, "code"):
                return getattr(response, "code", None)

        # Method 4: Check for HTTP status in message or args
        if hasattr(exception, "args") and exception.args:
            for arg in exception.args:
                if isinstance(arg, str) and "429" in arg:
                    return 429
                if isinstance(arg, dict) and "status_code" in arg:
                    return arg["status_code"]
                if isinstance(arg, dict) and "code" in arg:
                    return arg["code"]

        # Method 5: Check for specific exception types that are known to be 429
        exception_name = exception.__class__.__name__.lower()
        if "ratelimit" in exception_name or "rate_limit" in exception_name:
            return 429

        # Method 6: Check for Google GenAI specific patterns
        if hasattr(exception, "response_json"):
            response_json = exception.response_json
            if isinstance(response_json, dict) and "error" in response_json:
                error_info = response_json["error"]
                if "code" in error_info:
                    return error_info["code"]

        return None

    def _handle_error(self, error: Exception, context: str = "") -> None:
        """
        Handle and log errors.

        Args:
            error: The exception that occurred
            context: Additional context about the error
        """
        logger.error("Error: %s", error)
        error_msg = f"{context}: {error!s}" if context else str(error)
        self.errors.append(error_msg)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "BaseModel":
        """
        Create a model from configuration.

        Args:
            config: Configuration dictionary

        Returns:
            Configured model instance
        """
        return cls(**config)

    def __str__(self) -> str:
        """String representation of the model."""
        return (
            f"{self.__class__.__name__}(name='{self.name}', model='{self.model_name}')"
        )

    def __repr__(self) -> str:
        """Detailed string representation of the model."""
        return (
            f"{self.__class__.__name__}("
            f"name='{self.name}', "
            f"model_name='{self.model_name}', "
            f"provider='{self.get_provider()}'"
            f")"
        )
