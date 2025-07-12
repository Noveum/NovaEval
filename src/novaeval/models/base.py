"""
Base model class for NovaEval.

This module defines the abstract base class for all model implementations.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional, Union


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
        **kwargs,
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
        self.errors = []

    @abstractmethod
    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stop: Optional[Union[str, list[str]]] = None,
        **kwargs,
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
        **kwargs,
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

    def _handle_error(self, error: Exception, context: str = "") -> None:
        """
        Handle and log errors.

        Args:
            error: The exception that occurred
            context: Additional context about the error
        """
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
