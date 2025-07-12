"""
Anthropic model implementation for NovaEval.

This module provides an interface to Anthropic's Claude models.
"""

import os
import time
from typing import Any, Optional, Union

import anthropic

from novaeval.models.base import BaseModel


class AnthropicModel(BaseModel):
    """
    Anthropic Claude model implementation.

    Supports Claude 3 family models (Haiku, Sonnet, Opus).
    """

    # Token pricing per 1M tokens (input, output)
    PRICING = {
        "claude-3-haiku-20240307": (0.25, 1.25),
        "claude-3-sonnet-20240229": (3.0, 15.0),
        "claude-3-opus-20240229": (15.0, 75.0),
        "claude-3-5-sonnet-20241022": (3.0, 15.0),
        "claude-2.1": (8.0, 24.0),
        "claude-2.0": (8.0, 24.0),
        "claude-instant-1.2": (0.8, 2.4),
    }

    def __init__(
        self,
        model_name: str = "claude-3-sonnet-20240229",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        max_retries: int = 3,
        timeout: float = 60.0,
        **kwargs: Any,
    ):
        """
        Initialize the Anthropic model.

        Args:
            model_name: Anthropic model name (e.g., "claude-3-opus-20240229")
            api_key: Anthropic API key (defaults to ANTHROPIC_API_KEY env var)
            base_url: Custom base URL for API requests
            max_retries: Maximum number of retries for failed requests
            timeout: Request timeout in seconds
            **kwargs: Additional model parameters
        """
        super().__init__(
            name=f"anthropic_{model_name}",
            model_name=model_name,
            api_key=api_key,
            base_url=base_url,
            **kwargs,
        )

        self.max_retries = max_retries
        self.timeout = timeout

        # Initialize Anthropic client
        self.client = anthropic.Anthropic(
            api_key=api_key or os.getenv("ANTHROPIC_API_KEY"),
            base_url=base_url,
            max_retries=max_retries,
            timeout=timeout,
        )

    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stop: Optional[Union[str, list[str]]] = None,
        **kwargs: Any,
    ) -> str:
        """
        Generate text using Anthropic's API.

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            stop: Stop sequences
            **kwargs: Additional generation parameters

        Returns:
            Generated text
        """
        try:
            # Prepare parameters
            params = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens or 1024,
                "temperature": temperature,
                "stop_sequences": (
                    stop if isinstance(stop, list) else [stop] if stop else None
                ),
                **kwargs,
            }

            # Remove None values
            params = {k: v for k, v in params.items() if v is not None}

            # Make API call
            time.time()
            response = self.client.messages.create(**params)
            time.time()

            # Extract response
            generated_text = response.content[0].text

            # Track usage
            usage = response.usage
            tokens_used = usage.input_tokens + usage.output_tokens if usage else 0
            cost = self.estimate_cost(prompt, generated_text)

            self._track_request(
                prompt=prompt,
                response=generated_text,
                tokens_used=tokens_used,
                cost=cost,
            )

            return generated_text

        except Exception as e:
            self._handle_error(
                e, f"Failed to generate text for prompt: {prompt[:100]}..."
            )
            raise

    def generate_batch(
        self,
        prompts: list[str],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stop: Optional[Union[str, list[str]]] = None,
        **kwargs: Any,
    ) -> list[str]:
        """
        Generate text for multiple prompts.

        Args:
            prompts: List of input prompts
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            stop: Stop sequences
            **kwargs: Additional generation parameters

        Returns:
            List of generated texts
        """
        # Anthropic doesn't have native batch support
        # So we'll process them sequentially
        results = []
        for prompt in prompts:
            try:
                result = self.generate(
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stop=stop,
                    **kwargs,
                )
                results.append(result)
            except Exception as e:
                self._handle_error(
                    e, f"Failed in batch generation for prompt: {prompt[:100]}..."
                )
                results.append("")  # Add empty string for failed generations

        return results

    def get_provider(self) -> str:
        """Get the provider name."""
        return "anthropic"

    def estimate_cost(self, prompt: str, response: str = "") -> float:
        """
        Estimate the cost for a generation request.

        Args:
            prompt: Input prompt
            response: Generated response

        Returns:
            Estimated cost in USD
        """
        if self.model_name not in self.PRICING:
            return 0.0

        input_price, output_price = self.PRICING[self.model_name]

        # Estimate tokens (rough approximation)
        input_tokens = self.count_tokens(prompt)
        output_tokens = self.count_tokens(response)

        # Calculate cost (pricing is per 1M tokens)
        input_cost = (input_tokens / 1_000_000) * input_price
        output_cost = (output_tokens / 1_000_000) * output_price

        return input_cost + output_cost

    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text.

        Args:
            text: Text to count tokens for

        Returns:
            Number of tokens
        """
        try:
            # Use Anthropic's token counting if available
            response = self.client.count_tokens(text)  # type: ignore
            return response.count  # type: ignore
        except Exception:
            # Fallback to simple approximation
            return super().count_tokens(text)

    def validate_connection(self) -> bool:
        """
        Validate that the Anthropic API can be accessed.

        Returns:
            True if connection is valid, False otherwise
        """
        try:
            # Try a simple API call
            response = self.client.messages.create(
                model=self.model_name,
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=1,
            )
            return response is not None
        except Exception as e:
            self._handle_error(e, "Anthropic connection validation failed")
            return False

    def get_info(self) -> dict[str, Any]:
        """
        Get information about the Anthropic model.

        Returns:
            Dictionary containing model metadata
        """
        info = super().get_info()
        info.update(
            {
                "max_retries": self.max_retries,
                "timeout": self.timeout,
                "supports_batch": False,
                "pricing": self.PRICING.get(self.model_name, (0, 0)),
            }
        )
        return info
