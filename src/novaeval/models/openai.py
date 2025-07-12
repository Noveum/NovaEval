"""
OpenAI model implementation for NovaEval.

This module provides an interface to OpenAI's language models.
"""

import os
import time
from typing import Any, Optional, Union

from openai import OpenAI

from novaeval.models.base import BaseModel


class OpenAIModel(BaseModel):
    """
    OpenAI model implementation.

    Supports GPT-3.5, GPT-4, and other OpenAI models.
    """

    # Token pricing per 1K tokens (input, output)
    PRICING = {
        "gpt-4": (0.03, 0.06),
        "gpt-4-32k": (0.06, 0.12),
        "gpt-4-turbo": (0.01, 0.03),
        "gpt-4-turbo-preview": (0.01, 0.03),
        "gpt-3.5-turbo": (0.0015, 0.002),
        "gpt-3.5-turbo-16k": (0.003, 0.004),
        "gpt-3.5-turbo-instruct": (0.0015, 0.002),
    }

    def __init__(
        self,
        model_name: str = "gpt-4",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        organization: Optional[str] = None,
        max_retries: int = 3,
        timeout: float = 60.0,
        **kwargs: Any,
    ):
        """
        Initialize the OpenAI model.

        Args:
            model_name: OpenAI model name (e.g., "gpt-4", "gpt-3.5-turbo")
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            base_url: Custom base URL for API requests
            organization: OpenAI organization ID
            max_retries: Maximum number of retries for failed requests
            timeout: Request timeout in seconds
            **kwargs: Additional model parameters
        """
        super().__init__(
            name=f"openai_{model_name}",
            model_name=model_name,
            api_key=api_key,
            base_url=base_url,
            **kwargs,
        )

        self.organization = organization
        self.max_retries = max_retries
        self.timeout = timeout

        # Initialize OpenAI client
        self.client = OpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            base_url=base_url or os.getenv("OPENAI_API_BASE"),
            organization=organization,
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
        Generate text using OpenAI's API.

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
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stop": stop,
                **kwargs,
            }

            # Remove None values
            params = {k: v for k, v in params.items() if v is not None}

            # Make API call
            time.time()
            response = self.client.chat.completions.create(**params)
            time.time()

            # Extract response
            generated_text = response.choices[0].message.content

            # Track usage
            usage = response.usage
            tokens_used = usage.total_tokens if usage else 0
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
        # OpenAI doesn't have native batch support for chat completions
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
        return "openai"

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

        # Calculate cost
        input_cost = (input_tokens / 1000) * input_price
        output_cost = (output_tokens / 1000) * output_price

        return input_cost + output_cost

    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text using tiktoken.

        Args:
            text: Text to count tokens for

        Returns:
            Number of tokens
        """
        try:
            import tiktoken

            # Get encoding for the model
            if "gpt-4" in self.model_name:
                encoding = tiktoken.encoding_for_model("gpt-4")
            elif "gpt-3.5" in self.model_name:
                encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
            else:
                encoding = tiktoken.get_encoding("cl100k_base")

            return len(encoding.encode(text))

        except ImportError:
            # Fallback to simple approximation if tiktoken not available
            return super().count_tokens(text)

    def validate_connection(self) -> bool:
        """
        Validate that the OpenAI API can be accessed.

        Returns:
            True if connection is valid, False otherwise
        """
        try:
            # Try a simple API call
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=1,
            )
            return response is not None
        except Exception as e:
            self._handle_error(e, "OpenAI connection validation failed")
            return False

    def get_info(self) -> dict[str, Any]:
        """
        Get information about the OpenAI model.

        Returns:
            Dictionary containing model metadata
        """
        info = super().get_info()
        info.update(
            {
                "organization": self.organization,
                "max_retries": self.max_retries,
                "timeout": self.timeout,
                "supports_batch": False,
                "pricing": self.PRICING.get(self.model_name, (0, 0)),
            }
        )
        return info
