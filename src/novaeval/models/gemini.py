"""
Gemini model implementation for NovaEval.

This module provides an interface to Gemini's language models using the Google GenAI SDK.
"""

import os
import re
import time
from typing import Any, Optional, Union

from google import genai
from google.genai import types

from novaeval.models.base import BaseModel


class GeminiModel(BaseModel):
    """
    Gemini model implementation.

    Supports Gemini 2.5 Pro, 2.5 Flash, 2.0 Flash, 2.0 Flash Lite, 1.5 Flash, 1.5 Flash-8B, and 1.5 Pro.
    """

    # Pricing per 1,000,000 tokens (input, output)
    PRICING = {
        "gemini-2.5-pro": (1.25, 10.00),
        "gemini-2.5-flash": (
            0.0003,
            0.0025,
        ),
        "gemini-2.0-flash": (
            0.0001,
            0.0004,
        ),
        "gemini-2.0-flash-lite": (
            0.000075,
            0.0003,
        ),
        "gemini-1.5-flash": (
            0.000075,
            0.0003,
        ),
        "gemini-1.5-flash-8b": (
            0.0000375,
            0.00015,
        ),
        "gemini-1.5-pro": (1.25, 5.00),
    }

    # Supported model names
    SUPPORTED_MODELS = [
        "gemini-2.5-pro",
        "gemini-2.5-flash",
        "gemini-2.0-flash",
        "gemini-2.0-flash-lite",
        "gemini-1.5-flash",
        "gemini-1.5-flash-8b",
        "gemini-1.5-pro",
    ]

    @classmethod
    def create_from_config(cls, config: dict[str, Any]) -> "GeminiModel":
        """
        Create a GeminiModel instance from a configuration dictionary.

        Args:
            config: Configuration dictionary containing model parameters

        Returns:
            GeminiModel instance
        """
        model_name = config.get("model_name", "gemini-2.5-flash")
        api_key = config.get("api_key")
        max_retries = config.get("max_retries", 3)
        timeout = config.get("timeout", 60.0)

        # Extract any additional keyword arguments
        kwargs = {
            k: v
            for k, v in config.items()
            if k not in ["model_name", "api_key", "max_retries", "timeout"]
        }

        return cls(
            model_name=model_name,
            api_key=api_key,
            max_retries=max_retries,
            timeout=timeout,
            **kwargs,
        )

    def __init__(
        self,
        model_name: str = "gemini-2.5-flash",
        api_key: Optional[str] = None,
        max_retries: int = 3,
        timeout: float = 60.0,
        **kwargs: Any,
    ):
        """
        Initialize the Gemini model.

        Args:
            model_name: Gemini model name
            api_key: Gemini API key
            max_retries: Max retries on failure
            timeout: Request timeout
            **kwargs: Extra params

        Raises:
            ValueError: If model_name is not supported
            ValueError: If API key is missing or invalid
        """

        # Validate API key
        effective_api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not effective_api_key:
            raise ValueError(
                "API key is required. Provide it via the 'api_key' parameter "
                "or set the 'GOOGLE_API_KEY' environment variable."
            )

        if not isinstance(effective_api_key, str) or not effective_api_key.strip():
            raise ValueError("API key must be a non-empty string.")

        super().__init__(
            name=f"gemini_{model_name}",
            model_name=model_name,
            api_key=effective_api_key,
            **kwargs,
        )

        try:
            self.client = genai.Client(api_key=effective_api_key)
        except Exception as e:
            raise ValueError(f"Failed to initialize Gemini client: {e!s}")

        self.max_retries = max_retries
        self.timeout = timeout

    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stop: Optional[Union[str, list[str]]] = None,
        **kwargs: Any,
    ) -> str:
        """
        Generate text using Gemini's API.

        Args:
            prompt: Input prompt
            max_tokens: Max output tokens
            temperature: Sampling temperature
            stop: Not supported in Gemini currently
            **kwargs: Additional generation params

        Returns:
            Generated text
        """
        try:
            time.time()
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=temperature, max_output_tokens=max_tokens, **kwargs
                ),
            )
            time.time()

            output = response.text or ""

            tokens_used = self.count_tokens(prompt + output)
            cost = self.estimate_cost(prompt, output)

            self._track_request(
                prompt=prompt,
                response=output,
                tokens_used=tokens_used,
                cost=cost,
            )

            return output

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
        Generate responses for multiple prompts (sequentially).

        Gemini doesn't support batch generation natively.

        Returns:
            List of responses.
        """
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
                self._handle_error(e, f"Batch failure for: {prompt[:100]}...")
                results.append("")
        return results

    def get_provider(self) -> str:
        return "gemini"

    def estimate_cost(self, prompt: str, response: str = "") -> float:
        """
        Estimate Gemini API cost.

        Returns:
            Estimated cost in USD
        """
        pricing = self.PRICING.get(self.model_name)
        if not pricing:
            return 0.0

        input_price, output_price = pricing
        input_tokens = self.count_tokens(prompt)
        output_tokens = self.count_tokens(response)

        return (input_tokens / 1000) * input_price + (
            output_tokens / 1000
        ) * output_price

    def count_tokens(self, text: str) -> int:
        """
        Estimate token count using an improved heuristic.

        This method provides a more accurate token count estimation by:
        1. Splitting on whitespace and punctuation
        2. Accounting for subword tokenization patterns
        3. Adjusting for typical tokenization overhead

        Args:
            text: Input text to count tokens for

        Returns:
            Estimated token count
        """
        if not text:
            return 0

        # Split on whitespace and common punctuation
        tokens = re.findall(r"\w+|[^\w\s]", text)

        # Base token count
        base_count = len(tokens)

        # Account for subword tokenization
        # Longer words are more likely to be split into subwords
        subword_adjustment = 0
        for token in tokens:
            if len(token) > 6:  # Longer words likely split
                subword_adjustment += len(token) // 4
            elif len(token) > 3:  # Medium words sometimes split
                subword_adjustment += len(token) // 8

        # Add special token overhead (BOS, EOS, etc.)
        special_tokens = 2

        # Final estimate with bounds checking
        estimated_tokens = base_count + subword_adjustment + special_tokens

        # Apply a conservative multiplier for safety
        return int(estimated_tokens * 1.1)

    def validate_connection(self) -> bool:
        """
        Ping the Gemini API to check if it's alive.

        Returns:
            True if success
        """
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents="Ping!",
                config=types.GenerateContentConfig(max_output_tokens=1),
            )
            return bool(response.text)
        except Exception as e:
            self._handle_error(e, "Connection test failed")
            return False

    def get_info(self) -> dict[str, Any]:
        """
        Get metadata about the model.

        Returns:
            Info dict
        """
        info = super().get_info()
        info.update(
            {
                "max_retries": self.max_retries,
                "timeout": self.timeout,
                "supports_batch": False,
                "pricing": self.PRICING.get(self.model_name, (0, 0)),
                "supported_models": self.SUPPORTED_MODELS,
            }
        )
        return info
