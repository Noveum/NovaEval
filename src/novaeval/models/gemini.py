"""
Gemini model implementation for NovaEval.

This module provides an interface to Gemini's language models using the Google GenAI SDK.
"""

import os
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

    PRICING = {
        "gemini-2.5-pro": (1.25, 10.00),
        "gemini-2.5-flash": (0.30, 2.50),
        "gemini-2.0-flash": (0.10, 0.40),
        "gemini-2.0-flash-lite": (0.075, 0.30),
        "gemini-1.5-flash": (0.075, 0.30),
        "gemini-1.5-flash-8b": (0.0375, 0.15),
        "gemini-1.5-pro": (1.25, 5.00),
    }

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
        """
        super().__init__(
            name=f"gemini_{model_name}",
            model_name=model_name,
            api_key=api_key,
            **kwargs,
        )

        self.client = genai.Client(api_key=api_key or os.getenv("GOOGLE_API_KEY"))
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
        Estimate token count using a rough heuristic.

        Gemini doesn't expose tokenizers.
        """
        return int(len(text.split()) * 1.3)

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
            }
        )
        return info
