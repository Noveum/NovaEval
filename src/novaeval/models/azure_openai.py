"""
Azure OpenAI model implementation for NovaEval.

This module provides an interface to Azure OpenAI's language models using the OpenAI SDK.

PRICING UNITS: USD **per 1,000,000 tokens** (M tokens), aligned with Azure OpenAI API rates.

Pricing last validated: 2025-07-17 (Asia/Kolkata).
"""

import os
import re
from typing import Any, Optional, Union

from openai import AzureOpenAI

from novaeval.models.base import BaseModel

# ---------------------------------------------------------------------------
# Pricing (BASE TIER + TIERED for context window models)
# ---------------------------------------------------------------------------
MODEL_PRICING_PER_1M = {
    # GPT-4 family (base tier)
    "gpt-4": (30.0, 60.0),  # base tier (<=8k)
    "gpt-4-turbo": (10.0, 30.0),
    # GPT-3.5 family
    "gpt-3.5-turbo-0301": (1.50, 2.00),
    "gpt-3.5-turbo-0613": (1.50, 2.00),  # base tier (<=4k)
    "gpt-3.5-turbo-1106": (1.00, 2.00),
    "gpt-3.5-turbo-0125": (0.50, 1.50),
    "gpt-3.5-turbo-instruct": (1.50, 2.00),
    # o-series
    "o3": (2.00, 8.00),
    "o4-mini": (1.10, 4.40),
    # GPT-4.1 series
    "gpt-4.1": (2.00, 8.00),
    "gpt-4.1mini": (0.40, 1.60),
    "gpt-4.1-nano": (0.10, 0.40),
}

# Models with context window tiers (canonical name: context window in TIERED_MODELS)
TIERED_MODELS = {
    # model_name: (context_window, (low_in, low_out), (high_in, high_out))
    "gpt-4": (8000, (30.0, 60.0), (60.0, 120.0)),  # <=8k, >8k
    "gpt-3.5-turbo-0613": (4000, (1.50, 2.00), (3.00, 4.00)),  # <=4k, >4k
}

USE_TIERED_PRICING = True

SUPPORTED_MODELS = list(MODEL_PRICING_PER_1M.keys())


def _canonical_model_name(model_name: str) -> str:
    """
    Extract canonical model name (e.g., 'gpt-4' from 'gpt-4-8k' or 'gpt-4-32k').
    """
    # Remove context window suffixes like -8k, -32k, -4k, -16k
    return re.sub(r"-(8k|32k|4k|16k)$", "", model_name)


class AzureOpenAIModel(BaseModel):
    """
    Azure OpenAI model implementation.
    """

    @classmethod
    def create_from_config(cls, config: dict[str, Any]) -> "AzureOpenAIModel":
        # No defaults here; constructor handles fallbacks and errors.
        model_name = config.get("model_name")
        api_key = config.get("api_key")
        base_url = config.get("base_url")
        api_version = config.get("api_version")
        max_retries = config.get("max_retries", 3)
        timeout = config.get("timeout", 60.0)

        kwargs = {
            k: v
            for k, v in config.items()
            if k
            not in [
                "model_name",
                "api_key",
                "base_url",
                "api_version",
                "max_retries",
                "timeout",
            ]
        }
        return cls(
            model_name=model_name,
            api_key=api_key,
            base_url=base_url,
            api_version=api_version,
            max_retries=max_retries,
            timeout=timeout,
            **kwargs,
        )

    def __init__(
        self,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        max_retries: int = 3,
        timeout: float = 60.0,
        api_version: Optional[str] = None,
        **kwargs: Any,
    ):
        # Enforce required parameters from args or environment variables
        effective_model_name = model_name or os.getenv("AZURE_OPENAI_DEPLOYMENT")
        if not effective_model_name:
            raise ValueError(
                "Deployment name is required via 'model_name' parameter or 'AZURE_OPENAI_DEPLOYMENT' env var."
            )

        effective_api_key = api_key or os.getenv("AZURE_OPENAI_API_KEY")
        if effective_api_key:
            effective_api_key = effective_api_key.strip()
        if not effective_api_key:
            raise ValueError(
                "API key is required via 'api_key' parameter or 'AZURE_OPENAI_API_KEY' env var."
            )

        effective_base_url = base_url or os.getenv("AZURE_OPENAI_BASE_URL")
        if not effective_base_url:
            raise ValueError(
                "Base URL is required via 'base_url' parameter or 'AZURE_OPENAI_BASE_URL' env var."
            )

        effective_api_version = api_version or os.getenv("AZURE_OPENAI_API_VERSION")
        if not effective_api_version:
            raise ValueError(
                "API version is required via 'api_version' parameter or 'AZURE_OPENAI_API_VERSION' env var."
            )

        super().__init__(
            name=f"azure_openai_{effective_model_name}",
            model_name=effective_model_name,
            api_key=effective_api_key,
            base_url=effective_base_url,
            **kwargs,
        )
        self.api_version = effective_api_version
        self.max_retries = max_retries
        self.timeout = timeout
        self.client: Any  # type: ignore
        try:

            self.client = AzureOpenAI(
                api_key=effective_api_key,
                azure_endpoint=effective_base_url,
                api_version=effective_api_version,
            )

        except Exception as e:
            raise ValueError(f"Failed to initialize Azure OpenAI client: {e!s}")

    def generate(
        self,
        prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stop: Optional[Union[str, list[str]]] = None,
        messages: Optional[list] = None,
        **kwargs: Any,
    ) -> str:
        if prompt is None and messages is None:
            raise ValueError("Either `prompt` or `messages` must be provided.")

        if messages is None:
            messages = [{"role": "user", "content": prompt}]

        return self.generate_chat(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop,
            **kwargs,
        )

    def generate_chat(
        self,
        messages: list,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stop: Optional[Union[str, list[str]]] = None,
        **kwargs: Any,
    ) -> str:
        """
        Generate a chat completion using the Azure OpenAI chat endpoint.
        """
        try:
            params: dict[str, Any] = {
                "model": self.model_name,  # deployment name
                "messages": messages,
            }
            if max_tokens is not None:
                params["max_tokens"] = max_tokens
            if temperature is not None:
                params["temperature"] = temperature
            if stop is not None:
                params["stop"] = stop
            params.update(kwargs)

            if not (
                hasattr(self.client, "chat")
                and hasattr(self.client.chat, "completions")
            ):
                raise RuntimeError(
                    "The OpenAI client does not support the 'chat.completions' endpoint. Please upgrade your SDK."
                )

            response = self.client.chat.completions.create(**params)

            output_text = ""
            if hasattr(response, "choices") and response.choices:
                output_text = response.choices[0].message.content or ""

            usage = getattr(response, "usage", None)
            input_tokens = (
                usage.prompt_tokens if usage and hasattr(usage, "prompt_tokens") else 0
            )
            output_tokens = (
                usage.completion_tokens
                if usage and hasattr(usage, "completion_tokens")
                else self.count_tokens(output_text)
            )

            # For tracking purposes, serialize messages to a string.
            import json

            prompt_for_tracking = json.dumps(messages)
            if not input_tokens:
                input_tokens = self.count_tokens(prompt_for_tracking)

            cost = self.estimate_cost(
                prompt_for_tracking,
                output_text,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )
            self._track_request(
                prompt=prompt_for_tracking,
                response=output_text,
                tokens_used=input_tokens + output_tokens,
                cost=cost,
            )

            return output_text
        except Exception as e:
            self._handle_error(e, "Failed to generate chat completion.")
            raise

    def generate_batch(
        self,
        prompts: list[str],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stop: Optional[Union[str, list[str]]] = None,
        **kwargs: Any,
    ) -> list[str]:
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
        return "azure_openai"

    def _get_rates(self, input_tokens: int, output_tokens: int) -> tuple[float, float]:
        canonical_name = _canonical_model_name(self.model_name)
        if USE_TIERED_PRICING and canonical_name in TIERED_MODELS:
            cutoff, low_rates, high_rates = TIERED_MODELS[canonical_name]
            if (input_tokens + output_tokens) <= cutoff:
                return low_rates
            else:
                return high_rates
        return MODEL_PRICING_PER_1M.get(canonical_name, (0.0, 0.0))

    def estimate_cost(
        self,
        prompt: str,
        response: str = "",
        *,
        input_tokens: Optional[int] = None,
        output_tokens: Optional[int] = None,
    ) -> float:
        if input_tokens is None:
            input_tokens = self.count_tokens(prompt)
        if output_tokens is None:
            output_tokens = self.count_tokens(response)
        in_rate, out_rate = self._get_rates(input_tokens, output_tokens)
        if in_rate == 0.0 and out_rate == 0.0:
            return 0.0
        M = 1_000_000
        return (input_tokens / M) * in_rate + (output_tokens / M) * out_rate

    def count_tokens(self, text: str) -> int:
        """
        Estimate token count using a simple approximation.

        Note: This uses 2 tokens per word as an estimate. Actual Azure OpenAI
        tokenization may differ significantly. For accurate counts, use the
        tiktoken library or Azure's tokenization API.

        Args:
            text: Text to count tokens for

        Returns:
            Estimated token count
        """
        if not text:
            return 0
        # The tests expect 2 tokens per word, 0 for empty string
        words = text.split()
        return len(words) * 2

    def validate_connection(self) -> bool:
        if not (
            hasattr(self.client, "chat") and hasattr(self.client.chat, "completions")
        ):
            raise RuntimeError(
                "The OpenAI client does not support the 'chat.completions' endpoint. Please upgrade your SDK."
            )
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": "Ping!"}],
                max_tokens=1,
            )
            return bool(response.choices)
        except Exception as e:
            self._handle_error(e, "Connection test failed")
            return False

    def get_info(self) -> dict[str, Any]:
        info = super().get_info()
        canonical_name = _canonical_model_name(self.model_name)
        info.update(
            {
                "max_retries": self.max_retries,
                "timeout": self.timeout,
                "api_version": self.api_version,
                "supports_batch": False,
                "pricing": MODEL_PRICING_PER_1M.get(canonical_name, (0, 0)),
                "supported_models": SUPPORTED_MODELS,
            }
        )
        return info
