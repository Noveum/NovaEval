"""
Gemini model implementation for NovaEval.

This module provides an interface to Gemini's language models using the Google GenAI SDK.

PRICING UNITS: USD **per 1,000,000 tokens** (M tokens), aligned with the
published Gemini API paid-tier rates. Cache pricing is ignored here.

Pricing last validated: 2025-07-17 (Asia/Kolkata).
Sources: Google Gemini API Pricing docs; TechCrunch Gemini 2.5 Pro article.
"""

import os
import re
from typing import Any, Optional, Union

from google import genai
from google.genai import types

from novaeval.models.base import BaseModel

# ---------------------------------------------------------------------------
# Pricing (BASE TIER ONLY, cache ignored)
# ---------------------------------------------------------------------------
# NOTE: Some Gemini models have *higher* rates when prompt tokens exceed a threshold
# (e.g., 2.5 Pro >200k tokens; 1.5 family >128k). For simplicity, NovaEval defaults
# to the *base-tier* rates below. You can enable tiered billing by flipping
# `USE_TIERED_PRICING = True` and providing actual input token counts.
#
# USD per 1M tokens (input, output)
PRICING = {
    "gemini-2.5-pro": (1.25, 10.00),  # <=200k prompt tier.
    "gemini-2.5-flash": (0.30, 2.50),  # flat.
    "gemini-2.5-flash-lite": (0.10, 0.40),  # flat.
    "gemini-2.0-flash": (0.10, 0.40),  # flat.
    "gemini-2.0-flash-lite": (0.075, 0.30),  # flat.
    "gemini-1.5-flash": (0.075, 0.30),  # <=128k tier.
    "gemini-1.5-flash-8b": (0.0375, 0.15),  # <=128k tier.
    "gemini-1.5-pro": (1.25, 5.00),  # <=128k tier.
}

SUPPORTED_MODELS = list(PRICING.keys())

# Optional (not used unless enabled below): prompt-size tier cutoffs + high-tier rates.
# cutoff is inclusive for low tier (i.e., <= cutoff => low-tier pricing).
PRICING_TIERED = {
    "gemini-2.5-pro": (200_000, (1.25, 10.00), (2.50, 15.00)),
    "gemini-1.5-pro": (128_000, (1.25, 5.00), (2.50, 10.00)),
    "gemini-1.5-flash": (128_000, (0.075, 0.30), (0.15, 0.60)),
    "gemini-1.5-flash-8b": (128_000, (0.0375, 0.15), (0.075, 0.30)),
}

# Flip to True if you want NovaEval to auto-select the higher tier when prompt tokens exceed the cutoff.
USE_TIERED_PRICING = False


# ---------------------------------------------------------------------------
# Lightweight token estimation
# ---------------------------------------------------------------------------
def rough_token_estimate(text: str) -> int:
    """
    Very rough character-based token estimate.

    Gemini tokenization differs from GPT-style BPEs; for low-stakes *cost estimates*
    we approximate 4 chars/token (English-ish mix).  Adjust if you see large drifts.

    For tighter numbers, integrate:
      - Google Gemini SDK token counting endpoints; OR
      - a local tokenizer once Google publishes an official one.

    Returns integer token estimate.
    """
    if not text:
        return 0
    # crude heuristic: avg 4 chars/token; clamp floor 1 token if non-empty
    return max(1, len(text) // 4)


class GeminiModel(BaseModel):
    """
    Gemini model implementation.

    Supports Gemini 2.5 Pro, 2.5 Flash, 2.0 Flash, 2.0 Flash Lite, 1.5 Flash, 1.5 Flash-8B, and 1.5 Pro.
    """

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
        effective_api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not effective_api_key:
            raise ValueError(
                "API key is required. Provide it via the 'api_key' parameter "
                "or set the 'GEMINI_API_KEY' environment variable."
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
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=temperature, max_output_tokens=max_tokens, **kwargs
                ),
            )

            # Extract text from response, handling different response formats
            output = ""
            if response.text is not None:
                output = response.text
            elif response.candidates and len(response.candidates) > 0:
                candidate = response.candidates[0]
                if candidate.content and candidate.content.parts:
                    # Extract text from content parts
                    for part in candidate.content.parts:
                        if hasattr(part, "text") and part.text:
                            output += part.text

                # If still no output and finish reason is MAX_TOKENS,
                # the model couldn't generate due to token limit
                if not output and candidate.finish_reason == "MAX_TOKENS":
                    # Try with a higher token limit
                    if max_tokens is None or max_tokens < 50:
                        return self.generate(
                            prompt, max_tokens=50, temperature=temperature, **kwargs
                        )
                    else:
                        # If we already tried with higher tokens, return empty
                        output = ""

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

    def _get_rates(self, input_tokens: int) -> tuple[float, float]:
        """
        Return (input_rate, output_rate) in USD per 1M tokens.

        If USE_TIERED_PRICING is False, always return base-tier PRICING.
        If True, and model has a tier cutoff, choose high-tier when input_tokens > cutoff.
        """
        if not USE_TIERED_PRICING:
            return PRICING.get(self.model_name, (0.0, 0.0))

        tier_info = PRICING_TIERED.get(self.model_name)
        if not tier_info:
            return PRICING.get(self.model_name, (0.0, 0.0))

        cutoff, low_rates, high_rates = tier_info
        return low_rates if input_tokens <= cutoff else high_rates

    def estimate_cost(
        self,
        prompt: str,
        response: str = "",
        *,
        input_tokens: Optional[int] = None,
        output_tokens: Optional[int] = None,
    ) -> float:
        """
        Estimate USD cost for a single prompt/response pair.

        - If explicit token counts provided, they're used directly.
        - Else we perform a rough estimate via `rough_token_estimate()`.
        - Ignores context caching, min charges, taxes, etc.
        """
        if input_tokens is None:
            input_tokens = rough_token_estimate(prompt)
        if output_tokens is None:
            output_tokens = rough_token_estimate(response)

        in_rate, out_rate = self._get_rates(input_tokens)
        if in_rate == 0.0 and out_rate == 0.0:
            return 0.0

        M = 1_000_000  # billing unit
        return (input_tokens / M) * in_rate + (output_tokens / M) * out_rate

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
                "pricing": PRICING.get(self.model_name, (0, 0)),
                "supported_models": SUPPORTED_MODELS,
            }
        )
        return info
