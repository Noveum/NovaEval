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
        Initialize a GeminiModel instance with the specified model name, API key, retry limit, and timeout.
        
        Parameters:
            model_name (str): The Gemini model variant to use (e.g., "gemini-2.5-flash").
            api_key (Optional[str]): API key for authenticating with the Gemini service. If not provided, the environment variable 'GOOGLE_API_KEY' is used.
            max_retries (int): Maximum number of retry attempts for failed requests.
            timeout (float): Timeout in seconds for API requests.
            **kwargs: Additional keyword arguments passed to the base model initializer.
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
        Generates a text completion for a given prompt using the Gemini API.
        
        Parameters:
            prompt (str): The input prompt to generate text from.
            max_tokens (Optional[int]): Maximum number of tokens in the generated output.
            temperature (Optional[float]): Sampling temperature for generation randomness.
            stop (Optional[Union[str, list[str]]]): Stop sequences (not supported by Gemini).
            **kwargs: Additional generation parameters.
        
        Returns:
            str: The generated text completion.
        """
        try:
            start_time = time.time()

            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=temperature, max_output_tokens=max_tokens, **kwargs
                ),
            )

            end_time = time.time()
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
        Generates responses for a list of prompts sequentially.
        
        Each prompt is processed individually using the `generate` method. If an error occurs for a prompt, an empty string is returned for that prompt.
        
        Parameters:
            prompts (list[str]): List of input prompts to generate responses for.
        
        Returns:
            list[str]: List of generated responses corresponding to each prompt.
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
        """
        Return the name of the model provider.
        """
        return "gemini"

    def estimate_cost(self, prompt: str, response: str = "") -> float:
        """
        Estimate the API usage cost in USD for a given prompt and response based on model-specific token pricing.
        
        Parameters:
            prompt (str): The input text sent to the model.
            response (str, optional): The generated output text. Defaults to an empty string.
        
        Returns:
            float: The estimated cost in USD for processing the prompt and response.
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
        Estimate the number of tokens in the given text using a heuristic based on word count.
        
        Parameters:
            text (str): The input text to estimate token count for.
        
        Returns:
            int: Estimated token count, calculated as the number of words multiplied by 1.3.
        """
        return int(len(text.split()) * 1.3)

    def validate_connection(self) -> bool:
        """
        Checks connectivity to the Gemini API by sending a minimal prompt.
        
        Returns:
            bool: True if a valid response is received from the API, otherwise False.
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
        Return a dictionary containing metadata about the Gemini model, including retry settings, timeout, batch support, and pricing information.
        
        Returns:
            dict: Model metadata with keys for max retries, timeout, batch support, and pricing.
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
