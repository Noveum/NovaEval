"""
LLM utility functions for NovaEval.

This module contains shared utilities for interacting with language models,
    supporting various model types and interfaces.
"""

from typing import Any

# LangChain detection
try:
    from langchain_core.language_models.base import BaseLanguageModel

    _HAS_LANGCHAIN = True
except ImportError:
    _HAS_LANGCHAIN = False
    BaseLanguageModel = Any  # type: ignore


def call_llm(model: Any, prompt: str) -> str:
    """
    Helper to call the LLM, supporting both string (OpenAI) and LangChain LLM objects.

    Args:
        model: The language model to use. Can be:
            - A LangChain BaseLanguageModel instance
            - A string model name (not implemented)
            - A NovaEval-style model with .generate method
        prompt: The prompt to send to the model

    Returns:
        The model's response as a string

    Raises:
        ValueError: If LangChain LLM doesn't support invoke or generate
        NotImplementedError: If string-based model calling is attempted
    """
    if _HAS_LANGCHAIN and isinstance(model, BaseLanguageModel):
        # LangChain LLM: use .invoke or .generate
        if hasattr(model, "invoke"):
            return model.invoke(prompt)
        elif hasattr(model, "generate"):
            return model.generate([prompt]).generations[0][0].text
        else:
            raise ValueError("LangChain LLM does not support invoke or generate.")
    elif isinstance(model, str):
        # Built-in: string model name, use OpenAI API
        raise NotImplementedError(
            "String-based LLM calling not implemented. Please provide a LangChain LLM or implement OpenAI call here."
        )
    else:
        # Assume model has a .generate method (NovaEval style)
        return model.generate(prompt)
