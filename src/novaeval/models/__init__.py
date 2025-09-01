"""
Models package for NovaEval.

This package contains model implementations for different AI providers.
"""

from novaeval.models.anthropic import AnthropicModel
from novaeval.models.azure_openai import AzureOpenAIModel
from novaeval.models.base import BaseModel
from novaeval.models.gemini import GeminiModel
from novaeval.models.openai import OpenAIModel

# Optional imports for models with external dependencies
try:
    from novaeval.models.ollama import OllamaModel

    _OLLAMA_AVAILABLE = True
except ImportError:
    _OLLAMA_AVAILABLE = False

__all__ = [
    "AnthropicModel",
    "AzureOpenAIModel",
    "BaseModel",
    "GeminiModel",
    "OpenAIModel",
]

# Add optional models to __all__ if available
if _OLLAMA_AVAILABLE:
    __all__.append("OllamaModel")
