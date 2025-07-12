"""
NovaEval: A comprehensive, extensible AI model evaluation framework.

NovaEval provides a unified interface for evaluating AI models across different
providers, datasets, and metrics. It supports both standalone usage and integration
with the Noveum.ai platform for enhanced analytics and reporting.
"""

__version__ = "0.2.2"
__title__ = "novaeval"
__author__ = "Noveum Team"
__license__ = "Apache 2.0"

# Core imports
# Dataset imports
from novaeval.datasets.base import BaseDataset
from novaeval.evaluators.base import BaseEvaluator
from novaeval.evaluators.standard import Evaluator
from novaeval.models.anthropic import AnthropicModel

# Model imports
from novaeval.models.base import BaseModel
from novaeval.models.openai import OpenAIModel
from novaeval.scorers.accuracy import AccuracyScorer, ExactMatchScorer, F1Scorer

# Scorer imports
from novaeval.scorers.base import BaseScorer

# Utility imports
from novaeval.utils.config import Config
from novaeval.utils.logging import get_logger, setup_logging

__all__ = [
    "AccuracyScorer",
    "AnthropicModel",
    # Datasets
    "BaseDataset",
    # Core classes
    "BaseEvaluator",
    # Models
    "BaseModel",
    # Scorers
    "BaseScorer",
    # Utilities
    "Config",
    "Evaluator",
    "ExactMatchScorer",
    "F1Scorer",
    "OpenAIModel",
    "__author__",
    "__license__",
    "__title__",
    # Metadata
    "__version__",
    "get_logger",
    "setup_logging",
]
