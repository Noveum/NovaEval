"""
NovaEval: A comprehensive, extensible AI model evaluation framework.

NovaEval provides a unified interface for evaluating AI models across different
providers, datasets, and metrics. It supports both standalone usage and integration
with the Noveum.ai platform for enhanced analytics and reporting.
"""

__version__ = "0.1.0"
__title__ = "novaeval"
__author__ = "Noveum Team"
__license__ = "Apache 2.0"

# Core imports
from novaeval.evaluators.base import BaseEvaluator
from novaeval.evaluators.standard import Evaluator

# Dataset imports
from novaeval.datasets.base import BaseDataset

# Model imports  
from novaeval.models.base import BaseModel
from novaeval.models.openai import OpenAIModel
from novaeval.models.anthropic import AnthropicModel

# Scorer imports
from novaeval.scorers.base import BaseScorer
from novaeval.scorers.accuracy import AccuracyScorer

# Utility imports
from novaeval.utils.config import Config
from novaeval.utils.logging import setup_logging, get_logger

__all__ = [
    # Core classes
    "BaseEvaluator",
    "Evaluator",
    
    # Datasets
    "BaseDataset",
    
    # Models
    "BaseModel",
    "OpenAIModel",
    "AnthropicModel",
    
    # Scorers
    "BaseScorer", 
    "AccuracyScorer",
    
    # Utilities
    "Config",
    "setup_logging",
    "get_logger",
    
    # Metadata
    "__version__",
    "__title__",
    "__author__",
    "__license__"
]

