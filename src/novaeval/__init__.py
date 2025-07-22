"""
NovaEval: A comprehensive, extensible AI model evaluation framework.

NovaEval provides a unified interface for evaluating AI models across different
providers, datasets, and metrics. It supports both standalone usage and integration
with the Noveum.ai platform for enhanced analytics and reporting.
"""

__version__ = "0.4.0"
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

# Core scorer imports
from novaeval.scorers.accuracy import AccuracyScorer, ExactMatchScorer, F1Scorer
from novaeval.scorers.base import BaseScorer

# Conversational scorer imports - key ones for dialogue evaluation
from novaeval.scorers.conversational import (
    ConversationalMetricsScorer,
    ConversationCompletenessScorer,
    ConversationRelevancyScorer,
    KnowledgeRetentionScorer,
    RoleAdherenceScorer,
)

# Advanced scorer imports - key ones for common use cases
from novaeval.scorers.g_eval import GEvalScorer
from novaeval.scorers.panel_judge import PanelOfJudgesScorer
from novaeval.scorers.rag import AnswerRelevancyScorer, FaithfulnessScorer

# Utility imports
from novaeval.utils.config import Config
from novaeval.utils.logging import get_logger, setup_logging

__all__ = [
    "AccuracyScorer",
    "AnswerRelevancyScorer",
    "AnthropicModel",
    "BaseDataset",
    "BaseEvaluator",
    "BaseModel",
    "BaseScorer",
    "Config",
    "ConversationCompletenessScorer",
    "ConversationRelevancyScorer",
    "ConversationalMetricsScorer",
    "Evaluator",
    "ExactMatchScorer",
    "F1Scorer",
    "FaithfulnessScorer",
    "GEvalScorer",
    "KnowledgeRetentionScorer",
    "OpenAIModel",
    "PanelOfJudgesScorer",
    "RoleAdherenceScorer",
    "__author__",
    "__license__",
    "__title__",
    "__version__",
    "get_logger",
    "setup_logging",
]
