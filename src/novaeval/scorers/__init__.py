"""
Scorers package for NovaEval.

This package contains scoring mechanisms for evaluating AI model outputs.
"""

# Accuracy scorers
from novaeval.scorers.accuracy import AccuracyScorer, ExactMatchScorer, F1Scorer

# Base classes
from novaeval.scorers.base import BaseScorer, ScoreResult

# Conversational scorers
from novaeval.scorers.conversational import (
    Conversation,
    ConversationalMetricsScorer,
    ConversationCompletenessScorer,
    ConversationRelevancyScorer,
    ConversationTurn,
    KnowledgeRetentionScorer,
    RoleAdherenceScorer,
)

# G-Eval scorers
from novaeval.scorers.g_eval import GEvalCriteria, GEvalScorer

# Panel judge scorers
from novaeval.scorers.panel_judge import (
    AggregationMethod,
    JudgeConfig,
    PanelOfJudgesScorer,
    PanelResult,
    SpecializedPanelScorer,
)

# RAG scorers
from novaeval.scorers.rag import (
    AnswerRelevancyScorer,
    ContextualPrecisionScorer,
    ContextualRecallScorer,
    FaithfulnessScorer,
    RAGASScorer,
)

__all__ = [
    "AccuracyScorer",
    "AggregationMethod",
    "AnswerRelevancyScorer",
    "BaseScorer",
    "ContextualPrecisionScorer",
    "ContextualRecallScorer",
    "Conversation",
    "ConversationCompletenessScorer",
    "ConversationRelevancyScorer",
    "ConversationTurn",
    "ConversationalMetricsScorer",
    "ExactMatchScorer",
    "F1Scorer",
    "FaithfulnessScorer",
    "GEvalCriteria",
    "GEvalScorer",
    "JudgeConfig",
    "KnowledgeRetentionScorer",
    "PanelOfJudgesScorer",
    "PanelResult",
    "RAGASScorer",
    "RoleAdherenceScorer",
    "ScoreResult",
    "SpecializedPanelScorer",
]
