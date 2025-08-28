"""
Evaluators package for NovaEval.

This package contains the core evaluation logic and orchestration.
"""

from novaeval.evaluators.agent_evaluator import AgentEvaluator
from novaeval.evaluators.aggregators import (
    aggregate_by_agent_name,
    aggregate_by_task,
    aggregate_by_user,
    mean_callable,
)
from novaeval.evaluators.base import BaseEvaluator
from novaeval.evaluators.standard import Evaluator

__all__ = [
    "AgentEvaluator",
    "BaseEvaluator",
    "Evaluator",
    "aggregate_by_agent_name",
    "aggregate_by_task",
    "aggregate_by_user",
    "mean_callable",
]
