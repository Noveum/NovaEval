"""
Scorers package for NovaEval.

This package contains scoring mechanisms for evaluating AI model outputs.
"""

from novaeval.scorers.base import BaseScorer
from novaeval.scorers.accuracy import AccuracyScorer

__all__ = [
    "BaseScorer",
    "AccuracyScorer"
]

