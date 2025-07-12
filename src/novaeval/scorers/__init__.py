"""
Scorers package for NovaEval.

This package contains scoring mechanisms for evaluating AI model outputs.
"""

from novaeval.scorers.accuracy import AccuracyScorer
from novaeval.scorers.base import BaseScorer

__all__ = ["AccuracyScorer", "BaseScorer"]
