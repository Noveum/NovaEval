"""
Configuration package for NovaEval.

This package provides YAML-based configuration for evaluation jobs,
making it easy to integrate NovaEval into CI/CD pipelines.
"""

from novaeval.config.job_config import EvaluationJobConfig, JobRunner
from novaeval.config.schema import ConfigSchema

__all__ = ["ConfigSchema", "EvaluationJobConfig", "JobRunner"]
