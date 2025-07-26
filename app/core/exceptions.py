"""
Custom exception hierarchy for NovaEval API.

This module defines application-specific exceptions for better error handling
and structured error responses.
"""

from typing import Any, Optional


class NovaEvalAPIError(Exception):
    """Base exception for all NovaEval API errors."""

    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[dict[str, Any]] = None,
    ):
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.details = details or {}
        super().__init__(self.message)


class ValidationError(NovaEvalAPIError):
    """Raised when input validation fails."""

    pass


class ComponentNotFoundError(NovaEvalAPIError):
    """Raised when a requested component (model, dataset, scorer) is not found."""

    pass


class ConfigurationError(NovaEvalAPIError):
    """Raised when configuration is invalid or incomplete."""

    pass


class EvaluationError(NovaEvalAPIError):
    """Raised when evaluation operations fail."""

    pass


class ModelAPIError(NovaEvalAPIError):
    """Raised when model API calls fail."""

    pass


class DatasetError(NovaEvalAPIError):
    """Raised when dataset operations fail."""

    pass


class ScorerError(NovaEvalAPIError):
    """Raised when scorer operations fail."""

    pass


class TaskManagerError(NovaEvalAPIError):
    """Raised when task manager operations fail."""

    pass


class ResourceLimitError(NovaEvalAPIError):
    """Raised when resource limits are exceeded."""

    pass


class AuthenticationError(NovaEvalAPIError):
    """Raised when authentication fails."""

    pass


class RateLimitError(NovaEvalAPIError):
    """Raised when rate limits are exceeded."""

    pass


class TimeoutError(NovaEvalAPIError):
    """Raised when operations timeout."""

    pass
