"""
Shared utilities for Noveum Platform API modules.

This module provides common helper functions used across all API modules
to avoid code duplication and ensure consistent behavior.
"""

from typing import Any

import pydantic
import requests

from novaeval.utils.logging import get_logger

from .exceptions import (
    AuthenticationError,
    ConflictError,
    ForbiddenError,
    NotFoundError,
    RateLimitError,
    ServerError,
    ValidationError,
)

logger = get_logger(__name__)


def handle_response(response: requests.Response) -> dict[str, Any]:
    """
    Handle API response and raise appropriate exceptions for errors.

    Args:
        response: The requests.Response object from the API call

    Returns:
        Parsed JSON response as dictionary

    Raises:
        Various NoveumAPIError subclasses based on HTTP status code
    """
    try:
        response_body = response.json() if response.content else {}
    except ValueError:
        response_body = {"error": "Invalid JSON response"}

    # Check for specific error status codes
    if response.status_code == 400:
        raise ValidationError(
            message=response_body.get("message", "Invalid request format"),
            response_body=response_body,
        )
    elif response.status_code == 401:
        raise AuthenticationError(
            message=response_body.get("message", "Unauthorized - Invalid API key"),
            response_body=response_body,
        )
    elif response.status_code == 403:
        raise ForbiddenError(
            message=response_body.get(
                "message", "Forbidden (org mismatch or access denied)"
            ),
            response_body=response_body,
        )
    elif response.status_code == 404:
        raise NotFoundError(
            message=response_body.get("message", "Resource not found"),
            response_body=response_body,
        )
    elif response.status_code == 409:
        raise ConflictError(
            message=response_body.get("message", "Conflict - Trace is immutable"),
            response_body=response_body,
        )
    elif response.status_code == 429:
        raise RateLimitError(
            message=response_body.get("message", "Rate limit exceeded"),
            response_body=response_body,
        )
    elif response.status_code >= 500:
        raise ServerError(
            message=response_body.get("message", "Internal server error"),
            status_code=response.status_code,
            response_body=response_body,
        )

    # For successful responses, return the parsed JSON
    response.raise_for_status()  # This will raise for any other error status codes
    return response_body


def parse_model(model_cls: type[pydantic.BaseModel], data: dict[str, Any]) -> Any:
    """
    Parse data into a Pydantic model and convert ValidationError into our API ValidationError.

    Args:
        model_cls: The Pydantic model class to instantiate
        data: Dictionary of data to parse into the model

    Returns:
        Instantiated Pydantic model instance

    Raises:
        ValidationError: If validation fails, converted from pydantic.ValidationError
    """
    try:
        return model_cls(**data)
    except pydantic.ValidationError as e:
        error_messages = []
        for err in e.errors():
            loc = ".".join(str(p) for p in err.get("loc", []) or ["unknown"])
            msg = err.get("msg", "Validation error")
            error_messages.append(f"{loc}: {msg}")
        raise ValidationError(
            message="; ".join(error_messages),
            response_body={"validation_errors": e.errors()},
        ) from e


def create_authenticated_session(api_key: str) -> requests.Session:
    """
    Create an authenticated requests session for API calls.

    Args:
        api_key: Noveum API key

    Returns:
        Configured requests.Session with authentication headers
    """
    session = requests.Session()
    session.headers.update(
        {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "NovaEval/0.5.3",
        }
    )
    return session
