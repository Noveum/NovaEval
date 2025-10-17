"""
Custom exceptions for Noveum Platform Traces API.

This module defines all custom exceptions that can be raised by the
TracesClient when API calls fail with specific HTTP status codes.
"""

from typing import Any, Dict, Optional


class NoveumAPIError(Exception):
    """
    Base exception for all Noveum API errors.
    
    Attributes:
        status_code: HTTP status code of the failed request
        message: Human-readable error message
        response_body: Raw response body from the API (if available)
    """
    
    def __init__(
        self,
        message: str,
        status_code: int,
        response_body: Optional[Dict[str, Any]] = None
    ):
        self.status_code = status_code
        self.message = message
        self.response_body = response_body
        super().__init__(f"{message} (HTTP {status_code})")


class AuthenticationError(NoveumAPIError):
    """Raised when API authentication fails (401 Unauthorized)."""
    
    def __init__(self, message: str = "Authentication failed", response_body: Optional[Dict[str, Any]] = None):
        super().__init__(message, 401, response_body)


class ValidationError(NoveumAPIError):
    """Raised when request validation fails (400 Bad Request)."""
    
    def __init__(self, message: str = "Invalid request format", response_body: Optional[Dict[str, Any]] = None):
        super().__init__(message, 400, response_body)


class ForbiddenError(NoveumAPIError):
    """Raised when access is forbidden (403 Forbidden)."""
    
    def __init__(self, message: str = "Access forbidden", response_body: Optional[Dict[str, Any]] = None):
        super().__init__(message, 403, response_body)


class NotFoundError(NoveumAPIError):
    """Raised when a resource is not found (404 Not Found)."""
    
    def __init__(self, message: str = "Resource not found", response_body: Optional[Dict[str, Any]] = None):
        super().__init__(message, 404, response_body)


class ConflictError(NoveumAPIError):
    """Raised when there's a conflict with the request (409 Conflict)."""
    
    def __init__(self, message: str = "Request conflict", response_body: Optional[Dict[str, Any]] = None):
        super().__init__(message, 409, response_body)


class RateLimitError(NoveumAPIError):
    """Raised when rate limit is exceeded (429 Too Many Requests)."""
    
    def __init__(self, message: str = "Rate limit exceeded", response_body: Optional[Dict[str, Any]] = None):
        super().__init__(message, 429, response_body)


class ServerError(NoveumAPIError):
    """Raised when server encounters an error (500+ Internal Server Error)."""
    
    def __init__(self, message: str = "Internal server error", status_code: int = 500, response_body: Optional[Dict[str, Any]] = None):
        super().__init__(message, status_code, response_body)
