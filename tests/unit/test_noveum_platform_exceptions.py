"""
Unit tests for Noveum Platform API exceptions.
"""

import pytest

from novaeval.noveum_platform.exceptions import (
    AuthenticationError,
    ConflictError,
    ForbiddenError,
    NotFoundError,
    NoveumAPIError,
    RateLimitError,
    ServerError,
    ValidationError,
)


class TestNoveumAPIError:
    """Test cases for NoveumAPIError base exception."""

    @pytest.mark.unit
    def test_init_basic(self):
        """Test basic initialization."""
        error = NoveumAPIError("Test error", 500)

        assert error.status_code == 500
        assert error.message == "Test error"
        assert error.response_body is None
        assert str(error) == "Test error (HTTP 500)"

    @pytest.mark.unit
    def test_init_with_response_body(self):
        """Test initialization with response body."""
        response_body = {"error": "Something went wrong", "details": "More info"}
        error = NoveumAPIError("Test error", 500, response_body)

        assert error.status_code == 500
        assert error.message == "Test error"
        assert error.response_body == response_body

    @pytest.mark.unit
    def test_inheritance(self):
        """Test that NoveumAPIError inherits from Exception."""
        error = NoveumAPIError("Test error", 500)
        assert isinstance(error, Exception)


class TestAuthenticationError:
    """Test cases for AuthenticationError."""

    @pytest.mark.unit
    def test_init_default(self):
        """Test initialization with default message."""
        error = AuthenticationError()

        assert error.status_code == 401
        assert error.message == "Authentication failed"
        assert error.response_body is None
        assert str(error) == "Authentication failed (HTTP 401)"

    @pytest.mark.unit
    def test_init_custom_message(self):
        """Test initialization with custom message."""
        error = AuthenticationError("Invalid API key")

        assert error.status_code == 401
        assert error.message == "Invalid API key"

    @pytest.mark.unit
    def test_init_with_response_body(self):
        """Test initialization with response body."""
        response_body = {"error": "Invalid token"}
        error = AuthenticationError("Custom auth error", response_body)

        assert error.status_code == 401
        assert error.response_body == response_body

    @pytest.mark.unit
    def test_inheritance(self):
        """Test inheritance from NoveumAPIError."""
        error = AuthenticationError()
        assert isinstance(error, NoveumAPIError)
        assert isinstance(error, Exception)


class TestValidationError:
    """Test cases for ValidationError."""

    @pytest.mark.unit
    def test_init_default(self):
        """Test initialization with default message."""
        error = ValidationError()

        assert error.status_code == 400
        assert error.message == "Invalid request format"
        assert str(error) == "Invalid request format (HTTP 400)"

    @pytest.mark.unit
    def test_init_custom_message(self):
        """Test initialization with custom message."""
        error = ValidationError("Missing required field")

        assert error.status_code == 400
        assert error.message == "Missing required field"

    @pytest.mark.unit
    def test_init_with_response_body(self):
        """Test initialization with response body."""
        response_body = {"error": "Validation failed", "fields": ["name", "slug"]}
        error = ValidationError("Validation failed", response_body)

        assert error.status_code == 400
        assert error.response_body == response_body

    @pytest.mark.unit
    def test_inheritance(self):
        """Test inheritance from NoveumAPIError."""
        error = ValidationError()
        assert isinstance(error, NoveumAPIError)


class TestForbiddenError:
    """Test cases for ForbiddenError."""

    @pytest.mark.unit
    def test_init_default(self):
        """Test initialization with default message."""
        error = ForbiddenError()

        assert error.status_code == 403
        assert error.message == "Access forbidden"
        assert str(error) == "Access forbidden (HTTP 403)"

    @pytest.mark.unit
    def test_init_custom_message(self):
        """Test initialization with custom message."""
        error = ForbiddenError("Organization mismatch")

        assert error.status_code == 403
        assert error.message == "Organization mismatch"

    @pytest.mark.unit
    def test_inheritance(self):
        """Test inheritance from NoveumAPIError."""
        error = ForbiddenError()
        assert isinstance(error, NoveumAPIError)


class TestNotFoundError:
    """Test cases for NotFoundError."""

    @pytest.mark.unit
    def test_init_default(self):
        """Test initialization with default message."""
        error = NotFoundError()

        assert error.status_code == 404
        assert error.message == "Resource not found"
        assert str(error) == "Resource not found (HTTP 404)"

    @pytest.mark.unit
    def test_init_custom_message(self):
        """Test initialization with custom message."""
        error = NotFoundError("Dataset not found")

        assert error.status_code == 404
        assert error.message == "Dataset not found"

    @pytest.mark.unit
    def test_init_with_response_body(self):
        """Test initialization with response body."""
        response_body = {"error": "Not found", "resource": "dataset/test-slug"}
        error = NotFoundError("Dataset not found", response_body)

        assert error.status_code == 404
        assert error.response_body == response_body

    @pytest.mark.unit
    def test_inheritance(self):
        """Test inheritance from NoveumAPIError."""
        error = NotFoundError()
        assert isinstance(error, NoveumAPIError)


class TestConflictError:
    """Test cases for ConflictError."""

    @pytest.mark.unit
    def test_init_default(self):
        """Test initialization with default message."""
        error = ConflictError()

        assert error.status_code == 409
        assert error.message == "Request conflict"
        assert str(error) == "Request conflict (HTTP 409)"

    @pytest.mark.unit
    def test_init_custom_message(self):
        """Test initialization with custom message."""
        error = ConflictError("Trace is immutable")

        assert error.status_code == 409
        assert error.message == "Trace is immutable"

    @pytest.mark.unit
    def test_inheritance(self):
        """Test inheritance from NoveumAPIError."""
        error = ConflictError()
        assert isinstance(error, NoveumAPIError)


class TestRateLimitError:
    """Test cases for RateLimitError."""

    @pytest.mark.unit
    def test_init_default(self):
        """Test initialization with default message."""
        error = RateLimitError()

        assert error.status_code == 429
        assert error.message == "Rate limit exceeded"
        assert str(error) == "Rate limit exceeded (HTTP 429)"

    @pytest.mark.unit
    def test_init_custom_message(self):
        """Test initialization with custom message."""
        error = RateLimitError("Too many requests")

        assert error.status_code == 429
        assert error.message == "Too many requests"

    @pytest.mark.unit
    def test_init_with_response_body(self):
        """Test initialization with response body."""
        response_body = {"error": "Rate limit", "retry_after": 60}
        error = RateLimitError("Rate limit exceeded", response_body)

        assert error.status_code == 429
        assert error.response_body == response_body

    @pytest.mark.unit
    def test_inheritance(self):
        """Test inheritance from NoveumAPIError."""
        error = RateLimitError()
        assert isinstance(error, NoveumAPIError)


class TestServerError:
    """Test cases for ServerError."""

    @pytest.mark.unit
    def test_init_default(self):
        """Test initialization with default message."""
        error = ServerError()

        assert error.status_code == 500
        assert error.message == "Internal server error"
        assert str(error) == "Internal server error (HTTP 500)"

    @pytest.mark.unit
    def test_init_custom_message(self):
        """Test initialization with custom message."""
        error = ServerError("Database connection failed")

        assert error.status_code == 500
        assert error.message == "Database connection failed"

    @pytest.mark.unit
    def test_init_with_response_body(self):
        """Test initialization with response body."""
        response_body = {"error": "Server error", "trace_id": "abc123"}
        error = ServerError("Internal server error", 500, response_body)

        assert error.status_code == 500
        assert error.response_body == response_body

    @pytest.mark.unit
    def test_inheritance(self):
        """Test inheritance from NoveumAPIError."""
        error = ServerError()
        assert isinstance(error, NoveumAPIError)
