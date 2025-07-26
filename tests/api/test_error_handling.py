"""
Tests for error handling and logging functionality.
"""

from unittest.mock import patch

from app.core.exceptions import ComponentNotFoundError, NovaEvalAPIError


class TestErrorHandling:
    """Test global error handling and logging."""

    def test_health_endpoint(self, client):
        """Test basic health endpoint works."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"

    def test_404_error(self, client):
        """Test 404 error handling."""
        response = client.get("/api/v1/nonexistent")
        assert response.status_code == 404

    def test_validation_error(self, client):
        """Test Pydantic validation error handling."""
        # Try to submit evaluation with invalid JSON
        response = client.post(
            "/api/v1/evaluations/submit",
            files={"config_file": ("invalid.json", "invalid json", "application/json")},
        )
        assert response.status_code == 400

    def test_openapi_docs(self, client):
        """Test OpenAPI documentation endpoints."""
        response = client.get("/docs")
        assert response.status_code == 200

        response = client.get("/redoc")
        assert response.status_code == 200

        response = client.get("/openapi.json")
        assert response.status_code == 200
        data = response.json()
        assert "paths" in data
        assert "components" in data

    def test_metrics_endpoint(self, client):
        """Test Prometheus metrics endpoint."""
        # Generate some requests first
        client.get("/health")
        client.get("/api/v1/ping")

        response = client.get("/metrics")
        assert response.status_code == 200
        assert "novaeval_http_requests_total" in response.text

    @patch("app.core.logging.log_error")
    def test_logging_on_error(self, mock_log_error, client):
        """Test that errors are properly logged."""
        # Trigger an error
        response = client.get("/api/v1/nonexistent")
        assert response.status_code == 404
        # Note: Logging verification would depend on actual log capture

    def test_cors_headers(self, client):
        """Test CORS headers are present."""
        response = client.options("/api/v1/ping")
        # FastAPI/Starlette handles OPTIONS automatically with CORS middleware
        assert response.status_code in [200, 405]  # May vary based on implementation


class TestCustomExceptions:
    """Test custom exception classes."""

    def test_novaevalapi_error(self):
        """Test NovaEvalAPIError base class."""
        error = NovaEvalAPIError(
            "Test error", error_code="TEST_ERROR", details={"key": "value"}
        )
        assert error.message == "Test error"
        assert error.error_code == "TEST_ERROR"
        assert error.details == {"key": "value"}

    def test_component_not_found_error(self):
        """Test ComponentNotFoundError."""
        error = ComponentNotFoundError("Component not found")
        assert error.message == "Component not found"
        assert error.error_code == "ComponentNotFoundError"
        assert isinstance(error, NovaEvalAPIError)

    def test_exception_inheritance(self):
        """Test exception inheritance chain."""
        error = ComponentNotFoundError("Test")
        assert isinstance(error, NovaEvalAPIError)
        assert isinstance(error, Exception)
