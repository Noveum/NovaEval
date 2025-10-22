"""
Unit tests for Noveum Platform API client - Core functionality.
"""

import os
from unittest.mock import Mock, patch

import pytest

from novaeval.noveum_platform.client import NoveumClient
from novaeval.noveum_platform.exceptions import (
    AuthenticationError,
    ConflictError,
    ForbiddenError,
    NotFoundError,
    RateLimitError,
    ServerError,
    ValidationError,
)
from novaeval.noveum_platform.utils import handle_response

BASE_URL = "https://noveum.ai/api/v1/"


class TestNoveumClientInit:
    """Test cases for NoveumClient initialization."""

    @patch.dict(os.environ, {"NOVEUM_API_KEY": "test-key"})
    @pytest.mark.unit
    def test_init_with_env_vars(self):
        """Test initialization using environment variables."""
        with patch(
            "novaeval.noveum_platform.utils.requests.Session"
        ) as mock_session_class:
            client = NoveumClient()

            assert client.api_key == "test-key"
            assert client.base_url == "https://noveum.ai"
            assert client.timeout == 30.0

            # Check session headers
            mock_session_class.return_value.headers.update.assert_called_once_with(
                {
                    "Authorization": "Bearer test-key",
                    "Content-Type": "application/json",
                    "User-Agent": "NovaEval/0.5.3",
                }
            )

    @pytest.mark.unit
    def test_init_with_params(self):
        """Test initialization with direct parameters."""
        with patch("novaeval.noveum_platform.utils.requests.Session"):
            client = NoveumClient(
                api_key="direct-key",
                base_url="https://api.test.com",
                timeout=60.0,
            )

            assert client.api_key == "direct-key"
            assert client.base_url == "https://api.test.com"
            assert client.timeout == 60.0

    @pytest.mark.unit
    def test_init_base_url_stripping(self):
        """Test that base_url trailing slash is stripped."""
        with patch("novaeval.noveum_platform.utils.requests.Session"):
            client = NoveumClient(api_key="test-key", base_url="https://api.test.com/")
            assert client.base_url == "https://api.test.com"

    @pytest.mark.unit
    def test_init_no_api_key_raises_error(self):
        """Test that missing API key raises ValueError."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError) as exc_info:
                NoveumClient()

            assert "API key is required" in str(exc_info.value)

    @pytest.mark.unit
    def test_init_no_organization_id(self):
        """Test initialization without organization ID."""
        with (
            patch("novaeval.noveum_platform.utils.requests.Session") as mock_session,
            patch.dict(os.environ, {"NOVEUM_API_KEY": "test-key"}, clear=True),
        ):
            NoveumClient()

            # Should not set X-Organization-Id header
            mock_session.return_value.headers.update.assert_called_once_with(
                {
                    "Authorization": "Bearer test-key",
                    "Content-Type": "application/json",
                    "User-Agent": "NovaEval/0.5.3",
                }
            )


class TestNoveumClientHandleResponse:
    """Test cases for _handle_response method."""

    def setup_method(self):
        """Set up test client."""
        with patch("novaeval.noveum_platform.utils.requests.Session"):
            self.client = NoveumClient(api_key="test-key")

    @pytest.mark.unit
    def test_handle_response_success(self):
        """Test successful response handling."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"data": "test"}
        mock_response.content = b'{"data": "test"}'

        result = handle_response(mock_response)

        assert result == {"data": "test"}

    @pytest.mark.unit
    def test_handle_response_empty_content(self):
        """Test response with empty content."""
        mock_response = Mock()
        mock_response.status_code = 204
        mock_response.content = b""
        mock_response.json.return_value = {}

        result = handle_response(mock_response)

        assert result == {}

    @pytest.mark.unit
    def test_handle_response_invalid_json(self):
        """Test response with invalid JSON."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.content = b"invalid json"
        mock_response.json.side_effect = ValueError("Invalid JSON")

        result = handle_response(mock_response)

        assert result == {"error": "Invalid JSON response"}

    @pytest.mark.unit
    def test_handle_response_400_validation_error(self):
        """Test 400 status code raises ValidationError."""
        mock_response = Mock()
        mock_response.status_code = 400
        mock_response.json.return_value = {"message": "Invalid request"}
        mock_response.content = b'{"message": "Invalid request"}'

        with pytest.raises(ValidationError) as exc_info:
            handle_response(mock_response)

        assert exc_info.value.status_code == 400
        assert exc_info.value.message == "Invalid request"
        assert exc_info.value.response_body == {"message": "Invalid request"}

    @pytest.mark.unit
    def test_handle_response_401_authentication_error(self):
        """Test 401 status code raises AuthenticationError."""
        mock_response = Mock()
        mock_response.status_code = 401
        mock_response.json.return_value = {"message": "Invalid API key"}
        mock_response.content = b'{"message": "Invalid API key"}'

        with pytest.raises(AuthenticationError) as exc_info:
            handle_response(mock_response)

        assert exc_info.value.status_code == 401
        assert exc_info.value.message == "Invalid API key"

    @pytest.mark.unit
    def test_handle_response_403_forbidden_error(self):
        """Test 403 status code raises ForbiddenError."""
        mock_response = Mock()
        mock_response.status_code = 403
        mock_response.json.return_value = {"message": "Access denied"}
        mock_response.content = b'{"message": "Access denied"}'

        with pytest.raises(ForbiddenError) as exc_info:
            handle_response(mock_response)

        assert exc_info.value.status_code == 403
        assert exc_info.value.message == "Access denied"

    @pytest.mark.unit
    def test_handle_response_404_not_found_error(self):
        """Test 404 status code raises NotFoundError."""
        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.json.return_value = {"message": "Resource not found"}
        mock_response.content = b'{"message": "Resource not found"}'

        with pytest.raises(NotFoundError) as exc_info:
            handle_response(mock_response)

        assert exc_info.value.status_code == 404
        assert exc_info.value.message == "Resource not found"

    @pytest.mark.unit
    def test_handle_response_409_conflict_error(self):
        """Test 409 status code raises ConflictError."""
        mock_response = Mock()
        mock_response.status_code = 409
        mock_response.json.return_value = {"message": "Trace is immutable"}
        mock_response.content = b'{"message": "Trace is immutable"}'

        with pytest.raises(ConflictError) as exc_info:
            handle_response(mock_response)

        assert exc_info.value.status_code == 409
        assert exc_info.value.message == "Trace is immutable"

    @pytest.mark.unit
    def test_handle_response_429_rate_limit_error(self):
        """Test 429 status code raises RateLimitError."""
        mock_response = Mock()
        mock_response.status_code = 429
        mock_response.json.return_value = {"message": "Rate limit exceeded"}
        mock_response.content = b'{"message": "Rate limit exceeded"}'

        with pytest.raises(RateLimitError) as exc_info:
            handle_response(mock_response)

        assert exc_info.value.status_code == 429
        assert exc_info.value.message == "Rate limit exceeded"

    @pytest.mark.unit
    def test_handle_response_500_server_error(self):
        """Test 500 status code raises ServerError."""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.json.return_value = {"message": "Internal server error"}
        mock_response.content = b'{"message": "Internal server error"}'

        with pytest.raises(ServerError) as exc_info:
            handle_response(mock_response)

        assert exc_info.value.status_code == 500
        assert exc_info.value.message == "Internal server error"

    @pytest.mark.unit
    def test_handle_response_502_server_error(self):
        """Test 502 status code raises ServerError."""
        mock_response = Mock()
        mock_response.status_code = 502
        mock_response.json.return_value = {"message": "Bad gateway"}
        mock_response.content = b'{"message": "Bad gateway"}'

        with pytest.raises(ServerError) as exc_info:
            handle_response(mock_response)

        assert exc_info.value.status_code == 502
        assert exc_info.value.message == "Bad gateway"

    @pytest.mark.unit
    def test_handle_response_default_message(self):
        """Test default error messages when API doesn't provide message."""
        mock_response = Mock()
        mock_response.status_code = 400
        mock_response.json.return_value = {}
        mock_response.content = b"{}"

        with pytest.raises(ValidationError) as exc_info:
            handle_response(mock_response)

        assert exc_info.value.message == "Invalid request format"


class TestNoveumClientTraces:
    """Test cases for trace-related methods."""

    def setup_method(self):
        """Set up test client."""
        with patch("novaeval.noveum_platform.utils.requests.Session") as mock_session:
            self.client = NoveumClient(api_key="test-key")
            self.mock_session = mock_session.return_value

    @pytest.mark.unit
    def test_ingest_traces(self):
        """Test ingest_traces method."""
        traces = [{"trace_id": "1"}, {"trace_id": "2"}]
        expected_json = {"traces": traces}
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"ingested": 2}
        mock_response.content = b'{"ingested": 2}'
        self.mock_session.post.return_value = mock_response

        with patch(
            "novaeval.noveum_platform.utils.handle_response",
            return_value={"ingested": 2},
        ):
            result = self.client.ingest_traces(traces)

            assert result == {"ingested": 2}
            self.mock_session.post.assert_called_once_with(
                f"{BASE_URL}traces", json=expected_json, timeout=30.0
            )

    @pytest.mark.unit
    def test_ingest_trace(self):
        """Test ingest_trace method."""
        trace = {"trace_id": "1", "name": "test"}
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"ingested": 1}
        mock_response.content = b'{"ingested": 1}'
        self.mock_session.post.return_value = mock_response

        with patch(
            "novaeval.noveum_platform.utils.handle_response",
            return_value={"ingested": 1},
        ):
            result = self.client.ingest_trace(trace)

            assert result == {"ingested": 1}
            self.mock_session.post.assert_called_once_with(
                f"{BASE_URL}traces/single", json=trace, timeout=30.0
            )

    @pytest.mark.unit
    def test_query_traces(self):
        """Test query_traces method."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"traces": []}
        mock_response.content = b'{"traces": []}'
        self.mock_session.get.return_value = mock_response

        with patch(
            "novaeval.noveum_platform.utils.handle_response",
            return_value={"traces": []},
        ):
            result = self.client.query_traces(project="test", size=10)

            assert result == {"traces": []}
            self.mock_session.get.assert_called_once()
            call_args = self.mock_session.get.call_args
            assert call_args[0][0] == f"{BASE_URL}traces"
            assert "params" in call_args[1]

    @pytest.mark.unit
    def test_get_trace(self):
        """Test get_trace method."""
        trace_id = "trace-123"
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"trace_id": trace_id}
        mock_response.content = b'{"trace_id": "trace-123"}'
        self.mock_session.get.return_value = mock_response

        with patch(
            "novaeval.noveum_platform.utils.handle_response",
            return_value={"trace_id": trace_id},
        ):
            result = self.client.get_trace(trace_id)

            assert result == {"trace_id": trace_id}
            self.mock_session.get.assert_called_once_with(
                f"{BASE_URL}traces/{trace_id}", timeout=30.0
            )

    @pytest.mark.unit
    def test_get_connection_status(self):
        """Test get_connection_status method."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "connected"}
        mock_response.content = b'{"status": "connected"}'
        self.mock_session.get.return_value = mock_response

        with patch(
            "novaeval.noveum_platform.utils.handle_response",
            return_value={"status": "connected"},
        ):
            result = self.client.get_connection_status()

            assert result == {"status": "connected"}
            self.mock_session.get.assert_called_once_with(
                f"{BASE_URL}traces/connection-status", timeout=30.0
            )

    @pytest.mark.unit
    def test_get_trace_spans(self):
        """Test get_trace_spans method."""
        trace_id = "trace-123"
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"spans": []}
        mock_response.content = b'{"spans": []}'
        self.mock_session.get.return_value = mock_response

        with patch(
            "novaeval.noveum_platform.utils.handle_response", return_value={"spans": []}
        ):
            result = self.client.get_trace_spans(trace_id)

            assert result == {"spans": []}
            self.mock_session.get.assert_called_once_with(
                f"{BASE_URL}traces/{trace_id}/spans", timeout=30.0
            )
