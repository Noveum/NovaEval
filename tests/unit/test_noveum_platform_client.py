"""
Unit tests for Noveum Platform API client.
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


class TestNoveumClientInit:
    """Test cases for NoveumClient initialization."""

    @patch.dict(
        os.environ, {"NOVEUM_API_KEY": "test-key", "NOVEUM_ORGANIZATION_ID": "test-org"}
    )
    @pytest.mark.unit
    def test_init_with_env_vars(self):
        """Test initialization using environment variables."""
        with patch(
            "novaeval.noveum_platform.client.requests.Session"
        ) as mock_session_class:
            client = NoveumClient()

            assert client.api_key == "test-key"
            assert client.base_url == "https://noveum.ai"
            assert client.organization_id == "test-org"
            assert client.timeout == 30.0

            # Check session headers
            mock_session_class.return_value.headers.update.assert_any_call(
                {
                    "Authorization": "Bearer test-key",
                    "Content-Type": "application/json",
                    "User-Agent": "NovaEval/0.5.3",
                }
            )
            mock_session_class.return_value.headers.update.assert_any_call(
                {"X-Organization-Id": "test-org"}
            )

    @pytest.mark.unit
    def test_init_with_params(self):
        """Test initialization with direct parameters."""
        with patch("novaeval.noveum_platform.client.requests.Session"):
            client = NoveumClient(
                api_key="direct-key",
                base_url="https://api.test.com",
                organization_id="direct-org",
                timeout=60.0,
            )

            assert client.api_key == "direct-key"
            assert client.base_url == "https://api.test.com"
            assert client.organization_id == "direct-org"
            assert client.timeout == 60.0

    @pytest.mark.unit
    def test_init_base_url_stripping(self):
        """Test that base_url trailing slash is stripped."""
        with patch("novaeval.noveum_platform.client.requests.Session"):
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
            patch("novaeval.noveum_platform.client.requests.Session") as mock_session,
            patch.dict(os.environ, {"NOVEUM_API_KEY": "test-key"}, clear=True),
        ):
            client = NoveumClient()

            assert client.organization_id is None
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
        with patch("novaeval.noveum_platform.client.requests.Session"):
            self.client = NoveumClient(api_key="test-key")

    @pytest.mark.unit
    def test_handle_response_success(self):
        """Test successful response handling."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"data": "test"}
        mock_response.content = b'{"data": "test"}'

        result = self.client._handle_response(mock_response)

        assert result == {"data": "test"}

    @pytest.mark.unit
    def test_handle_response_empty_content(self):
        """Test response with empty content."""
        mock_response = Mock()
        mock_response.status_code = 204
        mock_response.content = b""
        mock_response.json.return_value = {}

        result = self.client._handle_response(mock_response)

        assert result == {}

    @pytest.mark.unit
    def test_handle_response_invalid_json(self):
        """Test response with invalid JSON."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.content = b"invalid json"
        mock_response.json.side_effect = ValueError("Invalid JSON")

        result = self.client._handle_response(mock_response)

        assert result == {"error": "Invalid JSON response"}

    @pytest.mark.unit
    def test_handle_response_400_validation_error(self):
        """Test 400 status code raises ValidationError."""
        mock_response = Mock()
        mock_response.status_code = 400
        mock_response.json.return_value = {"message": "Invalid request"}
        mock_response.content = b'{"message": "Invalid request"}'

        with pytest.raises(ValidationError) as exc_info:
            self.client._handle_response(mock_response)

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
            self.client._handle_response(mock_response)

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
            self.client._handle_response(mock_response)

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
            self.client._handle_response(mock_response)

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
            self.client._handle_response(mock_response)

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
            self.client._handle_response(mock_response)

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
            self.client._handle_response(mock_response)

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
            self.client._handle_response(mock_response)

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
            self.client._handle_response(mock_response)

        assert exc_info.value.message == "Invalid request format"


class TestNoveumClientTraces:
    """Test cases for trace-related methods."""

    def setup_method(self):
        """Set up test client."""
        with patch("novaeval.noveum_platform.client.requests.Session") as mock_session:
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

        with patch.object(
            self.client, "_handle_response", return_value={"ingested": 2}
        ) as mock_handle:
            result = self.client.ingest_traces(traces)

            assert result == {"ingested": 2}
            self.mock_session.post.assert_called_once_with(
                "https://noveum.ai/api/v1/traces", json=expected_json, timeout=30.0
            )
            mock_handle.assert_called_once_with(mock_response)

    @pytest.mark.unit
    def test_ingest_trace(self):
        """Test ingest_trace method."""
        trace = {"trace_id": "1", "name": "test"}
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"ingested": 1}
        mock_response.content = b'{"ingested": 1}'
        self.mock_session.post.return_value = mock_response

        with patch.object(
            self.client, "_handle_response", return_value={"ingested": 1}
        ) as mock_handle:
            result = self.client.ingest_trace(trace)

            assert result == {"ingested": 1}
            self.mock_session.post.assert_called_once_with(
                "https://noveum.ai/api/v1/traces/single", json=trace, timeout=30.0
            )
            mock_handle.assert_called_once_with(mock_response)

    @pytest.mark.unit
    def test_query_traces(self):
        """Test query_traces method."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"traces": []}
        mock_response.content = b'{"traces": []}'
        self.mock_session.get.return_value = mock_response

        with patch.object(
            self.client, "_handle_response", return_value={"traces": []}
        ) as mock_handle:
            result = self.client.query_traces(project="test", size=10)

            assert result == {"traces": []}
            self.mock_session.get.assert_called_once()
            call_args = self.mock_session.get.call_args
            assert call_args[0][0] == "https://noveum.ai/api/v1/traces"
            assert "params" in call_args[1]
            mock_handle.assert_called_once_with(mock_response)

    @pytest.mark.unit
    def test_get_trace(self):
        """Test get_trace method."""
        trace_id = "trace-123"
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"trace_id": trace_id}
        mock_response.content = b'{"trace_id": "trace-123"}'
        self.mock_session.get.return_value = mock_response

        with patch.object(
            self.client, "_handle_response", return_value={"trace_id": trace_id}
        ) as mock_handle:
            result = self.client.get_trace(trace_id)

            assert result == {"trace_id": trace_id}
            self.mock_session.get.assert_called_once_with(
                f"https://noveum.ai/api/v1/traces/{trace_id}", timeout=30.0
            )
            mock_handle.assert_called_once_with(mock_response)

    @pytest.mark.unit
    def test_get_directory_tree(self):
        """Test get_directory_tree method."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"tree": {}}
        mock_response.content = b'{"tree": {}}'
        self.mock_session.get.return_value = mock_response

        with patch.object(
            self.client, "_handle_response", return_value={"tree": {}}
        ) as mock_handle:
            result = self.client.get_directory_tree()

            assert result == {"tree": {}}
            self.mock_session.get.assert_called_once_with(
                "https://noveum.ai/api/v1/traces/directory-tree", timeout=30.0
            )
            mock_handle.assert_called_once_with(mock_response)

    @pytest.mark.unit
    def test_get_connection_status(self):
        """Test get_connection_status method."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "connected"}
        mock_response.content = b'{"status": "connected"}'
        self.mock_session.get.return_value = mock_response

        with patch.object(
            self.client, "_handle_response", return_value={"status": "connected"}
        ) as mock_handle:
            result = self.client.get_connection_status()

            assert result == {"status": "connected"}
            self.mock_session.get.assert_called_once_with(
                "https://noveum.ai/api/v1/traces/connection-status", timeout=30.0
            )
            mock_handle.assert_called_once_with(mock_response)

    @pytest.mark.unit
    def test_get_trace_spans(self):
        """Test get_trace_spans method."""
        trace_id = "trace-123"
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"spans": []}
        mock_response.content = b'{"spans": []}'
        self.mock_session.get.return_value = mock_response

        with patch.object(
            self.client, "_handle_response", return_value={"spans": []}
        ) as mock_handle:
            result = self.client.get_trace_spans(trace_id)

            assert result == {"spans": []}
            self.mock_session.get.assert_called_once_with(
                f"https://noveum.ai/api/v1/traces/{trace_id}/spans", timeout=30.0
            )
            mock_handle.assert_called_once_with(mock_response)


class TestNoveumClientDatasets:
    """Test cases for dataset-related methods."""

    def setup_method(self):
        """Set up test client."""
        with patch("novaeval.noveum_platform.client.requests.Session") as mock_session:
            self.client = NoveumClient(api_key="test-key")
            self.mock_session = mock_session.return_value

    @pytest.mark.unit
    def test_create_dataset(self):
        """Test create_dataset method."""
        mock_response = Mock()
        mock_response.status_code = 201
        mock_response.json.return_value = {"slug": "test-dataset"}
        mock_response.content = b'{"slug": "test-dataset"}'
        self.mock_session.post.return_value = mock_response

        with patch.object(
            self.client, "_handle_response", return_value={"slug": "test-dataset"}
        ) as mock_handle:
            result = self.client.create_dataset(
                name="Test Dataset", description="A test dataset"
            )

            assert result == {"slug": "test-dataset"}
            self.mock_session.post.assert_called_once()
            call_args = self.mock_session.post.call_args
            assert call_args[0][0] == "https://noveum.ai/api/v1/datasets"
            assert "json" in call_args[1]
            mock_handle.assert_called_once_with(mock_response)

    @pytest.mark.unit
    def test_list_datasets(self):
        """Test list_datasets method."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"datasets": []}
        mock_response.content = b'{"datasets": []}'
        self.mock_session.get.return_value = mock_response

        with patch.object(
            self.client, "_handle_response", return_value={"datasets": []}
        ) as mock_handle:
            result = self.client.list_datasets(limit=10, visibility="public")

            assert result == {"datasets": []}
            self.mock_session.get.assert_called_once()
            call_args = self.mock_session.get.call_args
            assert call_args[0][0] == "https://noveum.ai/api/v1/datasets"
            assert "params" in call_args[1]
            mock_handle.assert_called_once_with(mock_response)

    @pytest.mark.unit
    def test_get_dataset(self):
        """Test get_dataset method."""
        slug = "test-dataset"
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"slug": slug}
        mock_response.content = b'{"slug": "test-dataset"}'
        self.mock_session.get.return_value = mock_response

        with patch.object(
            self.client, "_handle_response", return_value={"slug": slug}
        ) as mock_handle:
            result = self.client.get_dataset(slug)

            assert result == {"slug": slug}
            self.mock_session.get.assert_called_once_with(
                f"https://noveum.ai/api/v1/datasets/{slug}", timeout=30.0
            )
            mock_handle.assert_called_once_with(mock_response)

    @pytest.mark.unit
    def test_update_dataset(self):
        """Test update_dataset method."""
        slug = "test-dataset"
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"slug": slug, "updated": True}
        mock_response.content = b'{"slug": "test-dataset", "updated": true}'
        self.mock_session.put.return_value = mock_response

        with patch.object(
            self.client,
            "_handle_response",
            return_value={"slug": slug, "updated": True},
        ) as mock_handle:
            result = self.client.update_dataset(slug, name="Updated Dataset")

            assert result == {"slug": slug, "updated": True}
            self.mock_session.put.assert_called_once()
            call_args = self.mock_session.put.call_args
            assert call_args[0][0] == f"https://noveum.ai/api/v1/datasets/{slug}"
            assert "json" in call_args[1]
            mock_handle.assert_called_once_with(mock_response)

    @pytest.mark.unit
    def test_delete_dataset(self):
        """Test delete_dataset method."""
        slug = "test-dataset"
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"deleted": True}
        mock_response.content = b'{"deleted": true}'
        self.mock_session.delete.return_value = mock_response

        with patch.object(
            self.client, "_handle_response", return_value={"deleted": True}
        ) as mock_handle:
            result = self.client.delete_dataset(slug)

            assert result == {"deleted": True}
            self.mock_session.delete.assert_called_once_with(
                f"https://noveum.ai/api/v1/datasets/{slug}", timeout=30.0
            )
            mock_handle.assert_called_once_with(mock_response)

    @pytest.mark.unit
    def test_list_dataset_versions(self):
        """Test list_dataset_versions method."""
        dataset_slug = "test-dataset"
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"versions": []}
        mock_response.content = b'{"versions": []}'
        self.mock_session.get.return_value = mock_response

        with patch.object(
            self.client, "_handle_response", return_value={"versions": []}
        ) as mock_handle:
            result = self.client.list_dataset_versions(dataset_slug)

            assert result == {"versions": []}
            self.mock_session.get.assert_called_once_with(
                f"https://noveum.ai/api/v1/datasets/{dataset_slug}/versions",
                params={"limit": 50, "offset": 0},
                timeout=30.0,
            )
            mock_handle.assert_called_once_with(mock_response)

    @pytest.mark.unit
    def test_create_dataset_version(self):
        """Test create_dataset_version method."""
        dataset_slug = "test-dataset"
        version_data = {"version": "1.0.0", "description": "Initial version"}
        mock_response = Mock()
        mock_response.status_code = 201
        mock_response.json.return_value = {"version": "1.0.0"}
        mock_response.content = b'{"version": "1.0.0"}'
        self.mock_session.post.return_value = mock_response

        with patch.object(
            self.client, "_handle_response", return_value={"version": "1.0.0"}
        ) as mock_handle:
            result = self.client.create_dataset_version(dataset_slug, version_data)

            assert result == {"version": "1.0.0"}
            self.mock_session.post.assert_called_once()
            call_args = self.mock_session.post.call_args
            assert (
                call_args[0][0]
                == f"https://noveum.ai/api/v1/datasets/{dataset_slug}/versions"
            )
            assert "json" in call_args[1]
            mock_handle.assert_called_once_with(mock_response)

    @pytest.mark.unit
    def test_get_dataset_version(self):
        """Test get_dataset_version method."""
        dataset_slug = "test-dataset"
        version = "1.0.0"
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"version": version}
        mock_response.content = b'{"version": "1.0.0"}'
        self.mock_session.get.return_value = mock_response

        with patch.object(
            self.client, "_handle_response", return_value={"version": version}
        ) as mock_handle:
            result = self.client.get_dataset_version(dataset_slug, version)

            assert result == {"version": version}
            self.mock_session.get.assert_called_once_with(
                f"https://noveum.ai/api/v1/datasets/{dataset_slug}/versions/{version}",
                timeout=30.0,
            )
            mock_handle.assert_called_once_with(mock_response)

    @pytest.mark.unit
    def test_publish_dataset_version(self):
        """Test publish_dataset_version method."""
        dataset_slug = "test-dataset"
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"published": True}
        mock_response.content = b'{"published": true}'
        self.mock_session.post.return_value = mock_response

        with patch.object(
            self.client, "_handle_response", return_value={"published": True}
        ) as mock_handle:
            result = self.client.publish_dataset_version(dataset_slug)

            assert result == {"published": True}
            self.mock_session.post.assert_called_once_with(
                f"https://noveum.ai/api/v1/datasets/{dataset_slug}/versions/publish",
                timeout=30.0,
            )
            mock_handle.assert_called_once_with(mock_response)

    @pytest.mark.unit
    def test_get_dataset_versions_diff(self):
        """Test get_dataset_versions_diff method."""
        dataset_slug = "test-dataset"
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"diff": {"added": 1, "deleted": 0}}
        mock_response.content = b'{"diff": {"added": 1, "deleted": 0}}'
        self.mock_session.get.return_value = mock_response

        with patch.object(
            self.client, "_handle_response", return_value={"diff": {"added": 1, "deleted": 0}}
        ) as mock_handle:
            result = self.client.get_dataset_versions_diff(dataset_slug)

            assert result == {"diff": {"added": 1, "deleted": 0}}
            self.mock_session.get.assert_called_once_with(
                f"https://noveum.ai/api/v1/datasets/{dataset_slug}/versions/diff",
                timeout=30.0,
            )
            mock_handle.assert_called_once_with(mock_response)

    @pytest.mark.unit
    def test_list_dataset_items(self):
        """Test list_dataset_items method."""
        dataset_slug = "test-dataset"
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"items": []}
        mock_response.content = b'{"items": []}'
        self.mock_session.get.return_value = mock_response

        with patch.object(
            self.client, "_handle_response", return_value={"items": []}
        ) as mock_handle:
            result = self.client.list_dataset_items(
                dataset_slug, limit=10, version="1.0.0"
            )

            assert result == {"items": []}
            self.mock_session.get.assert_called_once()
            call_args = self.mock_session.get.call_args
            assert (
                call_args[0][0]
                == f"https://noveum.ai/api/v1/datasets/{dataset_slug}/items"
            )
            assert "params" in call_args[1]
            mock_handle.assert_called_once_with(mock_response)

    @pytest.mark.unit
    def test_add_dataset_items(self):
        """Test add_dataset_items method."""
        dataset_slug = "test-dataset"
        items = [{"item_key": "item1", "item_type": "test", "content": {}}]
        mock_response = Mock()
        mock_response.status_code = 201
        mock_response.json.return_value = {"added": 1}
        mock_response.content = b'{"added": 1}'
        self.mock_session.post.return_value = mock_response

        with patch.object(
            self.client, "_handle_response", return_value={"added": 1}
        ) as mock_handle:
            result = self.client.add_dataset_items(dataset_slug, items)

            assert result == {"added": 1}
            self.mock_session.post.assert_called_once()
            call_args = self.mock_session.post.call_args
            assert (
                call_args[0][0]
                == f"https://noveum.ai/api/v1/datasets/{dataset_slug}/items"
            )
            assert "json" in call_args[1]
            mock_handle.assert_called_once_with(mock_response)

    @pytest.mark.unit
    def test_delete_all_dataset_items(self):
        """Test delete_all_dataset_items method."""
        dataset_slug = "test-dataset"
        item_ids = ["item-1", "item-2"]
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"deleted": True}
        mock_response.content = b'{"deleted": true}'
        self.mock_session.delete.return_value = mock_response

        with patch.object(
            self.client, "_handle_response", return_value={"deleted": True}
        ) as mock_handle:
            result = self.client.delete_all_dataset_items(dataset_slug, item_ids)

            assert result == {"deleted": True}
            self.mock_session.delete.assert_called_once()
            call_args = self.mock_session.delete.call_args
            assert (
                call_args[0][0]
                == f"https://noveum.ai/api/v1/datasets/{dataset_slug}/items"
            )
            assert call_args[1]["json"] == {"itemIds": ["item-1", "item-2"]}
            mock_handle.assert_called_once_with(mock_response)

    @pytest.mark.unit
    def test_delete_all_dataset_items_no_version(self):
        """Test delete_all_dataset_items method without version."""
        dataset_slug = "test-dataset"
        item_ids = ["item-1", "item-2"]
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"deleted": True}
        mock_response.content = b'{"deleted": true}'
        self.mock_session.delete.return_value = mock_response

        with patch.object(
            self.client, "_handle_response", return_value={"deleted": True}
        ) as mock_handle:
            result = self.client.delete_all_dataset_items(dataset_slug, item_ids)

            assert result == {"deleted": True}
            self.mock_session.delete.assert_called_once()
            call_args = self.mock_session.delete.call_args
            assert (
                call_args[0][0]
                == f"https://noveum.ai/api/v1/datasets/{dataset_slug}/items"
            )
            assert call_args[1]["json"] == {"itemIds": ["item-1", "item-2"]}
            mock_handle.assert_called_once_with(mock_response)

    @pytest.mark.unit
    def test_get_dataset_item(self):
        """Test get_dataset_item method."""
        dataset_slug = "test-dataset"
        item_key = "item-1"
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"item_key": item_key}
        mock_response.content = b'{"item_key": "item-1"}'
        self.mock_session.get.return_value = mock_response

        with patch.object(
            self.client, "_handle_response", return_value={"item_key": item_key}
        ) as mock_handle:
            result = self.client.get_dataset_item(dataset_slug, item_key)

            assert result == {"item_key": item_key}
            self.mock_session.get.assert_called_once_with(
                f"https://noveum.ai/api/v1/datasets/{dataset_slug}/items/{item_key}",
                timeout=30.0,
            )
            mock_handle.assert_called_once_with(mock_response)

    @pytest.mark.unit
    def test_delete_dataset_item(self):
        """Test delete_dataset_item method."""
        dataset_slug = "test-dataset"
        item_id = "item-123"
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"deleted": True}
        mock_response.content = b'{"deleted": true}'
        self.mock_session.delete.return_value = mock_response

        with patch.object(
            self.client, "_handle_response", return_value={"deleted": True}
        ) as mock_handle:
            result = self.client.delete_dataset_item(dataset_slug, item_id)

            assert result == {"deleted": True}
            self.mock_session.delete.assert_called_once_with(
                f"https://noveum.ai/api/v1/datasets/{dataset_slug}/items/{item_id}",
                timeout=30.0,
            )
            mock_handle.assert_called_once_with(mock_response)


class TestNoveumClientScorerResults:
    """Test cases for scorer results methods."""

    def setup_method(self):
        """Set up test client."""
        with patch("novaeval.noveum_platform.client.requests.Session") as mock_session:
            self.client = NoveumClient(api_key="test-key")
            self.mock_session = mock_session.return_value

    @pytest.mark.unit
    def test_list_scorer_results(self):
        """Test list_scorer_results method."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"results": []}
        mock_response.content = b'{"results": []}'
        self.mock_session.get.return_value = mock_response

        with patch.object(
            self.client, "_handle_response", return_value={"results": []}
        ) as mock_handle:
            result = self.client.list_scorer_results(
                organizationSlug="test-org", limit=10
            )

            assert result == {"results": []}
            self.mock_session.get.assert_called_once()
            call_args = self.mock_session.get.call_args
            assert call_args[0][0] == "https://noveum.ai/api/v1/scorers/results"
            assert "params" in call_args[1]
            mock_handle.assert_called_once_with(mock_response)

    @pytest.mark.unit
    def test_create_scorer_result(self):
        """Test create_scorer_result method."""
        result_data = {
            "datasetSlug": "test-dataset",
            "itemId": "item-1",
            "scorerId": "accuracy-scorer",
            "score": 0.95,
        }
        mock_response = Mock()
        mock_response.status_code = 201
        mock_response.json.return_value = {"id": "result-123"}
        mock_response.content = b'{"id": "result-123"}'
        self.mock_session.post.return_value = mock_response

        with patch.object(
            self.client, "_handle_response", return_value={"id": "result-123"}
        ) as mock_handle:
            result = self.client.create_scorer_result(
                result_data, organizationSlug="test-org"
            )

            assert result == {"id": "result-123"}
            self.mock_session.post.assert_called_once()
            call_args = self.mock_session.post.call_args
            assert call_args[0][0] == "https://noveum.ai/api/v1/scorers/results"
            assert "json" in call_args[1]
            mock_handle.assert_called_once_with(mock_response)

    @pytest.mark.unit
    def test_create_scorer_results_batch(self):
        """Test create_scorer_results_batch method."""
        results = [
            {
                "datasetSlug": "test-dataset",
                "itemId": "item-1",
                "scorerId": "scorer-1",
                "score": 0.95,
            },
            {
                "datasetSlug": "test-dataset",
                "itemId": "item-2",
                "scorerId": "scorer-1",
                "score": 0.87,
            },
        ]
        mock_response = Mock()
        mock_response.status_code = 201
        mock_response.json.return_value = {"created": 2}
        mock_response.content = b'{"created": 2}'
        self.mock_session.post.return_value = mock_response

        with patch.object(
            self.client, "_handle_response", return_value={"created": 2}
        ) as mock_handle:
            result = self.client.create_scorer_results_batch(
                results, organizationSlug="test-org"
            )

            assert result == {"created": 2}
            self.mock_session.post.assert_called_once()
            call_args = self.mock_session.post.call_args
            assert call_args[0][0] == "https://noveum.ai/api/v1/scorers/results/batch"
            assert "json" in call_args[1]
            mock_handle.assert_called_once_with(mock_response)

    @pytest.mark.unit
    def test_get_scorer_result(self):
        """Test get_scorer_result method."""
        dataset_slug = "test-dataset"
        item_id = "item-1"
        scorer_id = "accuracy-scorer"
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"score": 0.95}
        mock_response.content = b'{"score": 0.95}'
        self.mock_session.get.return_value = mock_response

        with patch.object(
            self.client, "_handle_response", return_value={"score": 0.95}
        ) as mock_handle:
            result = self.client.get_scorer_result(
                dataset_slug, item_id, scorer_id, organizationSlug="test-org"
            )

            assert result == {"score": 0.95}
            self.mock_session.get.assert_called_once_with(
                f"https://noveum.ai/api/v1/scorers/results/{dataset_slug}/{item_id}/{scorer_id}",
                params={"organizationSlug": "test-org"},
                timeout=30.0,
            )
            mock_handle.assert_called_once_with(mock_response)

    @pytest.mark.unit
    def test_update_scorer_result(self):
        """Test update_scorer_result method."""
        dataset_slug = "test-dataset"
        item_id = "item-1"
        scorer_id = "accuracy-scorer"
        result_data = {"score": 0.98, "metadata": {"updated": True}}
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"updated": True}
        mock_response.content = b'{"updated": true}'
        self.mock_session.put.return_value = mock_response

        with patch.object(
            self.client, "_handle_response", return_value={"updated": True}
        ) as mock_handle:
            result = self.client.update_scorer_result(
                dataset_slug,
                item_id,
                scorer_id,
                result_data,
                organizationSlug="test-org",
            )

            assert result == {"updated": True}
            self.mock_session.put.assert_called_once()
            call_args = self.mock_session.put.call_args
            assert (
                call_args[0][0]
                == f"https://noveum.ai/api/v1/scorers/results/{dataset_slug}/{item_id}/{scorer_id}"
            )
            assert "json" in call_args[1]
            mock_handle.assert_called_once_with(mock_response)

    @pytest.mark.unit
    def test_delete_scorer_result(self):
        """Test delete_scorer_result method."""
        dataset_slug = "test-dataset"
        item_id = "item-1"
        scorer_id = "accuracy-scorer"
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"deleted": True}
        mock_response.content = b'{"deleted": true}'
        self.mock_session.delete.return_value = mock_response

        with patch.object(
            self.client, "_handle_response", return_value={"deleted": True}
        ) as mock_handle:
            result = self.client.delete_scorer_result(
                dataset_slug, item_id, scorer_id, organizationSlug="test-org"
            )

            assert result == {"deleted": True}
            self.mock_session.delete.assert_called_once_with(
                f"https://noveum.ai/api/v1/scorers/results/{dataset_slug}/{item_id}/{scorer_id}",
                params={"organizationSlug": "test-org"},
                timeout=30.0,
            )
            mock_handle.assert_called_once_with(mock_response)
