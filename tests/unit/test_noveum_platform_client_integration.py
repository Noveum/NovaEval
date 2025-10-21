"""
Integration tests for Noveum Platform API client using responses library.
"""

import json

import pytest

try:
    import responses

    responses_available = True
except ImportError:
    responses = None
    responses_available = False

from novaeval.noveum_platform.client import NoveumClient
from novaeval.noveum_platform.exceptions import (
    AuthenticationError,
    NotFoundError,
    ValidationError,
)

# Create conditional decorator
if responses_available:
    responses_activate = responses.activate
else:
    responses_activate = pytest.mark.skip(reason="responses library not available")


@pytest.mark.skipif(not responses_available, reason="responses library not available")
class TestNoveumClientIntegration:
    """Integration tests for NoveumClient using responses library."""

    def setup_method(self):
        """Set up test client."""
        self.client = NoveumClient(
            api_key="test-key",
            base_url="https://api.test.com",
            organization_id="test-org",
            timeout=30.0,
        )

    @responses_activate
    @pytest.mark.unit
    def test_complete_dataset_workflow(self):
        """Test complete dataset workflow: create → add items → create scorer results."""
        # Mock create dataset
        responses.add(
            responses.POST,
            "https://api.test.com/api/v1/datasets",
            json={"slug": "test-dataset", "name": "Test Dataset"},
            status=201,
        )

        # Mock add dataset items
        responses.add(
            responses.POST,
            "https://api.test.com/api/v1/datasets/test-dataset/items",
            json={"added": 2},
            status=201,
        )

        # Mock create scorer result
        responses.add(
            responses.POST,
            "https://api.test.com/api/v1/scorers/results",
            json={"id": "result-123"},
            status=201,
        )

        # Create dataset
        dataset = self.client.create_dataset(
            name="Test Dataset", description="A test dataset", dataset_type="custom"
        )
        assert dataset["slug"] == "test-dataset"

        # Add items
        items = [
            {
                "item_key": "q1",
                "item_type": "question_answer",
                "content": {
                    "question": "What is AI?",
                    "answer": "Artificial Intelligence",
                },
            },
            {
                "item_key": "q2",
                "item_type": "question_answer",
                "content": {"question": "What is ML?", "answer": "Machine Learning"},
            },
        ]
        add_result = self.client.add_dataset_items("test-dataset", "1.0.0", items)
        assert add_result["added"] == 2

        # Create scorer result
        scorer_result = self.client.create_scorer_result(
            {
                "datasetSlug": "test-dataset",
                "itemId": "q1",
                "scorerId": "accuracy-scorer",
                "score": 0.95,
            }
        )
        assert scorer_result["id"] == "result-123"

    @responses_activate
    @pytest.mark.unit
    def test_trace_ingestion_workflow(self):
        """Test trace ingestion workflow."""
        # Mock trace ingestion
        responses.add(
            responses.POST,
            "https://api.test.com/api/v1/traces",
            json={"ingested": 2, "trace_ids": ["trace-1", "trace-2"]},
            status=200,
        )

        # Mock query traces
        responses.add(
            responses.GET,
            "https://api.test.com/api/v1/traces",
            json={
                "traces": [
                    {"trace_id": "trace-1", "name": "Test Trace 1"},
                    {"trace_id": "trace-2", "name": "Test Trace 2"},
                ],
                "total": 2,
            },
            status=200,
        )

        # Ingest traces
        traces = [
            {
                "trace_id": "trace-1",
                "name": "Test Trace 1",
                "start_time": "2024-01-01T10:00:00Z",
                "end_time": "2024-01-01T10:05:00Z",
                "status": "success",
            },
            {
                "trace_id": "trace-2",
                "name": "Test Trace 2",
                "start_time": "2024-01-01T11:00:00Z",
                "end_time": "2024-01-01T11:05:00Z",
                "status": "success",
            },
        ]
        ingest_result = self.client.ingest_traces(traces)
        assert ingest_result["ingested"] == 2

        # Query traces
        query_result = self.client.query_traces(project="test-project", size=10)
        assert len(query_result["traces"]) == 2
        assert query_result["total"] == 2

    @responses_activate
    @pytest.mark.unit
    def test_error_handling_workflow(self):
        """Test error handling across different scenarios."""
        # Mock authentication error
        responses.add(
            responses.GET,
            "https://api.test.com/api/v1/datasets",
            json={"message": "Invalid API key"},
            status=401,
        )

        # Mock validation error
        responses.add(
            responses.POST,
            "https://api.test.com/api/v1/datasets",
            json={"message": "Missing required field: name"},
            status=400,
        )

        # Mock not found error
        responses.add(
            responses.GET,
            "https://api.test.com/api/v1/datasets/non-existent",
            json={"message": "Dataset not found"},
            status=404,
        )

        # Test authentication error
        with pytest.raises(AuthenticationError) as exc_info:
            self.client.list_datasets()
        assert exc_info.value.message == "Invalid API key"
        assert exc_info.value.status_code == 401

        # Test validation error
        with pytest.raises(ValidationError) as exc_info:
            self.client.create_dataset(name="")  # Empty name should fail validation
        assert "at least 1 character" in str(exc_info.value)
        assert exc_info.value.status_code == 400

        # Test not found error
        with pytest.raises(NotFoundError) as exc_info:
            self.client.get_dataset("non-existent")
        assert exc_info.value.message == "Dataset not found"
        assert exc_info.value.status_code == 404

    @responses_activate
    @pytest.mark.unit
    def test_batch_scorer_results_workflow(self):
        """Test batch scorer results creation workflow."""
        # Mock batch scorer results creation
        responses.add(
            responses.POST,
            "https://api.test.com/api/v1/scorers/results/batch",
            json={
                "created": 3,
                "results": [
                    {"id": "result-1", "datasetSlug": "test-dataset", "itemId": "q1"},
                    {"id": "result-2", "datasetSlug": "test-dataset", "itemId": "q2"},
                    {"id": "result-3", "datasetSlug": "test-dataset", "itemId": "q3"},
                ],
            },
            status=201,
        )

        # Mock list scorer results
        responses.add(
            responses.GET,
            "https://api.test.com/api/v1/scorers/results",
            json={
                "results": [
                    {"id": "result-1", "score": 0.95},
                    {"id": "result-2", "score": 0.87},
                    {"id": "result-3", "score": 0.92},
                ],
                "total": 3,
            },
            status=200,
        )

        # Create batch scorer results
        results = [
            {
                "datasetSlug": "test-dataset",
                "itemId": "q1",
                "scorerId": "accuracy-scorer",
                "score": 0.95,
            },
            {
                "datasetSlug": "test-dataset",
                "itemId": "q2",
                "scorerId": "accuracy-scorer",
                "score": 0.87,
            },
            {
                "datasetSlug": "test-dataset",
                "itemId": "q3",
                "scorerId": "accuracy-scorer",
                "score": 0.92,
            },
        ]
        batch_result = self.client.create_scorer_results_batch(results)
        assert batch_result["created"] == 3
        assert len(batch_result["results"]) == 3

        # List scorer results
        list_result = self.client.list_scorer_results(
            organizationSlug="test-org", datasetSlug="test-dataset"
        )
        assert len(list_result["results"]) == 3
        assert list_result["total"] == 3

    @responses_activate
    @pytest.mark.unit
    def test_dataset_version_workflow(self):
        """Test dataset version management workflow."""
        # Mock create dataset version
        responses.add(
            responses.POST,
            "https://api.test.com/api/v1/datasets/test-dataset/versions",
            json={"version": "1.0.0", "status": "draft"},
            status=201,
        )

        # Mock publish dataset version
        responses.add(
            responses.POST,
            "https://api.test.com/api/v1/datasets/test-dataset/versions/1.0.0/publish",
            json={"version": "1.0.0", "status": "published"},
            status=200,
        )

        # Mock get dataset version
        responses.add(
            responses.GET,
            "https://api.test.com/api/v1/datasets/test-dataset/versions/1.0.0",
            json={"version": "1.0.0", "status": "published"},
            status=200,
        )

        # Create dataset version
        version_data = {
            "version": "1.0.0",
            "description": "Initial version",
            "metadata": {"changelog": "First release"},
        }
        create_result = self.client.create_dataset_version("test-dataset", version_data)
        assert create_result["version"] == "1.0.0"
        assert create_result["status"] == "draft"

        # Publish dataset version
        publish_result = self.client.publish_dataset_version("test-dataset", "1.0.0")
        assert publish_result["status"] == "published"

        # Get dataset version
        get_result = self.client.get_dataset_version("test-dataset", "1.0.0")
        assert get_result["version"] == "1.0.0"
        assert get_result["status"] == "published"

    @responses_activate
    @pytest.mark.unit
    def test_session_reuse_across_requests(self):
        """Test that session is reused across multiple requests."""
        # Mock multiple requests
        responses.add(
            responses.GET,
            "https://api.test.com/api/v1/datasets",
            json={"datasets": []},
            status=200,
        )
        responses.add(
            responses.GET,
            "https://api.test.com/api/v1/traces",
            json={"traces": []},
            status=200,
        )

        # Make multiple requests
        self.client.list_datasets()
        self.client.query_traces()

        # Verify both requests were made
        assert len(responses.calls) == 2

        # Verify headers are consistent across requests
        for call in responses.calls:
            assert call.request.headers["Authorization"] == "Bearer test-key"
            assert call.request.headers["Content-Type"] == "application/json"
            assert call.request.headers["X-Organization-Id"] == "test-org"

    @responses_activate
    @pytest.mark.unit
    def test_timeout_behavior(self):
        """Test timeout behavior with slow responses."""
        import time

        # Mock slow response
        def slow_response(_request):
            time.sleep(0.1)  # Simulate slow response
            return (200, {}, json.dumps({"data": "slow response"}))

        responses.add_callback(
            responses.GET,
            "https://api.test.com/api/v1/datasets",
            callback=slow_response,
        )

        # Test with short timeout should work
        client_short_timeout = NoveumClient(
            api_key="test-key", base_url="https://api.test.com", timeout=1.0
        )

        result = client_short_timeout.list_datasets()
        assert result["data"] == "slow response"

    @responses_activate
    @pytest.mark.unit
    def test_query_parameters_encoding(self):
        """Test that query parameters are properly encoded."""
        # Mock query traces with complex parameters
        responses.add(
            responses.GET,
            "https://api.test.com/api/v1/traces",
            json={"traces": []},
            status=200,
        )

        # Query with complex parameters
        self.client.query_traces(
            project="test-project",
            tags=["tag1", "tag2", "tag with spaces"],
            start_time="2024-01-01T00:00:00Z",
            end_time="2024-12-31T23:59:59Z",
            search_term="test query with spaces",
            size=50,
            from_=10,
        )

        # Verify request was made
        assert len(responses.calls) == 1
        call = responses.calls[0]

        # Verify URL parameters
        assert "project=test-project" in call.request.url
        assert "tags=tag1" in call.request.url
        assert "tags=tag2" in call.request.url
        assert "tags=tag+with+spaces" in call.request.url
        assert "start_time=2024-01-01T00%3A00%3A00Z" in call.request.url
        assert "end_time=2024-12-31T23%3A59%3A59Z" in call.request.url
        assert "search_term=test+query+with+spaces" in call.request.url
        assert "size=50" in call.request.url
        assert "from=10" in call.request.url

    @responses_activate
    @pytest.mark.unit
    def test_json_request_body_encoding(self):
        """Test that JSON request bodies are properly encoded."""
        # Mock create dataset
        responses.add(
            responses.POST,
            "https://api.test.com/api/v1/datasets",
            json={"slug": "test-dataset"},
            status=201,
        )

        # Create dataset with complex data
        self.client.create_dataset(
            name="Test Dataset",
            description="A dataset with special characters: éñ中文",
            tags=["tag1", "tag with spaces"],
            custom_attributes={
                "nested": {"key": "value with spaces"},
                "unicode": "éñ中文",
                "special_chars": "!@#$%^&*()",
            },
        )

        # Verify request was made
        assert len(responses.calls) == 1
        call = responses.calls[0]

        # Verify JSON body
        request_body = json.loads(call.request.body)
        assert request_body["name"] == "Test Dataset"
        assert (
            request_body["description"] == "A dataset with special characters: éñ中文"
        )
        assert request_body["tags"] == ["tag1", "tag with spaces"]
        assert request_body["custom_attributes"]["unicode"] == "éñ中文"
        assert request_body["custom_attributes"]["special_chars"] == "!@#$%^&*()"

    @responses_activate
    @pytest.mark.unit
    def test_empty_response_handling(self):
        """Test handling of empty responses."""
        # Mock empty response
        responses.add(
            responses.DELETE,
            "https://api.test.com/api/v1/datasets/test-dataset",
            body="",  # Empty response body
            status=204,
        )

        # Delete dataset
        result = self.client.delete_dataset("test-dataset")

        # Should return empty dict for empty response
        assert result == {}

    @responses_activate
    @pytest.mark.unit
    def test_large_response_handling(self):
        """Test handling of large responses."""
        # Create large response data
        large_traces = [
            {
                "trace_id": f"trace-{i}",
                "name": f"Trace {i}",
                "data": "x" * 1000,  # 1KB per trace
            }
            for i in range(100)  # 100 traces = ~100KB
        ]

        responses.add(
            responses.GET,
            "https://api.test.com/api/v1/traces",
            json={"traces": large_traces, "total": 100},
            status=200,
        )

        # Query traces
        result = self.client.query_traces(size=100)

        assert len(result["traces"]) == 100
        assert result["total"] == 100
        assert result["traces"][0]["trace_id"] == "trace-0"
        assert result["traces"][99]["trace_id"] == "trace-99"
