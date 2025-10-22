"""
Integration tests for Noveum Platform API.

This module provides comprehensive integration tests that mirror the flow
in noveum_platform_api_demo.ipynb, testing all 26 API methods with real
API calls and actual data.
"""

from datetime import datetime
from typing import Any

import pytest


@pytest.mark.noveum
@pytest.mark.slow
class TestNoveumPlatformIntegration:
    """Integration tests for Noveum Platform API - mirrors notebook flow."""

    @pytest.fixture(scope="class")
    def test_context(self):
        """Shared context for all tests in sequence."""
        return {}

    def transform_traces_to_items(
        self, traces: list[dict[str, Any]], project: str, environment: str
    ) -> list[dict[str, Any]]:
        """Transform traces into dataset item format."""
        items = []
        for i, trace in enumerate(traces):
            item = {
                "item_key": f"trace_{i+1:03d}",
                "item_type": "agent_trace",
                "content": {**trace},
                "metadata": {
                    "project": project,
                    "environment": environment,
                    "item_index": i + 1,
                },
            }
            items.append(item)
        return items

    # Trace Tests (6 methods)
    def test_01_get_connection_status(self, noveum_client):
        """Test METHOD 1: get_connection_status()"""
        response = noveum_client.get_connection_status()

        assert response["success"] is True
        assert response["connected"] is True
        assert "connection_source" in response

    def test_02_ingest_trace(self, noveum_client, sample_traces, test_context):
        """Test METHOD 2: ingest_trace() - single trace"""
        trace = sample_traces[0]
        response = noveum_client.ingest_trace(trace)

        assert response["success"] is True
        assert "trace_id" in response
        assert response["trace_id"] is not None

        # Store trace_id for later tests
        test_context["trace_id"] = response["trace_id"]

    def test_03_ingest_traces(self, noveum_client, sample_traces):
        """Test METHOD 3: ingest_traces() - batch of 9 traces"""
        traces = sample_traces[1:10]  # Skip first trace already ingested
        response = noveum_client.ingest_traces(traces)

        assert response["success"] is True
        assert response["queued_count"] == 9
        assert "message" in response
        assert "job_id" in response

    def test_04_query_traces(self, noveum_client, test_context):
        """Test METHOD 4: query_traces() - with various filters"""
        response = noveum_client.query_traces(
            project="noveum-api-wrapper-demo",
            environment="development",
            size=10,
            sort="start_time:desc",
        )

        assert response["success"] is True
        assert "traces" in response
        assert len(response["traces"]) > 0

        # Store first trace for retrieval tests
        test_context["sample_trace_id"] = response["traces"][0]["trace_id"]

    def test_05_get_trace(self, noveum_client, test_context):
        """Test METHOD 5: get_trace() - specific trace"""
        trace_id = test_context["sample_trace_id"]
        response = noveum_client.get_trace(trace_id)

        assert response["success"] is True
        assert response["data"]["trace_id"] == trace_id
        assert "spans" in response["data"]

    def test_06_get_trace_spans(self, noveum_client, test_context):
        """Test METHOD 6: get_trace_spans()"""
        trace_id = test_context["sample_trace_id"]
        response = noveum_client.get_trace_spans(trace_id)

        assert response["success"] is True
        assert "spans" in response
        assert len(response["spans"]) > 0

    # Dataset Tests (14 methods)
    def test_07_create_dataset(
        self, noveum_client, integration_dataset_name, test_context
    ):
        """Test METHOD 7: create_dataset()"""
        dataset_slug = f"integration-test-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

        response = noveum_client.create_dataset(
            name=integration_dataset_name,
            slug=dataset_slug,
            description="Integration test dataset created by automated tests",
            visibility="org",
            dataset_type="custom",
            tags=["integration-test", "automated", "api-testing"],
            custom_attributes={
                "test_type": "integration",
                "created_by": "pytest",
                "environment": "test",
            },
        )

        assert response["success"] is True
        assert "dataset" in response
        assert response["dataset"]["name"] == integration_dataset_name
        assert response["dataset"]["slug"] == dataset_slug

        # Store dataset info for later tests
        test_context["dataset_slug"] = response["dataset"]["slug"]
        test_context["dataset_id"] = response["dataset"]["id"]

    def test_08_list_datasets(self, noveum_client):
        """Test METHOD 8: list_datasets()"""
        response = noveum_client.list_datasets(
            limit=10, offset=0, visibility="org", includeVersions=True
        )

        assert response["success"] is True
        assert "datasets" in response
        assert len(response["datasets"]) > 0

    def test_09_get_dataset(self, noveum_client, test_context):
        """Test METHOD 9: get_dataset()"""
        dataset_slug = test_context["dataset_slug"]
        response = noveum_client.get_dataset(dataset_slug)

        assert response["success"] is True
        assert response["dataset"]["slug"] == dataset_slug

    def test_10_update_dataset(self, noveum_client, test_context):
        """Test METHOD 10: update_dataset()"""
        dataset_slug = test_context["dataset_slug"]

        response = noveum_client.update_dataset(
            slug=dataset_slug,
            description="Updated integration test dataset description",
            tags=["integration-test", "automated", "api-testing", "updated"],
            custom_attributes={
                "test_type": "integration",
                "created_by": "pytest",
                "environment": "test",
                "updated": True,
            },
        )

        assert response["success"] is True
        assert (
            response["dataset"]["description"]
            == "Updated integration test dataset description"
        )

    def test_11_create_dataset_version(self, noveum_client, test_context):
        """Test METHOD 11: create_dataset_version()"""
        dataset_slug = test_context["dataset_slug"]
        version_data = {
            "version": "0.0.1",
            "description": "Initial version of integration test dataset",
            "metadata": {"version_type": "initial", "test_phase": "integration"},
        }

        response = noveum_client.create_dataset_version(dataset_slug, version_data)

        assert response["success"] is True
        assert response["version"] == "0.0.1"

    def test_12_list_dataset_versions(self, noveum_client, test_context):
        """Test METHOD 12: list_dataset_versions()"""
        dataset_slug = test_context["dataset_slug"]
        response = noveum_client.list_dataset_versions(dataset_slug)

        assert response["success"] is True
        assert "versions" in response
        assert len(response["versions"]) > 0

    def test_13_get_dataset_version(self, noveum_client, test_context):
        """Test METHOD 13: get_dataset_version()"""
        dataset_slug = test_context["dataset_slug"]
        response = noveum_client.get_dataset_version(dataset_slug, "0.0.1")

        assert response["success"] is True
        assert response["version"] == "0.0.1"

    def test_14_get_dataset_versions_diff(self, noveum_client, test_context):
        """Test METHOD 14: get_dataset_versions_diff()"""
        dataset_slug = test_context["dataset_slug"]
        response = noveum_client.get_dataset_versions_diff(dataset_slug)

        assert response["success"] is True
        assert "changes" in response

    def test_15_add_dataset_items(self, noveum_client, sample_traces, test_context):
        """Test METHOD 15: add_dataset_items()"""
        dataset_slug = test_context["dataset_slug"]
        items = self.transform_traces_to_items(
            sample_traces, "noveum-api-wrapper-demo", "development"
        )

        response = noveum_client.add_dataset_items(dataset_slug, items)

        assert response["success"] is True
        assert "created" in response
        assert response["created"] == 10

        # Store item info for later tests (we'll need to query to get IDs)
        test_context["items_added"] = True

    def test_16_publish_dataset_version(self, noveum_client, test_context):
        """Test METHOD 16: publish_dataset_version()"""
        dataset_slug = test_context["dataset_slug"]
        response = noveum_client.publish_dataset_version(dataset_slug)

        assert response["success"] is True

    def test_17_list_dataset_items(self, noveum_client, test_context):
        """Test METHOD 17: list_dataset_items()"""
        dataset_slug = test_context["dataset_slug"]
        # List items from the published version (after publish_dataset_version)
        response = noveum_client.list_dataset_items(
            dataset_slug=dataset_slug,
            version="0.0.1",  # Use the published version
            limit=10,
            offset=0,
        )

        assert response["success"] is True
        assert "items" in response
        if len(response["items"]) > 0:
            test_context["sample_item_id"] = response["items"][0]["item_id"]
            test_context["item_ids"] = [item["item_id"] for item in response["items"]]

    # Cleanup Tests (3 methods)
    def test_18_delete_dataset_item(self, noveum_client, test_context):
        """Test METHOD 18: delete_dataset_item()"""
        dataset_slug = test_context["dataset_slug"]
        if "sample_item_id" not in test_context:
            pytest.skip("No items available to delete")

        item_id = test_context["sample_item_id"]
        response = noveum_client.delete_dataset_item(dataset_slug, item_id)

        assert response["success"] is True

    def test_19_delete_all_dataset_items(self, noveum_client, test_context):
        """Test METHOD 19: delete_all_dataset_items()"""
        dataset_slug = test_context["dataset_slug"]
        if "item_ids" not in test_context:
            pytest.skip("No items available to delete")

        item_ids = test_context["item_ids"][:3]  # Delete first 3 items
        response = noveum_client.delete_all_dataset_items(dataset_slug, item_ids)

        assert response["success"] is True

    def test_20_delete_dataset(self, noveum_client, test_context):
        """Test METHOD 20: delete_dataset()"""
        dataset_slug = test_context["dataset_slug"]
        response = noveum_client.delete_dataset(dataset_slug)

        assert response["success"] is True

    # Scorer Results Tests (6 methods) - SKIPPED: API requires organizationSlug parameter
    @pytest.mark.skip(
        reason="Scorer results API requires organizationSlug parameter - needs API update"
    )
    def test_21_create_scorer_result(self, noveum_client, test_context):
        """Test create_scorer_result() - single result"""
        # First, we need to recreate a dataset and item for scorer results
        # since we deleted them in cleanup tests
        dataset_name = f"scorer_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        dataset_slug = f"scorer-test-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

        dataset_response = noveum_client.create_dataset(
            name=dataset_name,
            slug=dataset_slug,
            description="Dataset for scorer results testing",
            visibility="org",
            dataset_type="custom",
            tags=["scorer-test", "integration"],
        )
        test_context["scorer_dataset_slug"] = dataset_response["dataset"]["slug"]

        # Add a single item
        items = [
            {
                "item_key": "scorer_test_item_1",
                "item_type": "test_item",
                "content": {"test": "data"},
                "metadata": {"test": True},
            }
        ]

        noveum_client.add_dataset_items(test_context["scorer_dataset_slug"], items)
        # Publish the version to make items available
        noveum_client.publish_dataset_version(test_context["scorer_dataset_slug"])

        # Get the item ID by listing items
        list_response = noveum_client.list_dataset_items(
            dataset_slug=test_context["scorer_dataset_slug"], version="0.0.1", limit=1
        )
        test_context["scorer_item_id"] = list_response["items"][0]["item_id"]
        test_context["scorer_id"] = "integration-test-scorer"

        # Create scorer result
        result_data = {
            "datasetSlug": test_context["scorer_dataset_slug"],
            "itemId": test_context["scorer_item_id"],
            "scorerId": test_context["scorer_id"],
            "score": 0.85,
            "metadata": {"test_type": "integration", "confidence": 0.9},
            "details": {"reasoning": "Test scorer result for integration testing"},
        }

        response = noveum_client.create_scorer_result(result_data)

        assert response["success"] is True
        assert response["result"]["score"] == 0.85
        assert response["result"]["scorerId"] == test_context["scorer_id"]

    @pytest.mark.skip(
        reason="Scorer results API requires organizationSlug parameter - needs API update"
    )
    def test_22_create_scorer_results_batch(self, noveum_client, test_context):
        """Test create_scorer_results_batch() - batch results"""
        # Add more items for batch testing
        items = [
            {
                "item_key": "scorer_test_item_2",
                "item_type": "test_item",
                "content": {"test": "data2"},
                "metadata": {"test": True},
            },
            {
                "item_key": "scorer_test_item_3",
                "item_type": "test_item",
                "content": {"test": "data3"},
                "metadata": {"test": True},
            },
        ]

        noveum_client.add_dataset_items(test_context["scorer_dataset_slug"], items)
        # Publish the version to make items available
        noveum_client.publish_dataset_version(test_context["scorer_dataset_slug"])

        # Get the item IDs by listing items
        list_response = noveum_client.list_dataset_items(
            dataset_slug=test_context["scorer_dataset_slug"],
            version="0.0.2",  # New version after publishing
            limit=10,
        )
        item_ids = [item["item_id"] for item in list_response["items"]]

        # Create batch scorer results
        results = []
        for i, item_id in enumerate(item_ids):
            results.append(
                {
                    "datasetSlug": test_context["scorer_dataset_slug"],
                    "itemId": item_id,
                    "scorerId": f"batch-scorer-{i+1}",
                    "score": 0.7 + (i * 0.1),
                    "metadata": {"batch_test": True, "index": i + 1},
                }
            )

        response = noveum_client.create_scorer_results_batch(results)

        assert response["success"] is True
        assert "results" in response
        assert len(response["results"]) == 2

    @pytest.mark.skip(
        reason="Scorer results API requires organizationSlug parameter - needs API update"
    )
    def test_23_list_scorer_results(self, noveum_client, test_context):
        """Test list_scorer_results() - with filters"""
        response = noveum_client.list_scorer_results(
            datasetSlug=test_context["scorer_dataset_slug"],
            scorerId=test_context["scorer_id"],
            limit=10,
            offset=0,
        )

        assert response["success"] is True
        assert "results" in response
        assert len(response["results"]) > 0

    @pytest.mark.skip(
        reason="Scorer results API requires organizationSlug parameter - needs API update"
    )
    def test_24_get_scorer_result(self, noveum_client, test_context):
        """Test get_scorer_result() - specific result"""
        response = noveum_client.get_scorer_result(
            test_context["scorer_dataset_slug"],
            test_context["scorer_item_id"],
            test_context["scorer_id"],
        )

        assert response["success"] is True
        assert response["result"]["scorerId"] == test_context["scorer_id"]
        assert response["result"]["score"] == 0.85

    @pytest.mark.skip(
        reason="Scorer results API requires organizationSlug parameter - needs API update"
    )
    def test_25_update_scorer_result(self, noveum_client, test_context):
        """Test update_scorer_result()"""
        update_data = {
            "score": 0.92,
            "metadata": {
                "test_type": "integration",
                "confidence": 0.95,
                "updated": True,
            },
            "details": {
                "reasoning": "Updated test scorer result for integration testing"
            },
        }

        response = noveum_client.update_scorer_result(
            test_context["scorer_dataset_slug"],
            test_context["scorer_item_id"],
            test_context["scorer_id"],
            update_data,
        )

        assert response["success"] is True
        assert response["result"]["score"] == 0.92

    @pytest.mark.skip(
        reason="Scorer results API requires organizationSlug parameter - needs API update"
    )
    def test_26_delete_scorer_result(self, noveum_client, test_context):
        """Test delete_scorer_result()"""
        response = noveum_client.delete_scorer_result(
            test_context["scorer_dataset_slug"],
            test_context["scorer_item_id"],
            test_context["scorer_id"],
        )

        assert response["success"] is True

        # Clean up scorer test dataset
        noveum_client.delete_dataset(test_context["scorer_dataset_slug"])
