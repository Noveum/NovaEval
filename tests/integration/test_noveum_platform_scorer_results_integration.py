"""
Integration tests for Noveum Platform Scorer Results API.

This module provides comprehensive integration tests for scorer results API methods
with real API calls and actual data. These tests are currently skipped due to
API requiring organizationSlug parameter.
"""

from datetime import datetime

import pytest


@pytest.mark.noveum
@pytest.mark.slow
class TestNoveumPlatformScorerResultsIntegration:
    """Integration tests for Noveum Platform Scorer Results API."""

    @pytest.fixture(scope="class")
    def scorer_test_context(self):
        """Shared context for all scorer results tests in sequence."""
        return {}

    # Scorer Results Tests (6 methods) - SKIPPED: API requires organizationSlug parameter
    @pytest.mark.skip(
        reason="Scorer results API requires organizationSlug parameter - needs API update"
    )
    def test_01_create_scorer_result(self, noveum_client, scorer_test_context):
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
        scorer_test_context["scorer_dataset_slug"] = dataset_response["dataset"]["slug"]

        # Add a single item
        items = [
            {
                "item_key": "scorer_test_item_1",
                "item_type": "test_item",
                "content": {"test": "data"},
                "metadata": {"test": True},
            }
        ]

        noveum_client.add_dataset_items(
            scorer_test_context["scorer_dataset_slug"], items
        )
        # Publish the version to make items available
        noveum_client.publish_dataset_version(
            scorer_test_context["scorer_dataset_slug"]
        )

        # Get the item ID by listing items
        list_response = noveum_client.list_dataset_items(
            dataset_slug=scorer_test_context["scorer_dataset_slug"],
            version="0.0.1",
            limit=1,
        )
        scorer_test_context["scorer_item_id"] = list_response["items"][0]["item_id"]
        scorer_test_context["scorer_id"] = "integration-test-scorer"

        # Create scorer result
        result_data = {
            "datasetSlug": scorer_test_context["scorer_dataset_slug"],
            "itemId": scorer_test_context["scorer_item_id"],
            "scorerId": scorer_test_context["scorer_id"],
            "score": 0.85,
            "metadata": {"test_type": "integration", "confidence": 0.9},
            "details": {"reasoning": "Test scorer result for integration testing"},
        }

        response = noveum_client.create_scorer_result(result_data)

        assert response["success"] is True
        assert response["result"]["score"] == 0.85
        assert response["result"]["scorerId"] == scorer_test_context["scorer_id"]

    @pytest.mark.skip(
        reason="Scorer results API requires organizationSlug parameter - needs API update"
    )
    def test_02_create_scorer_results_batch(self, noveum_client, scorer_test_context):
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

        noveum_client.add_dataset_items(
            scorer_test_context["scorer_dataset_slug"], items
        )
        # Publish the version to make items available
        noveum_client.publish_dataset_version(
            scorer_test_context["scorer_dataset_slug"]
        )

        # Get the item IDs by listing items
        list_response = noveum_client.list_dataset_items(
            dataset_slug=scorer_test_context["scorer_dataset_slug"],
            version="0.0.2",  # New version after publishing
            limit=10,
        )
        item_ids = [item["item_id"] for item in list_response["items"]]

        # Create batch scorer results
        results = []
        for i, item_id in enumerate(item_ids):
            results.append(
                {
                    "datasetSlug": scorer_test_context["scorer_dataset_slug"],
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
    def test_03_list_scorer_results(self, noveum_client, scorer_test_context):
        """Test list_scorer_results() - with filters"""
        response = noveum_client.list_scorer_results(
            datasetSlug=scorer_test_context["scorer_dataset_slug"],
            scorerId=scorer_test_context["scorer_id"],
            limit=10,
            offset=0,
        )

        assert response["success"] is True
        assert "results" in response
        assert len(response["results"]) > 0

    @pytest.mark.skip(
        reason="Scorer results API requires organizationSlug parameter - needs API update"
    )
    def test_04_get_scorer_result(self, noveum_client, scorer_test_context):
        """Test get_scorer_result() - specific result"""
        response = noveum_client.get_scorer_result(
            scorer_test_context["scorer_dataset_slug"],
            scorer_test_context["scorer_item_id"],
            scorer_test_context["scorer_id"],
        )

        assert response["success"] is True
        assert response["result"]["scorerId"] == scorer_test_context["scorer_id"]
        assert response["result"]["score"] == 0.85

    @pytest.mark.skip(
        reason="Scorer results API requires organizationSlug parameter - needs API update"
    )
    def test_05_update_scorer_result(self, noveum_client, scorer_test_context):
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
            scorer_test_context["scorer_dataset_slug"],
            scorer_test_context["scorer_item_id"],
            scorer_test_context["scorer_id"],
            update_data,
        )

        assert response["success"] is True
        assert response["result"]["score"] == 0.92

    @pytest.mark.skip(
        reason="Scorer results API requires organizationSlug parameter - needs API update"
    )
    def test_06_delete_scorer_result(self, noveum_client, scorer_test_context):
        """Test delete_scorer_result()"""
        response = noveum_client.delete_scorer_result(
            scorer_test_context["scorer_dataset_slug"],
            scorer_test_context["scorer_item_id"],
            scorer_test_context["scorer_id"],
        )

        assert response["success"] is True

        # Clean up scorer test dataset
        noveum_client.delete_dataset(scorer_test_context["scorer_dataset_slug"])
