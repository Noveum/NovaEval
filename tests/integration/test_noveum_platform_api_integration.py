"""
Integration tests for Noveum Platform API.

This module provides comprehensive integration tests that mirror the flow
in noveum_platform_api_demo.ipynb, testing 20 API methods with real
API calls and actual data. Scorer results tests are in a separate file.
"""

import time
from datetime import datetime
from typing import Any

import pytest


@pytest.mark.noveum
@pytest.mark.slow
class TestNoveumPlatformIntegration:
    """Integration tests for Noveum Platform API - mirrors notebook flow (20 methods)."""

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

        # Wait and try to retrieve the trace to verify ingestion worked
        trace_id = response["trace_id"]

        # Wait progressively: 5, 10, 20 seconds with failure at the end
        wait_times = [5, 10, 20]
        trace_retrieved = False

        for wait_time in wait_times:
            print(f"Waiting {wait_time} seconds before attempting trace retrieval...")
            time.sleep(wait_time)

            try:
                get_response = noveum_client.get_trace(trace_id)
                if (
                    get_response["success"]
                    and get_response["data"]["trace_id"] == trace_id
                ):
                    print(f"Successfully retrieved trace after {wait_time} seconds")
                    trace_retrieved = True
                    break
                else:
                    print(
                        f"Trace retrieval failed after {wait_time} seconds: {get_response}"
                    )
            except Exception as e:
                print(
                    f"Exception during trace retrieval after {wait_time} seconds: {e}"
                )

        # Fail the test if we couldn't retrieve the trace after all attempts
        if not trace_retrieved:
            pytest.fail(
                "Failed to retrieve trace after waiting 35 seconds total (5+10+20). Single trace ingestion may not be working properly."
            )

    def test_03_ingest_traces(self, noveum_client, sample_traces, test_context):
        """Test METHOD 3: ingest_traces() - batch of 9 traces"""
        traces = sample_traces[1:10]  # Skip first trace already ingested
        response = noveum_client.ingest_traces(traces)

        assert response["success"] is True
        assert response["queued_count"] == 9
        assert "message" in response
        assert "job_id" in response

        # Store job_id for potential trace retrieval
        test_context["batch_job_id"] = response["job_id"]

        # Wait and try to retrieve a trace from the batch to verify ingestion worked
        # First, query for traces to get one from the batch we just ingested
        wait_times = [5, 10, 20]
        trace_retrieved = False
        batch_trace_id = None

        for wait_time in wait_times:
            print(f"Waiting {wait_time} seconds before attempting trace retrieval...")
            time.sleep(wait_time)

            try:
                # Query for traces to find one from our batch
                query_response = noveum_client.query_traces(
                    project="noveum-api-wrapper-demo",
                    environment="development",
                    size=10,
                    sort="start_time:desc",
                )

                if query_response["success"] and len(query_response["traces"]) > 0:
                    # Find a trace that's not the single trace we ingested earlier
                    single_trace_id = test_context.get("trace_id")
                    for trace in query_response["traces"]:
                        if trace["trace_id"] != single_trace_id:
                            batch_trace_id = trace["trace_id"]
                            break

                    if batch_trace_id:
                        # Try to retrieve the specific trace
                        get_response = noveum_client.get_trace(batch_trace_id)
                        if (
                            get_response["success"]
                            and get_response["data"]["trace_id"] == batch_trace_id
                        ):
                            print(
                                f"Successfully retrieved batch trace after {wait_time} seconds"
                            )
                            trace_retrieved = True
                            break
                        else:
                            print(
                                f"Batch trace retrieval failed after {wait_time} seconds: {get_response}"
                            )
                    else:
                        print(f"No batch traces found after {wait_time} seconds")
                else:
                    print(
                        f"Trace query failed after {wait_time} seconds: {query_response}"
                    )

            except Exception as e:
                print(
                    f"Exception during trace retrieval after {wait_time} seconds: {e}"
                )

        # Fail the test if we couldn't retrieve a trace from the batch after all attempts
        if not trace_retrieved:
            pytest.fail(
                "Failed to retrieve a trace from the batch after waiting 35 seconds total (5+10+20). Batch trace ingestion may not be working properly."
            )

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

    def test_21_complete_flow(
        self, noveum_client, sample_traces, integration_dataset_name
    ):
        """Test COMPLETE FLOW: All 20 API methods in a single comprehensive test"""
        # Initialize local context for this test
        context = {}

        # METHOD 1: get_connection_status()
        response = noveum_client.get_connection_status()
        assert response["success"] is True
        assert response["connected"] is True
        assert "connection_source" in response

        # METHOD 2: ingest_trace() - single trace
        trace = sample_traces[0]
        response = noveum_client.ingest_trace(trace)
        assert response["success"] is True
        assert "trace_id" in response
        assert response["trace_id"] is not None
        context["trace_id"] = response["trace_id"]

        # Wait and verify trace retrieval
        trace_id = response["trace_id"]
        wait_times = [5, 10, 20]
        trace_retrieved = False

        for wait_time in wait_times:
            time.sleep(wait_time)

            try:
                get_response = noveum_client.get_trace(trace_id)
                if (
                    get_response["success"]
                    and get_response["data"]["trace_id"] == trace_id
                ):
                    trace_retrieved = True
                    break
            except Exception:
                pass

        if not trace_retrieved:
            pytest.fail(
                "Failed to retrieve trace after waiting 35 seconds total (5+10+20). Single trace ingestion may not be working properly."
            )

        # METHOD 3: ingest_traces() - batch of 9 traces
        traces = sample_traces[1:10]  # Skip first trace already ingested
        response = noveum_client.ingest_traces(traces)
        assert response["success"] is True
        assert response["queued_count"] == 9
        assert "message" in response
        assert "job_id" in response
        context["batch_job_id"] = response["job_id"]

        # Wait and verify batch trace retrieval
        wait_times = [5, 10, 20]
        trace_retrieved = False
        batch_trace_id = None

        for wait_time in wait_times:
            time.sleep(wait_time)

            try:
                query_response = noveum_client.query_traces(
                    project="noveum-api-wrapper-demo",
                    environment="development",
                    size=10,
                    sort="start_time:desc",
                )

                if query_response["success"] and len(query_response["traces"]) > 0:
                    single_trace_id = context.get("trace_id")
                    for trace in query_response["traces"]:
                        if trace["trace_id"] != single_trace_id:
                            batch_trace_id = trace["trace_id"]
                            break

                    if batch_trace_id:
                        get_response = noveum_client.get_trace(batch_trace_id)
                        if (
                            get_response["success"]
                            and get_response["data"]["trace_id"] == batch_trace_id
                        ):
                            trace_retrieved = True
                            break

            except Exception:
                pass

        if not trace_retrieved:
            pytest.fail(
                "Failed to retrieve a trace from the batch after waiting 35 seconds total (5+10+20). Batch trace ingestion may not be working properly."
            )

        # METHOD 4: query_traces()
        response = noveum_client.query_traces(
            project="noveum-api-wrapper-demo",
            environment="development",
            size=10,
            sort="start_time:desc",
        )
        assert response["success"] is True
        assert "traces" in response
        assert len(response["traces"]) > 0
        context["sample_trace_id"] = response["traces"][0]["trace_id"]

        # METHOD 5: get_trace()
        trace_id = context["sample_trace_id"]
        response = noveum_client.get_trace(trace_id)
        assert response["success"] is True
        assert response["data"]["trace_id"] == trace_id
        assert "spans" in response["data"]

        # METHOD 6: get_trace_spans()
        response = noveum_client.get_trace_spans(trace_id)
        assert response["success"] is True
        assert "spans" in response
        assert len(response["spans"]) > 0

        # METHOD 7: create_dataset()
        dataset_slug = f"complete-flow-test-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        response = noveum_client.create_dataset(
            name=integration_dataset_name,
            slug=dataset_slug,
            description="Complete flow test dataset created by automated tests",
            visibility="org",
            dataset_type="custom",
            tags=["complete-flow-test", "automated", "api-testing"],
            custom_attributes={
                "test_type": "complete_flow",
                "created_by": "pytest",
                "environment": "test",
            },
        )
        assert response["success"] is True
        assert "dataset" in response
        assert response["dataset"]["name"] == integration_dataset_name
        assert response["dataset"]["slug"] == dataset_slug
        context["dataset_slug"] = response["dataset"]["slug"]
        context["dataset_id"] = response["dataset"]["id"]

        # METHOD 8: list_datasets()
        response = noveum_client.list_datasets(
            limit=10, offset=0, visibility="org", includeVersions=True
        )
        assert response["success"] is True
        assert "datasets" in response
        assert len(response["datasets"]) > 0

        # METHOD 9: get_dataset()
        dataset_slug = context["dataset_slug"]
        response = noveum_client.get_dataset(dataset_slug)
        assert response["success"] is True
        assert response["dataset"]["slug"] == dataset_slug

        # METHOD 10: update_dataset()
        response = noveum_client.update_dataset(
            slug=dataset_slug,
            description="Updated complete flow test dataset description",
            tags=["complete-flow-test", "automated", "api-testing", "updated"],
            custom_attributes={
                "test_type": "complete_flow",
                "created_by": "pytest",
                "environment": "test",
                "updated": True,
            },
        )
        assert response["success"] is True
        assert (
            response["dataset"]["description"]
            == "Updated complete flow test dataset description"
        )

        # METHOD 11: create_dataset_version()
        version_data = {
            "version": "0.0.1",
            "description": "Initial version of complete flow test dataset",
            "metadata": {"version_type": "initial", "test_phase": "complete_flow"},
        }
        response = noveum_client.create_dataset_version(dataset_slug, version_data)
        assert response["success"] is True
        assert response["version"] == "0.0.1"

        # METHOD 12: list_dataset_versions()
        response = noveum_client.list_dataset_versions(dataset_slug)
        assert response["success"] is True
        assert "versions" in response
        assert len(response["versions"]) > 0

        # METHOD 13: get_dataset_version()
        response = noveum_client.get_dataset_version(dataset_slug, "0.0.1")
        assert response["success"] is True
        assert response["version"] == "0.0.1"

        # METHOD 14: get_dataset_versions_diff()
        response = noveum_client.get_dataset_versions_diff(dataset_slug)
        assert response["success"] is True
        assert "changes" in response

        # METHOD 15: add_dataset_items()
        items = self.transform_traces_to_items(
            sample_traces, "noveum-api-wrapper-demo", "development"
        )
        response = noveum_client.add_dataset_items(dataset_slug, items)
        assert response["success"] is True
        assert "created" in response
        assert response["created"] == 10
        context["items_added"] = True

        # METHOD 16: publish_dataset_version()
        response = noveum_client.publish_dataset_version(dataset_slug)
        assert response["success"] is True

        # METHOD 17: list_dataset_items()
        response = noveum_client.list_dataset_items(
            dataset_slug=dataset_slug,
            version="0.0.1",
            limit=10,
            offset=0,
        )
        assert response["success"] is True
        assert "items" in response
        if len(response["items"]) > 0:
            context["sample_item_id"] = response["items"][0]["item_id"]
            context["item_ids"] = [item["item_id"] for item in response["items"]]

        # METHOD 18: delete_dataset_item()
        if "sample_item_id" in context:
            item_id = context["sample_item_id"]
            response = noveum_client.delete_dataset_item(dataset_slug, item_id)
            assert response["success"] is True

        # METHOD 19: delete_all_dataset_items()
        if "item_ids" in context:
            item_ids = context["item_ids"][:3]  # Delete first 3 items
            response = noveum_client.delete_all_dataset_items(dataset_slug, item_ids)
            assert response["success"] is True

        # METHOD 20: delete_dataset()
        response = noveum_client.delete_dataset(dataset_slug)
        assert response["success"] is True
