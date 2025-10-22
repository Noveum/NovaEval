"""
Unit tests for Noveum Platform API client - Resource operations.
"""

from unittest.mock import Mock, patch

import pytest

from novaeval.noveum_platform.client import NoveumClient

BASE_URL = "https://noveum.ai/api/v1/"


class TestNoveumClientDatasets:
    """Test cases for dataset-related methods."""

    def setup_method(self):
        """Set up test client."""
        with patch("novaeval.noveum_platform.utils.requests.Session") as mock_session:
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

        with patch(
            "novaeval.noveum_platform.utils.handle_response",
            return_value={"slug": "test-dataset"},
        ):
            result = self.client.create_dataset(
                name="Test Dataset", description="A test dataset"
            )

            assert result == {"slug": "test-dataset"}
            self.mock_session.post.assert_called_once()
            call_args = self.mock_session.post.call_args
            assert call_args[0][0] == f"{BASE_URL}datasets"
            assert "json" in call_args[1]

    @pytest.mark.unit
    def test_list_datasets(self):
        """Test list_datasets method."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"datasets": []}
        mock_response.content = b'{"datasets": []}'
        self.mock_session.get.return_value = mock_response

        with patch(
            "novaeval.noveum_platform.utils.handle_response",
            return_value={"datasets": []},
        ):
            result = self.client.list_datasets(limit=10, visibility="public")

            assert result == {"datasets": []}
            self.mock_session.get.assert_called_once()
            call_args = self.mock_session.get.call_args
            assert call_args[0][0] == f"{BASE_URL}datasets"
            assert "params" in call_args[1]

    @pytest.mark.unit
    def test_get_dataset(self):
        """Test get_dataset method."""
        slug = "test-dataset"
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"slug": slug}
        mock_response.content = b'{"slug": "test-dataset"}'
        self.mock_session.get.return_value = mock_response

        with patch(
            "novaeval.noveum_platform.utils.handle_response",
            return_value={"slug": slug},
        ):
            result = self.client.get_dataset(slug)

            assert result == {"slug": slug}
            self.mock_session.get.assert_called_once_with(
                f"{BASE_URL}datasets/{slug}", timeout=30.0
            )

    @pytest.mark.unit
    def test_update_dataset(self):
        """Test update_dataset method."""
        slug = "test-dataset"
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"slug": slug, "updated": True}
        mock_response.content = b'{"slug": "test-dataset", "updated": true}'
        self.mock_session.put.return_value = mock_response

        with patch(
            "novaeval.noveum_platform.utils.handle_response",
            return_value={"slug": slug, "updated": True},
        ):
            result = self.client.update_dataset(slug, name="Updated Dataset")

            assert result == {"slug": slug, "updated": True}
            self.mock_session.put.assert_called_once()
            call_args = self.mock_session.put.call_args
            assert call_args[0][0] == f"{BASE_URL}datasets/{slug}"
            assert "json" in call_args[1]

    @pytest.mark.unit
    def test_delete_dataset(self):
        """Test delete_dataset method."""
        slug = "test-dataset"
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"deleted": True}
        mock_response.content = b'{"deleted": true}'
        self.mock_session.delete.return_value = mock_response

        with patch(
            "novaeval.noveum_platform.utils.handle_response",
            return_value={"deleted": True},
        ):
            result = self.client.delete_dataset(slug)

            assert result == {"deleted": True}
            self.mock_session.delete.assert_called_once_with(
                f"{BASE_URL}datasets/{slug}", timeout=30.0
            )

    @pytest.mark.unit
    def test_list_dataset_versions(self):
        """Test list_dataset_versions method."""
        dataset_slug = "test-dataset"
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"versions": []}
        mock_response.content = b'{"versions": []}'
        self.mock_session.get.return_value = mock_response

        with patch(
            "novaeval.noveum_platform.utils.handle_response",
            return_value={"versions": []},
        ):
            result = self.client.list_dataset_versions(dataset_slug)

            assert result == {"versions": []}
            self.mock_session.get.assert_called_once_with(
                f"{BASE_URL}datasets/{dataset_slug}/versions",
                params={"limit": 50, "offset": 0},
                timeout=30.0,
            )

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

        with patch(
            "novaeval.noveum_platform.utils.handle_response",
            return_value={"version": "1.0.0"},
        ):
            result = self.client.create_dataset_version(dataset_slug, version_data)

            assert result == {"version": "1.0.0"}
            self.mock_session.post.assert_called_once()
            call_args = self.mock_session.post.call_args
            assert call_args[0][0] == f"{BASE_URL}datasets/{dataset_slug}/versions"
            assert "json" in call_args[1]

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

        with patch(
            "novaeval.noveum_platform.utils.handle_response",
            return_value={"version": version},
        ):
            result = self.client.get_dataset_version(dataset_slug, version)

            assert result == {"version": version}
            self.mock_session.get.assert_called_once_with(
                f"{BASE_URL}datasets/{dataset_slug}/versions/{version}",
                timeout=30.0,
            )

    @pytest.mark.unit
    def test_publish_dataset_version(self):
        """Test publish_dataset_version method."""
        dataset_slug = "test-dataset"
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"published": True}
        mock_response.content = b'{"published": true}'
        self.mock_session.post.return_value = mock_response

        with patch(
            "novaeval.noveum_platform.utils.handle_response",
            return_value={"published": True},
        ):
            result = self.client.publish_dataset_version(dataset_slug)

            assert result == {"published": True}
            self.mock_session.post.assert_called_once_with(
                f"{BASE_URL}datasets/{dataset_slug}/versions/publish",
                timeout=30.0,
            )

    @pytest.mark.unit
    def test_get_dataset_versions_diff(self):
        """Test get_dataset_versions_diff method."""
        dataset_slug = "test-dataset"
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"dif": {"added": 1, "deleted": 0}}
        mock_response.content = b'{"dif": {"added": 1, "deleted": 0}}'
        self.mock_session.get.return_value = mock_response

        with patch(
            "novaeval.noveum_platform.utils.handle_response",
            return_value={"dif": {"added": 1, "deleted": 0}},
        ):
            result = self.client.get_dataset_versions_diff(dataset_slug)

            assert result == {"dif": {"added": 1, "deleted": 0}}
            self.mock_session.get.assert_called_once_with(
                f"{BASE_URL}datasets/{dataset_slug}/versions/diff",
                timeout=30.0,
            )

    @pytest.mark.unit
    def test_list_dataset_items(self):
        """Test list_dataset_items method."""
        dataset_slug = "test-dataset"
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"items": []}
        mock_response.content = b'{"items": []}'
        self.mock_session.get.return_value = mock_response

        with patch(
            "novaeval.noveum_platform.utils.handle_response", return_value={"items": []}
        ):
            result = self.client.list_dataset_items(
                dataset_slug, limit=10, version="1.0.0"
            )

            assert result == {"items": []}
            self.mock_session.get.assert_called_once()
            call_args = self.mock_session.get.call_args
            assert call_args[0][0] == f"{BASE_URL}datasets/{dataset_slug}/items"
            assert "params" in call_args[1]

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

        with patch(
            "novaeval.noveum_platform.utils.handle_response", return_value={"added": 1}
        ):
            result = self.client.add_dataset_items(dataset_slug, items)

            assert result == {"added": 1}
            self.mock_session.post.assert_called_once()
            call_args = self.mock_session.post.call_args
            assert call_args[0][0] == f"{BASE_URL}datasets/{dataset_slug}/items"
            assert "json" in call_args[1]

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

        with patch(
            "novaeval.noveum_platform.utils.handle_response",
            return_value={"deleted": True},
        ):
            result = self.client.delete_all_dataset_items(dataset_slug, item_ids)

            assert result == {"deleted": True}
            self.mock_session.delete.assert_called_once()
            call_args = self.mock_session.delete.call_args
            assert call_args[0][0] == f"{BASE_URL}datasets/{dataset_slug}/items"
            assert call_args[1]["json"] == {"itemIds": ["item-1", "item-2"]}

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

        with patch(
            "novaeval.noveum_platform.utils.handle_response",
            return_value={"deleted": True},
        ):
            result = self.client.delete_all_dataset_items(dataset_slug, item_ids)

            assert result == {"deleted": True}
            self.mock_session.delete.assert_called_once()
            call_args = self.mock_session.delete.call_args
            assert call_args[0][0] == f"{BASE_URL}datasets/{dataset_slug}/items"
            assert call_args[1]["json"] == {"itemIds": ["item-1", "item-2"]}

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

        with patch(
            "novaeval.noveum_platform.utils.handle_response",
            return_value={"item_key": item_key},
        ):
            result = self.client.get_dataset_item(dataset_slug, item_key)

            assert result == {"item_key": item_key}
            self.mock_session.get.assert_called_once_with(
                f"{BASE_URL}datasets/{dataset_slug}/items/{item_key}",
                timeout=30.0,
            )

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

        with patch(
            "novaeval.noveum_platform.utils.handle_response",
            return_value={"deleted": True},
        ):
            result = self.client.delete_dataset_item(dataset_slug, item_id)

            assert result == {"deleted": True}
            self.mock_session.delete.assert_called_once_with(
                f"{BASE_URL}datasets/{dataset_slug}/items/{item_id}",
                timeout=30.0,
            )


class TestNoveumClientScorerResults:
    """Test cases for scorer results methods."""

    def setup_method(self):
        """Set up test client."""
        with patch("novaeval.noveum_platform.utils.requests.Session") as mock_session:
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

        with patch(
            "novaeval.noveum_platform.utils.handle_response",
            return_value={"results": []},
        ):
            result = self.client.list_scorer_results(limit=10)

            assert result == {"results": []}
            self.mock_session.get.assert_called_once()
            call_args = self.mock_session.get.call_args
            assert call_args[0][0] == f"{BASE_URL}scorers/results"
            assert "params" in call_args[1]

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

        with patch(
            "novaeval.noveum_platform.utils.handle_response",
            return_value={"id": "result-123"},
        ):
            result = self.client.create_scorer_result(result_data)

            assert result == {"id": "result-123"}
            self.mock_session.post.assert_called_once()
            call_args = self.mock_session.post.call_args
            assert call_args[0][0] == f"{BASE_URL}scorers/results"
            assert "json" in call_args[1]

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

        with patch(
            "novaeval.noveum_platform.utils.handle_response",
            return_value={"created": 2},
        ):
            result = self.client.create_scorer_results_batch(results)

            assert result == {"created": 2}
            self.mock_session.post.assert_called_once()
            call_args = self.mock_session.post.call_args
            assert call_args[0][0] == f"{BASE_URL}scorers/results/batch"
            assert "json" in call_args[1]

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

        with patch(
            "novaeval.noveum_platform.utils.handle_response",
            return_value={"score": 0.95},
        ):
            result = self.client.get_scorer_result(dataset_slug, item_id, scorer_id)

            assert result == {"score": 0.95}
            self.mock_session.get.assert_called_once_with(
                f"{BASE_URL}scorers/results/{dataset_slug}/{item_id}/{scorer_id}",
                timeout=30.0,
            )

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

        with patch(
            "novaeval.noveum_platform.utils.handle_response",
            return_value={"updated": True},
        ):
            result = self.client.update_scorer_result(
                dataset_slug,
                item_id,
                scorer_id,
                result_data,
            )

            assert result == {"updated": True}
            self.mock_session.put.assert_called_once()
            call_args = self.mock_session.put.call_args
            assert (
                call_args[0][0]
                == f"{BASE_URL}scorers/results/{dataset_slug}/{item_id}/{scorer_id}"
            )
            assert "json" in call_args[1]

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

        with patch(
            "novaeval.noveum_platform.utils.handle_response",
            return_value={"deleted": True},
        ):
            result = self.client.delete_scorer_result(dataset_slug, item_id, scorer_id)

            assert result == {"deleted": True}
            self.mock_session.delete.assert_called_once_with(
                f"{BASE_URL}scorers/results/{dataset_slug}/{item_id}/{scorer_id}",
                timeout=30.0,
            )
