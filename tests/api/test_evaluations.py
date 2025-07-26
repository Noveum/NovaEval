"""
Tests for evaluation API endpoints.
"""

import io
import json
from unittest.mock import Mock, patch

import yaml


class TestEvaluationEndpoints:
    """Test evaluation job submission and management endpoints."""

    def test_get_stats(self, client, mock_task_manager):
        """Test getting task manager statistics."""
        # The fixture already sets the complete mock return value, so we don't need to override it

        response = client.get("/api/v1/evaluations/stats")
        assert response.status_code == 200
        data = response.json()
        assert data["total_tasks"] == 5
        assert data["running_tasks"] == 1

    def test_list_evaluations(self, client, mock_task_manager):
        """Test listing evaluation tasks."""
        # The fixture already sets the proper mock return value with real data types

        response = client.get("/api/v1/evaluations/")
        assert response.status_code == 200
        data = response.json()
        assert "tasks" in data
        assert "total" in data

    def test_get_task_status(self, client, mock_task_manager):
        """Test getting evaluation task status."""
        response = client.get("/api/v1/evaluations/test-task-id/status")
        assert response.status_code == 200
        data = response.json()
        assert data["task_id"] == "test-task-id"
        assert data["name"] == "test_evaluation"

    def test_get_nonexistent_task_status(self, client):
        """Test getting status of nonexistent task."""
        with patch("app.core.task_manager.get_task_manager") as mock_manager:
            mock_instance = Mock()
            mock_instance.get_task_info.return_value = None
            mock_manager.return_value = mock_instance

            response = client.get("/api/v1/evaluations/nonexistent/status")
            assert response.status_code == 404

    def test_validate_config_json(self, client, sample_evaluation_config):
        """Test validating JSON evaluation configuration."""
        json_content = json.dumps(sample_evaluation_config)
        files = {
            "config_file": (
                "config.json",
                io.BytesIO(json_content.encode()),
                "application/json",
            )
        }

        response = client.post("/api/v1/evaluations/validate-config", files=files)
        assert response.status_code == 200
        data = response.json()
        assert data["valid"] is True
        assert "config_summary" in data

    def test_validate_config_yaml(self, client, sample_evaluation_config):
        """Test validating YAML evaluation configuration."""
        yaml_content = yaml.dump(sample_evaluation_config)
        files = {
            "config_file": (
                "config.yaml",
                io.BytesIO(yaml_content.encode()),
                "application/yaml",
            )
        }

        response = client.post("/api/v1/evaluations/validate-config", files=files)
        assert response.status_code == 200
        data = response.json()
        assert data["valid"] is True

    def test_validate_invalid_config(self, client):
        """Test validating invalid configuration."""
        invalid_config = {"invalid": "config"}
        json_content = json.dumps(invalid_config)
        files = {
            "config_file": (
                "config.json",
                io.BytesIO(json_content.encode()),
                "application/json",
            )
        }

        response = client.post("/api/v1/evaluations/validate-config", files=files)
        assert response.status_code == 200
        data = response.json()
        assert data["valid"] is False
        assert "error_message" in data

    def test_submit_evaluation(
        self, client, mock_task_manager, sample_evaluation_config
    ):
        """Test submitting evaluation job."""
        json_content = json.dumps(sample_evaluation_config)
        files = {
            "config_file": (
                "config.json",
                io.BytesIO(json_content.encode()),
                "application/json",
            )
        }

        response = client.post("/api/v1/evaluations/submit", files=files)
        assert response.status_code == 200
        data = response.json()
        assert data["task_id"] == "test-task-id"
        assert data["name"] == "test_evaluation"

    def test_submit_large_file(self, client):
        """Test submitting file that exceeds size limit."""
        large_content = "x" * (101 * 1024 * 1024)  # 101MB
        files = {
            "config_file": (
                "large.json",
                io.BytesIO(large_content.encode()),
                "application/json",
            )
        }

        response = client.post("/api/v1/evaluations/submit", files=files)
        assert response.status_code == 413

    def test_cleanup_results(self, client, mock_task_manager):
        """Test manual cleanup of expired results."""
        mock_task_manager.cleanup_expired_results.return_value = 3

        response = client.post("/api/v1/evaluations/cleanup")
        assert response.status_code == 200
        data = response.json()
        assert data["cleaned_count"] == 3
