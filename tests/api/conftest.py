"""
Common test fixtures and utilities for API testing.
"""

from unittest.mock import Mock, patch

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client():
    """Test client for FastAPI app."""
    return TestClient(app)


@pytest.fixture
def mock_model_factory():
    """Mock ModelFactory for testing model endpoints."""
    with patch("app.core.discovery.get_registry") as mock_registry:
        mock_registry.return_value.get_model_config.return_value = {
            "openai": {
                "config_class": "ModelConfig",
                "schema": {
                    "type": "object",
                    "properties": {
                        "model_name": {"type": "string"},
                        "provider": {"type": "string"},
                    },
                },
            }
        }
        yield mock_registry


@pytest.fixture
def mock_dataset_factory():
    """Mock DatasetFactory for testing dataset endpoints."""
    with patch("app.core.discovery.get_registry") as mock_registry:
        mock_registry.return_value.get_dataset_config.return_value = {
            "mmlu": {
                "config_class": "DatasetConfig",
                "schema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "split": {"type": "string"},
                    },
                },
            }
        }
        yield mock_registry


@pytest.fixture
def mock_scorer_factory():
    """Mock ScorerFactory for testing scorer endpoints."""
    with patch("app.core.discovery.get_registry") as mock_registry:
        mock_registry.return_value.get_scorer_config.return_value = {
            "accuracy": {
                "config_class": "ScorerConfig",
                "schema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "type": {"type": "string"},
                    },
                },
            }
        }
        yield mock_registry


@pytest.fixture
def sample_evaluation_config():
    """Sample evaluation configuration for testing."""
    return {
        "name": "test_evaluation",
        "description": "Test evaluation job",
        "models": [
            {
                "provider": "openai",
                "model_name": "gpt-3.5-turbo",
                "temperature": 0.7,
                "max_tokens": 100,
            }
        ],
        "datasets": [
            {
                "name": "mmlu",
                "type": "huggingface",
                "split": "test",
                "limit": 10,
                "seed": 42,
            }
        ],
        "scorers": [{"name": "accuracy", "type": "accuracy"}],
        "parallel_models": False,
        "max_workers": 4,
        "timeout": 300,
    }


@pytest.fixture
def mock_task_manager():
    """Mock TaskManager for testing evaluation endpoints."""
    with patch("app.api.v1.evaluations.get_task_manager") as mock_manager:
        mock_instance = Mock()
        mock_instance.create_task.return_value = "test-task-id"
        from datetime import datetime

        from app.core.task_manager import TaskStatus

        mock_task_info = Mock()
        mock_task_info.task_id = "test-task-id"
        mock_task_info.name = "test_evaluation"
        mock_task_info.status = TaskStatus.PENDING
        mock_task_info.created_at = datetime(2024, 1, 1, 0, 0, 0)
        mock_task_info.started_at = datetime(2024, 1, 1, 0, 1, 0)
        mock_task_info.completed_at = None
        mock_task_info.progress = 0.0
        mock_task_info.error_message = None
        mock_task_info.result_size = None
        mock_instance.get_task_info.return_value = mock_task_info
        mock_instance.get_stats.return_value = {
            "total_tasks": 5,
            "total_results": 3,
            "running_tasks": 1,
            "status_counts": {"pending": 2, "completed": 2, "running": 1},
            "max_concurrent_tasks": 5,
            "result_ttl_seconds": 7200,
            "last_cleanup": "2024-01-01T00:00:00",
        }
        mock_instance.cleanup_expired_results.return_value = 3
        mock_instance.auto_cleanup_if_needed.return_value = None
        mock_instance.start_task.return_value = None

        # Mock list_tasks method
        mock_task_info2 = Mock()
        mock_task_info2.task_id = "task1"
        mock_task_info2.name = "test_eval"
        mock_task_info2.status = TaskStatus.COMPLETED
        mock_task_info2.created_at = datetime(2024, 1, 1, 0, 0, 0)
        mock_task_info2.started_at = datetime(2024, 1, 1, 0, 1, 0)
        mock_task_info2.completed_at = datetime(2024, 1, 1, 0, 2, 0)
        mock_task_info2.progress = 1.0
        mock_task_info2.error_message = None
        mock_task_info2.result_size = 1024
        mock_instance.list_tasks.return_value = {"task1": mock_task_info2}
        mock_manager.return_value = mock_instance
        yield mock_instance
