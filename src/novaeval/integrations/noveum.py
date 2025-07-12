"""
Noveum.ai platform integration for NovaEval.

This module provides integration with the Noveum.ai platform for
dataset management, model access, and evaluation result reporting.
"""

from pathlib import Path
from typing import Any, Optional, Union

import requests  # type: ignore

from novaeval.utils.logging import get_logger

logger = get_logger(__name__)


class NoveumIntegration:
    """
    Integration with the Noveum.ai platform.

    Provides access to datasets, models, and evaluation job management
    through the Noveum.ai API.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.noveum.ai",
        organization_id: Optional[str] = None,
        project_id: Optional[str] = None,
        timeout: float = 30.0,
    ):
        """
        Initialize Noveum integration.

        Args:
            api_key: Noveum API key
            base_url: Base URL for Noveum API
            organization_id: Organization ID
            project_id: Project ID for evaluation jobs
            timeout: Request timeout in seconds
        """
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.organization_id = organization_id
        self.project_id = project_id
        self.timeout = timeout

        # Setup session with authentication
        self.session = requests.Session()
        if api_key:
            self.session.headers.update(
                {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                    "User-Agent": "NovaEval/0.1.0",
                }
            )

    def get_datasets(
        self,
        category: Optional[str] = None,
        tags: Optional[list[str]] = None,
        public_only: bool = False,
    ) -> list[dict[str, Any]]:
        """
        Get available datasets from Noveum platform.

        Args:
            category: Filter by dataset category
            tags: Filter by tags
            public_only: Only return public datasets

        Returns:
            List of dataset information
        """
        params: dict[str, str] = {}
        if category:
            params["category"] = category
        if tags:
            params["tags"] = ",".join(tags)
        if public_only:
            params["public"] = "true"

        try:
            response = self.session.get(
                f"{self.base_url}/v1/datasets", params=params, timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()
            if isinstance(data, dict):
                return data.get("datasets", [])  # type: ignore
            return []
        except Exception as e:
            logger.error(f"Failed to fetch datasets: {e}")
            return []

    def get_dataset(
        self, dataset_id: str, version: Optional[str] = None
    ) -> Optional[dict[str, Any]]:
        """
        Get specific dataset information.

        Args:
            dataset_id: Dataset identifier
            version: Specific version (defaults to latest)

        Returns:
            Dataset information or None if not found
        """
        url = f"{self.base_url}/v1/datasets/{dataset_id}"
        if version:
            url += f"/versions/{version}"

        try:
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            if isinstance(data, dict):
                return data
            return None
        except Exception as e:
            logger.error(f"Failed to fetch dataset {dataset_id}: {e}")
            return None

    def download_dataset(
        self,
        dataset_id: str,
        output_path: Union[str, Path],
        version: Optional[str] = None,
        format: str = "jsonl",
    ) -> bool:
        """
        Download dataset from Noveum platform.

        Args:
            dataset_id: Dataset identifier
            output_path: Local path to save dataset
            version: Specific version (defaults to latest)
            format: Download format (jsonl, csv, json)

        Returns:
            True if successful, False otherwise
        """
        url = f"{self.base_url}/v1/datasets/{dataset_id}/download"
        params: dict[str, str] = {"format": format}
        if version:
            params["version"] = version

        try:
            response = self.session.get(
                url, params=params, timeout=self.timeout, stream=True
            )
            response.raise_for_status()

            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            logger.info(f"Downloaded dataset {dataset_id} to {output_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to download dataset {dataset_id}: {e}")
            return False

    def get_models(self, provider: Optional[str] = None) -> list[dict[str, Any]]:
        """
        Get available models from Noveum platform.

        Args:
            provider: Filter by provider (openai, anthropic, etc.)

        Returns:
            List of model information
        """
        params: dict[str, str] = {}
        if provider:
            params["provider"] = provider

        try:
            response = self.session.get(
                f"{self.base_url}/v1/models", params=params, timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()
            if isinstance(data, dict):
                return data.get("models", [])  # type: ignore
            return []
        except Exception as e:
            logger.error(f"Failed to fetch models: {e}")
            return []

    def create_evaluation_job(
        self,
        name: str,
        dataset_id: str,
        model_ids: list[str],
        primary_metric: str = "accuracy",
        evaluator_type: str = "text_quality",
        description: Optional[str] = None,
        config: Optional[dict[str, Any]] = None,
    ) -> Optional[str]:
        """
        Create evaluation job on Noveum platform.

        Args:
            name: Job name
            dataset_id: Dataset to evaluate on
            model_ids: List of model IDs to evaluate
            primary_metric: Primary evaluation metric
            evaluator_type: Type of evaluator (text_quality, code_quality, custom)
            description: Job description
            config: Additional configuration

        Returns:
            Job ID if successful, None otherwise
        """
        if not self.project_id:
            logger.error("Project ID is required for creating evaluation jobs")
            return None

        job_data: dict[str, Any] = {
            "name": name,
            "project_id": self.project_id,
            "dataset_id": dataset_id,
            "model_ids": model_ids,
            "primary_metric": primary_metric,
            "evaluator_type": evaluator_type,
        }

        if description:
            job_data["description"] = description
        if config:
            job_data["config"] = config

        try:
            response = self.session.post(
                f"{self.base_url}/v1/evaluation-jobs",
                json=job_data,
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()
            if isinstance(data, dict):
                return data.get("job_id")
            return None
        except Exception as e:
            logger.error(f"Failed to create evaluation job: {e}")
            return None

    def get_evaluation_job(self, job_id: str) -> Optional[dict[str, Any]]:
        """
        Get evaluation job status and results.

        Args:
            job_id: Job identifier

        Returns:
            Job information or None if not found
        """
        try:
            response = self.session.get(
                f"{self.base_url}/v1/evaluation-jobs/{job_id}", timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()
            if isinstance(data, dict):
                return data
            return None
        except Exception as e:
            logger.error(f"Failed to get evaluation job {job_id}: {e}")
            return None

    def upload_evaluation_results(
        self,
        job_id: str,
        results: dict[str, Any],
        artifacts: Optional[list[str]] = None,
    ) -> bool:
        """
        Upload evaluation results to Noveum platform.

        Args:
            job_id: Job identifier
            results: Results dictionary
            artifacts: List of artifact file paths

        Returns:
            True if successful, False otherwise
        """
        try:
            # Upload results
            response = self.session.post(
                f"{self.base_url}/v1/evaluation-jobs/{job_id}/results",
                json=results,
                timeout=self.timeout,
            )
            response.raise_for_status()

            # Upload artifacts if provided
            if artifacts:
                for artifact_path in artifacts:
                    success = self._upload_artifact(job_id, artifact_path)
                    if not success:
                        logger.warning(f"Failed to upload artifact: {artifact_path}")

            logger.info(f"Uploaded results for job {job_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to upload results for job {job_id}: {e}")
            return False

    def _upload_artifact(self, job_id: str, artifact_path: str) -> bool:
        """
        Upload artifact file to Noveum platform.

        Args:
            job_id: Job identifier
            artifact_path: Path to artifact file

        Returns:
            True if successful, False otherwise
        """
        artifact_path_obj = Path(artifact_path)
        if not artifact_path_obj.exists():
            logger.error(f"Artifact file not found: {artifact_path}")
            return False

        try:
            with open(artifact_path_obj, "rb") as f:
                files = {
                    "file": (artifact_path_obj.name, f, "application/octet-stream")
                }
                response = self.session.post(
                    f"{self.base_url}/v1/evaluation-jobs/{job_id}/artifacts",
                    files=files,
                    timeout=self.timeout,
                )
                response.raise_for_status()

            logger.info(f"Uploaded artifact: {artifact_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to upload artifact {artifact_path}: {e}")
            return False

    def get_request_logs(
        self,
        project_id: Optional[str] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        limit: int = 100,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
    ) -> list[dict[str, Any]]:
        """
        Get request logs from Noveum platform.

        Args:
            project_id: Project ID (uses default if not provided)
            provider: Filter by provider
            model: Filter by model
            limit: Maximum number of logs to return
            start_time: Start time as Unix timestamp
            end_time: End time as Unix timestamp

        Returns:
            List of request log entries
        """
        params: dict[str, str] = {"limit": str(limit)}

        if project_id:
            params["project_id"] = project_id
        elif self.project_id:
            params["project_id"] = self.project_id

        if provider:
            params["provider"] = provider
        if model:
            params["model"] = model
        if start_time:
            params["start_time"] = str(int(start_time))
        if end_time:
            params["end_time"] = str(int(end_time))

        try:
            response = self.session.get(
                f"{self.base_url}/v1/request-logs", params=params, timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()
            if isinstance(data, dict):
                return data.get("logs", [])  # type: ignore
            return []
        except Exception as e:
            logger.error(f"Failed to fetch request logs: {e}")
            return []

    def create_dataset_from_logs(
        self,
        name: str,
        log_ids: list[str],
        description: Optional[str] = None,
        tags: Optional[list[str]] = None,
    ) -> Optional[str]:
        """
        Create dataset from request logs.

        Args:
            name: Dataset name
            log_ids: List of log IDs to include
            description: Dataset description
            tags: List of tags

        Returns:
            Dataset ID if successful, None otherwise
        """
        dataset_data = {
            "name": name,
            "log_ids": log_ids,
            "project_id": self.project_id,
        }

        if description:
            dataset_data["description"] = description
        if tags:
            dataset_data["tags"] = tags

        try:
            response = self.session.post(
                f"{self.base_url}/v1/datasets/from-logs",
                json=dataset_data,
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()
            if isinstance(data, dict):
                return data.get("dataset_id")
            return None
        except Exception as e:
            logger.error(f"Failed to create dataset from logs: {e}")
            return None

    def validate_connection(self) -> bool:
        """
        Validate connection to Noveum platform.

        Returns:
            True if connection is valid, False otherwise
        """
        try:
            response = self.session.get(
                f"{self.base_url}/v1/health", timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()
            return isinstance(data, dict) and data.get("status") == "ok"
        except Exception as e:
            logger.error(f"Failed to validate connection: {e}")
            return False
