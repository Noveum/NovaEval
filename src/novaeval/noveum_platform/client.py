"""
Noveum Platform API Client.

This module provides a synchronous client for interacting with the Noveum Platform
API. It handles authentication, request/response processing, and error handling for
traces, datasets, and scorer results.
"""

import os
from typing import Any, Literal, Optional

import pydantic
import requests
from dotenv import load_dotenv

from novaeval.utils.logging import get_logger

from .exceptions import (
    AuthenticationError,
    ConflictError,
    ForbiddenError,
    NotFoundError,
    RateLimitError,
    ServerError,
    ValidationError,
)
from .models import (
    DatasetCreateRequest,
    DatasetItem,
    DatasetItemsCreateRequest,
    DatasetItemsQueryParams,
    DatasetsQueryParams,
    DatasetUpdateRequest,
    DatasetVersionCreateRequest,
    ScorerResultCreateRequest,
    ScorerResultsBatchRequest,
    ScorerResultsQueryParams,
    ScorerResultUpdateRequest,
    TracesQueryParams,
)

# Load environment variables
load_dotenv()

logger = get_logger(__name__)


class NoveumClient:
    """
    Unified client for Noveum Platform API.

    Provides methods for traces, datasets, and scorer results with clear
    prefixed method names to avoid ambiguity.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://noveum.ai",
        organization_id: Optional[str] = None,
        timeout: float = 30.0,
    ):
        """
        Initialize the NoveumClient.

        Args:
            api_key: Noveum API key. If not provided, will try to load from
                     NOVEUM_API_KEY environment variable.
            base_url: Base URL for the Noveum API. Defaults to https://noveum.ai
            organization_id: Organization ID for API calls. If not provided,
                           will try to load from NOVEUM_ORGANIZATION_ID env var.
            timeout: Request timeout in seconds. Defaults to 30.0.
        """
        self.api_key = api_key or os.getenv("NOVEUM_API_KEY")
        self.base_url = base_url.rstrip("/")
        self.organization_id = organization_id or os.getenv("NOVEUM_ORGANIZATION_ID")
        self.timeout = timeout

        if not self.api_key:
            raise ValueError(
                "API key is required. Provide it directly or set NOVEUM_API_KEY environment variable."
            )

        # Setup session with authentication
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "User-Agent": "NovaEval/0.5.3",
            }
        )

        # Add organization header if provided
        if self.organization_id:
            self.session.headers.update({"X-Organization-Id": self.organization_id})

    def _handle_response(self, response: requests.Response) -> dict[str, Any]:
        """
        Handle API response and raise appropriate exceptions for errors.

        Args:
            response: The requests.Response object from the API call

        Returns:
            Parsed JSON response as dictionary

        Raises:
            Various NoveumAPIError subclasses based on HTTP status code
        """
        try:
            response_body = response.json() if response.content else {}
        except ValueError:
            response_body = {"error": "Invalid JSON response"}

        # Check for specific error status codes
        if response.status_code == 400:
            raise ValidationError(
                message=response_body.get("message", "Invalid request format"),
                response_body=response_body,
            )
        elif response.status_code == 401:
            raise AuthenticationError(
                message=response_body.get("message", "Unauthorized - Invalid API key"),
                response_body=response_body,
            )
        elif response.status_code == 403:
            raise ForbiddenError(
                message=response_body.get(
                    "message", "Forbidden (org mismatch or access denied)"
                ),
                response_body=response_body,
            )
        elif response.status_code == 404:
            raise NotFoundError(
                message=response_body.get("message", "Resource not found"),
                response_body=response_body,
            )
        elif response.status_code == 409:
            raise ConflictError(
                message=response_body.get("message", "Conflict - Trace is immutable"),
                response_body=response_body,
            )
        elif response.status_code == 429:
            raise RateLimitError(
                message=response_body.get("message", "Rate limit exceeded"),
                response_body=response_body,
            )
        elif response.status_code >= 500:
            raise ServerError(
                message=response_body.get("message", "Internal server error"),
                status_code=response.status_code,
                response_body=response_body,
            )

        # For successful responses, return the parsed JSON
        response.raise_for_status()  # This will raise for any other error status codes
        return response_body

    def _parse_model(self, model_cls, data: dict[str, Any]):
        """
        Parse data into a Pydantic model and convert ValidationError into our API ValidationError.

        Args:
            model_cls: The Pydantic model class to instantiate
            data: Dictionary of data to parse into the model

        Returns:
            Instantiated Pydantic model instance

        Raises:
            ValidationError: If validation fails, converted from pydantic.ValidationError
        """
        try:
            return model_cls(**data)
        except pydantic.ValidationError as e:
            error_messages = []
            for err in e.errors():
                loc = ".".join(str(p) for p in err.get("loc", []) or ["unknown"])
                msg = err.get("msg", "Validation error")
                error_messages.append(f"{loc}: {msg}")
            raise ValidationError(
                message="; ".join(error_messages),
                response_body={"validation_errors": e.errors()},
            ) from e

    def ingest_traces(self, traces: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Ingest multiple traces in a single batch request.

        Args:
            traces: List of trace dictionaries to ingest

        Returns:
            API response containing ingestion results

        Raises:
            NoveumAPIError: If the API request fails
        """
        logger.info("Ingesting %d traces", len(traces))

        response = self.session.post(
            f"{self.base_url}/api/v1/traces", json=traces, timeout=self.timeout
        )

        return self._handle_response(response)

    def ingest_trace(self, trace: dict[str, Any]) -> dict[str, Any]:
        """
        Ingest a single trace.

        Args:
            trace: Trace dictionary to ingest

        Returns:
            API response containing ingestion results

        Raises:
            NoveumAPIError: If the API request fails
        """
        logger.info("Ingesting single trace")

        response = self.session.post(
            f"{self.base_url}/api/v1/traces/single", json=trace, timeout=self.timeout
        )

        return self._handle_response(response)

    def query_traces(
        self,
        organization_id: Optional[str] = None,
        from_: Optional[int] = None,
        size: Optional[int] = 20,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        project: Optional[str] = None,
        environment: Optional[str] = None,
        status: Optional[str] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        tags: Optional[list[str]] = None,
        sort: Optional[
            Literal[
                "start_time:asc",
                "start_time:desc",
                "end_time:asc",
                "end_time:desc",
                "duration_ms:asc",
                "duration_ms:desc",
            ]
        ] = "start_time:desc",
        search_term: Optional[str] = None,
        include_spans: Optional[bool] = False,
    ) -> dict[str, Any]:
        """
        Query traces with optional filters and pagination.

        Args:
            organization_id: Organization ID filter
            from_: Pagination offset (0-based)
            size: Number of traces to return (1-100, default 20)
            start_time: Start time filter (ISO datetime)
            end_time: End time filter (ISO datetime)
            project: Project name filter
            environment: Environment filter
            status: Status filter
            user_id: User ID filter
            session_id: Session ID filter
            tags: List of tags to filter by
            sort: Sort order (e.g., "start_time:desc")
            search_term: Text search term
            include_spans: Whether to include spans (default False)

        Returns:
            API response containing traces and metadata

        Raises:
            NoveumAPIError: If the API request fails
        """
        # Build kwargs dict from parameters
        kwargs = {
            "organization_id": organization_id,
            "from_": from_,
            "size": size,
            "start_time": start_time,
            "end_time": end_time,
            "project": project,
            "environment": environment,
            "status": status,
            "user_id": user_id,
            "session_id": session_id,
            "tags": tags,
            "sort": sort,
            "search_term": search_term,
            "include_spans": include_spans,
        }
        
        # Validate parameters using Pydantic model
        query_params = self._parse_model(TracesQueryParams, kwargs)
        params = query_params.to_query_params()

        logger.info("Querying traces with params: %s", params)

        response = self.session.get(
            f"{self.base_url}/api/v1/traces", params=params, timeout=self.timeout
        )

        return self._handle_response(response)

    def get_trace(self, trace_id: str) -> dict[str, Any]:
        """
        Get a specific trace by its ID.

        Args:
            trace_id: The ID of the trace to retrieve

        Returns:
            Trace data dictionary

        Raises:
            NoveumAPIError: If the API request fails
        """
        logger.info("Getting trace: %s", trace_id)

        response = self.session.get(
            f"{self.base_url}/api/v1/traces/{trace_id}", timeout=self.timeout
        )

        return self._handle_response(response)

    def get_directory_tree(self) -> dict[str, Any]:
        """
        Get the directory tree for the organization.

        Returns:
            Directory structure data

        Raises:
            NoveumAPIError: If the API request fails
        """
        logger.info("Getting directory tree")

        response = self.session.get(
            f"{self.base_url}/api/v1/traces/directory-tree", timeout=self.timeout
        )

        return self._handle_response(response)

    def get_connection_status(self) -> dict[str, Any]:
        """
        Get the connection status for the organization.

        Returns:
            Connection status data

        Raises:
            NoveumAPIError: If the API request fails
        """
        logger.info("Getting connection status")

        response = self.session.get(
            f"{self.base_url}/api/v1/traces/connection-status", timeout=self.timeout
        )

        return self._handle_response(response)

    def get_trace_spans(self, trace_id: str) -> dict[str, Any]:
        """
        Get all spans for a specific trace.

        Args:
            trace_id: The ID of the trace to get spans for

        Returns:
            Spans data dictionary

        Raises:
            NoveumAPIError: If the API request fails
        """
        logger.info("Getting spans for trace: %s", trace_id)

        response = self.session.get(
            f"{self.base_url}/api/v1/traces/{trace_id}/spans", timeout=self.timeout
        )

        return self._handle_response(response)

    # Dataset Methods

    def create_dataset(
        self,
        name: str,
        slug: Optional[str] = None,
        description: Optional[str] = None,
        visibility: Literal["public", "org", "private"] = "org",
        dataset_type: Literal["agent", "conversational", "g-eval", "custom"] = "custom",
        environment: Optional[str] = None,
        schema_version: Optional[str] = None,
        tags: Optional[list[str]] = None,
        custom_attributes: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """
        Create a new dataset.

        Args:
            name: Dataset name (required)
            slug: Dataset slug (auto-generated if not provided)
            description: Dataset description
            visibility: Dataset visibility (default: "org")
            dataset_type: Dataset type (default: "custom")
            environment: Environment
            schema_version: Schema version
            tags: Dataset tags
            custom_attributes: Custom attributes

        Returns:
            API response containing created dataset data

        Raises:
            NoveumAPIError: If the API request fails
        """
        # Build kwargs dict from parameters
        kwargs = {
            "name": name,
            "slug": slug,
            "description": description,
            "visibility": visibility,
            "dataset_type": dataset_type,
            "environment": environment,
            "schema_version": schema_version,
            "tags": tags,
            "custom_attributes": custom_attributes,
        }
        
        # Validate request data
        request_data = self._parse_model(DatasetCreateRequest, kwargs)

        logger.info("Creating dataset: %s", name)

        response = self.session.post(
            f"{self.base_url}/api/v1/datasets",
            json=request_data.model_dump(exclude_none=True),
            timeout=self.timeout,
        )

        return self._handle_response(response)

    def list_datasets(
        self,
        limit: Optional[int] = 20,
        offset: Optional[int] = 0,
        visibility: Optional[Literal["public", "org", "private"]] = None,
        organizationSlug: Optional[str] = None,
        includeVersions: Optional[bool] = False,
    ) -> dict[str, Any]:
        """
        List datasets with optional filters and pagination.

        Args:
            limit: Number of datasets to return (1-1000, default 20)
            offset: Number of datasets to skip (default 0)
            visibility: Filter by visibility (public, org, private)
            organizationSlug: Filter by organization slug
            includeVersions: Whether to include versions (default False)

        Returns:
            API response containing datasets and metadata

        Raises:
            NoveumAPIError: If the API request fails
        """
        # Build kwargs dict from parameters
        kwargs = {
            "limit": limit,
            "offset": offset,
            "visibility": visibility,
            "organizationSlug": organizationSlug,
            "includeVersions": includeVersions,
        }
        
        # Validate parameters using Pydantic model
        query_params = self._parse_model(DatasetsQueryParams, kwargs)
        params = query_params.to_query_params()

        logger.info("Listing datasets with params: %s", params)

        response = self.session.get(
            f"{self.base_url}/api/v1/datasets", params=params, timeout=self.timeout
        )

        return self._handle_response(response)

    def get_dataset(self, slug: str) -> dict[str, Any]:
        """
        Get a specific dataset by its slug.

        Args:
            slug: The slug of the dataset to retrieve

        Returns:
            Dataset data dictionary

        Raises:
            NoveumAPIError: If the API request fails
        """
        logger.info("Getting dataset: %s", slug)

        response = self.session.get(
            f"{self.base_url}/api/v1/datasets/{slug}", timeout=self.timeout
        )

        return self._handle_response(response)

    def update_dataset(
        self,
        slug: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        visibility: Optional[Literal["public", "org", "private"]] = None,
        dataset_type: Optional[Literal["agent", "conversational", "g-eval", "custom"]] = None,
        environment: Optional[str] = None,
        schema_version: Optional[str] = None,
        tags: Optional[list[str]] = None,
        custom_attributes: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """
        Update an existing dataset.

        Args:
            slug: Dataset slug
            name: Dataset name
            description: Dataset description
            visibility: Dataset visibility
            dataset_type: Dataset type
            environment: Environment
            schema_version: Schema version
            tags: Dataset tags
            custom_attributes: Custom attributes

        Returns:
            API response containing updated dataset data

        Raises:
            NoveumAPIError: If the API request fails
        """
        # Build kwargs dict from parameters
        kwargs = {
            "name": name,
            "description": description,
            "visibility": visibility,
            "dataset_type": dataset_type,
            "environment": environment,
            "schema_version": schema_version,
            "tags": tags,
            "custom_attributes": custom_attributes,
        }
        
        # Validate request data
        request_data = self._parse_model(DatasetUpdateRequest, kwargs)

        logger.info("Updating dataset: %s", slug)

        response = self.session.put(
            f"{self.base_url}/api/v1/datasets/{slug}",
            json=request_data.model_dump(exclude_none=True),
            timeout=self.timeout,
        )

        return self._handle_response(response)

    def delete_dataset(self, slug: str) -> dict[str, Any]:
        """
        Delete a dataset.

        Args:
            slug: Dataset slug to delete

        Returns:
            API response confirming deletion

        Raises:
            NoveumAPIError: If the API request fails
        """
        logger.info("Deleting dataset: %s", slug)

        response = self.session.delete(
            f"{self.base_url}/api/v1/datasets/{slug}", timeout=self.timeout
        )

        return self._handle_response(response)

    def list_dataset_versions(self, dataset_slug: str) -> dict[str, Any]:
        """
        List versions for a dataset.

        Args:
            dataset_slug: Dataset slug

        Returns:
            API response containing dataset versions

        Raises:
            NoveumAPIError: If the API request fails
        """
        logger.info("Listing versions for dataset: %s", dataset_slug)

        response = self.session.get(
            f"{self.base_url}/api/v1/datasets/{dataset_slug}/versions",
            timeout=self.timeout,
        )

        return self._handle_response(response)

    def create_dataset_version(
        self, dataset_slug: str, version_data: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Create a new version for a dataset.

        Args:
            dataset_slug: Dataset slug
            version_data: Version data dictionary

        Returns:
            API response containing created version data

        Raises:
            NoveumAPIError: If the API request fails
        """
        # Validate request data
        request_data = self._parse_model(DatasetVersionCreateRequest, version_data)

        logger.info("Creating version for dataset: %s", dataset_slug)

        response = self.session.post(
            f"{self.base_url}/api/v1/datasets/{dataset_slug}/versions",
            json=request_data.model_dump(exclude_none=True),
            timeout=self.timeout,
        )

        return self._handle_response(response)

    def get_dataset_version(self, dataset_slug: str, version: str) -> dict[str, Any]:
        """
        Get a specific dataset version.

        Args:
            dataset_slug: Dataset slug
            version: Version identifier

        Returns:
            Version data dictionary

        Raises:
            NoveumAPIError: If the API request fails
        """
        logger.info("Getting version %s for dataset: %s", version, dataset_slug)

        response = self.session.get(
            f"{self.base_url}/api/v1/datasets/{dataset_slug}/versions/{version}",
            timeout=self.timeout,
        )

        return self._handle_response(response)

    def publish_dataset_version(
        self, dataset_slug: str, version: str
    ) -> dict[str, Any]:
        """
        Publish a dataset version.

        Args:
            dataset_slug: Dataset slug
            version: Version identifier to publish

        Returns:
            API response confirming publication

        Raises:
            NoveumAPIError: If the API request fails
        """
        logger.info("Publishing version %s for dataset: %s", version, dataset_slug)

        response = self.session.post(
            f"{self.base_url}/api/v1/datasets/{dataset_slug}/versions/{version}/publish",
            timeout=self.timeout,
        )

        return self._handle_response(response)

    def list_dataset_items(
        self,
        dataset_slug: str,
        version: Optional[str] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
    ) -> dict[str, Any]:
        """
        List items in a dataset with optional filters and pagination.

        Args:
            dataset_slug: Dataset slug
            version: Filter by version
            limit: Number of items to return (1-1000)
            offset: Number of items to skip

        Returns:
            API response containing dataset items and metadata

        Raises:
            NoveumAPIError: If the API request fails
        """
        # Build kwargs dict from parameters
        kwargs = {
            "version": version,
            "limit": limit,
            "offset": offset,
        }
        
        # Validate parameters using Pydantic model
        query_params = self._parse_model(DatasetItemsQueryParams, kwargs)
        params = query_params.to_query_params()

        logger.info(
            "Listing items for dataset %s with params: %s", dataset_slug, params
        )

        response = self.session.get(
            f"{self.base_url}/api/v1/datasets/{dataset_slug}/items",
            params=params,
            timeout=self.timeout,
        )

        return self._handle_response(response)

    def add_dataset_items(
        self, dataset_slug: str, version: str, items: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """
        Add items to a dataset.

        Args:
            dataset_slug: Dataset slug
            version: Dataset version
            items: List of items to add (each must have item_key, item_type, content)

        Returns:
            API response containing added items data

        Raises:
            NoveumAPIError: If the API request fails
        """
        # Convert dict items to DatasetItem objects
        dataset_items = [self._parse_model(DatasetItem, item) for item in items]

        # Validate request data
        request_data = self._parse_model(
            DatasetItemsCreateRequest, 
            {"version": version, "items": dataset_items}
        )

        logger.info("Adding %d items to dataset %s", len(items), dataset_slug)

        response = self.session.post(
            f"{self.base_url}/api/v1/datasets/{dataset_slug}/items",
            json=request_data.model_dump(exclude_none=True),
            timeout=self.timeout,
        )

        return self._handle_response(response)

    def delete_all_dataset_items(
        self, dataset_slug: str, version: Optional[str] = None
    ) -> dict[str, Any]:
        """
        Delete all items from a dataset.

        Args:
            dataset_slug: Dataset slug
            version: Optional version filter

        Returns:
            API response confirming deletion

        Raises:
            NoveumAPIError: If the API request fails
        """
        params = {"version": version} if version else {}

        logger.info("Deleting all items from dataset %s", dataset_slug)

        response = self.session.delete(
            f"{self.base_url}/api/v1/datasets/{dataset_slug}/items",
            params=params,
            timeout=self.timeout,
        )

        return self._handle_response(response)

    def get_dataset_item(self, dataset_slug: str, item_key: str) -> dict[str, Any]:
        """
        Get a specific dataset item by its key.

        Args:
            dataset_slug: Dataset slug
            item_key: Item key

        Returns:
            Item data dictionary

        Raises:
            NoveumAPIError: If the API request fails
        """
        logger.info("Getting item %s from dataset %s", item_key, dataset_slug)

        response = self.session.get(
            f"{self.base_url}/api/v1/datasets/{dataset_slug}/items/{item_key}",
            timeout=self.timeout,
        )

        return self._handle_response(response)

    def delete_dataset_item(self, dataset_slug: str, item_id: str) -> dict[str, Any]:
        """
        Delete a specific dataset item by its ID.

        Args:
            dataset_slug: Dataset slug
            item_id: Item ID

        Returns:
            API response confirming deletion

        Raises:
            NoveumAPIError: If the API request fails
        """
        logger.info("Deleting item %s from dataset %s", item_id, dataset_slug)

        response = self.session.delete(
            f"{self.base_url}/api/v1/datasets/{dataset_slug}/items/{item_id}",
            timeout=self.timeout,
        )

        return self._handle_response(response)

    # Scorer Results Methods

    def list_scorer_results(
        self,
        organizationSlug: str,
        datasetSlug: Optional[str] = None,
        itemId: Optional[str] = None,
        scorerId: Optional[str] = None,
        limit: Optional[int] = 100,
        offset: Optional[int] = 0,
    ) -> dict[str, Any]:
        """
        List scorer results with optional filters and pagination.

        Args:
            organizationSlug: Organization slug (required)
            datasetSlug: Filter by dataset slug
            itemId: Filter by item ID
            scorerId: Filter by scorer ID
            limit: Number of results to return (1-1000, default 100)
            offset: Number of results to skip (default 0)

        Returns:
            API response containing scorer results and metadata

        Raises:
            NoveumAPIError: If the API request fails
        """
        # Build kwargs dict from parameters
        kwargs = {
            "organizationSlug": organizationSlug,
            "datasetSlug": datasetSlug,
            "itemId": itemId,
            "scorerId": scorerId,
            "limit": limit,
            "offset": offset,
        }
        
        # Validate parameters using Pydantic model
        query_params = self._parse_model(ScorerResultsQueryParams, kwargs)
        params = query_params.to_query_params()

        logger.info("Listing scorer results with params: %s", params)

        response = self.session.get(
            f"{self.base_url}/api/v1/scorers/results",
            params=params,
            timeout=self.timeout,
        )

        return self._handle_response(response)

    def create_scorer_result(self, result_data: dict[str, Any]) -> dict[str, Any]:
        """
        Create a single scorer result.

        Args:
            result_data: Result data dictionary (datasetSlug, itemId, scorerId, score, etc.)

        Returns:
            API response containing created result data

        Raises:
            NoveumAPIError: If the API request fails
        """
        # Validate request data
        request_data = self._parse_model(ScorerResultCreateRequest, result_data)

        logger.info(
            "Creating scorer result for dataset %s, item %s, scorer %s",
            result_data.get("datasetSlug"),
            result_data.get("itemId"),
            result_data.get("scorerId"),
        )

        response = self.session.post(
            f"{self.base_url}/api/v1/scorers/results",
            json=request_data.model_dump(exclude_none=True),
            timeout=self.timeout,
        )

        return self._handle_response(response)

    def create_scorer_results_batch(
        self, results: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """
        Create multiple scorer results in a single batch request.

        Args:
            results: List of result data dictionaries

        Returns:
            API response containing batch creation results

        Raises:
            NoveumAPIError: If the API request fails
        """
        # Convert dict results to ScorerResultCreateRequest objects
        scorer_results = [self._parse_model(ScorerResultCreateRequest, result) for result in results]

        # Validate request data
        request_data = self._parse_model(ScorerResultsBatchRequest, {"results": scorer_results})

        logger.info("Creating %d scorer results in batch", len(results))

        response = self.session.post(
            f"{self.base_url}/api/v1/scorers/results/batch",
            json=request_data.model_dump(exclude_none=True),
            timeout=self.timeout,
        )

        return self._handle_response(response)

    def get_scorer_result(
        self, dataset_slug: str, item_id: str, scorer_id: str
    ) -> dict[str, Any]:
        """
        Get a specific scorer result.

        Args:
            dataset_slug: Dataset slug
            item_id: Item ID
            scorer_id: Scorer ID

        Returns:
            Scorer result data dictionary

        Raises:
            NoveumAPIError: If the API request fails
        """
        logger.info(
            "Getting scorer result for dataset %s, item %s, scorer %s",
            dataset_slug,
            item_id,
            scorer_id,
        )

        response = self.session.get(
            f"{self.base_url}/api/v1/scorers/results/{dataset_slug}/{item_id}/{scorer_id}",
            timeout=self.timeout,
        )

        return self._handle_response(response)

    def update_scorer_result(
        self,
        dataset_slug: str,
        item_id: str,
        scorer_id: str,
        result_data: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Update a scorer result.

        Args:
            dataset_slug: Dataset slug
            item_id: Item ID
            scorer_id: Scorer ID
            result_data: Updated result data (score, metadata, details)

        Returns:
            API response containing updated result data

        Raises:
            NoveumAPIError: If the API request fails
        """
        # Validate request data
        request_data = self._parse_model(ScorerResultUpdateRequest, result_data)

        logger.info(
            "Updating scorer result for dataset %s, item %s, scorer %s",
            dataset_slug,
            item_id,
            scorer_id,
        )

        response = self.session.put(
            f"{self.base_url}/api/v1/scorers/results/{dataset_slug}/{item_id}/{scorer_id}",
            json=request_data.model_dump(exclude_none=True),
            timeout=self.timeout,
        )

        return self._handle_response(response)

    def delete_scorer_result(
        self, dataset_slug: str, item_id: str, scorer_id: str
    ) -> dict[str, Any]:
        """
        Delete a scorer result.

        Args:
            dataset_slug: Dataset slug
            item_id: Item ID
            scorer_id: Scorer ID

        Returns:
            API response confirming deletion

        Raises:
            NoveumAPIError: If the API request fails
        """
        logger.info(
            "Deleting scorer result for dataset %s, item %s, scorer %s",
            dataset_slug,
            item_id,
            scorer_id,
        )

        response = self.session.delete(
            f"{self.base_url}/api/v1/scorers/results/{dataset_slug}/{item_id}/{scorer_id}",
            timeout=self.timeout,
        )

        return self._handle_response(response)
