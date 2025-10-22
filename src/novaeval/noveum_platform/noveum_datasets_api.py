"""
Noveum Platform Datasets API.

This module provides methods for interacting with datasets in the Noveum Platform API.
It handles dataset creation, management, versioning, and item operations.
"""

from typing import Any, Literal, Optional

import requests

from novaeval.utils.logging import get_logger

from .exceptions import ValidationError
from .models import (
    DatasetCreateRequest,
    DatasetItem,
    DatasetItemsCreateRequest,
    DatasetItemsQueryParams,
    DatasetsQueryParams,
    DatasetUpdateRequest,
    DatasetVersionCreateRequest,
)
from .utils import handle_response, parse_model

logger = get_logger(__name__)


class DatasetsAPI:
    """
    API class for dataset-related operations.

    Provides methods for creating, managing, and querying datasets
    from the Noveum Platform API.
    """

    def __init__(self, session: requests.Session, base_url: str, timeout: float = 30.0):
        """
        Initialize the DatasetsAPI.

        Args:
            session: Authenticated requests session
            base_url: Base URL for the Noveum API
            timeout: Request timeout in seconds
        """
        self.session = session
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

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
        request_data = parse_model(DatasetCreateRequest, kwargs)

        logger.info("Creating dataset: %s", name)

        response = self.session.post(
            f"{self.base_url}/api/v1/datasets",
            json=request_data.model_dump(exclude_none=True),
            timeout=self.timeout,
        )

        return handle_response(response)

    def list_datasets(
        self,
        limit: Optional[int] = 20,
        offset: Optional[int] = 0,
        visibility: Optional[Literal["public", "org", "private"]] = None,
        includeVersions: Optional[bool] = False,
    ) -> dict[str, Any]:
        """
        List datasets with optional filters and pagination.

        Args:
            limit: Number of datasets to return (1-1000, default 20)
            offset: Number of datasets to skip (default 0)
            visibility: Filter by visibility (public, org, private)
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
            "includeVersions": includeVersions,
        }

        # Validate parameters using Pydantic model
        query_params = parse_model(DatasetsQueryParams, kwargs)
        params = query_params.to_query_params()

        logger.info("Listing datasets with params: %s", params)

        response = self.session.get(
            f"{self.base_url}/api/v1/datasets", params=params, timeout=self.timeout
        )

        return handle_response(response)

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

        return handle_response(response)

    def update_dataset(
        self,
        slug: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        visibility: Optional[Literal["public", "org", "private"]] = None,
        dataset_type: Optional[
            Literal["agent", "conversational", "g-eval", "custom"]
        ] = None,
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
        request_data = parse_model(DatasetUpdateRequest, kwargs)

        logger.info("Updating dataset: %s", slug)

        response = self.session.put(
            f"{self.base_url}/api/v1/datasets/{slug}",
            json=request_data.model_dump(exclude_none=True),
            timeout=self.timeout,
        )

        return handle_response(response)

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

        return handle_response(response)

    def list_dataset_versions(
        self, dataset_slug: str, limit: Optional[int] = 50, offset: Optional[int] = 0
    ) -> dict[str, Any]:
        """
        List versions for a dataset.

        Args:
            dataset_slug: Dataset slug
            limit: Number of versions to return (1-100, default 50)
            offset: Number of versions to skip (default 0)

        Returns:
            API response containing dataset versions

        Raises:
            NoveumAPIError: If the API request fails
        """
        params = {"limit": limit, "offset": offset}

        logger.info(
            "Listing versions for dataset: %s with params: %s", dataset_slug, params
        )

        response = self.session.get(
            f"{self.base_url}/api/v1/datasets/{dataset_slug}/versions",
            params=params,
            timeout=self.timeout,
        )

        return handle_response(response)

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
        request_data = parse_model(DatasetVersionCreateRequest, version_data)

        logger.info("Creating version for dataset: %s", dataset_slug)

        response = self.session.post(
            f"{self.base_url}/api/v1/datasets/{dataset_slug}/versions",
            json=request_data.model_dump(exclude_none=True),
            timeout=self.timeout,
        )

        return handle_response(response)

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

        return handle_response(response)

    def publish_dataset_version(self, dataset_slug: str) -> dict[str, Any]:
        """
        Publish the next dataset version and automatically increment version number.

        Args:
            dataset_slug: Dataset slug

        Returns:
            API response confirming publication

        Raises:
            NoveumAPIError: If the API request fails
        """
        logger.info("Publishing next version for dataset: %s", dataset_slug)

        response = self.session.post(
            f"{self.base_url}/api/v1/datasets/{dataset_slug}/versions/publish",
            timeout=self.timeout,
        )

        return handle_response(response)

    def get_dataset_versions_diff(self, dataset_slug: str) -> dict[str, Any]:
        """
        Get changes between current_release and next_release.

        Args:
            dataset_slug: Dataset slug

        Returns:
            API response containing diff between current and next release

        Raises:
            NoveumAPIError: If the API request fails
        """
        logger.info("Getting version diff for dataset: %s", dataset_slug)

        response = self.session.get(
            f"{self.base_url}/api/v1/datasets/{dataset_slug}/versions/diff",
            timeout=self.timeout,
        )

        return handle_response(response)

    def list_dataset_items(
        self,
        dataset_slug: str,
        version: Optional[str] = None,
        limit: Optional[int] = 50,
        offset: Optional[int] = 0,
        item_type: Optional[str] = None,
        search: Optional[str] = None,
        sort_by: Optional[str] = None,
        sort_order: Optional[Literal["asc", "desc"]] = "asc",
    ) -> dict[str, Any]:
        """
        List items in a dataset with optional filters and pagination.

        Args:
            dataset_slug: Dataset slug
            version: Filter by version (defaults to current_release if not specified)
            limit: Number of items to return (1-1000, default 50)
            offset: Number of items to skip (default 0)
            item_type: Filter by item type (max 100 chars)
            search: Search term for filtering items (max 256 chars)
            sort_by: Field to sort by (scorer.field_name or field_name)
            sort_order: Sort order (asc or desc, default asc)

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
            "item_type": item_type,
            "search": search,
            "sort_by": sort_by,
            "sort_order": sort_order,
        }

        # Validate parameters using Pydantic model
        query_params = parse_model(DatasetItemsQueryParams, kwargs)
        params = query_params.to_query_params()

        logger.info(
            "Listing items for dataset %s with params: %s", dataset_slug, params
        )

        response = self.session.get(
            f"{self.base_url}/api/v1/datasets/{dataset_slug}/items",
            params=params,
            timeout=self.timeout,
        )

        return handle_response(response)

    def add_dataset_items(
        self, dataset_slug: str, items: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """
        Add items to a dataset.

        Args:
            dataset_slug: Dataset slug
            items: List of items to add (each must have item_key, item_type, content)
                   Items will be added to the next_release version

        Returns:
            API response containing added items data

        Raises:
            NoveumAPIError: If the API request fails
        """
        # Convert dict items to DatasetItem objects
        dataset_items = [parse_model(DatasetItem, item) for item in items]

        # Validate request data
        request_data = parse_model(DatasetItemsCreateRequest, {"items": dataset_items})

        logger.info("Adding %d items to dataset %s", len(items), dataset_slug)

        response = self.session.post(
            f"{self.base_url}/api/v1/datasets/{dataset_slug}/items",
            json=request_data.model_dump(exclude_none=True),
            timeout=self.timeout,
        )

        return handle_response(response)

    def delete_all_dataset_items(
        self,
        dataset_slug: str,
        item_ids: list[str],
    ) -> dict[str, Any]:
        """
        Delete items from a dataset.

        Args:
            dataset_slug: Dataset slug
            item_ids: List of item IDs to delete. Required parameter.

        Returns:
            API response confirming deletion

        Raises:
            ValidationError: If no item_ids are provided
            NoveumAPIError: If the API request fails
        """
        if not item_ids:
            raise ValidationError(
                message="item_ids parameter is required and cannot be empty",
                response_body={"error": "item_ids is required for deletion"},
            )

        logger.info("Deleting %d items from dataset %s", len(item_ids), dataset_slug)

        # Send DELETE request with item IDs in request body
        response = self.session.delete(
            f"{self.base_url}/api/v1/datasets/{dataset_slug}/items",
            json={"itemIds": item_ids},
            timeout=self.timeout,
        )

        return handle_response(response)

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

        return handle_response(response)

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

        return handle_response(response)
