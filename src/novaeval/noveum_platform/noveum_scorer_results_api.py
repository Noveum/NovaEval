"""
Noveum Platform Scorer Results API.

This module provides methods for interacting with scorer results in the Noveum Platform API.
It handles scorer result creation, querying, updating, and deletion operations.
"""

from typing import Any, Optional

import requests

from novaeval.utils.logging import get_logger

# NoveumAPIError is used in docstrings for type hints
from .models import (
    ScorerResultCreateRequest,
    ScorerResultsBatchRequest,
    ScorerResultsQueryParams,
    ScorerResultUpdateRequest,
)
from .utils import handle_response, parse_model

logger = get_logger(__name__)


class ScorerResultsAPI:
    """
    API class for scorer results operations.

    Provides methods for creating, querying, updating, and deleting scorer results
    from the Noveum Platform API.
    """

    def __init__(self, session: requests.Session, base_url: str, timeout: float = 30.0):
        """
        Initialize the ScorerResultsAPI.

        Args:
            session: Authenticated requests session
            base_url: Base URL for the Noveum API
            timeout: Request timeout in seconds
        """
        self.session = session
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def list_scorer_results(
        self,
        datasetSlug: Optional[str] = None,
        itemId: Optional[str] = None,
        scorerId: Optional[str] = None,
        limit: Optional[int] = 100,
        offset: Optional[int] = 0,
    ) -> dict[str, Any]:
        """
        List scorer results with optional filters and pagination.

        Args:
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
            "datasetSlug": datasetSlug,
            "itemId": itemId,
            "scorerId": scorerId,
            "limit": limit,
            "offset": offset,
        }

        # Validate parameters using Pydantic model
        query_params = parse_model(ScorerResultsQueryParams, kwargs)
        params = query_params.to_query_params()

        logger.info("Listing scorer results with params: %s", params)

        response = self.session.get(
            f"{self.base_url}/api/v1/scorers/results",
            params=params,
            timeout=self.timeout,
        )

        return handle_response(response)

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
        request_data = parse_model(ScorerResultCreateRequest, result_data)

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

        return handle_response(response)

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
        scorer_results = [
            parse_model(ScorerResultCreateRequest, result) for result in results
        ]

        # Validate request data
        request_data = parse_model(
            ScorerResultsBatchRequest, {"results": scorer_results}
        )

        logger.info("Creating %d scorer results in batch", len(results))

        response = self.session.post(
            f"{self.base_url}/api/v1/scorers/results/batch",
            json=request_data.model_dump(exclude_none=True),
            timeout=self.timeout,
        )

        return handle_response(response)

    def get_scorer_result(
        self,
        dataset_slug: str,
        item_id: str,
        scorer_id: str,
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

        return handle_response(response)

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
        request_data = parse_model(ScorerResultUpdateRequest, result_data)

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

        return handle_response(response)

    def delete_scorer_result(
        self,
        dataset_slug: str,
        item_id: str,
        scorer_id: str,
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

        return handle_response(response)
