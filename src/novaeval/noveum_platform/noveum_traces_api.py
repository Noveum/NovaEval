"""
Noveum Platform Traces API.

This module provides methods for interacting with traces in the Noveum Platform API.
It handles trace ingestion, querying, and retrieval operations.
"""

from typing import Any, Literal, Optional, Union

import requests

from novaeval.utils.logging import get_logger

# NoveumAPIError is used in docstrings for type hints
from .models import TracesQueryParams
from .utils import handle_response, parse_model

logger = get_logger(__name__)


class TracesAPI:
    """
    API class for trace-related operations.

    Provides methods for ingesting, querying, and retrieving traces
    from the Noveum Platform API.
    """

    def __init__(self, session: requests.Session, base_url: str, timeout: float = 30.0):
        """
        Initialize the TracesAPI.

        Args:
            session: Authenticated requests session
            base_url: Base URL for the Noveum API
            timeout: Request timeout in seconds
        """
        self.session = session
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def ingest_traces(
        self, traces: Union[list[dict[str, Any]], dict[str, Any]]
    ) -> dict[str, Any]:
        """
        Ingest multiple traces in a single batch request.

        Args:
            traces: List of trace dictionaries to ingest, or dict with 'traces' key
                   containing the list of traces (already wrapped format)

        Returns:
            API response containing ingestion results

        Raises:
            NoveumAPIError: If the API request fails
        """
        logger.debug("Ingesting %d traces", len(traces))

        # Check if traces are already wrapped in the expected format
        if isinstance(traces, dict) and "traces" in traces:
            # Already wrapped, use as-is
            batch_data = traces
        else:
            # Not wrapped, wrap in the format expected by the API
            batch_data = {"traces": traces}

        response = self.session.post(
            f"{self.base_url}/api/v1/traces", json=batch_data, timeout=self.timeout
        )

        return handle_response(response)

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
        logger.debug("Ingesting single trace")

        response = self.session.post(
            f"{self.base_url}/api/v1/traces/single", json=trace, timeout=self.timeout
        )

        return handle_response(response)

    def query_traces(
        self,
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
        query_params = parse_model(TracesQueryParams, kwargs)
        params = query_params.to_query_params()

        logger.debug("Querying traces with params: %s", params)

        response = self.session.get(
            f"{self.base_url}/api/v1/traces", params=params, timeout=self.timeout
        )

        return handle_response(response)

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
        logger.debug("Getting trace: %s", trace_id)

        response = self.session.get(
            f"{self.base_url}/api/v1/traces/{trace_id}", timeout=self.timeout
        )

        return handle_response(response)

    def get_connection_status(self) -> dict[str, Any]:
        """
        Get the connection status.

        Returns:
            Connection status data

        Raises:
            NoveumAPIError: If the API request fails
        """
        logger.debug("Getting connection status")

        response = self.session.get(
            f"{self.base_url}/api/v1/traces/connection-status", timeout=self.timeout
        )

        return handle_response(response)

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
        logger.debug("Getting spans for trace: %s", trace_id)

        response = self.session.get(
            f"{self.base_url}/api/v1/traces/{trace_id}/spans", timeout=self.timeout
        )

        return handle_response(response)
