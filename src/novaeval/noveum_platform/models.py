"""
Pydantic models for Noveum Platform API.

This module defines data models for request validation and type safety.
"""

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class TracesQueryParams(BaseModel):
    """
    Pydantic model for validating query parameters for the traces endpoint.

    This model ensures all query parameters are properly typed and validated
    before being sent to the API.
    """

    from_: Optional[int] = Field(
        None, alias="from", ge=0, description="Pagination offset (0-based)"
    )
    size: Optional[int] = Field(
        default=20, ge=1, le=100, description="Number of traces to return (1-100)"
    )
    start_time: Optional[str] = Field(
        None, description="Start time filter (ISO datetime)"
    )
    end_time: Optional[str] = Field(None, description="End time filter (ISO datetime)")
    project: Optional[str] = Field(None, description="Project name filter")
    environment: Optional[str] = Field(None, description="Environment filter")
    status: Optional[str] = Field(None, description="Status filter")
    user_id: Optional[str] = Field(None, description="User ID filter")
    session_id: Optional[str] = Field(None, description="Session ID filter")
    tags: Optional[list[str]] = Field(None, description="Tags filter")
    sort: Optional[
        Literal[
            "start_time:asc",
            "start_time:desc",
            "end_time:asc",
            "end_time:desc",
            "duration_ms:asc",
            "duration_ms:desc",
        ]
    ] = Field("start_time:desc", description="Sort order for results")
    search_term: Optional[str] = Field(None, description="Search term for text search")
    include_spans: Optional[bool] = Field(
        False, description="Whether to include spans in response"
    )

    def to_query_params(self) -> dict:
        """
        Convert the model to a dictionary suitable for URL query parameters.

        Returns:
            Dictionary with non-None values, ready for requests.get(params=...)
        """
        params = {}

        # Handle the 'from' alias properly
        if self.from_ is not None:
            params["from"] = self.from_

        # Add all other fields that are not None
        params.update(
            {
                field_name: field_value
                for field_name, field_value in self.model_dump(
                    exclude_none=True, exclude={"from_"}
                ).items()
                if field_value is not None
            }
        )

        return params

    class Config:
        """Pydantic configuration."""

        populate_by_name = True
        validate_assignment = True


# Dataset Models


class DatasetCreateRequest(BaseModel):
    """Model for creating a new dataset."""

    name: str = Field(..., min_length=1, description="Dataset name")
    slug: Optional[str] = Field(
        None, description="Dataset slug (auto-generated if not provided)"
    )
    description: Optional[str] = Field(None, description="Dataset description")
    visibility: Literal["public", "org", "private"] = Field(
        "org", description="Dataset visibility"
    )
    dataset_type: Literal["agent", "conversational", "g-eval", "custom"] = Field(
        "custom", description="Dataset type"
    )
    environment: Optional[str] = Field(None, description="Environment")
    schema_version: Optional[str] = Field(None, description="Schema version")
    tags: Optional[list[str]] = Field(None, description="Dataset tags")
    custom_attributes: Optional[dict[str, Any]] = Field(
        None, description="Custom attributes"
    )


class DatasetUpdateRequest(BaseModel):
    """Model for updating an existing dataset."""

    name: Optional[str] = Field(None, min_length=1, description="Dataset name")
    description: Optional[str] = Field(None, description="Dataset description")
    visibility: Optional[Literal["public", "org", "private"]] = Field(
        None, description="Dataset visibility"
    )
    dataset_type: Optional[Literal["agent", "conversational", "g-eval", "custom"]] = (
        Field(None, description="Dataset type")
    )
    environment: Optional[str] = Field(None, description="Environment")
    schema_version: Optional[str] = Field(None, description="Schema version")
    tags: Optional[list[str]] = Field(None, description="Dataset tags")
    custom_attributes: Optional[dict[str, Any]] = Field(
        None, description="Custom attributes"
    )


class DatasetsQueryParams(BaseModel):
    """Query parameters for listing datasets."""

    limit: Optional[int] = Field(
        20, ge=1, le=1000, description="Number of datasets to return"
    )
    offset: Optional[int] = Field(0, ge=0, description="Number of datasets to skip")
    visibility: Optional[Literal["public", "org", "private"]] = Field(
        None, description="Filter by visibility"
    )
    includeVersions: Optional[bool] = Field(
        False, description="Include dataset versions"
    )

    def to_query_params(self) -> dict:
        """Convert to query parameters dictionary."""
        return dict(self.model_dump(exclude_none=True).items())


class DatasetVersionCreateRequest(BaseModel):
    """Model for creating a dataset version."""

    version: str = Field(..., min_length=1, description="Version identifier")
    description: Optional[str] = Field(None, description="Version description")
    metadata: Optional[dict[str, Any]] = Field(None, description="Version metadata")


class DatasetItem(BaseModel):
    """Model for a dataset item."""

    item_key: str = Field(..., min_length=1, description="Unique item key")
    item_type: str = Field(..., min_length=1, description="Item type")
    content: dict[str, Any] = Field(..., description="Item content")
    metadata: Optional[dict[str, Any]] = Field(None, description="Item metadata")
    trace_id: Optional[str] = Field(None, description="Associated trace ID")
    span_id: Optional[str] = Field(None, description="Associated span ID")


class DatasetItemsCreateRequest(BaseModel):
    """Model for creating dataset items."""

    items: list[DatasetItem] = Field(..., min_length=1, description="Items to add")


class DatasetItemsQueryParams(BaseModel):
    """Query parameters for listing dataset items."""

    version: Optional[str] = Field(
        None, pattern=r"^\s*\d+\.\d+\.\d+\s*$", description="Filter by version"
    )
    limit: Optional[int] = Field(
        50, ge=1, le=1000, description="Number of items to return"
    )
    offset: Optional[int] = Field(0, ge=0, description="Number of items to skip")
    item_type: Optional[str] = Field(
        None, max_length=100, description="Filter by item type"
    )
    search: Optional[str] = Field(
        None, max_length=256, description="Search term for filtering items"
    )
    sort_by: Optional[str] = Field(
        None,
        pattern=r"^(scorer\.[a-zA-Z0-9_-]{1,64}|[a-zA-Z_][a-zA-Z0-9_]{0,63})$",
        description="Field to sort by",
    )
    sort_order: Optional[Literal["asc", "desc"]] = Field(
        "asc", description="Sort order"
    )

    def to_query_params(self) -> dict:
        """Convert to query parameters dictionary."""
        return dict(self.model_dump(exclude_none=True).items())


# Scorer Results Models


class ScorerResultCreateRequest(BaseModel):
    """Model for creating a scorer result."""

    datasetSlug: str = Field(..., description="Dataset slug")
    itemId: str = Field(..., description="Item ID")
    scorerId: str = Field(..., description="Scorer ID")
    score: Optional[float] = Field(None, description="Score value")
    metadata: Optional[dict[str, Any]] = Field(None, description="Result metadata")
    details: Optional[dict[str, Any]] = Field(
        None, description="Detailed scoring information"
    )


class ScorerResultUpdateRequest(BaseModel):
    """Model for updating a scorer result."""

    score: Optional[float] = Field(None, description="Score value")
    metadata: Optional[dict[str, Any]] = Field(None, description="Result metadata")
    details: Optional[dict[str, Any]] = Field(
        None, description="Detailed scoring information"
    )


class ScorerResultsQueryParams(BaseModel):
    """Query parameters for listing scorer results."""

    datasetSlug: Optional[str] = Field(None, description="Filter by dataset slug")
    itemId: Optional[str] = Field(None, description="Filter by item ID")
    scorerId: Optional[str] = Field(None, description="Filter by scorer ID")
    limit: Optional[int] = Field(
        100, ge=1, le=1000, description="Number of results to return"
    )
    offset: Optional[int] = Field(0, ge=0, description="Number of results to skip")

    def to_query_params(self) -> dict:
        """Convert to query parameters dictionary."""
        return dict(self.model_dump(exclude_none=True).items())


class ScorerResultsBatchRequest(BaseModel):
    """Model for batch creating scorer results."""

    results: list[ScorerResultCreateRequest] = Field(
        ..., min_length=1, description="Results to create"
    )
