"""
Request and response schemas for dataset operations.

This module defines Pydantic models for dataset loading, querying, and sampling.
"""

from typing import Any, Optional

from pydantic import BaseModel, Field

from novaeval.config.schema import DatasetConfig


class DatasetLoadRequest(BaseModel):
    """Request schema for loading a dataset."""

    config: DatasetConfig = Field(..., description="Dataset configuration")


class DatasetQueryRequest(BaseModel):
    """Request schema for querying dataset with pagination."""

    limit: Optional[int] = Field(
        default=50, description="Maximum number of samples to return", ge=1, le=1000
    )
    offset: Optional[int] = Field(
        default=0, description="Number of samples to skip", ge=0
    )
    shuffle: Optional[bool] = Field(
        default=False, description="Whether to shuffle the results"
    )
    seed: Optional[int] = Field(default=None, description="Random seed for shuffling")
    filter_params: Optional[dict[str, Any]] = Field(
        default_factory=dict, description="Additional filtering parameters"
    )


class DatasetSampleRequest(BaseModel):
    """Request schema for sampling from a dataset."""

    count: int = Field(..., description="Number of samples to return", ge=1, le=100)
    method: str = Field(
        default="random",
        description="Sampling method: 'random', 'first', 'last'",
        pattern="^(random|first|last)$",
    )
    seed: Optional[int] = Field(
        default=None, description="Random seed for reproducible sampling"
    )
    filter_params: Optional[dict[str, Any]] = Field(
        default_factory=dict, description="Additional filtering parameters"
    )


class DatasetInfo(BaseModel):
    """Dataset information schema."""

    name: str = Field(..., description="Dataset name")
    dataset_type: str = Field(..., description="Dataset type")
    split: str = Field(..., description="Dataset split")
    num_samples: int = Field(..., description="Total number of samples")
    seed: int = Field(..., description="Random seed used")


class DatasetSample(BaseModel):
    """Single dataset sample schema."""

    index: int = Field(..., description="Sample index")
    data: dict[str, Any] = Field(..., description="Sample data")


class DatasetLoadResponse(BaseModel):
    """Response schema for dataset loading."""

    dataset_info: DatasetInfo = Field(..., description="Dataset information")
    loaded: bool = Field(..., description="Whether dataset was successfully loaded")
    message: str = Field(..., description="Status message")


class DatasetQueryResponse(BaseModel):
    """Response schema for dataset querying."""

    samples: list[DatasetSample] = Field(..., description="Dataset samples")
    dataset_info: DatasetInfo = Field(..., description="Dataset information")
    pagination: dict[str, Any] = Field(..., description="Pagination information")
    count: int = Field(..., description="Number of samples returned")


class DatasetSampleResponse(BaseModel):
    """Response schema for dataset sampling."""

    samples: list[DatasetSample] = Field(..., description="Sampled dataset samples")
    dataset_info: DatasetInfo = Field(..., description="Dataset information")
    sampling_info: dict[str, Any] = Field(
        ..., description="Sampling method and parameters"
    )
    count: int = Field(..., description="Number of samples returned")


class DatasetInstantiateRequest(BaseModel):
    """Request schema for instantiating a dataset with custom configuration."""

    dataset_type: str = Field(..., description="Type of dataset to instantiate")
    config: DatasetConfig = Field(..., description="Dataset configuration")


class ErrorResponse(BaseModel):
    """Error response schema."""

    error: str = Field(..., description="Error message")
    error_type: str = Field(..., description="Type of error")
    details: Optional[dict[str, Any]] = Field(
        default=None, description="Additional error details"
    )
