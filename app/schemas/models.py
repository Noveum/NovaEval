"""
Request and response schemas for model operations.

This module defines Pydantic models for model prediction requests and responses.
"""

from typing import Any, Optional, Union

from pydantic import BaseModel, Field

from novaeval.config.schema import ModelConfig


class PredictRequest(BaseModel):
    """Request schema for single prediction."""

    prompt: str = Field(..., description="Input prompt for the model")
    max_tokens: Optional[int] = Field(
        default=None,
        description="Maximum number of tokens to generate (1-1 048 576; adjust per model)",
        ge=1,
        le=1_048_576,
    )
    temperature: Optional[float] = Field(
        default=None,
        description="Sampling temperature (0.0 to 2.0; note Anthropic Claude caps at 1.0)",
        ge=0.0,
        le=2.0,
    )
    stop: Optional[Union[str, list[str]]] = Field(
        default=None, description="Stop sequences for generation"
    )
    additional_params: dict[str, Any] = Field(
        default_factory=dict, description="Additional model-specific parameters"
    )


class BatchPredictRequest(BaseModel):
    """Request schema for batch predictions."""

    prompts: list[str] = Field(
        ...,
        description="List of input prompts for the model",
        min_items=1,
        max_items=100,
    )
    max_tokens: Optional[int] = Field(
        default=None,
        description="Maximum number of tokens to generate (1-1 048 576; adjust per model)",
        ge=1,
        le=1_048_576,
    )
    temperature: Optional[float] = Field(
        default=None,
        description="Sampling temperature (0.0 to 2.0; note Anthropic Claude caps at 1.0)",
        ge=0.0,
        le=2.0,
    )
    stop: Optional[Union[str, list[str]]] = Field(
        default=None, description="Stop sequences for generation"
    )
    additional_params: dict[str, Any] = Field(
        default_factory=dict, description="Additional model-specific parameters"
    )


class PredictResponse(BaseModel):
    """Response schema for single prediction."""

    prediction: str = Field(..., description="Generated text response")
    inference_details: dict[str, Any] = Field(
        ..., description="Model information and statistics"
    )
    processing_time: float = Field(..., description="Processing time in seconds")


class BatchPredictResponse(BaseModel):
    """Response schema for batch predictions."""

    predictions: list[str] = Field(..., description="List of generated text responses")
    inference_details: dict[str, Any] = Field(
        ..., description="Model information and statistics"
    )
    processing_time: float = Field(..., description="Total processing time in seconds")
    count: int = Field(..., description="Number of predictions generated")


class ModelInstantiateRequest(BaseModel):
    """Request schema for instantiating a model with custom configuration."""

    config: ModelConfig = Field(..., description="Model configuration")


class ModelInfo(BaseModel):
    """Model information schema."""

    name: str = Field(..., description="Model name")
    identifier: str = Field(..., description="Specific model identifier")
    provider: str = Field(..., description="Model provider")
    total_requests: int = Field(..., description="Total number of requests made")
    total_tokens: int = Field(..., description="Total tokens processed")
    total_cost: float = Field(..., description="Total cost incurred")
    errors: list[str] = Field(..., description="List of errors encountered")


class ErrorResponse(BaseModel):
    """Error response schema."""

    error: str = Field(..., description="Error message")
    error_type: str = Field(..., description="Type of error")
    details: Optional[dict[str, Any]] = Field(
        default=None, description="Additional error details"
    )
