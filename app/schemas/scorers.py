"""
Request and response schemas for scorer operations.

This module defines Pydantic models for scorer operations including single and batch scoring.
"""

from typing import Any, Optional

from pydantic import BaseModel, Field

from novaeval.config.schema import ScorerConfig


class ScoringRequest(BaseModel):
    """Request schema for single scoring operation."""

    prediction: str = Field(..., description="Model's prediction/output")
    ground_truth: str = Field(..., description="Expected/correct output")
    context: Optional[dict[str, Any]] = Field(
        default=None, description="Additional context information"
    )


class BatchScoringRequest(BaseModel):
    """Request schema for batch scoring operations."""

    predictions: list[str] = Field(
        ..., description="List of model predictions", min_items=1, max_items=100
    )
    ground_truths: list[str] = Field(
        ..., description="List of expected outputs", min_items=1, max_items=100
    )
    contexts: Optional[list[dict[str, Any]]] = Field(
        default=None, description="List of context information for each prediction"
    )


class ScoringResponse(BaseModel):
    """Response schema for single scoring operation."""

    score: float = Field(..., description="Numerical score value")
    passed: bool = Field(..., description="Whether the score passes the threshold")
    reasoning: str = Field(..., description="Explanation of the scoring decision")
    metadata: dict[str, Any] = Field(..., description="Additional scoring metadata")
    scorer_info: dict[str, Any] = Field(
        ..., description="Information about the scorer used"
    )
    processing_time: float = Field(..., description="Processing time in seconds")


class BatchScoringResponse(BaseModel):
    """Response schema for batch scoring operations."""

    scores: list[float] = Field(..., description="List of numerical score values")
    passed: list[bool] = Field(
        ..., description="List of pass/fail status for each score"
    )
    reasonings: list[str] = Field(
        ..., description="List of explanations for each score"
    )
    metadata: list[dict[str, Any]] = Field(
        ..., description="List of metadata for each score"
    )
    scorer_info: dict[str, Any] = Field(
        ..., description="Information about the scorer used"
    )
    processing_time: float = Field(..., description="Total processing time in seconds")
    count: int = Field(..., description="Number of scores generated")
    statistics: dict[str, Any] = Field(
        ..., description="Statistical summary of the scores"
    )


class ScorerInstantiateRequest(BaseModel):
    """Request schema for instantiating a scorer with custom configuration."""

    scorer_config: ScorerConfig = Field(..., description="Scorer configuration")
    inference_settings: Optional[dict[str, Any]] = Field(
        default=None, description="Model configuration for scorers that require a model"
    )


class ScorerInfo(BaseModel):
    """Scorer information schema."""

    name: str = Field(..., description="Scorer name")
    description: str = Field(..., description="Scorer description")
    type: str = Field(..., description="Scorer type/class name")
    total_scores: int = Field(..., description="Total number of scores processed")
    average_score: float = Field(..., description="Average score")
    requires_model: bool = Field(..., description="Whether scorer requires a model")
    config: dict[str, Any] = Field(..., description="Scorer configuration")


class ScorerStatsResponse(BaseModel):
    """Response schema for scorer statistics."""

    scorer_info: ScorerInfo = Field(..., description="Basic scorer information")
    statistics: dict[str, Any] = Field(..., description="Detailed scoring statistics")


class ErrorResponse(BaseModel):
    """Error response schema."""

    error: str = Field(..., description="Error message")
    error_type: str = Field(..., description="Type of error")
    details: Optional[dict[str, Any]] = Field(
        default=None, description="Additional error details"
    )
