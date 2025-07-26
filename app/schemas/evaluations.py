"""
Request and response schemas for evaluation operations.

This module defines Pydantic models for evaluation job submission, status tracking,
and result retrieval.
"""

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field

from app.core.task_manager import TaskStatus


class EvaluationSubmissionResponse(BaseModel):
    """Response schema for evaluation submission."""

    task_id: str = Field(..., description="Unique task identifier")
    name: str = Field(..., description="Evaluation job name")
    status: TaskStatus = Field(..., description="Initial task status")
    created_at: datetime = Field(..., description="Task creation timestamp")
    estimated_duration: Optional[str] = Field(
        default=None, description="Estimated completion time"
    )


class EvaluationStatusResponse(BaseModel):
    """Response schema for evaluation status queries."""

    task_id: str = Field(..., description="Task identifier")
    name: str = Field(..., description="Evaluation job name")
    status: TaskStatus = Field(..., description="Current task status")
    created_at: datetime = Field(..., description="Task creation timestamp")
    started_at: Optional[datetime] = Field(
        default=None, description="Task start timestamp"
    )
    completed_at: Optional[datetime] = Field(
        default=None, description="Task completion timestamp"
    )
    progress: float = Field(..., description="Task progress (0.0 to 1.0)")
    error_message: Optional[str] = Field(
        default=None, description="Error message if failed"
    )
    result_size: Optional[int] = Field(
        default=None, description="Size of result data in bytes"
    )
    duration: Optional[float] = Field(
        default=None, description="Current/final duration in seconds"
    )


class EvaluationResultResponse(BaseModel):
    """Response schema for evaluation results."""

    task_id: str = Field(..., description="Task identifier")
    status: TaskStatus = Field(..., description="Final task status")
    result: Optional[dict[str, Any]] = Field(
        default=None, description="Evaluation result data"
    )
    error_message: Optional[str] = Field(
        default=None, description="Error message if failed"
    )
    duration: float = Field(..., description="Task execution duration in seconds")
    retrieved_at: datetime = Field(..., description="Result retrieval timestamp")


class EvaluationListResponse(BaseModel):
    """Response schema for listing evaluations."""

    tasks: list[EvaluationStatusResponse] = Field(
        ..., description="List of evaluation tasks"
    )
    total: int = Field(..., description="Total number of tasks")
    status_counts: dict[str, int] = Field(..., description="Count of tasks by status")


class TaskManagerStatsResponse(BaseModel):
    """Response schema for task manager statistics."""

    total_tasks: int = Field(..., description="Total number of tasks")
    total_results: int = Field(..., description="Total cached results")
    running_tasks: int = Field(..., description="Currently running tasks")
    status_counts: dict[str, int] = Field(..., description="Tasks by status")
    max_concurrent_tasks: int = Field(
        ..., description="Maximum concurrent tasks allowed"
    )
    result_ttl_seconds: int = Field(..., description="Result cache TTL in seconds")
    last_cleanup: str = Field(..., description="Last cleanup timestamp")


class ConfigValidationResponse(BaseModel):
    """Response schema for configuration validation."""

    valid: bool = Field(..., description="Whether configuration is valid")
    error_message: Optional[str] = Field(
        default=None, description="Validation error message"
    )
    config_summary: Optional[dict[str, Any]] = Field(
        default=None, description="Summary of validated configuration"
    )


class ErrorResponse(BaseModel):
    """Error response schema."""

    error: str = Field(..., description="Error message")
    error_type: str = Field(..., description="Type of error")
    details: Optional[dict[str, Any]] = Field(
        default=None, description="Additional error details"
    )
