"""
Evaluation API endpoints for NovaEval.

This module provides REST endpoints for evaluation orchestration including
job submission, status tracking, and result retrieval.
"""

import json
from datetime import datetime
from typing import Any, Optional

import yaml
from fastapi import APIRouter, BackgroundTasks, File, HTTPException, Query, UploadFile
from pydantic import ValidationError

from app.core.task_manager import TaskStatus, get_task_manager
from app.schemas.evaluations import (
    ConfigValidationResponse,
    EvaluationListResponse,
    EvaluationResultResponse,
    EvaluationStatusResponse,
    EvaluationSubmissionResponse,
    TaskManagerStatsResponse,
)
from novaeval.config.schema import EvaluationJobConfig

router = APIRouter()


def parse_config_file(file_content: bytes, filename: str) -> dict[str, Any]:
    """
    Parse YAML or JSON configuration file.

    Args:
        file_content: Raw file content
        filename: Original filename for format detection

    Returns:
        Parsed configuration dictionary

    Raises:
        ValueError: If file format is unsupported or parsing fails
    """
    content_str = file_content.decode("utf-8")

    # Determine format from filename or try to parse
    filename_lower = filename.lower()

    if filename_lower.endswith((".yml", ".yaml")):
        try:
            return yaml.safe_load(content_str)
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML format: {e!s}")

    elif filename_lower.endswith(".json"):
        try:
            return json.loads(content_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format: {e!s}")

    else:
        # Try to auto-detect format
        try:
            # Try JSON first
            return json.loads(content_str)
        except json.JSONDecodeError:
            try:
                # Try YAML
                return yaml.safe_load(content_str)
            except yaml.YAMLError as e:
                raise ValueError(
                    f"Unsupported file format. Please upload YAML or JSON. YAML error: {e!s}"
                )


def validate_and_create_config(config_dict: dict[str, Any]) -> EvaluationJobConfig:
    """
    Validate and create EvaluationJobConfig from dictionary.

    Args:
        config_dict: Configuration dictionary

    Returns:
        Validated EvaluationJobConfig

    Raises:
        ValueError: If configuration is invalid
    """
    try:
        return EvaluationJobConfig(**config_dict)
    except ValidationError as e:
        raise ValueError(f"Invalid configuration: {e}")
    except Exception as e:
        raise ValueError(f"Unexpected error validating configuration: {e!s}")


@router.post("/submit", response_model=EvaluationSubmissionResponse)
async def submit_evaluation(
    background_tasks: BackgroundTasks,
    config_file: UploadFile = File(..., description="YAML or JSON configuration file"),
):
    """
    Submit an evaluation job for execution.

    Args:
        background_tasks: FastAPI background tasks
        config_file: Uploaded configuration file (YAML or JSON)

    Returns:
        Evaluation submission response with task ID
    """
    try:
        # Validate file size (100MB limit)
        from app.core.config import get_settings

        settings = get_settings()
        max_size = settings.max_file_size_mb * 1024 * 1024

        # Check file size before reading
        if config_file.size and config_file.size > max_size:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum size: {settings.max_file_size_mb}MB",
            )

        # Read file content
        file_content = await config_file.read()

        # Parse configuration
        config_dict = parse_config_file(file_content, config_file.filename or "config")
        config = validate_and_create_config(config_dict)

        # Get task manager
        task_manager = get_task_manager()

        # Trigger auto-cleanup if needed
        task_manager.auto_cleanup_if_needed()

        # Create task
        task_id = task_manager.create_task(config.name, config)

        # Start background execution
        background_tasks.add_task(task_manager.start_task, task_id, config)

        # Get task info for response
        task_info = task_manager.get_task_info(task_id)
        if not task_info:
            raise HTTPException(status_code=500, detail="Failed to create task")

        return EvaluationSubmissionResponse(
            task_id=task_id,
            name=config.name,
            status=task_info.status,
            created_at=task_info.created_at,
            estimated_duration="Variable (depends on dataset size and model complexity)",
        )

    except HTTPException:
        # Re-raise HTTP exceptions as-is (like the 413 file size error)
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=429, detail=str(e))  # Too many concurrent tasks
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {e!s}")


@router.get("/{task_id}/status", response_model=EvaluationStatusResponse)
async def get_evaluation_status(task_id: str):
    """
    Get the status of an evaluation task.

    Args:
        task_id: Task identifier

    Returns:
        Evaluation status response
    """
    try:
        task_manager = get_task_manager()
        task_info = task_manager.get_task_info(task_id)

        if not task_info:
            raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

        # Calculate duration if task is running or completed
        duration = None
        if task_info.started_at:
            end_time = task_info.completed_at or datetime.now()
            duration = (end_time - task_info.started_at).total_seconds()

        return EvaluationStatusResponse(
            task_id=task_info.task_id,
            name=task_info.name,
            status=task_info.status,
            created_at=task_info.created_at,
            started_at=task_info.started_at,
            completed_at=task_info.completed_at,
            progress=task_info.progress,
            error_message=task_info.error_message,
            result_size=task_info.result_size,
            duration=duration,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error retrieving task status: {e!s}"
        )


@router.get("/{task_id}/result", response_model=EvaluationResultResponse)
async def get_evaluation_result(task_id: str):
    """
    Get the result of a completed evaluation task.

    Args:
        task_id: Task identifier

    Returns:
        Evaluation result response
    """
    try:
        task_manager = get_task_manager()
        task_result = task_manager.get_task_result(task_id, mark_retrieved=True)

        if not task_result:
            # Check if task exists but result is not ready
            task_info = task_manager.get_task_info(task_id)
            if not task_info:
                raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
            elif task_info.status in [TaskStatus.PENDING, TaskStatus.RUNNING]:
                raise HTTPException(
                    status_code=202,
                    detail=f"Task {task_id} is still {task_info.status.value}. Result not ready.",
                )
            else:
                raise HTTPException(
                    status_code=404,
                    detail=f"Result for task {task_id} not found or expired",
                )

        return EvaluationResultResponse(
            task_id=task_result.task_id,
            status=task_result.status,
            result=task_result.result,
            error_message=task_result.error_message,
            duration=task_result.duration,
            retrieved_at=task_result.retrieved_at,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error retrieving task result: {e!s}"
        )


@router.delete("/{task_id}")
async def cancel_evaluation(task_id: str):
    """
    Cancel a running evaluation task.

    Args:
        task_id: Task identifier

    Returns:
        Cancellation status
    """
    try:
        task_manager = get_task_manager()
        success = task_manager.cancel_task(task_id)

        if not success:
            task_info = task_manager.get_task_info(task_id)
            if not task_info:
                raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
            else:
                raise HTTPException(
                    status_code=400,
                    detail=f"Task {task_id} cannot be cancelled (status: {task_info.status.value})",
                )

        return {"message": f"Task {task_id} cancelled successfully"}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error cancelling task: {e!s}")


@router.get("/", response_model=EvaluationListResponse)
async def list_evaluations(
    status: Optional[TaskStatus] = Query(
        default=None, description="Filter by task status"
    ),
    limit: int = Query(
        default=50, ge=1, le=100, description="Maximum number of tasks to return"
    ),
    offset: int = Query(default=0, ge=0, description="Number of tasks to skip"),
):
    """
    List evaluation tasks with optional filtering.

    Args:
        status: Optional status filter
        limit: Maximum number of tasks to return
        offset: Number of tasks to skip

    Returns:
        List of evaluation tasks
    """
    try:
        task_manager = get_task_manager()

        # Get all tasks or filtered by status
        all_tasks = task_manager.list_tasks(status_filter=status)

        # Sort by creation time (newest first)
        sorted_tasks = sorted(
            all_tasks.values(), key=lambda x: x.created_at, reverse=True
        )

        # Apply pagination
        paginated_tasks = sorted_tasks[offset : offset + limit]

        # Convert to response format
        task_responses = []
        for task_info in paginated_tasks:
            # Calculate duration if applicable
            duration = None
            if task_info.started_at:
                end_time = task_info.completed_at or datetime.now()
                duration = (end_time - task_info.started_at).total_seconds()

            task_responses.append(
                EvaluationStatusResponse(
                    task_id=task_info.task_id,
                    name=task_info.name,
                    status=task_info.status,
                    created_at=task_info.created_at,
                    started_at=task_info.started_at,
                    completed_at=task_info.completed_at,
                    progress=task_info.progress,
                    error_message=task_info.error_message,
                    result_size=task_info.result_size,
                    duration=duration,
                )
            )

        # Get status counts
        status_counts = {}
        for status_val in TaskStatus:
            status_counts[status_val.value] = sum(
                1 for task in all_tasks.values() if task.status == status_val
            )

        return EvaluationListResponse(
            tasks=task_responses, total=len(all_tasks), status_counts=status_counts
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing tasks: {e!s}")


@router.post("/validate-config", response_model=ConfigValidationResponse)
async def validate_config(
    config_file: UploadFile = File(..., description="YAML or JSON configuration file")
):
    """
    Validate an evaluation configuration file without executing it.

    Args:
        config_file: Configuration file to validate

    Returns:
        Validation result
    """
    try:
        # Validate file size (100MB limit)
        from app.core.config import get_settings

        settings = get_settings()
        max_size = settings.max_file_size_mb * 1024 * 1024

        # Check file size before reading
        if config_file.size and config_file.size > max_size:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum size: {settings.max_file_size_mb}MB",
            )

        # Read and parse file
        file_content = await config_file.read()
        config_dict = parse_config_file(file_content, config_file.filename or "config")

        # Validate configuration
        config = validate_and_create_config(config_dict)

        # Create summary
        summary = {
            "name": config.name,
            "description": config.description,
            "models": len(config.models),
            "datasets": len(config.datasets),
            "scorers": len(config.scorers),
            "parallel_models": config.parallel_models,
            "max_workers": config.max_workers,
            "timeout": config.timeout,
        }

        return ConfigValidationResponse(valid=True, config_summary=summary)

    except ValueError as e:
        return ConfigValidationResponse(valid=False, error_message=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error validating config: {e!s}")


@router.get("/stats", response_model=TaskManagerStatsResponse)
async def get_task_manager_stats():
    """
    Get task manager statistics.

    Returns:
        Task manager statistics
    """
    try:
        task_manager = get_task_manager()
        stats = task_manager.get_stats()

        return TaskManagerStatsResponse(**stats)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving stats: {e!s}")


@router.post("/cleanup")
async def cleanup_expired_results():
    """
    Manually trigger cleanup of expired results.

    Returns:
        Cleanup status
    """
    try:
        task_manager = get_task_manager()
        cleaned_count = task_manager.cleanup_expired_results()

        return {
            "message": f"Cleanup completed. Removed {cleaned_count} expired items.",
            "cleaned_count": cleaned_count,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during cleanup: {e!s}")
