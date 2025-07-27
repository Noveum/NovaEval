"""
Background Task Management for NovaEval API.

This module provides the TaskManager class for managing background evaluation tasks
with thread safety, TTL caching, status tracking, and automatic cleanup.
"""

import asyncio
import threading
import time
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Optional
from uuid import uuid4

from pydantic import BaseModel, Field

from novaeval.config.job_config import JobRunner
from novaeval.config.schema import EvaluationJobConfig


class TaskStatus(str, Enum):
    """Task execution status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskInfo(BaseModel):
    """Information about a background task."""

    task_id: str = Field(..., description="Unique task identifier")
    name: str = Field(..., description="Task name")
    status: TaskStatus = Field(..., description="Current task status")
    created_at: datetime = Field(..., description="Task creation timestamp")
    started_at: Optional[datetime] = Field(
        default=None, description="Task start timestamp"
    )
    completed_at: Optional[datetime] = Field(
        default=None, description="Task completion timestamp"
    )
    progress: float = Field(default=0.0, description="Task progress (0.0 to 1.0)")
    error_message: Optional[str] = Field(
        default=None, description="Error message if failed"
    )
    result_size: Optional[int] = Field(
        default=None, description="Size of result data in bytes"
    )


class TaskResult(BaseModel):
    """Task execution result."""

    task_id: str = Field(..., description="Task identifier")
    status: TaskStatus = Field(..., description="Final task status")
    result: Optional[dict[str, Any]] = Field(
        default=None, description="Task result data"
    )
    error_message: Optional[str] = Field(
        default=None, description="Error message if failed"
    )
    duration: float = Field(..., description="Task execution duration in seconds")
    retrieved_at: datetime = Field(
        default_factory=datetime.now, description="Result retrieval timestamp"
    )


class TaskManager:
    """
    Thread-safe background task manager for evaluation jobs.

    Manages the lifecycle of background evaluation tasks including:
    - Task creation and execution
    - Status and progress tracking
    - Result caching with TTL
    - Automatic cleanup of expired results
    - Thread-safe access to task registry
    """

    def __init__(
        self,
        result_ttl_seconds: int = 7200,  # 2 hours
        cleanup_interval_seconds: int = 300,  # 5 minutes
        max_concurrent_tasks: int = 5,
    ):
        """
        Initialize the task manager.

        Args:
            result_ttl_seconds: Time to live for cached results
            cleanup_interval_seconds: Interval for automatic cleanup
            max_concurrent_tasks: Maximum number of concurrent tasks
        """
        self.result_ttl_seconds = result_ttl_seconds
        self.cleanup_interval_seconds = cleanup_interval_seconds
        self.max_concurrent_tasks = max_concurrent_tasks

        # Thread-safe storage
        self._lock = threading.RLock()
        self._tasks: dict[str, TaskInfo] = {}
        self._results: dict[str, TaskResult] = {}
        self._running_tasks: dict[str, asyncio.Task] = {}

        # Cleanup tracking
        self._last_cleanup = time.time()

    def create_task(self, name: str, _config: EvaluationJobConfig) -> str:
        """
        Create a new evaluation task.

        Args:
            name: Human-readable task name
            _config: Evaluation job configuration (for future use)

        Returns:
            Task ID string

        Raises:
            RuntimeError: If maximum concurrent tasks exceeded
        """
        with self._lock:
            # Check if we can accept more tasks
            running_count = sum(
                1
                for task in self._tasks.values()
                if task.status in [TaskStatus.PENDING, TaskStatus.RUNNING]
            )

            if running_count >= self.max_concurrent_tasks:
                raise RuntimeError(
                    f"Maximum concurrent tasks ({self.max_concurrent_tasks}) exceeded"
                )

            # Create new task
            task_id = str(uuid4())
            task_info = TaskInfo(
                task_id=task_id,
                name=name,
                status=TaskStatus.PENDING,
                created_at=datetime.now(),
            )

            self._tasks[task_id] = task_info

            return task_id

    async def start_task(self, task_id: str, config: EvaluationJobConfig) -> None:
        """
        Start executing a task in the background.

        Args:
            task_id: Task identifier
            config: Evaluation job configuration
        """
        with self._lock:
            if task_id not in self._tasks:
                raise ValueError(f"Task {task_id} not found")

            task_info = self._tasks[task_id]
            if task_info.status != TaskStatus.PENDING:
                raise ValueError(f"Task {task_id} is not in pending status")

            # Update status
            task_info.status = TaskStatus.RUNNING
            task_info.started_at = datetime.now()

        # Create and start the background task
        background_task = asyncio.create_task(self._run_evaluation(task_id, config))

        with self._lock:
            self._running_tasks[task_id] = background_task

    async def _run_evaluation(self, task_id: str, config: EvaluationJobConfig) -> None:
        """
        Execute the evaluation job.

        Args:
            task_id: Task identifier
            config: Evaluation job configuration
        """
        start_time = time.time()

        try:
            # Create job runner and execute
            runner = JobRunner(config)
            result = await runner.run()

            duration = time.time() - start_time

            # Store result
            task_result = TaskResult(
                task_id=task_id,
                status=TaskStatus.COMPLETED,
                result=result,
                duration=duration,
            )

            with self._lock:
                # Update task info
                if task_id in self._tasks:
                    self._tasks[task_id].status = TaskStatus.COMPLETED
                    self._tasks[task_id].completed_at = datetime.now()
                    self._tasks[task_id].progress = 1.0

                    # Calculate result size
                    import json

                    result_size = len(json.dumps(result, default=str).encode("utf-8"))
                    self._tasks[task_id].result_size = result_size

                # Store result
                self._results[task_id] = task_result

                # Remove from running tasks
                self._running_tasks.pop(task_id, None)

        except Exception as e:
            duration = time.time() - start_time
            error_message = str(e)

            # Store error result
            task_result = TaskResult(
                task_id=task_id,
                status=TaskStatus.FAILED,
                error_message=error_message,
                duration=duration,
            )

            with self._lock:
                # Update task info
                if task_id in self._tasks:
                    self._tasks[task_id].status = TaskStatus.FAILED
                    self._tasks[task_id].completed_at = datetime.now()
                    self._tasks[task_id].error_message = error_message

                # Store result
                self._results[task_id] = task_result

                # Remove from running tasks
                self._running_tasks.pop(task_id, None)

    def get_task_info(self, task_id: str) -> Optional[TaskInfo]:
        """
        Get information about a task.

        Args:
            task_id: Task identifier

        Returns:
            Task information or None if not found
        """
        with self._lock:
            return self._tasks.get(task_id)

    def get_task_result(
        self, task_id: str, mark_retrieved: bool = True
    ) -> Optional[TaskResult]:
        """
        Get the result of a completed task.

        Args:
            task_id: Task identifier
            mark_retrieved: Whether to mark the result as retrieved (for cleanup)

        Returns:
            Task result or None if not found/not ready
        """
        with self._lock:
            result = self._results.get(task_id)

            if result and mark_retrieved:
                # Update retrieval timestamp for cleanup
                result.retrieved_at = datetime.now()

            return result

    def cancel_task(self, task_id: str) -> bool:
        """
        Cancel a running task.

        Args:
            task_id: Task identifier

        Returns:
            True if task was cancelled, False if not found or not cancellable
        """
        with self._lock:
            task_info = self._tasks.get(task_id)
            if not task_info:
                return False

            if task_info.status not in [TaskStatus.PENDING, TaskStatus.RUNNING]:
                return False

            # Cancel the background task if running
            background_task = self._running_tasks.get(task_id)
            if background_task:
                background_task.cancel()
                self._running_tasks.pop(task_id, None)

            # Update status
            task_info.status = TaskStatus.CANCELLED
            task_info.completed_at = datetime.now()

            return True

    def list_tasks(
        self, status_filter: Optional[TaskStatus] = None
    ) -> dict[str, TaskInfo]:
        """
        List all tasks, optionally filtered by status.

        Args:
            status_filter: Optional status to filter by

        Returns:
            Dictionary of task ID to task info
        """
        with self._lock:
            if status_filter:
                return {
                    task_id: task_info
                    for task_id, task_info in self._tasks.items()
                    if task_info.status == status_filter
                }
            else:
                return self._tasks.copy()

    def cleanup_expired_results(self) -> int:
        """
        Clean up expired results and completed tasks.

        Returns:
            Number of items cleaned up
        """
        current_time = datetime.now()
        cutoff_time = current_time - timedelta(seconds=self.result_ttl_seconds)

        with self._lock:
            # Find expired results
            expired_task_ids = []

            for task_id, result in self._results.items():
                # Clean up if result was retrieved and TTL expired
                task_info = self._tasks.get(task_id)
                if (result.retrieved_at and result.retrieved_at < cutoff_time) or (
                    result.status
                    in [
                        TaskStatus.COMPLETED,
                        TaskStatus.FAILED,
                    ]
                    and task_info
                    and task_info.completed_at
                    and (current_time - task_info.completed_at) > timedelta(hours=24)
                ):
                    expired_task_ids.append(task_id)

            # Remove expired items
            for task_id in expired_task_ids:
                self._results.pop(task_id, None)
                self._tasks.pop(task_id, None)

            # Update cleanup timestamp
            self._last_cleanup = time.time()

            return len(expired_task_ids)

    def auto_cleanup_if_needed(self) -> None:
        """
        Perform automatic cleanup if enough time has passed.
        """
        current_time = time.time()
        if current_time - self._last_cleanup >= self.cleanup_interval_seconds:
            self.cleanup_expired_results()

    def get_stats(self) -> dict[str, Any]:
        """
        Get task manager statistics.

        Returns:
            Dictionary with statistics
        """
        with self._lock:
            status_counts = {}
            for status in TaskStatus:
                status_counts[status.value] = sum(
                    1 for task in self._tasks.values() if task.status == status
                )

            return {
                "total_tasks": len(self._tasks),
                "total_results": len(self._results),
                "running_tasks": len(self._running_tasks),
                "status_counts": status_counts,
                "max_concurrent_tasks": self.max_concurrent_tasks,
                "result_ttl_seconds": self.result_ttl_seconds,
                "last_cleanup": datetime.fromtimestamp(self._last_cleanup).isoformat(),
            }


# Global task manager instance
_task_manager: Optional[TaskManager] = None
_task_manager_lock = threading.Lock()


def get_task_manager() -> TaskManager:
    """
    Get the global task manager instance.

    Returns:
        TaskManager instance
    """
    global _task_manager
    if _task_manager is None:
        with _task_manager_lock:
            # Double-check pattern
            if _task_manager is None:
                # Get settings from config
                from app.core.config import get_settings

                settings = get_settings()

                _task_manager = TaskManager(
                    result_ttl_seconds=settings.result_cache_ttl_seconds,
                    max_concurrent_tasks=settings.max_concurrent_evaluations,
                )

    return _task_manager
