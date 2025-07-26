"""
Structured logging configuration for NovaEval API.

This module provides centralized logging configuration with structured output
for better observability and debugging.
"""

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any, Optional


class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured JSON logging."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as structured JSON."""

        # Build structured log entry
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        # Add extra fields from the record
        excluded_keys = {
            "name",
            "msg",
            "args",
            "levelname",
            "levelno",
            "pathname",
            "filename",
            "module",
            "lineno",
            "funcName",
            "created",
            "msecs",
            "relativeCreated",
            "thread",
            "threadName",
            "processName",
            "process",
            "getMessage",
            "exc_info",
            "exc_text",
            "stack_info",
        }
        log_entry.update(
            {
                key: value
                for key, value in record.__dict__.items()
                if key not in excluded_keys
            }
        )

        return json.dumps(log_entry, default=str)


def setup_logging(
    level: str = "INFO",
    structured: bool = True,
    enable_file_logging: bool = False,
    log_file: str = "novaeval_api.log",
) -> None:
    """
    Set up application logging configuration.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        structured: Whether to use structured JSON logging
        enable_file_logging: Whether to enable file logging
        log_file: Log file path (if file logging enabled)
    """

    # Clear any existing handlers
    root_logger = logging.getLogger()
    root_logger.handlers.clear()

    # Set logging level
    log_level = getattr(logging, level.upper(), logging.INFO)
    root_logger.setLevel(log_level)

    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)

    # Set formatter
    if structured:
        formatter = StructuredFormatter()
    else:
        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)s in %(name)s: %(message)s"
        )

    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # Add file handler if requested
    if enable_file_logging:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    # Set specific logger levels
    logging.getLogger("uvicorn").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    # Create API logger
    api_logger = logging.getLogger("novaeval.api")
    api_logger.setLevel(log_level)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for the given name.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Logger instance
    """
    return logging.getLogger(f"novaeval.api.{name}")


def log_request(
    method: str,
    path: str,
    status_code: int,
    duration_ms: float,
    user_id: Optional[str] = None,
    **kwargs,
) -> None:
    """
    Log HTTP request details.

    Args:
        method: HTTP method
        path: Request path
        status_code: Response status code
        duration_ms: Request duration in milliseconds
        user_id: Optional user identifier
        **kwargs: Additional context
    """
    logger = get_logger("requests")

    log_data = {
        "request_method": method,
        "request_path": path,
        "response_status": status_code,
        "duration_ms": duration_ms,
        "user_id": user_id,
        **kwargs,
    }

    level = logging.INFO
    if status_code >= 500:
        level = logging.ERROR
    elif status_code >= 400:
        level = logging.WARNING

    logger.log(
        level, f"{method} {path} {status_code} {duration_ms:.2f}ms", extra=log_data
    )


def log_error(
    error: Exception,
    context: Optional[dict[str, Any]] = None,
    user_id: Optional[str] = None,
) -> None:
    """
    Log error with structured context.

    Args:
        error: Exception instance
        context: Additional context information
        user_id: Optional user identifier
    """
    logger = get_logger("errors")

    log_data = {
        "error_type": error.__class__.__name__,
        "error_message": str(error),
        "user_id": user_id,
        **(context or {}),
    }

    logger.error(f"Error occurred: {error}", extra=log_data, exc_info=True)


def log_evaluation_event(
    event_type: str, task_id: str, details: Optional[dict[str, Any]] = None
) -> None:
    """
    Log evaluation-related events.

    Args:
        event_type: Type of event (submit, start, complete, fail, cancel)
        task_id: Task identifier
        details: Additional event details
    """
    logger = get_logger("evaluations")

    log_data = {"event_type": event_type, "task_id": task_id, **(details or {})}

    logger.info(f"Evaluation {event_type}: {task_id}", extra=log_data)


def log_component_event(
    component_type: str,
    component_name: str,
    operation: str,
    success: bool = True,
    details: Optional[dict[str, Any]] = None,
) -> None:
    """
    Log component-related events.

    Args:
        component_type: Type of component (model, dataset, scorer)
        component_name: Component name
        operation: Operation performed
        success: Whether operation was successful
        details: Additional operation details
    """
    logger = get_logger("components")

    log_data = {
        "component_type": component_type,
        "component_name": component_name,
        "operation": operation,
        "success": success,
        **(details or {}),
    }

    level = logging.INFO if success else logging.ERROR
    status = "succeeded" if success else "failed"

    logger.log(
        level,
        f"{component_type.title()} {operation} {status}: {component_name}",
        extra=log_data,
    )
