"""
Logging utilities for NovaEval.

This module provides logging setup and configuration functionality.
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Union


def setup_logging(
    level: Union[str, int] = "INFO",
    log_file: Optional[Union[str, Path]] = None,
    format_string: Optional[str] = None,
    include_timestamp: bool = True,
    include_level: bool = True,
    include_name: bool = True,
) -> logging.Logger:
    """
    Setup logging configuration for NovaEval.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path to write logs
        format_string: Custom format string for log messages
        include_timestamp: Whether to include timestamp in logs
        include_level: Whether to include log level in logs
        include_name: Whether to include logger name in logs

    Returns:
        Configured logger instance
    """
    # Convert string level to logging constant
    if isinstance(level, str):
        level = getattr(logging, level.upper())

    # Create format string if not provided
    if format_string is None:
        format_parts = []

        if include_timestamp:
            format_parts.append("%(asctime)s")

        if include_level:
            format_parts.append("%(levelname)s")

        if include_name:
            format_parts.append("%(name)s")

        format_parts.append("%(message)s")
        format_string = " - ".join(format_parts)

    # Configure root logger
    logging.basicConfig(
        level=level,
        format=format_string,
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[],  # Clear existing handlers
    )

    # Get root logger
    logger = logging.getLogger()
    logger.handlers.clear()  # Remove any existing handlers

    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_formatter = logging.Formatter(format_string, datefmt="%Y-%m-%d %H:%M:%S")
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # Create file handler if log_file is specified
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_formatter = logging.Formatter(format_string, datefmt="%Y-%m-%d %H:%M:%S")
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    # Set level
    logger.setLevel(level)

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the specified name.

    Args:
        name: Logger name

    Returns:
        Logger instance
    """
    return logging.getLogger(name)


class LoggerMixin:
    """
    Mixin class to add logging capabilities to other classes.
    """

    @property
    def logger(self) -> logging.Logger:
        """Get logger for this class."""
        return logging.getLogger(self.__class__.__name__)

    def log_info(self, message: str, *args, **kwargs) -> None:
        """Log info message."""
        self.logger.info(message, *args, **kwargs)

    def log_warning(self, message: str, *args, **kwargs) -> None:
        """Log warning message."""
        self.logger.warning(message, *args, **kwargs)

    def log_error(self, message: str, *args, **kwargs) -> None:
        """Log error message."""
        self.logger.error(message, *args, **kwargs)

    def log_debug(self, message: str, *args, **kwargs) -> None:
        """Log debug message."""
        self.logger.debug(message, *args, **kwargs)


def configure_third_party_loggers(level: Union[str, int] = "WARNING") -> None:
    """
    Configure third-party library loggers to reduce noise.

    Args:
        level: Logging level for third-party loggers
    """
    if isinstance(level, str):
        level = getattr(logging, level.upper())

    # Common noisy loggers
    noisy_loggers = [
        "urllib3",
        "requests",
        "httpx",
        "openai",
        "anthropic",
        "boto3",
        "botocore",
        "transformers",
        "datasets",
    ]

    for logger_name in noisy_loggers:
        logging.getLogger(logger_name).setLevel(level)


def log_evaluation_start(
    dataset_name: str, model_names: list, scorer_names: list, num_samples: int
) -> None:
    """
    Log evaluation start information.

    Args:
        dataset_name: Name of the dataset
        model_names: List of model names
        scorer_names: List of scorer names
        num_samples: Number of samples to evaluate
    """
    logger = get_logger("novaeval.evaluation")

    logger.info("=" * 60)
    logger.info("Starting NovaEval evaluation")
    logger.info("=" * 60)
    logger.info(f"Dataset: {dataset_name}")
    logger.info(f"Models: {', '.join(model_names)}")
    logger.info(f"Scorers: {', '.join(scorer_names)}")
    logger.info(f"Samples: {num_samples}")
    logger.info("=" * 60)


def log_evaluation_end(
    duration: float, total_requests: int, total_tokens: int, total_cost: float
) -> None:
    """
    Log evaluation end information.

    Args:
        duration: Evaluation duration in seconds
        total_requests: Total number of API requests
        total_tokens: Total number of tokens used
        total_cost: Total cost in USD
    """
    logger = get_logger("novaeval.evaluation")

    logger.info("=" * 60)
    logger.info("NovaEval evaluation completed")
    logger.info("=" * 60)
    logger.info(f"Duration: {duration:.2f} seconds")
    logger.info(f"Total requests: {total_requests}")
    logger.info(f"Total tokens: {total_tokens:,}")
    logger.info(f"Total cost: ${total_cost:.4f}")
    logger.info("=" * 60)


def log_model_results(model_name: str, results: dict) -> None:
    """
    Log model evaluation results.

    Args:
        model_name: Name of the model
        results: Results dictionary
    """
    logger = get_logger("novaeval.evaluation")

    logger.info(f"Results for {model_name}:")

    if "scores" in results:
        for scorer_name, score_info in results["scores"].items():
            if isinstance(score_info, dict) and "mean" in score_info:
                logger.info(f"  {scorer_name}: {score_info['mean']:.4f}")
            else:
                logger.info(f"  {scorer_name}: {score_info}")

    if results.get("errors"):
        logger.warning(f"  Errors: {len(results['errors'])}")


# Configure third-party loggers by default
configure_third_party_loggers()
