"""
Configuration management for NovaEval API.

This module provides environment variable management using Pydantic BaseSettings.
"""

import os
from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    """

    # API Configuration
    api_title: str = "NovaEval API"
    api_description: str = "HTTP API for NovaEval evaluation framework"
    api_version: str = "1.0.0"
    debug: bool = False

    # Server Configuration
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    log_level: str = "INFO"

    # Model API Keys
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    azure_openai_api_key: Optional[str] = None
    azure_openai_endpoint: Optional[str] = None
    google_api_key: Optional[str] = None

    # Evaluation Configuration
    max_concurrent_evaluations: int = 5
    evaluation_timeout_seconds: int = 3600  # 1 hour
    result_cache_ttl_seconds: int = 7200  # 2 hours

    # File Upload Limits
    max_file_size_mb: int = 100
    max_dataset_rows: int = 10000

    # Default Model Configuration
    default_model_provider: str = "openai"
    default_model_name: str = "gpt-3.5-turbo"
    default_model_temperature: float = 0.0
    default_model_max_tokens: int = 1000

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"  # Ignore extra environment variables


@lru_cache
def get_settings() -> Settings:
    """
    Get cached application settings.

    Returns:
        Settings: Application configuration settings
    """
    return Settings()


def get_env_var(name: str, default: Optional[str] = None) -> Optional[str]:
    """
    Get environment variable with optional default.

    Args:
        name: Environment variable name
        default: Default value if variable not found

    Returns:
        Optional[str]: Environment variable value or default
    """
    return os.getenv(name, default)
