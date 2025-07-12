"""
Configuration utilities for NovaEval.

This module provides configuration loading and management functionality.
"""

import json
import os
from pathlib import Path
from typing import Any, Optional, Union

import yaml


class Config:
    """
    Configuration manager for NovaEval.

    Supports loading from YAML, JSON files and environment variables.
    """

    def __init__(self, config_dict: Optional[dict[str, Any]] = None):
        """
        Initialize configuration.

        Args:
            config_dict: Optional initial configuration dictionary
        """
        self._config = config_dict or {}
        self._env_prefix = "NOVAEVAL_"

    @classmethod
    def load(cls, config_path: Union[str, Path]) -> "Config":
        """
        Load configuration from file.

        Args:
            config_path: Path to configuration file (YAML or JSON)

        Returns:
            Config instance
        """
        config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        suffix = config_path.suffix.lower()

        if suffix in [".yaml", ".yml"]:
            return cls._load_yaml(config_path)
        elif suffix == ".json":
            return cls._load_json(config_path)
        else:
            raise ValueError(f"Unsupported configuration file format: {suffix}")

    @classmethod
    def _load_yaml(cls, config_path: Path) -> "Config":
        """Load configuration from YAML file."""
        with open(config_path, encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)
        return cls(config_dict)

    @classmethod
    def _load_json(cls, config_path: Path) -> "Config":
        """Load configuration from JSON file."""
        with open(config_path, encoding="utf-8") as f:
            config_dict = json.load(f)
        return cls(config_dict)

    @classmethod
    def from_env(cls, prefix: str = "NOVAEVAL_") -> "Config":
        """
        Load configuration from environment variables.

        Args:
            prefix: Environment variable prefix

        Returns:
            Config instance
        """
        config_dict = {}

        for key, value in os.environ.items():
            if key.startswith(prefix):
                # Remove prefix and convert to lowercase
                config_key = key[len(prefix) :].lower()

                # Try to parse as JSON for complex values
                try:
                    config_dict[config_key] = json.loads(value)
                except json.JSONDecodeError:
                    config_dict[config_key] = value

        instance = cls(config_dict)
        instance._env_prefix = prefix
        return instance

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value.

        Args:
            key: Configuration key (supports dot notation)
            default: Default value if key not found

        Returns:
            Configuration value
        """
        # Check environment variable first
        env_key = f"{self._env_prefix}{key.upper().replace('.', '_')}"
        if env_key in os.environ:
            try:
                return json.loads(os.environ[env_key])
            except json.JSONDecodeError:
                return os.environ[env_key]

        # Navigate through nested dictionary
        keys = key.split(".")
        value = self._config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value.

        Args:
            key: Configuration key (supports dot notation)
            value: Value to set
        """
        keys = key.split(".")
        config = self._config

        # Navigate to parent dictionary
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]

        # Set the value
        config[keys[-1]] = value

    def update(self, other: Union["Config", dict[str, Any]]) -> None:
        """
        Update configuration with another config or dictionary.

        Args:
            other: Config instance or dictionary to merge
        """
        other_dict = other._config if isinstance(other, Config) else other

        self._deep_update(self._config, other_dict)

    def _deep_update(
        self, base_dict: dict[str, Any], update_dict: dict[str, Any]
    ) -> None:
        """
        Deep update dictionary.

        Args:
            base_dict: Base dictionary to update
            update_dict: Dictionary with updates
        """
        for key, value in update_dict.items():
            if (
                key in base_dict
                and isinstance(base_dict[key], dict)
                and isinstance(value, dict)
            ):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value

    def to_dict(self) -> dict[str, Any]:
        """
        Convert configuration to dictionary.

        Returns:
            Configuration as dictionary
        """
        return self._config.copy()

    def save(self, config_path: Union[str, Path], format: str = "yaml") -> None:
        """
        Save configuration to file.

        Args:
            config_path: Path to save configuration
            format: File format ("yaml" or "json")
        """
        config_path = Path(config_path)

        if format.lower() == "yaml":
            with open(config_path, "w", encoding="utf-8") as f:
                yaml.dump(self._config, f, default_flow_style=False, indent=2)
        elif format.lower() == "json":
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(self._config, f, indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def validate(self, schema: dict[str, Any]) -> bool:
        """
        Validate configuration against schema.

        Args:
            schema: Validation schema

        Returns:
            True if valid, False otherwise
        """
        # Simple validation - can be extended with jsonschema
        for key, expected_type in schema.items():
            value = self.get(key)
            if value is not None and not isinstance(value, expected_type):
                return False
        return True

    def __getitem__(self, key: str) -> Any:
        """Get configuration value using bracket notation."""
        return self.get(key)

    def __setitem__(self, key: str, value: Any) -> None:
        """Set configuration value using bracket notation."""
        self.set(key, value)

    def __contains__(self, key: str) -> bool:
        """Check if configuration key exists."""
        return self.get(key) is not None

    def __str__(self) -> str:
        """String representation of configuration."""
        return json.dumps(self._config, indent=2)

    def __repr__(self) -> str:
        """Detailed string representation of configuration."""
        return f"Config({self._config})"
