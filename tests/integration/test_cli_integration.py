"""Integration tests for CLI and configuration functionality."""

import json
import tempfile
from pathlib import Path

import yaml

from novaeval.utils.config import Config


class TestCLIIntegration:
    """Test CLI integration with the evaluation system."""

    def test_config_file_loading(self):
        """Test loading configuration from YAML file."""
        config_data = {
            "dataset": {"type": "custom", "path": "test_data/qa_dataset.jsonl"},
            "models": [
                {"type": "openai", "model_name": "gpt-3.5-turbo", "temperature": 0.7}
            ],
            "scorers": [{"type": "accuracy"}],
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name

        try:
            config = Config.load(temp_path)
            assert config.get("dataset")["type"] == "custom"
            assert config.get("models")[0]["type"] == "openai"
            assert config.get("scorers")[0]["type"] == "accuracy"
        finally:
            Path(temp_path).unlink()

    def test_json_config_loading(self):
        """Test loading configuration from JSON file."""
        config_data = {
            "dataset": {"type": "mmlu", "categories": ["mathematics", "science"]},
            "models": [{"type": "anthropic", "model_name": "claude-3-sonnet"}],
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config_data, f)
            temp_path = f.name

        try:
            config = Config.load(temp_path)
            assert config.get("dataset")["type"] == "mmlu"
            assert config.get("models")[0]["type"] == "anthropic"
        finally:
            Path(temp_path).unlink()

    def test_config_validation(self):
        """Test basic configuration validation."""
        config_data = {
            "dataset": {"type": "custom", "path": "test.jsonl"},
            "models": [{"type": "openai", "model_name": "gpt-4"}],
            "scorers": [{"type": "accuracy"}],
        }

        config = Config(config_data)

        # Test that required fields exist
        assert config.get("dataset") is not None
        assert config.get("models") is not None
        assert config.get("scorers") is not None

    def test_environment_variable_substitution(self):
        """Test environment variable handling."""
        config_data = {
            "models": [
                {
                    "type": "openai",
                    "api_key": "${OPENAI_API_KEY}",
                    "model_name": "gpt-4",
                }
            ]
        }

        config = Config(config_data)
        assert config.get("models")[0]["api_key"] == "${OPENAI_API_KEY}"

    def test_config_merging(self):
        """Test configuration merging functionality."""
        base_config = {
            "dataset": {"type": "custom", "path": "base.jsonl"},
            "models": [{"type": "openai", "model_name": "gpt-3.5-turbo"}],
            "scorers": [{"type": "accuracy"}],
        }

        override_config = {
            "dataset": {"num_samples": 50},
            "models": [{"type": "anthropic", "model_name": "claude-3-sonnet"}],
            "output": {"directory": "./custom_results"},
        }

        config = Config(base_config)
        config.update(override_config)

        # Test that merging worked correctly
        assert config.get("dataset")["type"] == "custom"
        assert config.get("dataset")["num_samples"] == 50
        assert config.get("models")[0]["type"] == "anthropic"
        assert config.get("output")["directory"] == "./custom_results"

    def test_config_defaults(self):
        """Test configuration with default values."""
        config_data = {
            "dataset": {"type": "custom", "path": "test.jsonl"},
            "models": [{"type": "openai", "model_name": "gpt-4"}],
            "scorers": [{"type": "accuracy"}],
        }

        config = Config(config_data)
        config.update({"output": {"directory": "./results", "format": "json"}})

        assert config.get("models")[0]["type"] == "openai"
        assert config.get("output")["directory"] == "./results"

    def test_cli_config_integration(self):
        """Test CLI configuration integration."""
        # This test verifies that configurations can be loaded and used
        # in a way that would work with the CLI

        config_data = {
            "dataset": {"type": "custom", "path": "test_data/qa_dataset.jsonl"},
            "models": [
                {"type": "openai", "model_name": "gpt-3.5-turbo", "temperature": 0.7}
            ],
            "scorers": [{"type": "accuracy"}],
            "output": {"directory": "./results", "format": "json"},
        }

        config = Config(config_data)

        # Test that all components can be accessed
        assert config.get("dataset") is not None
        assert config.get("models") is not None
        assert config.get("scorers") is not None
        assert config.get("output") is not None

        # Test that nested values work
        assert config.get("dataset")["type"] == "custom"
        assert config.get("models")[0]["model_name"] == "gpt-3.5-turbo"


class TestConfigurationEdgeCases:
    """Test edge cases and error conditions."""

    def test_config_with_none_values(self):
        """Test configuration with None values."""
        config_data = {"dataset": None, "models": [{"type": "openai", "api_key": None}]}

        config = Config(config_data)
        assert config.get("dataset") is None
        assert config.get("models")[0]["api_key"] is None

    def test_config_key_not_found(self):
        """Test accessing non-existent keys."""
        config = Config({})

        assert config.get("nonexistent") is None
        assert config.get("nonexistent", "default") == "default"

    def test_config_set_nested_values(self):
        """Test setting nested configuration values."""
        config = Config({})

        config.set("dataset.type", "custom")
        config.set("models.0.type", "openai")

        assert config.get("dataset")["type"] == "custom"

    def test_config_update_with_complex_structure(self):
        """Test updating with complex nested structures."""
        base = {"models": [{"type": "openai", "params": {"temperature": 0.5}}]}

        update = {"models": [{"type": "anthropic", "params": {"max_tokens": 1000}}]}

        config = Config(base)
        config.update(update)

        # Update should replace the entire models array
        assert config.get("models")[0]["type"] == "anthropic"


class TestConfigurationIntegration:
    """Test configuration integration with real-world scenarios."""

    def test_config_with_real_file_paths(self):
        """Test configuration with real file paths."""
        config_data = {
            "dataset": {"type": "custom", "path": "test_data/qa_dataset.jsonl"},
            "output": {"directory": "./results"},
        }

        config = Config(config_data)

        # Test that paths are preserved as strings
        assert isinstance(config.get("dataset")["path"], str)
        assert config.get("dataset")["path"] == "test_data/qa_dataset.jsonl"
