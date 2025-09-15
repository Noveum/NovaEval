"""
Unit tests for configuration utilities.
"""

import json
import tempfile
from pathlib import Path

import pytest
import yaml

from novaeval.utils.config import Config


class TestConfig:
    """Test cases for Config class."""

    def test_init_empty(self):
        """Test initialization with empty config."""
        config = Config()
        assert config.to_dict() == {}

    def test_init_with_dict(self):
        """Test initialization with dictionary."""
        data = {"key": "value", "nested": {"inner": 123}}
        config = Config(data)
        assert config.to_dict() == data

    def test_get_simple_key(self):
        """Test getting simple key."""
        config = Config({"key": "value"})
        assert config.get("key") == "value"
        assert config.get("missing") is None
        assert config.get("missing", "default") == "default"

    def test_get_nested_key(self):
        """Test getting nested key with dot notation."""
        config = Config({"level1": {"level2": {"key": "value"}}})
        assert config.get("level1.level2.key") == "value"
        assert config.get("level1.missing") is None

    def test_set_simple_key(self):
        """Test setting simple key."""
        config = Config()
        config.set("key", "value")
        assert config.get("key") == "value"

    def test_set_nested_key(self):
        """Test setting nested key with dot notation."""
        config = Config()
        config.set("level1.level2.key", "value")
        assert config.get("level1.level2.key") == "value"
        assert isinstance(config.get("level1"), dict)
        assert isinstance(config.get("level1.level2"), dict)

    def test_update_config(self):
        """Test updating config with another config."""
        config1 = Config({"a": 1, "b": {"c": 2}})
        config2 = Config({"b": {"d": 3}, "e": 4})

        config1.update(config2)

        assert config1.get("a") == 1
        assert config1.get("b.c") == 2
        assert config1.get("b.d") == 3
        assert config1.get("e") == 4

    def test_update_dict(self):
        """Test updating config with dictionary."""
        config = Config({"a": 1})
        config.update({"b": 2})

        assert config.get("a") == 1
        assert config.get("b") == 2

    def test_load_yaml(self):
        """Test loading from YAML file."""
        data = {"key": "value", "number": 123}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(data, f)
            yaml_path = f.name

        try:
            config = Config.load(yaml_path)
            assert config.to_dict() == data
        finally:
            Path(yaml_path).unlink()

    def test_load_json(self):
        """Test loading from JSON file."""
        data = {"key": "value", "number": 123}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            json_path = f.name

        try:
            config = Config.load(json_path)
            assert config.to_dict() == data
        finally:
            Path(json_path).unlink()

    def test_load_nonexistent_file(self):
        """Test loading from non-existent file."""
        with pytest.raises(FileNotFoundError):
            Config.load("nonexistent.yaml")

    def test_load_unsupported_format(self):
        """Test loading from unsupported file format."""
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            txt_path = f.name

        try:
            with pytest.raises(
                ValueError, match="Unsupported configuration file format"
            ):
                Config.load(txt_path)
        finally:
            Path(txt_path).unlink()

    def test_save_yaml(self):
        """Test saving to YAML file."""
        data = {"key": "value", "number": 123}
        config = Config(data)

        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            yaml_path = f.name

        try:
            config.save(yaml_path, format="yaml")

            # Load and verify
            with open(yaml_path) as f:
                loaded_data = yaml.safe_load(f)
            assert loaded_data == data
        finally:
            Path(yaml_path).unlink()

    def test_save_json(self):
        """Test saving to JSON file."""
        data = {"key": "value", "number": 123}
        config = Config(data)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            json_path = f.name

        try:
            config.save(json_path, format="json")

            # Load and verify
            with open(json_path) as f:
                loaded_data = json.load(f)
            assert loaded_data == data
        finally:
            Path(json_path).unlink()

    def test_bracket_notation(self):
        """Test bracket notation for getting and setting."""
        config = Config({"key": "value"})

        # Test getting
        assert config["key"] == "value"

        # Test setting
        config["new_key"] = "new_value"
        assert config["new_key"] == "new_value"

        # Test contains
        assert "key" in config
        assert "missing" not in config

    def test_validate(self):
        """Test configuration validation."""
        config = Config({"string_key": "value", "int_key": 123, "list_key": [1, 2, 3]})

        # Valid schema
        schema = {"string_key": str, "int_key": int, "list_key": list}
        assert config.validate(schema) is True

        # Invalid schema
        invalid_schema = {
            "string_key": int,  # Wrong type
        }
        assert config.validate(invalid_schema) is False
