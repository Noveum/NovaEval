"""
Unit tests for base model functionality.
"""

import pytest

from novaeval.models.base import BaseModel


class ConcreteModel(BaseModel):
    """Concrete implementation of BaseModel for testing."""

    def __init__(self, **kwargs):
        super().__init__(name="test_model", model_name="test-model-v1", **kwargs)

    def generate(self, prompt, max_tokens=None, temperature=None, stop=None, **kwargs):
        """Mock implementation."""
        return f"Generated response for: {prompt}"

    def generate_batch(
        self, prompts, max_tokens=None, temperature=None, stop=None, **kwargs
    ):
        """Mock implementation."""
        return [f"Generated response for: {prompt}" for prompt in prompts]

    def get_provider(self):
        return "test_provider"


class TestBaseModel:
    """Test cases for BaseModel class."""

    def test_init_default(self):
        """Test initialization with default parameters."""
        model = ConcreteModel()
        assert model.name == "test_model"
        assert model.model_name == "test-model-v1"
        assert model.api_key is None
        assert model.base_url is None
        assert model.total_requests == 0
        assert model.total_tokens == 0
        assert model.total_cost == 0.0
        assert model.errors == []

    def test_init_with_params(self):
        """Test initialization with custom parameters."""
        model = ConcreteModel(
            api_key="test_key",
            base_url="https://api.test.com",
            custom_param="custom_value",
        )
        assert model.api_key == "test_key"
        assert model.base_url == "https://api.test.com"
        assert model.kwargs["custom_param"] == "custom_value"

    def test_track_request(self):
        """Test request tracking functionality."""
        model = ConcreteModel()

        # Track a request
        model._track_request(
            prompt="test prompt", response="test response", tokens_used=50, cost=0.01
        )

        assert model.total_requests == 1
        assert model.total_tokens == 50
        assert model.total_cost == 0.01

        # Track another request
        model._track_request(
            prompt="another prompt",
            response="another response",
            tokens_used=30,
            cost=0.02,
        )

        assert model.total_requests == 2
        assert model.total_tokens == 80
        assert model.total_cost == 0.03

    def test_handle_error(self):
        """Test error handling functionality."""
        model = ConcreteModel()

        error = ValueError("Test error")
        model._handle_error(error, "Test context")

        assert len(model.errors) == 1
        assert "Test context" in model.errors[0]
        assert "Test error" in model.errors[0]

    def test_count_tokens(self):
        """Test token counting functionality."""
        model = ConcreteModel()

        # Test simple text
        tokens = model.count_tokens("Hello world")
        assert tokens == 2  # 11 chars / 4 = 2.75 -> 2

        # Test longer text
        long_text = "This is a longer text with many words"
        tokens = model.count_tokens(long_text)
        assert tokens == len(long_text) // 4

    def test_estimate_cost_default(self):
        """Test default cost estimation."""
        model = ConcreteModel()

        cost = model.estimate_cost("test prompt", "test response")
        assert cost == 0.0  # Default implementation returns 0

    def test_validate_connection_with_generate(self):
        """Test connection validation using generate method."""
        model = ConcreteModel()

        # Should return True since our mock generate works
        assert model.validate_connection() is True

    def test_generate_batch_implementation(self):
        """Test the concrete generate_batch implementation."""
        model = ConcreteModel()
        prompts = ["prompt1", "prompt2", "prompt3"]
        responses = model.generate_batch(prompts)

        assert len(responses) == 3
        assert responses[0] == "Generated response for: prompt1"
        assert responses[1] == "Generated response for: prompt2"
        assert responses[2] == "Generated response for: prompt3"

    def test_get_info(self):
        """Test model info retrieval."""
        model = ConcreteModel(api_key="test_key", custom_param="custom_value")

        info = model.get_info()

        assert info["name"] == "test_model"
        assert info["model_name"] == "test-model-v1"
        assert info["provider"] == "test_provider"
        assert info["type"] == "ConcreteModel"
        assert info["total_requests"] == 0
        assert info["total_tokens"] == 0
        assert info["total_cost"] == 0.0
        assert info["error_count"] == 0

    def test_abstract_methods_not_implemented(self):
        """Test that abstract methods raise NotImplementedError."""
        with pytest.raises(TypeError):
            # Cannot instantiate abstract class
            BaseModel(name="test", model_name="test")

    def test_generate_implementation(self):
        """Test the concrete generate implementation."""
        model = ConcreteModel()
        response = model.generate("Test prompt")
        assert response == "Generated response for: Test prompt"

    def test_from_config_classmethod(self):
        """Test creating model from configuration."""
        config = {"api_key": "test_key", "custom_param": "custom_value"}

        model = ConcreteModel.from_config(config)
        assert model.api_key == "test_key"
        assert model.kwargs["custom_param"] == "custom_value"

    def test_str_representation(self):
        """Test string representation."""
        model = ConcreteModel()
        str_repr = str(model)
        assert "ConcreteModel" in str_repr
        assert "test_model" in str_repr
        assert "test-model-v1" in str_repr

    def test_repr_representation(self):
        """Test detailed string representation."""
        model = ConcreteModel()
        repr_str = repr(model)
        assert "ConcreteModel" in repr_str
        assert "name='test_model'" in repr_str
        assert "model_name='test-model-v1'" in repr_str
        assert "provider='test_provider'" in repr_str
