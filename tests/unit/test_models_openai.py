"""
Unit tests for OpenAI model functionality.
"""

import os
from unittest.mock import Mock, patch

import pytest

from novaeval.models.openai import OpenAIModel


class TestOpenAIModel:
    """Test cases for OpenAIModel class."""

    def test_init_default(self):
        """Test initialization with default parameters."""
        with patch("novaeval.models.openai.OpenAI") as mock_openai:
            model = OpenAIModel()

            assert model.name == "openai_gpt-4"
            assert model.model_name == "gpt-4"
            assert model.max_retries == 3
            assert model.timeout == 60.0
            assert model.organization is None
            mock_openai.assert_called_once()

    def test_init_with_params(self):
        """Test initialization with custom parameters."""
        with patch("novaeval.models.openai.OpenAI") as mock_openai:
            model = OpenAIModel(
                model_name="gpt-3.5-turbo",
                api_key="test_key",
                base_url="https://api.test.com",
                organization="test_org",
                max_retries=5,
                timeout=30.0,
            )

            assert model.name == "openai_gpt-3.5-turbo"
            assert model.model_name == "gpt-3.5-turbo"
            assert model.api_key == "test_key"
            assert model.base_url == "https://api.test.com"
            assert model.organization == "test_org"
            assert model.max_retries == 5
            assert model.timeout == 30.0

            mock_openai.assert_called_once_with(
                api_key="test_key",
                base_url="https://api.test.com",
                organization="test_org",
                max_retries=5,
                timeout=30.0,
            )

    @patch.dict(
        os.environ, {"OPENAI_API_KEY": "env_key", "OPENAI_API_BASE": "env_base"}
    )
    def test_init_with_env_vars(self):
        """Test initialization using environment variables."""
        with patch("novaeval.models.openai.OpenAI") as mock_openai:
            OpenAIModel()

            mock_openai.assert_called_once_with(
                api_key="env_key",
                base_url="env_base",
                organization=None,
                max_retries=3,
                timeout=60.0,
            )

    def test_generate_success(self):
        """Test successful text generation."""
        with patch("novaeval.models.openai.OpenAI") as mock_openai:
            # Setup mock response
            mock_usage = Mock()
            mock_usage.total_tokens = 100

            mock_choice = Mock()
            mock_choice.message.content = "Generated response"

            mock_response = Mock()
            mock_response.choices = [mock_choice]
            mock_response.usage = mock_usage

            mock_client = Mock()
            mock_client.chat.completions.create.return_value = mock_response
            mock_openai.return_value = mock_client

            model = OpenAIModel()

            # Mock the estimate_cost method
            model.estimate_cost = Mock(return_value=0.01)

            response = model.generate("Test prompt")

            assert response == "Generated response"
            assert model.total_requests == 1
            assert model.total_tokens == 100
            assert model.total_cost == 0.01

            mock_client.chat.completions.create.assert_called_once_with(
                model="gpt-4", messages=[{"role": "user", "content": "Test prompt"}]
            )

    def test_generate_with_params(self):
        """Test text generation with additional parameters."""
        with patch("novaeval.models.openai.OpenAI") as mock_openai:
            mock_choice = Mock()
            mock_choice.message.content = "Generated response"

            mock_response = Mock()
            mock_response.choices = [mock_choice]
            mock_response.usage = Mock()
            mock_response.usage.total_tokens = 50

            mock_client = Mock()
            mock_client.chat.completions.create.return_value = mock_response
            mock_openai.return_value = mock_client

            model = OpenAIModel()
            model.estimate_cost = Mock(return_value=0.005)

            response = model.generate(
                "Test prompt", max_tokens=100, temperature=0.5, stop=["<END>"]
            )

            assert response == "Generated response"

            mock_client.chat.completions.create.assert_called_once_with(
                model="gpt-4",
                messages=[{"role": "user", "content": "Test prompt"}],
                max_tokens=100,
                temperature=0.5,
                stop=["<END>"],
            )

    def test_generate_error_handling(self):
        """Test error handling during text generation."""
        with patch("novaeval.models.openai.OpenAI") as mock_openai:
            mock_client = Mock()
            mock_client.chat.completions.create.side_effect = Exception("API Error")
            mock_openai.return_value = mock_client

            model = OpenAIModel()

            with pytest.raises(Exception, match="API Error"):
                model.generate("Test prompt")

            # Check that error was tracked
            assert len(model.errors) == 1
            assert "Failed to generate text" in model.errors[0]

    def test_generate_batch(self):
        """Test batch text generation."""
        with patch("novaeval.models.openai.OpenAI") as mock_openai:
            mock_choice = Mock()
            mock_choice.message.content = "Generated response"

            mock_response = Mock()
            mock_response.choices = [mock_choice]
            mock_response.usage = Mock()
            mock_response.usage.total_tokens = 50

            mock_client = Mock()
            mock_client.chat.completions.create.return_value = mock_response
            mock_openai.return_value = mock_client

            model = OpenAIModel()
            model.estimate_cost = Mock(return_value=0.005)

            prompts = ["Prompt 1", "Prompt 2", "Prompt 3"]
            responses = model.generate_batch(prompts)

            assert len(responses) == 3
            assert all(response == "Generated response" for response in responses)
            assert mock_client.chat.completions.create.call_count == 3

    def test_generate_batch_with_error(self):
        """Test batch generation with errors."""
        with patch("novaeval.models.openai.OpenAI") as mock_openai:
            mock_client = Mock()
            mock_client.chat.completions.create.side_effect = Exception("API Error")
            mock_openai.return_value = mock_client

            model = OpenAIModel()

            # Call generate_batch with multiple prompts
            responses = model.generate_batch(["prompt1", "prompt2"])

            # Should return empty strings for failed generations
            assert responses == ["", ""]

            # Should track errors - at least one error per failed prompt
            assert len(model.errors) >= 2  # At least one error per failed prompt
            assert all("API Error" in error for error in model.errors)

    def test_get_provider(self):
        """Test provider name."""
        with patch("novaeval.models.openai.OpenAI"):
            model = OpenAIModel()
            assert model.get_provider() == "openai"

    def test_estimate_cost_known_model(self):
        """Test cost estimation for known model."""
        with patch("novaeval.models.openai.OpenAI"):
            model = OpenAIModel(model_name="gpt-4")

            # Mock token counting
            model.count_tokens = Mock(return_value=1000)

            prompt = "Test prompt"
            response = "Test response"

            cost = model.estimate_cost(prompt, response)

            # Expected cost: (1000 input + 1000 output) / 1K * pricing
            # gpt-4 pricing: $0.03 input, $0.06 output per 1K tokens
            expected_cost = (1000 / 1000) * 0.03 + (1000 / 1000) * 0.06

            # Use floating point comparison with reasonable tolerance
            assert abs(cost - expected_cost) < 1e-6

    def test_estimate_cost_unknown_model(self):
        """Test cost estimation for unknown model."""
        with patch("novaeval.models.openai.OpenAI"):
            model = OpenAIModel(model_name="unknown-model")

            cost = model.estimate_cost("Test prompt", "Test response")

            # Should return 0.0 for unknown models
            assert cost == 0.0

    def test_count_tokens(self):
        """Test token counting."""
        with patch("novaeval.models.openai.OpenAI"):
            model = OpenAIModel()

            # Test simple estimation (chars / 4)
            tokens = model.count_tokens("Hello world")
            assert tokens == 11 // 4  # 2

            tokens = model.count_tokens("This is a longer text")
            assert tokens == 21 // 4  # 5

    def test_count_tokens_tiktoken_fallback(self):
        """Test token counting when tiktoken is not available."""
        with patch("novaeval.models.openai.OpenAI"):
            model = OpenAIModel()

            # Mock the import of tiktoken to raise ImportError
            def mock_import(name, *args, **kwargs):
                if name == "tiktoken":
                    raise ImportError("tiktoken not available")
                return __import__(name, *args, **kwargs)

            with patch("builtins.__import__", side_effect=mock_import):
                # This should fall back to base model's count_tokens
                tokens = model.count_tokens("Hello world")
                assert isinstance(tokens, int)
                assert tokens > 0

    def test_count_tokens_different_models(self):
        """Test token counting for different model types."""
        with patch("novaeval.models.openai.OpenAI"):
            # Test GPT-4
            model_gpt4 = OpenAIModel(model_name="gpt-4")
            tokens_gpt4 = model_gpt4.count_tokens("Hello world")
            assert isinstance(tokens_gpt4, int)
            assert tokens_gpt4 > 0

            # Test GPT-3.5
            model_gpt35 = OpenAIModel(model_name="gpt-3.5-turbo")
            tokens_gpt35 = model_gpt35.count_tokens("Hello world")
            assert isinstance(tokens_gpt35, int)
            assert tokens_gpt35 > 0

            # Test other model (uses cl100k_base)
            model_other = OpenAIModel(model_name="text-davinci-003")
            tokens_other = model_other.count_tokens("Hello world")
            assert isinstance(tokens_other, int)
            assert tokens_other > 0

    def test_validate_connection_success(self):
        """Test successful connection validation."""
        with patch("novaeval.models.openai.OpenAI") as mock_openai:
            mock_choice = Mock()
            mock_choice.message.content = "Hi"

            mock_response = Mock()
            mock_response.choices = [mock_choice]
            mock_response.usage = Mock()
            mock_response.usage.total_tokens = 1

            mock_client = Mock()
            mock_client.chat.completions.create.return_value = mock_response
            mock_openai.return_value = mock_client

            model = OpenAIModel()
            model.estimate_cost = Mock(return_value=0.001)

            assert model.validate_connection() is True

    def test_validate_connection_failure(self):
        """Test connection validation failure."""
        with patch("novaeval.models.openai.OpenAI") as mock_openai:
            mock_client = Mock()
            mock_client.chat.completions.create.side_effect = Exception(
                "Connection failed"
            )
            mock_openai.return_value = mock_client

            model = OpenAIModel()

            result = model.validate_connection()

            assert result is False
            assert len(model.errors) == 1
            assert "OpenAI connection validation failed" in model.errors[0]

    def test_get_info(self):
        """Test model info retrieval."""
        with patch("novaeval.models.openai.OpenAI"):
            model = OpenAIModel(model_name="gpt-3.5-turbo")

            info = model.get_info()

            assert info["name"] == "openai_gpt-3.5-turbo"
            assert info["model_name"] == "gpt-3.5-turbo"
            assert info["provider"] == "openai"
            assert info["type"] == "OpenAIModel"

    def test_pricing_constants(self):
        """Test that pricing constants are defined correctly."""
        assert "gpt-4" in OpenAIModel.PRICING
        assert "gpt-3.5-turbo" in OpenAIModel.PRICING

        # Check that pricing is a tuple of (input_price, output_price)
        gpt4_pricing = OpenAIModel.PRICING["gpt-4"]
        assert len(gpt4_pricing) == 2
        assert isinstance(gpt4_pricing[0], (int, float))
        assert isinstance(gpt4_pricing[1], (int, float))
