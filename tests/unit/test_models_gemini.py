"""
Unit tests for Gemini model functionality.
"""

import os
from unittest.mock import Mock, patch

import pytest

from novaeval.models.gemini import GeminiModel


@pytest.mark.unit
class TestGeminiModel:
    """Test cases for GeminiModel class."""

    def test_init_default(self):
        """Test initialization with default parameters."""
        with patch("novaeval.models.gemini.genai.Client") as mock_client:
            model = GeminiModel()

            assert model.name == "gemini_gemini-2.5-flash"
            assert model.model_name == "gemini-2.5-flash"
            assert model.max_retries == 3
            assert model.timeout == 60.0
            mock_client.assert_called_once()

    def test_init_with_params(self):
        """Test initialization with custom parameters."""
        with patch("novaeval.models.gemini.genai.Client") as mock_client:
            model = GeminiModel(
                model_name="gemini-2.5-pro",
                api_key="test_key",
                max_retries=5,
                timeout=30.0,
            )

            assert model.name == "gemini_gemini-2.5-pro"
            assert model.model_name == "gemini-2.5-pro"
            assert model.api_key == "test_key"
            assert model.max_retries == 5
            assert model.timeout == 30.0

            mock_client.assert_called_once_with(api_key="test_key")

    @patch.dict(os.environ, {"GOOGLE_API_KEY": "env_key"})
    def test_init_with_env_vars(self):
        """Test initialization using environment variables."""
        with patch("novaeval.models.gemini.genai.Client") as mock_client:
            GeminiModel()

            mock_client.assert_called_once_with(api_key="env_key")

    def test_generate_success(self):
        """Test successful text generation."""
        with patch("novaeval.models.gemini.genai.Client") as mock_client:
            # Setup mock response
            mock_response = Mock()
            mock_response.text = "Generated response"

            mock_client_instance = Mock()
            mock_client_instance.models.generate_content.return_value = mock_response
            mock_client.return_value = mock_client_instance

            model = GeminiModel()

            # Mock the estimate_cost method
            model.estimate_cost = Mock(return_value=0.01)

            response = model.generate("Test prompt")

            assert response == "Generated response"
            assert model.total_requests == 1
            assert model.total_cost == 0.01

            mock_client_instance.models.generate_content.assert_called_once()
            call_args = mock_client_instance.models.generate_content.call_args
            assert call_args[1]["model"] == "gemini-2.5-flash"
            assert call_args[1]["contents"] == "Test prompt"

    def test_generate_with_params(self):
        """Test text generation with additional parameters."""
        with (
            patch("novaeval.models.gemini.genai.Client") as mock_client,
            patch("novaeval.models.gemini.types.GenerateContentConfig") as mock_config,
        ):
            mock_response = Mock()
            mock_response.text = "Generated response"

            mock_client_instance = Mock()
            mock_client_instance.models.generate_content.return_value = mock_response
            mock_client.return_value = mock_client_instance

            model = GeminiModel()
            model.estimate_cost = Mock(return_value=0.005)

            response = model.generate(
                "Test prompt", max_tokens=100, temperature=0.5, custom_param="value"
            )

            assert response == "Generated response"

            mock_config.assert_called_once_with(
                temperature=0.5, max_output_tokens=100, custom_param="value"
            )

    def test_generate_empty_response(self):
        """Test generation with empty response."""
        with patch("novaeval.models.gemini.genai.Client") as mock_client:
            mock_response = Mock()
            mock_response.text = None

            mock_client_instance = Mock()
            mock_client_instance.models.generate_content.return_value = mock_response
            mock_client.return_value = mock_client_instance

            model = GeminiModel()
            model.estimate_cost = Mock(return_value=0.001)

            response = model.generate("Test prompt")

            assert response == ""

    def test_generate_error_handling(self):
        """Test error handling during text generation."""
        with patch("novaeval.models.gemini.genai.Client") as mock_client:
            mock_client_instance = Mock()
            mock_client_instance.models.generate_content.side_effect = Exception(
                "API Error"
            )
            mock_client.return_value = mock_client_instance

            model = GeminiModel()

            with pytest.raises(Exception, match="API Error"):
                model.generate("Test prompt")

            # Check that error was tracked
            assert len(model.errors) == 1
            assert "Failed to generate text" in model.errors[0]

    def test_generate_batch(self):
        """Test batch text generation."""
        with patch("novaeval.models.gemini.genai.Client") as mock_client:
            mock_response = Mock()
            mock_response.text = "Generated response"

            mock_client_instance = Mock()
            mock_client_instance.models.generate_content.return_value = mock_response
            mock_client.return_value = mock_client_instance

            model = GeminiModel()
            model.estimate_cost = Mock(return_value=0.005)

            prompts = ["Prompt 1", "Prompt 2", "Prompt 3"]
            responses = model.generate_batch(prompts)

            assert len(responses) == 3
            assert all(response == "Generated response" for response in responses)
            assert mock_client_instance.models.generate_content.call_count == 3

    def test_generate_batch_with_error(self):
        """Test batch generation with errors."""
        with patch("novaeval.models.gemini.genai.Client") as mock_client:
            mock_client_instance = Mock()
            mock_client_instance.models.generate_content.side_effect = Exception(
                "API Error"
            )
            mock_client.return_value = mock_client_instance

            model = GeminiModel()

            # Call generate_batch with multiple prompts
            responses = model.generate_batch(["prompt1", "prompt2"])

            # Should return empty strings for failed generations
            assert responses == ["", ""]

            # Should track errors - at least one error per failed prompt
            assert len(model.errors) >= 2  # At least one error per failed prompt
            assert all("API Error" in error for error in model.errors)

    def test_get_provider(self):
        """Test provider name."""
        with patch("novaeval.models.gemini.genai.Client"):
            model = GeminiModel()
            assert model.get_provider() == "gemini"

    def test_estimate_cost_known_model(self):
        """Test cost estimation for known model."""
        with patch("novaeval.models.gemini.genai.Client"):
            model = GeminiModel(model_name="gemini-2.5-flash")

            # Mock token counting
            model.count_tokens = Mock(return_value=1000)

            prompt = "Test prompt"
            response = "Test response"

            cost = model.estimate_cost(prompt, response)

            # Expected cost: (1000 input + 1000 output) / 1K * pricing
            # gemini-2.5-flash pricing: $0.30 input, $2.50 output per 1K tokens
            expected_cost = (1000 / 1000) * 0.30 + (1000 / 1000) * 2.50

            # Use floating point comparison with reasonable tolerance
            assert abs(cost - expected_cost) < 1e-6

    def test_estimate_cost_unknown_model(self):
        """Test cost estimation for unknown model."""
        with patch("novaeval.models.gemini.genai.Client"):
            model = GeminiModel(model_name="unknown-model")

            cost = model.estimate_cost("Test prompt", "Test response")

            # Should return 0.0 for unknown models
            assert cost == 0.0

    def test_count_tokens(self):
        """Test token counting with specific input strings and expected values."""
        with patch("novaeval.models.gemini.genai.Client"):
            model = GeminiModel()

            # Test cases with known inputs and expected outputs based on heuristic (word count * 1.3)
            test_cases = [
                ("Hello world", 2, int(2 * 1.3)),  # 2 words -> 2 tokens
                ("This is a test", 4, int(4 * 1.3)),  # 4 words -> 5 tokens
                ("The quick brown fox jumps", 5, int(5 * 1.3)),  # 5 words -> 6 tokens
                ("Single", 1, int(1 * 1.3)),  # 1 word -> 1 token
                ("", 0, int(0 * 1.3)),  # 0 words -> 0 tokens
                (
                    "One two three four five six seven eight nine ten",
                    10,
                    int(10 * 1.3),
                ),  # 10 words -> 13 tokens
            ]

            for input_text, word_count, expected_tokens in test_cases:
                actual_tokens = model.count_tokens(input_text)
                assert (
                    actual_tokens == expected_tokens
                ), f"For input '{input_text}' with {word_count} words, expected {expected_tokens} tokens but got {actual_tokens}"
                assert isinstance(
                    actual_tokens, int
                ), f"Token count should be an integer, got {type(actual_tokens)}"

    def test_validate_connection_success(self):
        """Test successful connection validation."""
        with patch("novaeval.models.gemini.genai.Client") as mock_client:
            mock_response = Mock()
            mock_response.text = "Pong"

            mock_client_instance = Mock()
            mock_client_instance.models.generate_content.return_value = mock_response
            mock_client.return_value = mock_client_instance

            model = GeminiModel()

            assert model.validate_connection() is True

            mock_client_instance.models.generate_content.assert_called_once()
            call_args = mock_client_instance.models.generate_content.call_args
            assert call_args[1]["contents"] == "Ping!"

    def test_validate_connection_failure(self):
        """Test connection validation failure."""
        with patch("novaeval.models.gemini.genai.Client") as mock_client:
            mock_client_instance = Mock()
            mock_client_instance.models.generate_content.side_effect = Exception(
                "Connection failed"
            )
            mock_client.return_value = mock_client_instance

            model = GeminiModel()

            result = model.validate_connection()

            assert result is False
            assert len(model.errors) == 1
            assert "Connection test failed" in model.errors[0]

    def test_validate_connection_empty_response(self):
        """Test connection validation with empty response."""
        with patch("novaeval.models.gemini.genai.Client") as mock_client:
            mock_response = Mock()
            mock_response.text = None

            mock_client_instance = Mock()
            mock_client_instance.models.generate_content.return_value = mock_response
            mock_client.return_value = mock_client_instance

            model = GeminiModel()

            result = model.validate_connection()

            assert result is False

    def test_get_info(self):
        """Test model info retrieval."""
        with patch("novaeval.models.gemini.genai.Client"):
            model = GeminiModel(model_name="gemini-1.5-pro")

            info = model.get_info()

            assert info["name"] == "gemini_gemini-1.5-pro"
            assert info["model_name"] == "gemini-1.5-pro"
            assert info["provider"] == "gemini"
            assert info["type"] == "GeminiModel"
            assert info["max_retries"] == 3
            assert info["timeout"] == 60.0
            assert info["supports_batch"] is False
            assert info["pricing"] == (1.25, 5.00)

    def test_get_info_unknown_model(self):
        """Test model info retrieval for unknown model."""
        with patch("novaeval.models.gemini.genai.Client"):
            model = GeminiModel(model_name="unknown-model")

            info = model.get_info()

            assert info["pricing"] == (0, 0)

    def test_pricing_constants(self):
        """Test that pricing constants are defined correctly."""
        assert "gemini-2.5-pro" in GeminiModel.PRICING
        assert "gemini-2.5-flash" in GeminiModel.PRICING
        assert "gemini-2.0-flash" in GeminiModel.PRICING
        assert "gemini-1.5-pro" in GeminiModel.PRICING
        assert "gemini-1.5-flash" in GeminiModel.PRICING
        assert "gemini-1.5-flash-8b" in GeminiModel.PRICING

        # Check that pricing is a tuple of (input_price, output_price)
        for _model_name, pricing in GeminiModel.PRICING.items():
            assert len(pricing) == 2
            assert isinstance(pricing[0], (int, float))
            assert isinstance(pricing[1], (int, float))

    def test_different_model_names(self):
        """Test initialization with different model names."""
        model_names = [
            "gemini-2.5-pro",
            "gemini-2.5-flash",
            "gemini-2.0-flash",
            "gemini-2.0-flash-lite",
            "gemini-1.5-flash",
            "gemini-1.5-flash-8b",
            "gemini-1.5-pro",
        ]

        with patch("novaeval.models.gemini.genai.Client"):
            for model_name in model_names:
                model = GeminiModel(model_name=model_name)
                assert model.model_name == model_name
                assert model.name == f"gemini_{model_name}"

    def test_time_tracking(self):
        """Test that time tracking works during generation."""
        with (
            patch("novaeval.models.gemini.genai.Client") as mock_client,
            patch("novaeval.models.gemini.time.time") as mock_time,
        ):
            mock_time.side_effect = [100.0, 101.0]  # start_time, end_time

            mock_response = Mock()
            mock_response.text = "Generated response"

            mock_client_instance = Mock()
            mock_client_instance.models.generate_content.return_value = mock_response
            mock_client.return_value = mock_client_instance

            model = GeminiModel()
            model.estimate_cost = Mock(return_value=0.01)

            response = model.generate("Test prompt")

            assert response == "Generated response"
            # Verify time.time() was called twice (start and end)
            assert mock_time.call_count == 2

    def test_generate_with_stop_parameter(self):
        """Test generate method with stop parameter (should be ignored)."""
        with patch("novaeval.models.gemini.genai.Client") as mock_client:
            mock_response = Mock()
            mock_response.text = "Generated response"

            mock_client_instance = Mock()
            mock_client_instance.models.generate_content.return_value = mock_response
            mock_client.return_value = mock_client_instance

            model = GeminiModel()
            model.estimate_cost = Mock(return_value=0.01)

            # The stop parameter should be accepted but not used
            response = model.generate("Test prompt", stop=["<END>"])

            assert response == "Generated response"
            # Verify that the stop parameter doesn't affect the API call
            mock_client_instance.models.generate_content.assert_called_once()
