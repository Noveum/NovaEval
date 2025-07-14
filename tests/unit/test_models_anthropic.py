"""
Unit tests for Anthropic model functionality.
"""

import os
from unittest.mock import Mock, patch

import pytest

from novaeval.models.anthropic import AnthropicModel


class TestAnthropicModel:
    """Test cases for AnthropicModel class."""

    def test_init_default(self):
        """Test initialization with default parameters."""
        with patch("novaeval.models.anthropic.anthropic.Anthropic") as mock_anthropic:
            model = AnthropicModel()

            assert model.name == "anthropic_claude-3-sonnet-20240229"
            assert model.model_name == "claude-3-sonnet-20240229"
            assert model.max_retries == 3
            assert model.timeout == 60.0
            assert model.api_key is None
            assert model.base_url is None
            mock_anthropic.assert_called_once()

    def test_init_with_params(self):
        """Test initialization with custom parameters."""
        with patch("novaeval.models.anthropic.anthropic.Anthropic") as mock_anthropic:
            model = AnthropicModel(
                model_name="claude-3-opus-20240229",
                api_key="test_key",
                base_url="https://api.test.com",
                max_retries=5,
                timeout=30.0,
            )

            assert model.name == "anthropic_claude-3-opus-20240229"
            assert model.model_name == "claude-3-opus-20240229"
            assert model.api_key == "test_key"
            assert model.base_url == "https://api.test.com"
            assert model.max_retries == 5
            assert model.timeout == 30.0

            mock_anthropic.assert_called_once_with(
                api_key="test_key",
                base_url="https://api.test.com",
                max_retries=5,
                timeout=30.0,
            )

    def test_init_with_env_api_key(self):
        """Test initialization with API key from environment."""
        with (
            patch("novaeval.models.anthropic.anthropic.Anthropic") as mock_anthropic,
            patch.dict(os.environ, {"ANTHROPIC_API_KEY": "env_key"}),
        ):

            AnthropicModel()

            mock_anthropic.assert_called_once_with(
                api_key="env_key",
                base_url=None,
                max_retries=3,
                timeout=60.0,
            )

    def test_generate_success(self):
        """Test successful text generation."""
        with patch("novaeval.models.anthropic.anthropic.Anthropic") as mock_anthropic:
            # Mock the client and response
            mock_client = Mock()
            mock_anthropic.return_value = mock_client

            # Mock response
            mock_response = Mock()
            mock_response.content = [Mock(text="Generated response")]
            mock_response.usage = Mock(input_tokens=50, output_tokens=30)
            mock_client.messages.create.return_value = mock_response

            model = AnthropicModel()

            with patch.object(model, "estimate_cost", return_value=0.01) as mock_cost:
                result = model.generate("Test prompt")

                assert result == "Generated response"
                mock_client.messages.create.assert_called_once()
                mock_cost.assert_called_once_with("Test prompt", "Generated response")

                # Verify the API call parameters
                call_args = mock_client.messages.create.call_args
                assert call_args[1]["model"] == "claude-3-sonnet-20240229"
                assert call_args[1]["messages"] == [
                    {"role": "user", "content": "Test prompt"}
                ]
                assert call_args[1]["max_tokens"] == 1024

    def test_generate_with_params(self):
        """Test text generation with custom parameters."""
        with patch("novaeval.models.anthropic.anthropic.Anthropic") as mock_anthropic:
            mock_client = Mock()
            mock_anthropic.return_value = mock_client

            mock_response = Mock()
            mock_response.content = [Mock(text="Generated response")]
            mock_response.usage = Mock(input_tokens=50, output_tokens=30)
            mock_client.messages.create.return_value = mock_response

            model = AnthropicModel()

            with patch.object(model, "estimate_cost", return_value=0.01):
                result = model.generate(
                    "Test prompt",
                    max_tokens=500,
                    temperature=0.7,
                    stop=["END"],
                    custom_param="value",
                )

                assert result == "Generated response"

                # Verify the API call parameters
                call_args = mock_client.messages.create.call_args
                assert call_args[1]["max_tokens"] == 500
                assert call_args[1]["temperature"] == 0.7
                assert call_args[1]["stop_sequences"] == ["END"]
                assert call_args[1]["custom_param"] == "value"

    def test_generate_with_string_stop(self):
        """Test text generation with string stop sequence."""
        with patch("novaeval.models.anthropic.anthropic.Anthropic") as mock_anthropic:
            mock_client = Mock()
            mock_anthropic.return_value = mock_client

            mock_response = Mock()
            mock_response.content = [Mock(text="Generated response")]
            mock_response.usage = Mock(input_tokens=50, output_tokens=30)
            mock_client.messages.create.return_value = mock_response

            model = AnthropicModel()

            with patch.object(model, "estimate_cost", return_value=0.01):
                result = model.generate("Test prompt", stop="END")

                assert result == "Generated response"

                # Verify stop sequence is converted to list
                call_args = mock_client.messages.create.call_args
                assert call_args[1]["stop_sequences"] == ["END"]

    def test_generate_with_exception(self):
        """Test text generation with API exception."""
        with patch("novaeval.models.anthropic.anthropic.Anthropic") as mock_anthropic:
            mock_client = Mock()
            mock_anthropic.return_value = mock_client
            mock_client.messages.create.side_effect = Exception("API Error")

            model = AnthropicModel()

            with patch.object(model, "_handle_error") as mock_handle_error:
                with pytest.raises(Exception, match="API Error"):
                    model.generate("Test prompt")

                mock_handle_error.assert_called_once()

    def test_generate_batch_success(self):
        """Test successful batch text generation."""
        with patch("novaeval.models.anthropic.anthropic.Anthropic") as mock_anthropic:
            mock_client = Mock()
            mock_anthropic.return_value = mock_client

            # Mock responses for each prompt
            mock_responses = [
                Mock(
                    content=[Mock(text="Response 1")],
                    usage=Mock(input_tokens=10, output_tokens=5),
                ),
                Mock(
                    content=[Mock(text="Response 2")],
                    usage=Mock(input_tokens=12, output_tokens=6),
                ),
            ]
            mock_client.messages.create.side_effect = mock_responses

            model = AnthropicModel()

            with patch.object(model, "estimate_cost", return_value=0.01):
                results = model.generate_batch(["Prompt 1", "Prompt 2"])

                assert results == ["Response 1", "Response 2"]
                assert mock_client.messages.create.call_count == 2

    def test_generate_batch_with_exception(self):
        """Test batch generation with some failures."""
        with patch("novaeval.models.anthropic.anthropic.Anthropic") as mock_anthropic:
            mock_client = Mock()
            mock_anthropic.return_value = mock_client

            # First call succeeds, second fails
            mock_response = Mock(
                content=[Mock(text="Response 1")],
                usage=Mock(input_tokens=10, output_tokens=5),
            )
            mock_client.messages.create.side_effect = [
                mock_response,
                Exception("API Error"),
            ]

            model = AnthropicModel()

            with (
                patch.object(model, "estimate_cost", return_value=0.01),
                patch.object(model, "_handle_error") as mock_handle_error,
            ):

                results = model.generate_batch(["Prompt 1", "Prompt 2"])

                assert results == [
                    "Response 1",
                    "",
                ]  # Empty string for failed generation
                mock_handle_error.assert_called_once()

    def test_get_provider(self):
        """Test get_provider method."""
        with patch("novaeval.models.anthropic.anthropic.Anthropic"):
            model = AnthropicModel()
            assert model.get_provider() == "anthropic"

    def test_estimate_cost_known_model(self):
        """Test cost estimation for known model."""
        with patch("novaeval.models.anthropic.anthropic.Anthropic"):
            model = AnthropicModel(model_name="claude-3-sonnet-20240229")

            with patch.object(
                model, "count_tokens", side_effect=[100, 50]
            ):  # input, output tokens
                cost = model.estimate_cost("Test prompt", "Response")

                # claude-3-sonnet-20240229 pricing: $3.0 input, $15.0 output per 1M tokens
                expected_cost = (100 / 1_000_000) * 3.0 + (50 / 1_000_000) * 15.0
                assert cost == expected_cost

    def test_estimate_cost_unknown_model(self):
        """Test cost estimation for unknown model."""
        with patch("novaeval.models.anthropic.anthropic.Anthropic"):
            model = AnthropicModel(model_name="unknown-model")

            cost = model.estimate_cost("Test prompt", "Response")
            assert cost == 0.0

    def test_count_tokens_with_anthropic_api(self):
        """Test token counting using Anthropic API."""
        with patch("novaeval.models.anthropic.anthropic.Anthropic") as mock_anthropic:
            mock_client = Mock()
            mock_anthropic.return_value = mock_client

            mock_response = Mock(count=42)
            mock_client.count_tokens.return_value = mock_response

            model = AnthropicModel()

            count = model.count_tokens("Test text")
            assert count == 42
            mock_client.count_tokens.assert_called_once_with("Test text")

    def test_count_tokens_fallback(self):
        """Test token counting fallback when API fails."""
        with patch("novaeval.models.anthropic.anthropic.Anthropic") as mock_anthropic:
            mock_client = Mock()
            mock_anthropic.return_value = mock_client
            mock_client.count_tokens.side_effect = Exception("API Error")

            model = AnthropicModel()

            with patch.object(
                model.__class__.__bases__[0], "count_tokens", return_value=25
            ) as mock_fallback:
                count = model.count_tokens("Test text")
                assert count == 25
                mock_fallback.assert_called_once_with("Test text")

    def test_validate_connection_success(self):
        """Test successful connection validation."""
        with patch("novaeval.models.anthropic.anthropic.Anthropic") as mock_anthropic:
            mock_client = Mock()
            mock_anthropic.return_value = mock_client

            mock_response = Mock()
            mock_client.messages.create.return_value = mock_response

            model = AnthropicModel()

            result = model.validate_connection()
            assert result is True
            mock_client.messages.create.assert_called_once()

    def test_validate_connection_failure(self):
        """Test connection validation failure."""
        with patch("novaeval.models.anthropic.anthropic.Anthropic") as mock_anthropic:
            mock_client = Mock()
            mock_anthropic.return_value = mock_client
            mock_client.messages.create.side_effect = Exception("Connection failed")

            model = AnthropicModel()

            with patch.object(model, "_handle_error") as mock_handle_error:
                result = model.validate_connection()
                assert result is False
                mock_handle_error.assert_called_once()

    def test_get_info(self):
        """Test get_info method."""
        with patch("novaeval.models.anthropic.anthropic.Anthropic"):
            model = AnthropicModel(
                model_name="claude-3-opus-20240229", max_retries=5, timeout=30.0
            )

            info = model.get_info()

            assert info["name"] == "anthropic_claude-3-opus-20240229"
            assert info["model_name"] == "claude-3-opus-20240229"
            assert info["provider"] == "anthropic"
            assert info["max_retries"] == 5
            assert info["timeout"] == 30.0
            assert info["supports_batch"] is False
            assert info["pricing"] == (15.0, 75.0)  # Opus pricing

    def test_get_info_unknown_model(self):
        """Test get_info for unknown model."""
        with patch("novaeval.models.anthropic.anthropic.Anthropic"):
            model = AnthropicModel(model_name="unknown-model")

            info = model.get_info()
            assert info["pricing"] == (0, 0)

    def test_pricing_constants(self):
        """Test that pricing constants are defined for all models."""
        expected_models = [
            "claude-3-haiku-20240307",
            "claude-3-sonnet-20240229",
            "claude-3-opus-20240229",
            "claude-3-5-sonnet-20241022",
            "claude-2.1",
            "claude-2.0",
            "claude-instant-1.2",
        ]

        for model_name in expected_models:
            assert model_name in AnthropicModel.PRICING
            pricing = AnthropicModel.PRICING[model_name]
            assert isinstance(pricing, tuple)
            assert len(pricing) == 2
            assert all(isinstance(price, (int, float)) for price in pricing)

    def test_usage_tracking(self):
        """Test that usage is properly tracked."""
        with patch("novaeval.models.anthropic.anthropic.Anthropic") as mock_anthropic:
            mock_client = Mock()
            mock_anthropic.return_value = mock_client

            mock_response = Mock()
            mock_response.content = [Mock(text="Response")]
            mock_response.usage = Mock(input_tokens=100, output_tokens=50)
            mock_client.messages.create.return_value = mock_response

            model = AnthropicModel()

            with (
                patch.object(model, "estimate_cost", return_value=0.05),
                patch.object(model, "_track_request") as mock_track,
            ):

                model.generate("Test prompt")

                mock_track.assert_called_once_with(
                    prompt="Test prompt",
                    response="Response",
                    tokens_used=150,  # 100 + 50
                    cost=0.05,
                )

    def test_generate_no_usage_info(self):
        """Test generation when usage info is not available."""
        with patch("novaeval.models.anthropic.anthropic.Anthropic") as mock_anthropic:
            mock_client = Mock()
            mock_anthropic.return_value = mock_client

            mock_response = Mock()
            mock_response.content = [Mock(text="Response")]
            mock_response.usage = None  # No usage info
            mock_client.messages.create.return_value = mock_response

            model = AnthropicModel()

            with (
                patch.object(model, "estimate_cost", return_value=0.05),
                patch.object(model, "_track_request") as mock_track,
            ):

                result = model.generate("Test prompt")

                assert result == "Response"
                mock_track.assert_called_once_with(
                    prompt="Test prompt",
                    response="Response",
                    tokens_used=0,  # 0 when no usage info
                    cost=0.05,
                )

    def test_none_parameters_filtered(self):
        """Test that None parameters are filtered out."""
        with patch("novaeval.models.anthropic.anthropic.Anthropic") as mock_anthropic:
            mock_client = Mock()
            mock_anthropic.return_value = mock_client

            mock_response = Mock()
            mock_response.content = [Mock(text="Response")]
            mock_response.usage = Mock(input_tokens=10, output_tokens=5)
            mock_client.messages.create.return_value = mock_response

            model = AnthropicModel()

            with patch.object(model, "estimate_cost", return_value=0.01):
                model.generate("Test prompt", temperature=None, stop=None)

                call_args = mock_client.messages.create.call_args
                # None values should be filtered out
                assert "temperature" not in call_args[1]
                assert "stop_sequences" not in call_args[1]
