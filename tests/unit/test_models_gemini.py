"""
Unit tests for Gemini model functionality.
"""

import os
from unittest.mock import Mock, patch

import pytest

from novaeval.models.gemini import GeminiModel


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
                model_name="gemini-2.0-flash",
                api_key="abc123",
                max_retries=5,
                timeout=30.0,
            )
            assert model.name == "gemini_gemini-2.0-flash"
            assert model.api_key == "abc123"
            assert model.max_retries == 5
            assert model.timeout == 30.0
            mock_client.assert_called_once_with(api_key="abc123")

    @patch.dict(os.environ, {"GOOGLE_API_KEY": "env_key"})
    def test_init_with_env_key(self):
        """Test initialization using environment variable for API key."""
        with patch("novaeval.models.gemini.genai.Client") as mock_client:
            GeminiModel()
            mock_client.assert_called_once_with(api_key="env_key")

    def test_generate_success(self):
        """Test successful text generation."""
        with patch("novaeval.models.gemini.genai.Client") as mock_client:
            mock_response = Mock()
            mock_response.text = "Test response"
            mock_gen = mock_client.return_value.models.generate_content
            mock_gen.return_value = mock_response

            model = GeminiModel()
            model.estimate_cost = Mock(return_value=0.01)

            response = model.generate("Hello")

            assert response == "Test response"
            assert model.total_requests == 1
            assert model.total_tokens > 0
            assert model.total_cost == 0.01
            mock_gen.assert_called_once()

    def test_generate_error(self):
        """Test error during generation."""
        with patch("novaeval.models.gemini.genai.Client") as mock_client:
            mock_gen = mock_client.return_value.models.generate_content
            mock_gen.side_effect = Exception("API Error")

            model = GeminiModel()

            with pytest.raises(Exception, match="API Error"):
                model.generate("Hello")

            assert len(model.errors) == 1
            assert "Failed to generate text" in model.errors[0]

    def test_generate_batch(self):
        """Test batch generation."""
        with patch("novaeval.models.gemini.genai.Client") as mock_client:
            mock_response = Mock()
            mock_response.text = "Batch response"
            mock_gen = mock_client.return_value.models.generate_content
            mock_gen.return_value = mock_response

            model = GeminiModel()
            model.estimate_cost = Mock(return_value=0.01)

            prompts = ["Prompt 1", "Prompt 2"]
            results = model.generate_batch(prompts)

            assert results == ["Batch response", "Batch response"]
            assert model.total_requests == 2
            assert mock_gen.call_count == 2

    def test_generate_batch_with_error(self):
        """Test batch generation with an error."""
        with patch("novaeval.models.gemini.genai.Client") as mock_client:
            mock_gen = mock_client.return_value.models.generate_content
            mock_gen.side_effect = Exception("Batch fail")

            model = GeminiModel()
            results = model.generate_batch(["P1", "P2"])

            assert results == ["", ""]
            assert len(model.errors) == 2
            assert all("Batch failure" in e for e in model.errors)

    def test_estimate_cost_known_model(self):
        """Test cost estimation with known pricing."""
        with patch("novaeval.models.gemini.genai.Client"):
            model = GeminiModel(model_name="gemini-1.5-flash")
            model.count_tokens = Mock(return_value=1000)

            cost = model.estimate_cost("Prompt", "Resp")
            expected = (1000 / 1000) * 0.075 + (1000 / 1000) * 0.30
            assert abs(cost - expected) < 1e-6

    def test_estimate_cost_unknown_model(self):
        """Test cost estimation with unknown model."""
        with patch("novaeval.models.gemini.genai.Client"):
            model = GeminiModel(model_name="unknown-model")
            cost = model.estimate_cost("Prompt", "Resp")
            assert cost == 0.0

    def test_count_tokens(self):
        """Test token estimation heuristic."""
        with patch("novaeval.models.gemini.genai.Client"):
            model = GeminiModel()
            assert isinstance(model.count_tokens("This is a test"), int)

    def test_validate_connection_success(self):
        """Test successful connection validation."""
        with patch("novaeval.models.gemini.genai.Client") as mock_client:
            mock_response = Mock()
            mock_response.text = "pong"
            mock_client.return_value.models.generate_content.return_value = (
                mock_response
            )

            model = GeminiModel()
            assert model.validate_connection() is True

    def test_validate_connection_failure(self):
        """Test failed connection validation."""
        with patch("novaeval.models.gemini.genai.Client") as mock_client:
            mock_client.return_value.models.generate_content.side_effect = Exception(
                "fail"
            )

            model = GeminiModel()
            result = model.validate_connection()

            assert result is False
            assert "Connection test failed" in model.errors[0]

    def test_get_provider(self):
        with patch("novaeval.models.gemini.genai.Client"):
            model = GeminiModel()
            assert model.get_provider() == "gemini"

    def test_get_info(self):
        with patch("novaeval.models.gemini.genai.Client"):
            model = GeminiModel()
            info = model.get_info()
            assert info["name"].startswith("gemini_")
            assert info["provider"] == "gemini"
            assert "pricing" in info

    def test_pricing_constants(self):
        """Test pricing constants are defined correctly."""
        assert "gemini-2.5-pro" in GeminiModel.PRICING
        assert isinstance(GeminiModel.PRICING["gemini-2.5-pro"], tuple)
