"""
Unit tests for AzureOpenAIModel functionality.
"""

import os
from unittest.mock import Mock, patch

import pytest


@pytest.mark.unit
class TestAzureOpenAIModel:
    """Test cases for AzureOpenAIModel class."""

    @patch.dict(
        os.environ,
        {
            "AZURE_OPENAI_API_KEY": "test_key",
            "AZURE_OPENAI_BASE_URL": "https://test.azure.com",
        },
    )
    def test_init_default(self):
        """Test initialization with default parameters."""
        from novaeval.models.azure_openai import AzureOpenAIModel
        with patch("novaeval.models.azure_openai.AzureOpenAI") as mock_client:
            model = AzureOpenAIModel()
            assert model.name == "azure_openai_gpt-4-8k"
            assert model.model_name == "gpt-4-8k"
            assert model.max_retries == 3
            assert model.timeout == 60.0
            assert model.api_version == "preview"
            mock_client.assert_called_once()

    def test_init_with_params(self):
        """Test initialization with custom parameters."""
        from novaeval.models.azure_openai import AzureOpenAIModel
        with patch("novaeval.models.azure_openai.AzureOpenAI") as mock_client:
            model = AzureOpenAIModel(
                model_name="gpt-3.5-turbo-0613",
                api_key="test_key",
                base_url="https://test.azure.com",
                max_retries=5,
                timeout=30.0,
                api_version="2024-01-01",
            )
            assert model.name == "azure_openai_gpt-3.5-turbo-0613"
            assert model.model_name == "gpt-3.5-turbo-0613"
            assert model.api_key == "test_key"
            assert model.base_url == "https://test.azure.com"
            assert model.max_retries == 5
            assert model.timeout == 30.0
            assert model.api_version == "2024-01-01"
            mock_client.assert_called_once_with(
                api_key="test_key",
                base_url="https://test.azure.com",
                api_version="2024-01-01",
            )

    @patch.dict(
        os.environ,
        {"AZURE_OPENAI_API_KEY": "env_key", "AZURE_OPENAI_BASE_URL": "env_base"},
    )
    def test_init_with_env_vars(self):
        """Test initialization using environment variables."""
        from novaeval.models.azure_openai import AzureOpenAIModel
        with patch("novaeval.models.azure_openai.AzureOpenAI") as mock_client:
            AzureOpenAIModel()
            mock_client.assert_called_once_with(
                api_key="env_key",
                base_url="env_base",
                api_version="preview",
            )

    def test_init_missing_api_key_raises(self, monkeypatch):
        """No api_key param + no AZURE_OPENAI_API_KEY env -> ValueError."""
        from novaeval.models.azure_openai import AzureOpenAIModel
        monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
        monkeypatch.setenv("AZURE_OPENAI_BASE_URL", "https://test.azure.com")
        with (
            patch("novaeval.models.azure_openai.AzureOpenAI"),
            pytest.raises(ValueError, match="API key is required"),
        ):
            AzureOpenAIModel(api_key=None)

    def test_init_missing_base_url_raises(self, monkeypatch):
        """No base_url param + no AZURE_OPENAI_BASE_URL env -> ValueError."""
        from novaeval.models.azure_openai import AzureOpenAIModel
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test_key")
        monkeypatch.delenv("AZURE_OPENAI_BASE_URL", raising=False)
        with (
            patch("novaeval.models.azure_openai.AzureOpenAI"),
            pytest.raises(ValueError, match="Base URL is required"),
        ):
            AzureOpenAIModel(base_url=None)

    def test_init_blank_api_key_raises(self):
        """Blank api_key trips validation."""
        from novaeval.models.azure_openai import AzureOpenAIModel
        with (
            patch("novaeval.models.azure_openai.AzureOpenAI"),
            pytest.raises(ValueError, match="API key is required"),
        ):
            AzureOpenAIModel(api_key="   ", base_url="https://test.azure.com")

    def test_init_client_failure_raises(self):
        """If AzureOpenAI blows up, we wrap in ValueError."""
        from novaeval.models.azure_openai import AzureOpenAIModel
        with (
            patch(
                "novaeval.models.azure_openai.AzureOpenAI",
                side_effect=RuntimeError("boom"),
            ),
            pytest.raises(ValueError, match="Failed to initialize Azure OpenAI client"),
        ):
            AzureOpenAIModel(api_key="test_key", base_url="https://test.azure.com")

    @patch.dict(
        os.environ,
        {
            "AZURE_OPENAI_API_KEY": "test_key",
            "AZURE_OPENAI_BASE_URL": "https://test.azure.com",
        },
    )
    def test_generate_success(self):
        """Test successful text generation."""
        from novaeval.models.azure_openai import AzureOpenAIModel
        with patch("novaeval.models.azure_openai.AzureOpenAI") as mock_client:
            mock_part = {"type": "output_text", "text": "Generated response"}
            mock_output_obj = Mock()
            mock_output_obj.content = [mock_part]
            mock_response = Mock()
            mock_response.output = [mock_output_obj]
            mock_response.usage = Mock(input_tokens=10, output_tokens=20)
            mock_client_instance = Mock()
            mock_client_instance.responses.create.return_value = mock_response
            mock_client.return_value = mock_client_instance
            model = AzureOpenAIModel()
            model.estimate_cost = Mock(return_value=0.01)
            response = model.generate("Test prompt")
            assert response == "Generated response"
            assert model.total_requests == 1
            mock_client_instance.responses.create.assert_called_once()
            call_args = mock_client_instance.responses.create.call_args
            assert call_args[1]["model"] == "gpt-4-8k"
            assert call_args[1]["input"] == "Test prompt"

    @patch.dict(
        os.environ,
        {
            "AZURE_OPENAI_API_KEY": "test_key",
            "AZURE_OPENAI_BASE_URL": "https://test.azure.com",
        },
    )
    def test_generate_with_params(self):
        """Test text generation with additional parameters."""
        from novaeval.models.azure_openai import AzureOpenAIModel
        with patch("novaeval.models.azure_openai.AzureOpenAI") as mock_client:
            mock_part = {"type": "output_text", "text": "Generated response"}
            mock_output_obj = Mock()
            mock_output_obj.content = [mock_part]
            mock_response = Mock()
            mock_response.output = [mock_output_obj]
            mock_response.usage = Mock(input_tokens=5, output_tokens=15)
            mock_client_instance = Mock()
            mock_client_instance.responses.create.return_value = mock_response
            mock_client.return_value = mock_client_instance
            model = AzureOpenAIModel()
            model.estimate_cost = Mock(return_value=0.005)
            response = model.generate(
                "Test prompt",
                max_tokens=100,
                temperature=0.5,
                stop=["<END>"],
                custom_param="value",
            )
            assert response == "Generated response"
            mock_client_instance.responses.create.assert_called_once_with(
                model="gpt-4-8k",
                input="Test prompt",
                max_output_tokens=100,
                temperature=0.5,
                stop=["<END>"],
                custom_param="value",
            )

    @patch.dict(
        os.environ,
        {
            "AZURE_OPENAI_API_KEY": "test_key",
            "AZURE_OPENAI_BASE_URL": "https://test.azure.com",
        },
    )
    def test_generate_empty_response(self):
        """Test generation with empty response."""
        from novaeval.models.azure_openai import AzureOpenAIModel
        with patch("novaeval.models.azure_openai.AzureOpenAI") as mock_client:
            mock_output_obj = Mock()
            mock_output_obj.content = []
            mock_response = Mock()
            mock_response.output = [mock_output_obj]
            mock_response.usage = Mock(input_tokens=5, output_tokens=0)
            mock_client_instance = Mock()
            mock_client_instance.responses.create.return_value = mock_response
            mock_client.return_value = mock_client_instance
            model = AzureOpenAIModel()
            model.estimate_cost = Mock(return_value=0.001)
            response = model.generate("Test prompt")
            assert response == ""

    @patch.dict(
        os.environ,
        {
            "AZURE_OPENAI_API_KEY": "test_key",
            "AZURE_OPENAI_BASE_URL": "https://test.azure.com",
        },
    )
    def test_generate_error_handling(self):
        """Test error handling during text generation."""
        from novaeval.models.azure_openai import AzureOpenAIModel
        with patch("novaeval.models.azure_openai.AzureOpenAI") as mock_client:
            mock_client_instance = Mock()
            mock_client_instance.responses.create.side_effect = Exception("API Error")
            mock_client.return_value = mock_client_instance
            model = AzureOpenAIModel()
            with pytest.raises(Exception, match="API Error"):
                model.generate("Test prompt")
            assert len(model.errors) == 1
            assert "Failed to generate text" in model.errors[0]

    @patch.dict(
        os.environ,
        {
            "AZURE_OPENAI_API_KEY": "test_key",
            "AZURE_OPENAI_BASE_URL": "https://test.azure.com",
        },
    )
    def test_generate_batch(self):
        """Test batch text generation."""
        from novaeval.models.azure_openai import AzureOpenAIModel
        with patch("novaeval.models.azure_openai.AzureOpenAI") as mock_client:
            mock_part = {"type": "output_text", "text": "Generated response"}
            mock_output_obj = Mock()
            mock_output_obj.content = [mock_part]
            mock_response = Mock()
            mock_response.output = [mock_output_obj]
            mock_response.usage = Mock(input_tokens=5, output_tokens=15)
            mock_client_instance = Mock()
            mock_client_instance.responses.create.return_value = mock_response
            mock_client.return_value = mock_client_instance
            model = AzureOpenAIModel()
            model.estimate_cost = Mock(return_value=0.005)
            prompts = ["Prompt 1", "Prompt 2", "Prompt 3"]
            responses = model.generate_batch(prompts)
            assert len(responses) == 3
            assert all(response == "Generated response" for response in responses)
            assert mock_client_instance.responses.create.call_count == 3

    @patch.dict(
        os.environ,
        {
            "AZURE_OPENAI_API_KEY": "test_key",
            "AZURE_OPENAI_BASE_URL": "https://test.azure.com",
        },
    )
    def test_generate_batch_with_error(self):
        """Test batch generation with errors."""
        from novaeval.models.azure_openai import AzureOpenAIModel
        with patch("novaeval.models.azure_openai.AzureOpenAI") as mock_client:
            mock_client_instance = Mock()
            mock_client_instance.responses.create.side_effect = Exception("API Error")
            mock_client.return_value = mock_client_instance
            model = AzureOpenAIModel()
            responses = model.generate_batch(["prompt1", "prompt2"])
            assert responses == ["", ""]
            assert len(model.errors) >= 2
            assert all("API Error" in error for error in model.errors)

    @patch.dict(
        os.environ,
        {
            "AZURE_OPENAI_API_KEY": "test_key",
            "AZURE_OPENAI_BASE_URL": "https://test.azure.com",
        },
    )
    def test_get_provider(self):
        """Test provider name."""
        from novaeval.models.azure_openai import AzureOpenAIModel
        with patch("novaeval.models.azure_openai.AzureOpenAI"):
            model = AzureOpenAIModel()
            assert model.get_provider() == "azure_openai"

    @patch.dict(
        os.environ,
        {
            "AZURE_OPENAI_API_KEY": "test_key",
            "AZURE_OPENAI_BASE_URL": "https://test.azure.com",
        },
    )
    def test_estimate_cost_known_model(self):
        """Test cost estimation for known model."""
        from novaeval.models.azure_openai import AzureOpenAIModel, MODEL_PRICING_PER_1M
        with patch("novaeval.models.azure_openai.AzureOpenAI"):
            model = AzureOpenAIModel(model_name="gpt-4")
            model.count_tokens = Mock(return_value=1000)
            prompt = "Test prompt"
            response = "Test response"
            cost = model.estimate_cost(prompt, response)
            input_price, output_price = MODEL_PRICING_PER_1M["gpt-4"]
            expected_cost = (1000 / 1_000_000) * input_price + (
                1000 / 1_000_000
            ) * output_price
            assert abs(cost - expected_cost) < 1e-2

    @patch.dict(
        os.environ,
        {
            "AZURE_OPENAI_API_KEY": "test_key",
            "AZURE_OPENAI_BASE_URL": "https://test.azure.com",
        },
    )
    def test_estimate_cost_unknown_model(self):
        """Test cost estimation for unknown model."""
        from novaeval.models.azure_openai import AzureOpenAIModel
        with patch("novaeval.models.azure_openai.AzureOpenAI"):
            model = AzureOpenAIModel(model_name="unknown-model")
            cost = model.estimate_cost("Test prompt", "Test response")
            assert cost == 0.0

    @patch.dict(
        os.environ,
        {
            "AZURE_OPENAI_API_KEY": "test_key",
            "AZURE_OPENAI_BASE_URL": "https://test.azure.com",
        },
    )
    def test_count_tokens(self):
        """Test token counting with actual implementation behavior."""
        from novaeval.models.azure_openai import AzureOpenAIModel
        with patch("novaeval.models.azure_openai.AzureOpenAI"):
            model = AzureOpenAIModel()
            test_cases = [
                ("Hello world", 4),
                ("This is a test", 8),
                ("The quick brown fox jumps", 10),
                ("Single", 2),
                ("", 0),
                ("One two three four five six seven eight nine ten", 20),
            ]
            for input_text, expected_tokens in test_cases:
                actual_tokens = model.count_tokens(input_text)
                assert isinstance(actual_tokens, int)
                assert actual_tokens == expected_tokens

    @patch.dict(
        os.environ,
        {
            "AZURE_OPENAI_API_KEY": "test_key",
            "AZURE_OPENAI_BASE_URL": "https://test.azure.com",
        },
    )
    def test_openai_fallback_in_init(self):
        """Test the else branch in __init__ for OpenAI fallback."""
        from novaeval.models.azure_openai import AzureOpenAIModel
        with (
            patch("novaeval.models.azure_openai._AZURE_OPENAI_AVAILABLE", False),
            patch("novaeval.models.azure_openai.OpenAI") as mock_openai,
        ):
            model = AzureOpenAIModel(
                api_key="test_key", base_url="https://test.azure.com"
            )
            mock_openai.assert_called_once()
            assert hasattr(model, "client")

    @patch.dict(
        os.environ,
        {
            "AZURE_OPENAI_API_KEY": "test_key",
            "AZURE_OPENAI_BASE_URL": "https://test.azure.com",
        },
    )
    def test_generate_raises_runtimeerror_on_missing_responses(self):
        """Test the else branch in generate for missing 'responses' attribute."""
        from novaeval.models.azure_openai import AzureOpenAIModel
        with patch("novaeval.models.azure_openai.AzureOpenAI") as mock_client:
            mock_client_instance = Mock()
            delattr(mock_client_instance, "responses")
            mock_client.return_value = mock_client_instance
            model = AzureOpenAIModel()
            model.client = Mock()
            delattr(model.client, "responses")
            with pytest.raises(
                RuntimeError, match="does not support the 'responses' endpoint"
            ):
                model.generate("Test prompt")

    @patch.dict(
        os.environ,
        {
            "AZURE_OPENAI_API_KEY": "test_key",
            "AZURE_OPENAI_BASE_URL": "https://test.azure.com",
        },
    )
    def test_validate_connection_else_branch(self):
        """Test the else branch in validate_connection for missing 'responses' attribute."""
        from novaeval.models.azure_openai import AzureOpenAIModel
        with patch("novaeval.models.azure_openai.AzureOpenAI") as mock_client:
            mock_client_instance = Mock()
            delattr(mock_client_instance, "responses")
            mock_client.return_value = mock_client_instance
            model = AzureOpenAIModel()
            model.client = Mock()
            delattr(model.client, "responses")
            with pytest.raises(
                RuntimeError, match="does not support the 'responses' endpoint"
            ):
                model.validate_connection()

    @patch.dict(
        os.environ,
        {
            "AZURE_OPENAI_API_KEY": "test_key",
            "AZURE_OPENAI_BASE_URL": "https://test.azure.com",
        },
    )
    def test_validate_connection_error_branch(self):
        """Test the error branch in validate_connection (simulate exception)."""
        from novaeval.models.azure_openai import AzureOpenAIModel
        with patch("novaeval.models.azure_openai.AzureOpenAI") as mock_client:
            mock_client_instance = Mock()
            mock_client_instance.responses.create.side_effect = Exception("fail")
            mock_client.return_value = mock_client_instance
            model = AzureOpenAIModel()
            model.client = mock_client_instance
            result = model.validate_connection()
            assert result is False
            assert len(model.errors) > 0
            assert "Connection test failed" in model.errors[-1]

    @patch.dict(
        os.environ,
        {
            "AZURE_OPENAI_API_KEY": "test_key",
            "AZURE_OPENAI_BASE_URL": "https://test.azure.com",
        },
    )
    def test_estimate_cost_zero_rates(self):
        """Test the 0.0 cost branch in estimate_cost."""
        from novaeval.models.azure_openai import AzureOpenAIModel
        with patch("novaeval.models.azure_openai.AzureOpenAI"):
            model = AzureOpenAIModel(model_name="unknown-model")
            cost = model.estimate_cost(
                "prompt", "response", input_tokens=10, output_tokens=10
            )
            assert cost == 0.0

    @patch.dict(
        os.environ,
        {
            "AZURE_OPENAI_API_KEY": "test_key",
            "AZURE_OPENAI_BASE_URL": "https://test.azure.com",
        },
    )
    def test_validate_connection_success(self):
        """Test successful connection validation."""
        from novaeval.models.azure_openai import AzureOpenAIModel
        with patch("novaeval.models.azure_openai.AzureOpenAI") as mock_client:
            mock_response = Mock()
            mock_response.output = [[{"type": "output_text", "text": "Pong"}]]
            mock_client_instance = Mock()
            mock_client_instance.responses.create.return_value = mock_response
            mock_client.return_value = mock_client_instance
            model = AzureOpenAIModel()
            assert model.validate_connection() is True
            mock_client_instance.responses.create.assert_called_once()
            call_args = mock_client_instance.responses.create.call_args
            assert call_args[1]["input"] == "Ping!"

    @patch.dict(
        os.environ,
        {
            "AZURE_OPENAI_API_KEY": "test_key",
            "AZURE_OPENAI_BASE_URL": "https://test.azure.com",
        },
    )
    def test_validate_connection_failure(self):
        """Test connection validation failure."""
        from novaeval.models.azure_openai import AzureOpenAIModel
        with patch("novaeval.models.azure_openai.AzureOpenAI") as mock_client:
            mock_client_instance = Mock()
            mock_client_instance.responses.create.side_effect = Exception(
                "Connection failed"
            )
            mock_client.return_value = mock_client_instance
            model = AzureOpenAIModel()
            result = model.validate_connection()
            assert result is False
            assert len(model.errors) == 1
            assert "Connection test failed" in model.errors[0]

    @patch.dict(
        os.environ,
        {
            "AZURE_OPENAI_API_KEY": "test_key",
            "AZURE_OPENAI_BASE_URL": "https://test.azure.com",
        },
    )
    def test_validate_connection_empty_response(self):
        """Test connection validation with empty response."""
        from novaeval.models.azure_openai import AzureOpenAIModel
        with patch("novaeval.models.azure_openai.AzureOpenAI") as mock_client:
            mock_response = Mock()
            mock_response.output = []
            mock_client_instance = Mock()
            mock_client_instance.responses.create.return_value = mock_response
            mock_client.return_value = mock_client_instance
            model = AzureOpenAIModel()
            result = model.validate_connection()
            assert result is False

    @patch.dict(
        os.environ,
        {
            "AZURE_OPENAI_API_KEY": "test_key",
            "AZURE_OPENAI_BASE_URL": "https://test.azure.com",
        },
    )
    def test_get_info(self):
        """Test model info retrieval."""
        from novaeval.models.azure_openai import AzureOpenAIModel
        with patch("novaeval.models.azure_openai.AzureOpenAI"):
            model = AzureOpenAIModel(model_name="gpt-4")
            info = model.get_info()
            assert info["name"] == "azure_openai_gpt-4"
            assert info["model_name"] == "gpt-4"
            assert info["provider"] == "azure_openai"
            assert info["type"] == "AzureOpenAIModel"
            assert info["max_retries"] == 3
            assert info["timeout"] == 60.0
            assert info["supports_batch"] is False
            assert info["pricing"] == (30.0, 60.0)

    def test_get_info_unknown_model(self):
        """Test model info retrieval for unknown model."""
        from novaeval.models.azure_openai import AzureOpenAIModel
        with patch("novaeval.models.azure_openai.AzureOpenAI"):
            model = AzureOpenAIModel(
                model_name="unknown-model",
                api_key="test_key",
                base_url="https://test.azure.com",
            )
            info = model.get_info()
            assert info["pricing"] == (0, 0)

    def test_pricing_constants(self):
        """Test that pricing constants are defined correctly."""
        from novaeval.models.azure_openai import MODEL_PRICING_PER_1M
        assert "gpt-4" in MODEL_PRICING_PER_1M
        assert "gpt-4-turbo" in MODEL_PRICING_PER_1M
        assert "gpt-3.5-turbo-0301" in MODEL_PRICING_PER_1M
        assert "gpt-3.5-turbo-0613" in MODEL_PRICING_PER_1M
        assert "gpt-3.5-turbo-1106" in MODEL_PRICING_PER_1M
        assert "gpt-3.5-turbo-0125" in MODEL_PRICING_PER_1M
        assert "gpt-3.5-turbo-instruct" in MODEL_PRICING_PER_1M
        assert "o3" in MODEL_PRICING_PER_1M
        assert "o4-mini" in MODEL_PRICING_PER_1M
        assert "gpt-4.1" in MODEL_PRICING_PER_1M
        assert "gpt-4.1mini" in MODEL_PRICING_PER_1M
        assert "gpt-4.1-nano" in MODEL_PRICING_PER_1M
        for _model_name, pricing in MODEL_PRICING_PER_1M.items():
            assert len(pricing) == 2
            assert isinstance(pricing[0], (int, float))
            assert isinstance(pricing[1], (int, float))

    @patch.dict(
        os.environ,
        {"AZURE_OPENAI_API_KEY": "env_key", "AZURE_OPENAI_BASE_URL": "env_base"},
    )
    def test_create_from_config_roundtrip(self):
        """create_from_config builds an AzureOpenAIModel w/ defaults + extra kwargs."""
        from novaeval.models.azure_openai import AzureOpenAIModel
        cfg = {
            "model_name": "gpt-4-turbo",
            # api_key omitted on purpose -> picked up from env
            "base_url": "env_base",
            "max_retries": 7,
            "timeout": 12.5,
            "api_version": "2024-01-01",
            "foo": "bar",  # extra kw to prove passthrough
        }
        with patch("novaeval.models.azure_openai.AzureOpenAI") as mock_client:
            model = AzureOpenAIModel.create_from_config(cfg)
            mock_client.assert_called_once_with(
                api_key="env_key", base_url="env_base", api_version="2024-01-01"
            )
            assert isinstance(model, AzureOpenAIModel)
            assert model.model_name == "gpt-4-turbo"
            assert model.max_retries == 7
            assert model.timeout == 12.5
            assert model.api_version == "2024-01-01"
            assert "foo" in model.kwargs

    def test_importerror_on_openai_import_sets_flag_and_fallback(self):
        """Simulate ImportError for openai.AzureOpenAI and ensure fallback to OpenAI is used."""
        import importlib
        import sys
        import builtins
        real_import = builtins.__import__
        def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "openai" and "AzureOpenAI" in fromlist:
                raise ImportError("No AzureOpenAI!")
            return real_import(name, globals, locals, fromlist, level)
        with patch("builtins.__import__", side_effect=fake_import):
            import novaeval.models.azure_openai as mod
            importlib.reload(mod)
            assert getattr(mod, "_AZURE_OPENAI_AVAILABLE") is False
            with patch.object(mod, "OpenAI") as mock_openai:
                model = getattr(mod, "AzureOpenAIModel")(api_key="test_key", base_url="https://test.azure.com")
                mock_openai.assert_called_once()
                assert hasattr(model, "client")

    @patch.dict(
        os.environ,
        {
            "AZURE_OPENAI_API_KEY": "test_key",
            "AZURE_OPENAI_BASE_URL": "https://test.azure.com",
        },
    )
    def test_get_rates_high_tier_branch(self):
        """Test _get_rates returns high_rates when tokens > cutoff for tiered model."""
        from novaeval.models.azure_openai import AzureOpenAIModel
        with patch("novaeval.models.azure_openai.AzureOpenAI"):
            model = AzureOpenAIModel(model_name="gpt-4")
            # gpt-4 cutoff is 8000, so use 9000
            high_rates = model._get_rates(5000, 4000)
            assert high_rates == (60.0, 120.0)

    @patch.dict(
        os.environ,
        {
            "AZURE_OPENAI_API_KEY": "test_key",
            "AZURE_OPENAI_BASE_URL": "https://test.azure.com",
        },
    )
    def test_get_rates_non_tiered_model(self):
        """Test _get_rates returns MODEL_PRICING_PER_1M for non-tiered model."""
        from novaeval.models.azure_openai import AzureOpenAIModel
        with patch("novaeval.models.azure_openai.AzureOpenAI"):
            model = AzureOpenAIModel(model_name="gpt-4-turbo")
            rates = model._get_rates(10, 10)
            assert rates == (10.0, 30.0)

    @patch.dict(
        os.environ,
        {
            "AZURE_OPENAI_API_KEY": "test_key",
            "AZURE_OPENAI_BASE_URL": "https://test.azure.com",
        },
    )
    def test_estimate_cost_zero_rates_branch(self):
        """Test estimate_cost returns 0.0 when _get_rates returns (0.0, 0.0)."""
        from novaeval.models.azure_openai import AzureOpenAIModel
        with patch("novaeval.models.azure_openai.AzureOpenAI"):
            model = AzureOpenAIModel(model_name="nonexistent-model")
            cost = model.estimate_cost(
                "prompt", "response", input_tokens=10, output_tokens=10
            )
            assert cost == 0.0

    @patch.dict(
        os.environ,
        {
            "AZURE_OPENAI_API_KEY": "test_key",
            "AZURE_OPENAI_BASE_URL": "https://test.azure.com",
        },
    )
    def test_get_rates_high_tier_else_branch(self):
        """Test _get_rates returns high_rates (else branch) when tokens > cutoff for tiered model."""
        from novaeval.models.azure_openai import AzureOpenAIModel
        with patch("novaeval.models.azure_openai.AzureOpenAI"):
            model = AzureOpenAIModel(model_name="gpt-4")
            # gpt-4 cutoff is 8000, so use 9000 to trigger else branch
            rates = model._get_rates(8000, 1000)
            assert rates == (60.0, 120.0)

    @patch.dict(
        os.environ,
        {
            "AZURE_OPENAI_API_KEY": "test_key",
            "AZURE_OPENAI_BASE_URL": "https://test.azure.com",
        },
    )
    def test_generate_empty_response_output(self):
        """Test generate() when response.output is empty (line 181 coverage)."""
        from novaeval.models.azure_openai import AzureOpenAIModel
        model = AzureOpenAIModel()
        model.estimate_cost = Mock(return_value=0.001)
        mock_response = Mock()
        mock_response.output = []
        mock_response.usage = Mock(input_tokens=5, output_tokens=0)
        model.client.responses = Mock()
        model.client.responses.create = Mock(return_value=mock_response)
        response = model.generate("Test prompt")
        assert response == ""

    @patch.dict(
        os.environ,
        {
            "AZURE_OPENAI_API_KEY": "test_key",
            "AZURE_OPENAI_BASE_URL": "https://test.azure.com",
        },
    )
    def test_generate_empty_content(self):
        """Test generate() when response.output[0].content is empty (line 183 coverage)."""
        from novaeval.models.azure_openai import AzureOpenAIModel
        model = AzureOpenAIModel()
        model.estimate_cost = Mock(return_value=0.001)
        mock_output_obj = Mock()
        mock_output_obj.content = []
        mock_response = Mock()
        mock_response.output = [mock_output_obj]
        mock_response.usage = Mock(input_tokens=5, output_tokens=0)
        model.client.responses = Mock()
        model.client.responses.create = Mock(return_value=mock_response)
        response = model.generate("Test prompt")
        assert response == ""
