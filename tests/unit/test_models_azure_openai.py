"""
Unit tests for AzureOpenAIModel functionality.
"""

import os
from unittest.mock import Mock, patch

import pytest


@pytest.fixture
def mock_azure_env():
    """Fixture to set required environment variables for Azure OpenAI."""
    with patch.dict(
        os.environ,
        {
            "AZURE_OPENAI_API_KEY": "test_key",
            "AZURE_OPENAI_BASE_URL": "https://test.azure.com",
            "AZURE_OPENAI_API_VERSION": "2024-02-01",
            "AZURE_OPENAI_DEPLOYMENT": "gpt-4",
        },
    ):
        yield


@pytest.mark.unit
class TestAzureOpenAIModel:
    """Test cases for AzureOpenAIModel class."""

    def test_init_with_params(self):
        """Test initialization with custom parameters."""
        from novaeval.models.azure_openai import AzureOpenAIModel

        with patch("novaeval.models.azure_openai.AzureOpenAI") as mock_client:
            model = AzureOpenAIModel(
                model_name="gpt-3.5-turbo-0613",
                api_key="custom_key",
                base_url="https://custom.azure.com",
                api_version="2024-01-01",
                max_retries=5,
                timeout=30.0,
            )
            assert model.name == "azure_openai_gpt-3.5-turbo-0613"
            assert model.model_name == "gpt-3.5-turbo-0613"
            assert model.api_key == "custom_key"
            assert model.base_url == "https://custom.azure.com"
            assert model.max_retries == 5
            assert model.timeout == 30.0
            assert model.api_version == "2024-01-01"
            mock_client.assert_called_once_with(
                api_key="custom_key",
                azure_endpoint="https://custom.azure.com",
                api_version="2024-01-01",
            )

    def test_init_with_env_vars(self, mock_azure_env):
        """Test initialization using environment variables."""
        from novaeval.models.azure_openai import AzureOpenAIModel

        with patch("novaeval.models.azure_openai.AzureOpenAI") as mock_client:
            model = AzureOpenAIModel()
            assert model.model_name == "gpt-4"
            assert model.api_key == "test_key"
            assert model.base_url == "https://test.azure.com"
            assert model.api_version == "2024-02-01"
            mock_client.assert_called_once_with(
                api_key="test_key",
                azure_endpoint="https://test.azure.com",
                api_version="2024-02-01",
            )

    def test_init_missing_deployment_raises(self, monkeypatch):
        """No model_name param + no AZURE_OPENAI_DEPLOYMENT env -> ValueError."""
        from novaeval.models.azure_openai import AzureOpenAIModel

        monkeypatch.delenv("AZURE_OPENAI_DEPLOYMENT", raising=False)
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test_key")
        monkeypatch.setenv("AZURE_OPENAI_BASE_URL", "https://test.azure.com")
        monkeypatch.setenv("AZURE_OPENAI_API_VERSION", "2024-02-01")
        with pytest.raises(ValueError, match="Deployment name is required"):
            AzureOpenAIModel(model_name=None)

    def test_init_missing_api_key_raises(self, monkeypatch):
        """No api_key param + no AZURE_OPENAI_API_KEY env -> ValueError."""
        from novaeval.models.azure_openai import AzureOpenAIModel

        monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
        monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4")
        monkeypatch.setenv("AZURE_OPENAI_BASE_URL", "https://test.azure.com")
        monkeypatch.setenv("AZURE_OPENAI_API_VERSION", "2024-02-01")
        with pytest.raises(ValueError, match="API key is required"):
            AzureOpenAIModel(api_key=None)

    def test_init_missing_base_url_raises(self, monkeypatch):
        """No base_url param + no AZURE_OPENAI_BASE_URL env -> ValueError."""
        from novaeval.models.azure_openai import AzureOpenAIModel

        monkeypatch.delenv("AZURE_OPENAI_BASE_URL", raising=False)
        monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4")
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test_key")
        monkeypatch.setenv("AZURE_OPENAI_API_VERSION", "2024-02-01")
        with pytest.raises(ValueError, match="Base URL is required"):
            AzureOpenAIModel(base_url=None)

    def test_init_missing_api_version_raises(self, monkeypatch):
        """No api_version param + no AZURE_OPENAI_API_VERSION env -> ValueError."""
        from novaeval.models.azure_openai import AzureOpenAIModel

        monkeypatch.delenv("AZURE_OPENAI_API_VERSION", raising=False)
        monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4")
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test_key")
        monkeypatch.setenv("AZURE_OPENAI_BASE_URL", "https://test.azure.com")
        with pytest.raises(ValueError, match="API version is required"):
            AzureOpenAIModel(api_version=None)

    def test_init_blank_api_key_raises(self, mock_azure_env):
        """Blank api_key trips validation."""
        from novaeval.models.azure_openai import AzureOpenAIModel

        with pytest.raises(ValueError, match="API key is required"):
            AzureOpenAIModel(api_key="   ")

    def test_init_client_failure_raises(self, mock_azure_env):
        """If AzureOpenAI blows up, we wrap in ValueError."""
        from novaeval.models.azure_openai import AzureOpenAIModel

        with (
            patch(
                "novaeval.models.azure_openai.AzureOpenAI",
                side_effect=RuntimeError("boom"),
            ),
            pytest.raises(ValueError, match="Failed to initialize Azure OpenAI client"),
        ):
            AzureOpenAIModel()

    def test_generate_chat_success(self, mock_azure_env):
        """Test successful chat generation."""
        from novaeval.models.azure_openai import AzureOpenAIModel

        with patch("novaeval.models.azure_openai.AzureOpenAI") as mock_client:
            mock_choice = Mock()
            mock_choice.message.content = "Chat response"
            mock_chat_response = Mock()
            mock_chat_response.choices = [mock_choice]
            mock_chat_response.usage = Mock(prompt_tokens=5, completion_tokens=7)
            mock_client_instance = Mock()
            mock_client_instance.chat.completions.create.return_value = (
                mock_chat_response
            )
            mock_client.return_value = mock_client_instance
            model = AzureOpenAIModel()
            messages = [{"role": "user", "content": "Hello"}]
            response = model.generate_chat(messages=messages)
            assert response == "Chat response"
            mock_client_instance.chat.completions.create.assert_called_once_with(
                model="gpt-4", messages=messages
            )

    def test_generate_delegates_to_generate_chat(self, mock_azure_env):
        """Test that generate() calls generate_chat() when messages are provided."""
        from novaeval.models.azure_openai import AzureOpenAIModel

        with patch("novaeval.models.azure_openai.AzureOpenAI"):
            model = AzureOpenAIModel()
            model.generate_chat = Mock(return_value="Delegated response")
            messages = [{"role": "user", "content": "Test"}]
            response = model.generate(messages=messages)
            assert response == "Delegated response"
            model.generate_chat.assert_called_once_with(
                messages=messages, max_tokens=None, temperature=None, stop=None
            )

    def test_generate_legacy_prompt_removed(self, mock_azure_env):
        """Test that legacy prompt endpoint is not supported and falls back to chat."""
        from novaeval.models.azure_openai import AzureOpenAIModel

        with patch("novaeval.models.azure_openai.AzureOpenAI") as mock_client:
            mock_choice = Mock()
            mock_choice.message.content = "Prompt as chat"
            mock_chat_response = Mock()
            mock_chat_response.choices = [mock_choice]
            mock_chat_response.usage = Mock(prompt_tokens=3, completion_tokens=2)
            mock_client_instance = Mock()
            mock_client_instance.chat.completions.create.return_value = (
                mock_chat_response
            )
            mock_client.return_value = mock_client_instance
            model = AzureOpenAIModel()
            response = model.generate(prompt="Test prompt")
            assert response == "Prompt as chat"
            mock_client_instance.chat.completions.create.assert_called_once()
            call_args = mock_client_instance.chat.completions.create.call_args[1]
            assert call_args["model"] == "gpt-4"
            assert call_args["messages"][0]["content"] == "Test prompt"

    def test_generate_chat_error_handling(self, mock_azure_env):
        """Test error handling during chat generation."""
        from novaeval.models.azure_openai import AzureOpenAIModel

        with patch("novaeval.models.azure_openai.AzureOpenAI") as mock_client:
            mock_client_instance = Mock()
            mock_client_instance.chat.completions.create.side_effect = Exception(
                "API Error"
            )
            mock_client.return_value = mock_client_instance
            model = AzureOpenAIModel()
            with pytest.raises(Exception, match="API Error"):
                model.generate_chat(messages=[{"role": "user", "content": "Hello"}])
            assert len(model.errors) == 1
            assert "Failed to generate chat completion" in model.errors[0]

    def test_generate_batch(self, mock_azure_env):
        """Test batch text generation."""
        from novaeval.models.azure_openai import AzureOpenAIModel

        with patch("novaeval.models.azure_openai.AzureOpenAI"):
            model = AzureOpenAIModel()
            model.generate = Mock(side_effect=["Response 1", "Response 2"])
            prompts = ["Prompt 1", "Prompt 2"]
            responses = model.generate_batch(
                prompts, max_tokens=10, temperature=0.5, stop="."
            )
            assert responses == ["Response 1", "Response 2"]
            assert model.generate.call_count == 2
            model.generate.assert_any_call(
                prompt="Prompt 1", max_tokens=10, temperature=0.5, stop="."
            )
            model.generate.assert_any_call(
                prompt="Prompt 2", max_tokens=10, temperature=0.5, stop="."
            )

    def test_get_provider(self, mock_azure_env):
        """Test provider name."""
        from novaeval.models.azure_openai import AzureOpenAIModel

        with patch("novaeval.models.azure_openai.AzureOpenAI"):
            model = AzureOpenAIModel()
            assert model.get_provider() == "azure_openai"

    def test_estimate_cost_non_tiered_model(self, mock_azure_env):
        """Test cost estimation for a non-tiered model."""
        from novaeval.models.azure_openai import MODEL_PRICING_PER_1M, AzureOpenAIModel

        with patch("novaeval.models.azure_openai.AzureOpenAI"):
            model = AzureOpenAIModel(model_name="gpt-4-turbo")
            in_rate, out_rate = MODEL_PRICING_PER_1M["gpt-4-turbo"]
            cost = model.estimate_cost("", "", input_tokens=1000, output_tokens=2000)
            expected_cost = (1000 / 1_000_000) * in_rate + (2000 / 1_000_000) * out_rate
            assert abs(cost - expected_cost) < 1e-9

    def test_estimate_cost_tiered_model_low_tier(self, mock_azure_env):
        """Test cost estimation for a tiered model in the low tier."""
        from novaeval.models.azure_openai import TIERED_MODELS, AzureOpenAIModel

        with patch("novaeval.models.azure_openai.AzureOpenAI"):
            model = AzureOpenAIModel(model_name="gpt-4")
            _cutoff, low_rates, _high_rates = TIERED_MODELS["gpt-4"]
            cost = model.estimate_cost("", "", input_tokens=1000, output_tokens=2000)
            expected_cost = (1000 / 1_000_000) * low_rates[0] + (
                2000 / 1_000_000
            ) * low_rates[1]
            assert abs(cost - expected_cost) < 1e-9

    def test_estimate_cost_tiered_model_high_tier(self, mock_azure_env):
        """Test cost estimation for a tiered model in the high tier."""
        from novaeval.models.azure_openai import TIERED_MODELS, AzureOpenAIModel

        with patch("novaeval.models.azure_openai.AzureOpenAI"):
            model = AzureOpenAIModel(model_name="gpt-4")
            _cutoff, _low_rates, high_rates = TIERED_MODELS["gpt-4"]
            cost = model.estimate_cost("", "", input_tokens=5000, output_tokens=4000)
            expected_cost = (5000 / 1_000_000) * high_rates[0] + (
                4000 / 1_000_000
            ) * high_rates[1]
            assert abs(cost - expected_cost) < 1e-9

    def test_estimate_cost_unknown_model(self, mock_azure_env):
        """Test cost estimation for unknown model returns 0."""
        from novaeval.models.azure_openai import AzureOpenAIModel

        with patch("novaeval.models.azure_openai.AzureOpenAI"):
            model = AzureOpenAIModel(model_name="unknown-model")
            cost = model.estimate_cost("prompt", "response")
            assert cost == 0.0

    def test_count_tokens(self, mock_azure_env):
        """Test token counting approximation."""
        from novaeval.models.azure_openai import AzureOpenAIModel

        with patch("novaeval.models.azure_openai.AzureOpenAI"):
            model = AzureOpenAIModel()
            assert model.count_tokens("Hello world") == 4
            assert model.count_tokens("") == 0

    def test_validate_connection_success(self, mock_azure_env):
        """Test successful connection validation."""
        from novaeval.models.azure_openai import AzureOpenAIModel

        with patch("novaeval.models.azure_openai.AzureOpenAI") as mock_client:
            mock_choice = Mock()
            mock_choice.message.content = "Pong"
            mock_response = Mock()
            mock_response.choices = [mock_choice]
            mock_client_instance = Mock()
            mock_client_instance.chat.completions.create.return_value = mock_response
            mock_client.return_value = mock_client_instance
            model = AzureOpenAIModel()
            assert model.validate_connection() is True
            mock_client_instance.chat.completions.create.assert_called_once_with(
                model="gpt-4",
                messages=[{"role": "user", "content": "Ping!"}],
                max_tokens=1,
            )

    def test_validate_connection_failure(self, mock_azure_env):
        """Test connection validation failure on API error."""
        from novaeval.models.azure_openai import AzureOpenAIModel

        with patch("novaeval.models.azure_openai.AzureOpenAI") as mock_client:
            mock_client_instance = Mock()
            mock_client_instance.chat.completions.create.side_effect = Exception(
                "Connection failed"
            )
            mock_client.return_value = mock_client_instance
            model = AzureOpenAIModel()
            result = model.validate_connection()
            assert result is False
            assert len(model.errors) == 1
            assert "Connection test failed" in model.errors[0]

    def test_validate_connection_empty_response(self, mock_azure_env):
        """Test connection validation with empty response."""
        from novaeval.models.azure_openai import AzureOpenAIModel

        with patch("novaeval.models.azure_openai.AzureOpenAI") as mock_client:
            mock_response = Mock()
            mock_response.choices = []
            mock_client_instance = Mock()
            mock_client_instance.chat.completions.create.return_value = mock_response
            mock_client.return_value = mock_client_instance
            model = AzureOpenAIModel()
            result = model.validate_connection()
            assert result is False

    def test_get_info(self, mock_azure_env):
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

    def test_create_from_config_roundtrip(self):
        """create_from_config builds an AzureOpenAIModel w/ env vars + kwargs."""
        from novaeval.models.azure_openai import AzureOpenAIModel

        with patch.dict(
            os.environ,
            {
                "AZURE_OPENAI_API_KEY": "env_key",
                "AZURE_OPENAI_API_VERSION": "env_version",
            },
        ):
            cfg = {
                "model_name": "gpt-4-turbo",
                "base_url": "https://config.azure.com",
                "max_retries": 7,
                "timeout": 12.5,
                "foo": "bar",
            }
            with patch("novaeval.models.azure_openai.AzureOpenAI") as mock_client:
                model = AzureOpenAIModel.create_from_config(cfg)
                mock_client.assert_called_once_with(
                    api_key="env_key",
                    azure_endpoint="https://config.azure.com",
                    api_version="env_version",
                )
                assert isinstance(model, AzureOpenAIModel)
                assert model.model_name == "gpt-4-turbo"
                assert model.max_retries == 7
                assert model.timeout == 12.5
                assert "foo" in model.kwargs

    def test_canonical_model_name(self):
        from novaeval.models.azure_openai import _canonical_model_name

        assert _canonical_model_name("gpt-4-8k") == "gpt-4"
        assert _canonical_model_name("gpt-4-32k") == "gpt-4"
        assert _canonical_model_name("gpt-3.5-turbo-0613") == "gpt-3.5-turbo-0613"

    def test_generate_raises_if_no_prompt_or_messages(self, mock_azure_env):
        from novaeval.models.azure_openai import AzureOpenAIModel

        with patch("novaeval.models.azure_openai.AzureOpenAI"):
            model = AzureOpenAIModel()
            with pytest.raises(
                ValueError, match="Either `prompt` or `messages` must be provided"
            ):
                model.generate()

    def test_generate_chat_no_chat_completions(self, mock_azure_env):
        from novaeval.models.azure_openai import AzureOpenAIModel

        with patch("novaeval.models.azure_openai.AzureOpenAI") as mock_client:
            mock_client_instance = Mock()
            # Remove chat.completions
            mock_client_instance.chat = Mock()
            del mock_client_instance.chat.completions
            mock_client.return_value = mock_client_instance
            model = AzureOpenAIModel()
            with pytest.raises(
                RuntimeError, match="does not support the 'chat.completions' endpoint"
            ):
                model.generate_chat(messages=[{"role": "user", "content": "hi"}])

    def test_generate_chat_empty_choices(self, mock_azure_env):
        from novaeval.models.azure_openai import AzureOpenAIModel

        with patch("novaeval.models.azure_openai.AzureOpenAI") as mock_client:
            mock_chat_response = Mock()
            mock_chat_response.choices = []
            mock_chat_response.usage = None
            mock_client_instance = Mock()
            mock_client_instance.chat.completions.create.return_value = (
                mock_chat_response
            )
            mock_client.return_value = mock_client_instance
            model = AzureOpenAIModel()
            messages = [{"role": "user", "content": "Hello"}]
            # Should not error, just return ""
            response = model.generate_chat(messages=messages)
            assert response == ""

    def test_generate_chat_usage_missing_prompt_and_completion_tokens(
        self, mock_azure_env
    ):
        # Covers lines 182, 184, 186: usage present but missing prompt_tokens/completion_tokens
        from novaeval.models.azure_openai import AzureOpenAIModel

        with patch("novaeval.models.azure_openai.AzureOpenAI") as mock_client:
            mock_choice = Mock()
            mock_choice.message.content = "Test"
            mock_chat_response = Mock()
            mock_chat_response.choices = [mock_choice]

            class Usage:
                pass  # no prompt_tokens or completion_tokens

            mock_chat_response.usage = Usage()
            mock_client_instance = Mock()
            mock_client_instance.chat.completions.create.return_value = (
                mock_chat_response
            )
            mock_client.return_value = mock_client_instance
            model = AzureOpenAIModel()
            messages = [{"role": "user", "content": "Hello"}]
            response = model.generate_chat(messages=messages)
            assert response == "Test"

    def test_generate_batch_error_handling(self, mock_azure_env):
        from novaeval.models.azure_openai import AzureOpenAIModel

        with patch("novaeval.models.azure_openai.AzureOpenAI"):
            model = AzureOpenAIModel()
            model._handle_error = Mock()
            # The first call returns "ok", the second call raises an Exception instance
            model.generate = Mock(side_effect=["ok", Exception("fail")])
            responses = model.generate_batch(["a", "b"])
            assert responses == ["ok", ""]
            assert model._handle_error.called

    def test_generate_chat_no_usage(self, mock_azure_env):
        # Covers line 182, 184, 186: usage is None, input_tokens fallback, output_tokens fallback
        from novaeval.models.azure_openai import AzureOpenAIModel

        with patch("novaeval.models.azure_openai.AzureOpenAI") as mock_client:
            mock_choice = Mock()
            mock_choice.message.content = "Test"
            mock_chat_response = Mock()
            mock_chat_response.choices = [mock_choice]
            mock_chat_response.usage = None
            mock_client_instance = Mock()
            mock_client_instance.chat.completions.create.return_value = (
                mock_chat_response
            )
            mock_client.return_value = mock_client_instance
            model = AzureOpenAIModel()
            messages = [{"role": "user", "content": "Hello"}]
            response = model.generate_chat(messages=messages)
            assert response == "Test"

    def test_generate_batch_handles_exception_and_appends_empty_string(
        self, mock_azure_env
    ):
        # Covers lines 247-249
        from novaeval.models.azure_openai import AzureOpenAIModel

        with patch("novaeval.models.azure_openai.AzureOpenAI"):
            model = AzureOpenAIModel()

            def raise_exc(*a, **kw):
                raise Exception("fail")

            model.generate = Mock(side_effect=raise_exc)
            model._handle_error = Mock()
            responses = model.generate_batch(["a"])
            assert responses == [""]
            assert model._handle_error.called

    def test_validate_connection_no_chat_completions(self, mock_azure_env):
        # Covers line 305: validate_connection raises RuntimeError if chat.completions missing
        from novaeval.models.azure_openai import AzureOpenAIModel

        with patch("novaeval.models.azure_openai.AzureOpenAI") as mock_client:
            mock_client_instance = Mock()
            mock_client_instance.chat = Mock()
            del mock_client_instance.chat.completions
            mock_client.return_value = mock_client_instance
            model = AzureOpenAIModel()
            with pytest.raises(
                RuntimeError, match="does not support the 'chat.completions' endpoint"
            ):
                model.validate_connection()

    def test_get_info_supported_models_and_pricing(self, mock_azure_env):
        from novaeval.models.azure_openai import SUPPORTED_MODELS, AzureOpenAIModel

        with patch("novaeval.models.azure_openai.AzureOpenAI"):
            model = AzureOpenAIModel(model_name="gpt-4")
            info = model.get_info()
            assert info["supported_models"] == SUPPORTED_MODELS
            assert info["pricing"] == (30.0, 60.0)
            # Unknown model fallback
            model = AzureOpenAIModel(model_name="unknown-model")
            info = model.get_info()
            assert info["pricing"] == (0, 0)

    def test_supported_models_and_pricing_constants(self):
        from novaeval.models.azure_openai import MODEL_PRICING_PER_1M, SUPPORTED_MODELS

        assert isinstance(SUPPORTED_MODELS, list)
        assert isinstance(MODEL_PRICING_PER_1M, dict)
        for k, v in MODEL_PRICING_PER_1M.items():
            assert isinstance(k, str)
            assert isinstance(v, tuple)
            assert len(v) == 2

    def test_get_rates_all_branches(self, mock_azure_env):
        from novaeval.models.azure_openai import AzureOpenAIModel

        with patch("novaeval.models.azure_openai.AzureOpenAI"):
            # Tiered model, low tier
            model = AzureOpenAIModel(model_name="gpt-4")
            in_rate, out_rate = model._get_rates(100, 100)
            assert in_rate == 30.0 and out_rate == 60.0
            # Tiered model, high tier
            in_rate, out_rate = model._get_rates(9000, 0)
            assert in_rate == 60.0 and out_rate == 120.0
            # Non-tiered model
            model = AzureOpenAIModel(model_name="gpt-4-turbo")
            in_rate, out_rate = model._get_rates(100, 100)
            assert in_rate == 10.0 and out_rate == 30.0
            # Unknown model
            model = AzureOpenAIModel(model_name="unknown")
            in_rate, out_rate = model._get_rates(100, 100)
            assert in_rate == 0.0 and out_rate == 0.0

    def test_estimate_cost_calls_count_tokens(self, mock_azure_env):
        from novaeval.models.azure_openai import AzureOpenAIModel

        with patch("novaeval.models.azure_openai.AzureOpenAI"):
            model = AzureOpenAIModel()
            model.count_tokens = Mock(return_value=42)
            cost = model.estimate_cost(
                "prompt", "response", input_tokens=None, output_tokens=None
            )
            assert model.count_tokens.call_count == 2
            assert isinstance(cost, float)

    def test_count_tokens_edge_cases(self, mock_azure_env):
        from novaeval.models.azure_openai import AzureOpenAIModel

        with patch("novaeval.models.azure_openai.AzureOpenAI"):
            model = AzureOpenAIModel()
            assert model.count_tokens("   ") == 0
            assert model.count_tokens("word") == 2
            assert model.count_tokens("word word") == 4
            assert model.count_tokens("word, word!") == 4
            assert model.count_tokens("你好 世界") == 4

    def test_create_from_config_with_extra_kwargs(self, mock_azure_env):
        from novaeval.models.azure_openai import AzureOpenAIModel

        with patch("novaeval.models.azure_openai.AzureOpenAI") as mock_client:
            cfg = {
                "model_name": "gpt-4",
                "base_url": "https://test.azure.com",
                "api_key": "test_key",
                "api_version": "2024-02-01",
                "extra_param": "extra",
            }
            model = AzureOpenAIModel.create_from_config(cfg)
            assert hasattr(model, "kwargs")
            assert "extra_param" in model.kwargs
            mock_client.assert_called_once()

    def test_get_info_custom_model_and_kwargs(self, mock_azure_env):
        from novaeval.models.azure_openai import AzureOpenAIModel

        with patch("novaeval.models.azure_openai.AzureOpenAI"):
            model = AzureOpenAIModel(model_name="gpt-4-turbo", foo="bar")
            info = model.get_info()
            assert info["name"] == "azure_openai_gpt-4-turbo"
            assert info["model_name"] == "gpt-4-turbo"
            # get_info does not include arbitrary kwargs, but model.kwargs should
            assert "foo" in model.kwargs and model.kwargs["foo"] == "bar"

    def test_generate_with_both_prompt_and_messages(self, mock_azure_env):
        from novaeval.models.azure_openai import AzureOpenAIModel

        with patch("novaeval.models.azure_openai.AzureOpenAI"):
            model = AzureOpenAIModel()
            model.generate_chat = Mock(return_value="msg")
            response = model.generate(
                prompt="hi", messages=[{"role": "user", "content": "msg"}]
            )
            assert response == "msg"
            model.generate_chat.assert_called_once()

    def test_generate_with_only_prompt(self, mock_azure_env):
        from novaeval.models.azure_openai import AzureOpenAIModel

        with patch("novaeval.models.azure_openai.AzureOpenAI"):
            model = AzureOpenAIModel()
            model.generate_chat = Mock(return_value="wrapped")
            response = model.generate(prompt="hi")
            assert response == "wrapped"
            model.generate_chat.assert_called_once()
            args, kwargs = model.generate_chat.call_args
            assert kwargs["messages"][0]["content"] == "hi"

    def test_generate_batch_empty(self, mock_azure_env):
        from novaeval.models.azure_openai import AzureOpenAIModel

        with patch("novaeval.models.azure_openai.AzureOpenAI"):
            model = AzureOpenAIModel()
            result = model.generate_batch([])
            assert result == []

    def test_validate_connection_choices_none(self, mock_azure_env):
        from novaeval.models.azure_openai import AzureOpenAIModel

        with patch("novaeval.models.azure_openai.AzureOpenAI") as mock_client:
            mock_response = Mock()
            mock_response.choices = None
            mock_client_instance = Mock()
            mock_client_instance.chat.completions.create.return_value = mock_response
            mock_client.return_value = mock_client_instance
            model = AzureOpenAIModel()
            assert model.validate_connection() is False

    def test_generate_chat_error_handling_and_error_tracking(self, mock_azure_env):
        from novaeval.models.azure_openai import AzureOpenAIModel

        with patch("novaeval.models.azure_openai.AzureOpenAI") as mock_client:
            mock_client_instance = Mock()
            mock_client_instance.chat.completions.create.side_effect = Exception("fail")
            mock_client.return_value = mock_client_instance
            model = AzureOpenAIModel()
            with pytest.raises(Exception, match="fail"):
                model.generate_chat(messages=[{"role": "user", "content": "hi"}])
            assert any(
                "Failed to generate chat completion" in err for err in model.errors
            )

    def test_base_model_methods_called(self, mock_azure_env):
        from novaeval.models.azure_openai import AzureOpenAIModel

        with patch("novaeval.models.azure_openai.AzureOpenAI"):
            model = AzureOpenAIModel()
            model._track_request = Mock()
            model.estimate_cost = Mock(return_value=0.1)
            model.count_tokens = Mock(return_value=2)
            model.client = Mock()
            mock_choice = Mock()
            mock_choice.message.content = "foo"
            mock_response = Mock()
            mock_response.choices = [mock_choice]
            mock_response.usage = Mock(prompt_tokens=1, completion_tokens=1)
            model.client.chat.completions.create.return_value = mock_response
            messages = [{"role": "user", "content": "bar"}]
            model.generate_chat(messages=messages)
            assert model._track_request.called

    def test_get_info_keys(self, mock_azure_env):
        from novaeval.models.azure_openai import AzureOpenAIModel

        with patch("novaeval.models.azure_openai.AzureOpenAI"):
            model = AzureOpenAIModel(model_name="gpt-4")
            info = model.get_info()
            for key in [
                "name",
                "model_name",
                "provider",
                "type",
                "max_retries",
                "timeout",
                "api_version",
                "supports_batch",
                "pricing",
                "supported_models",
            ]:
                assert key in info
