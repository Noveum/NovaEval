"""
Unit tests for Ollama model functionality.
"""

import os
from unittest.mock import Mock, patch

import pytest

from novaeval.models.ollama import OllamaModel

pytestmark = pytest.mark.unit


class TestOllamaModel:
    """Test cases for OllamaModel class."""

    # -------------------------- Initialization Tests --------------------------

    def test_init_default(self):
        """Test initialization with default parameters."""
        with patch("novaeval.models.ollama.Client") as mock_client:
            mock_client.return_value.pull.return_value = iter([])
            model = OllamaModel()

            assert model.name == "ollama_llama3"
            assert model.model_name == "llama3"
            assert model.base_url == "http://localhost:11434"
            assert model.gpu_cost_per_sec is None
            assert model.headers == {}
            mock_client.assert_called_once_with(
                host="http://localhost:11434", headers={}
            )

    def test_init_with_custom_params(self):
        """Test initialization with custom parameters."""
        with patch("novaeval.models.ollama.Client") as mock_client:
            mock_client.return_value.pull.return_value = iter([])
            model = OllamaModel(
                model_name="custom-model",
                base_url="http://custom-host:8080",
                headers={"Authorization": "Bearer token"},
                gpu_cost_per_sec=0.001,
                custom_param="value",
            )

            assert model.name == "ollama_custom-model"
            assert model.model_name == "custom-model"
            assert model.base_url == "http://custom-host:8080"
            assert model.gpu_cost_per_sec == 0.001
            assert model.headers == {"Authorization": "Bearer token"}
            assert model.kwargs["custom_param"] == "value"
            mock_client.assert_called_once_with(
                host="http://custom-host:8080",
                headers={"Authorization": "Bearer token"},
            )

    def test_init_with_env_vars(self):
        """Test initialization using environment variables."""
        with (
            patch.dict(os.environ, {"OLLAMA_HOST": "http://env-host:9090"}),
            patch("novaeval.models.ollama.Client") as mock_client,
        ):
            mock_client.return_value.pull.return_value = iter([])
            model = OllamaModel()

            assert model.base_url == "http://env-host:9090"
            mock_client.assert_called_once_with(host="http://env-host:9090", headers={})

    def test_init_with_pull_on_init_false(self):
        """Test initialization with pull_on_init=False."""
        with patch("novaeval.models.ollama.Client") as mock_client:
            model = OllamaModel(pull_on_init=False)

            assert model.name == "ollama_llama3"
            mock_client.assert_called_once()
            mock_client.return_value.pull.assert_not_called()

    def test_init_missing_ollama_package(self):
        """Test import error handling when ollama package is missing."""
        with (
            patch("novaeval.models.ollama.Client", None),
            pytest.raises(ImportError, match="The 'ollama' package is required"),
        ):
            OllamaModel()

    def test_init_pull_model_success(self):
        """Test successful model pulling during initialization."""
        with patch("novaeval.models.ollama.Client") as mock_client:
            mock_client.return_value.pull.return_value = iter(["chunk1", "chunk2"])
            OllamaModel()

            mock_client.return_value.pull.assert_called_once_with(
                model="llama3", stream=True
            )

    def test_init_pull_model_failure(self):
        """Test failed model pulling during initialization (non-blocking)."""
        with patch("novaeval.models.ollama.Client") as mock_client:
            mock_client.return_value.pull.side_effect = Exception("Pull failed")
            model = OllamaModel()

            # Should not raise exception, just log error
            assert len(model.errors) == 1
            assert "Failed to pull model 'llama3' on init" in model.errors[0]

    def test_init_pull_model_non_iterable_result(self):
        """Test handling of non-iterable pull result."""
        with patch("novaeval.models.ollama.Client") as mock_client:
            mock_client.return_value.pull.return_value = "non-iterable"
            model = OllamaModel()

            # Should handle non-iterable result gracefully
            assert model.name == "ollama_llama3"

    # -------------------------- Helper Method Tests --------------------------

    def test_extract_content_from_response_attribute(self):
        """Test content extraction using attribute access."""
        mock_response = Mock()
        mock_message = Mock()
        mock_message.content = "test content"
        mock_response.message = mock_message

        content = OllamaModel._extract_content_from_response(mock_response)
        assert content == "test content"

    def test_extract_content_from_response_mapping(self):
        """Test content extraction using mapping access."""
        mock_response = {"message": {"content": "test content"}}

        content = OllamaModel._extract_content_from_response(mock_response)
        assert content == "test content"

    def test_extract_content_from_response_fallback(self):
        """Test content extraction fallback to empty string."""
        mock_response = {"message": {"other_field": "value"}}

        content = OllamaModel._extract_content_from_response(mock_response)
        assert content == ""

    def test_extract_metric_attribute(self):
        """Test metric extraction using attribute access."""
        mock_response = Mock()
        mock_response.total_duration = 1000000000

        metric = OllamaModel._extract_metric(mock_response, "total_duration")
        assert metric == 1000000000

    def test_extract_metric_mapping(self):
        """Test metric extraction using mapping access."""
        mock_response = {"total_duration": 1000000000}

        metric = OllamaModel._extract_metric(mock_response, "total_duration")
        assert metric == 1000000000

    def test_extract_metric_none(self):
        """Test metric extraction when metric is missing."""
        mock_response = {"other_field": "value"}

        metric = OllamaModel._extract_metric(mock_response, "total_duration")
        assert metric is None

    def test_build_options_basic(self):
        """Test basic options building."""
        base_options = {"existing": "value"}
        options = OllamaModel._build_options(
            base_options, max_tokens=100, temperature=0.7, stop="<END>", extra_kwargs={}
        )

        assert options["existing"] == "value"
        assert options["num_predict"] == 100
        assert options["temperature"] == 0.7
        assert options["stop"] == ["<END>"]

    def test_build_options_ollama_specific(self):
        """Test Ollama-specific options merging."""
        base_options = {"existing": "value"}
        extra_kwargs = {
            "top_p": 0.9,
            "num_ctx": 2048,
            "mirostat": 2,
            "unsupported": "ignored",
        }

        options = OllamaModel._build_options(
            base_options,
            max_tokens=50,
            temperature=0.5,
            stop=None,
            extra_kwargs=extra_kwargs,
        )

        assert options["top_p"] == 0.9
        assert options["num_ctx"] == 2048
        assert options["mirostat"] == 2
        assert "unsupported" not in options
        assert "unsupported" in extra_kwargs  # Should be popped

    def test_build_options_stop_sequences(self):
        """Test stop sequence handling."""
        # Single string
        options = OllamaModel._build_options(
            {}, max_tokens=None, temperature=None, stop="<END>", extra_kwargs={}
        )
        assert options["stop"] == ["<END>"]

        # List of strings
        options = OllamaModel._build_options(
            {},
            max_tokens=None,
            temperature=None,
            stop=["<END>", "STOP"],
            extra_kwargs={},
        )
        assert options["stop"] == ["<END>", "STOP"]

        # None
        options = OllamaModel._build_options(
            {}, max_tokens=None, temperature=None, stop=None, extra_kwargs={}
        )
        assert "stop" not in options

    def test_build_messages_from_prompt(self):
        """Test message format conversion."""
        messages = OllamaModel._build_messages_from_prompt("test prompt")
        assert messages == [{"role": "user", "content": "test prompt"}]

        messages = OllamaModel._build_messages_from_prompt(None)
        assert messages == [{"role": "user", "content": ""}]

    def test_extract_thinking_from_response(self):
        """Test thinking extraction from various formats."""
        # Top-level thinking attribute
        mock_response = Mock()
        mock_response.thinking = "test thought"
        thinking = OllamaModel._extract_thinking_from_response(mock_response)
        assert thinking == "test thought"

        # Nested under message as attribute
        mock_response = Mock()
        mock_message = Mock()
        mock_message.thinking = "nested thought"
        mock_response.message = mock_message
        thinking = OllamaModel._extract_thinking_from_response(mock_response)
        assert thinking == "nested thought"

        # Mapping top-level
        mock_response = {"thinking": "mapped thought"}
        thinking = OllamaModel._extract_thinking_from_response(mock_response)
        assert thinking == "mapped thought"

        # Mapping nested under message
        mock_response = {"message": {"reasoning": "nested mapped thought"}}
        thinking = OllamaModel._extract_thinking_from_response(mock_response)
        assert thinking == "nested mapped thought"

        # No thinking field
        mock_response = {"message": {"content": "no thinking"}}
        thinking = OllamaModel._extract_thinking_from_response(mock_response)
        assert thinking == ""

    # -------------------------- Core Generation Tests --------------------------

    def test_generate_with_prompt(self):
        """Test basic prompt generation."""
        with patch("novaeval.models.ollama.Client") as mock_client:
            mock_response = Mock()
            mock_response.message.content = "generated response"
            mock_response.total_duration = 1000000000
            mock_response.prompt_eval_count = 10
            mock_response.eval_count = 20
            mock_client.return_value.chat.return_value = mock_response

            model = OllamaModel(pull_on_init=False)
            response = model.generate("test prompt")

            assert response == "generated response"
            assert model.total_requests == 1

    def test_generate_with_messages(self):
        """Test messages-based generation."""
        with patch("novaeval.models.ollama.Client") as mock_client:
            mock_response = Mock()
            mock_response.message.content = "generated response"
            mock_response.total_duration = 1000000000
            mock_response.prompt_eval_count = 10
            mock_response.eval_count = 20
            mock_client.return_value.chat.return_value = mock_response

            model = OllamaModel(pull_on_init=False)
            messages = [{"role": "user", "content": "test message"}]
            response = model.generate(messages=messages)

            assert response == "generated response"

    def test_generate_missing_inputs(self):
        """Test error when neither prompt nor messages provided."""
        with patch("novaeval.models.ollama.Client"):
            model = OllamaModel(pull_on_init=False)

            with pytest.raises(
                ValueError, match="Either `prompt` or `messages` must be provided"
            ):
                model.generate()

    def test_generate_with_params(self):
        """Test generation with additional parameters."""
        with patch("novaeval.models.ollama.Client") as mock_client:
            mock_response = Mock()
            mock_response.message.content = "generated response"
            mock_response.total_duration = 1000000000
            mock_response.prompt_eval_count = 10
            mock_response.eval_count = 20
            mock_client.return_value.chat.return_value = mock_response

            model = OllamaModel(pull_on_init=False)
            model.generate(
                "test prompt", max_tokens=100, temperature=0.7, stop=["<END>"]
            )

            # Check that options were built correctly
            call_args = mock_client.return_value.chat.call_args[1]
            assert call_args["options"]["num_predict"] == 100
            assert call_args["options"]["temperature"] == 0.7
            assert call_args["options"]["stop"] == ["<END>"]

    def test_generate_non_streaming(self):
        """Test non-streaming generation."""
        with patch("novaeval.models.ollama.Client") as mock_client:
            mock_response = Mock()
            mock_response.message.content = "generated response"
            mock_response.total_duration = 1000000000
            mock_response.prompt_eval_count = 10
            mock_response.eval_count = 20
            mock_client.return_value.chat.return_value = mock_response

            model = OllamaModel(pull_on_init=False)
            response = model.generate("test prompt")

            assert response == "generated response"

    def test_generate_with_ollama_options(self):
        """Test Ollama-specific options."""
        with patch("novaeval.models.ollama.Client") as mock_client:
            mock_response = Mock()
            mock_response.message.content = "generated response"
            mock_response.total_duration = 1000000000
            mock_response.prompt_eval_count = 10
            mock_response.eval_count = 20
            mock_client.return_value.chat.return_value = mock_response

            model = OllamaModel(pull_on_init=False)
            model.generate("test prompt", options={"num_ctx": 2048, "top_p": 0.9})

            call_args = mock_client.return_value.chat.call_args[1]
            assert call_args["options"]["num_ctx"] == 2048
            assert call_args["options"]["top_p"] == 0.9

    def test_generate_with_format_keep_alive(self):
        """Test format and keep_alive parameters."""
        with patch("novaeval.models.ollama.Client") as mock_client:
            mock_response = Mock()
            mock_response.message.content = "generated response"
            mock_response.total_duration = 1000000000
            mock_response.prompt_eval_count = 10
            mock_response.eval_count = 20
            mock_client.return_value.chat.return_value = mock_response

            model = OllamaModel(pull_on_init=False)
            model.generate("test prompt", format="json", keep_alive="5m")

            call_args = mock_client.return_value.chat.call_args[1]
            assert call_args["format"] == "json"
            assert call_args["keep_alive"] == "5m"

    def test_generate_with_thinking(self):
        """Test think/reasoning parameters."""
        with patch("novaeval.models.ollama.Client") as mock_client:
            mock_response = Mock()
            mock_response.message.content = "generated response"
            mock_response.total_duration = 1000000000
            mock_response.prompt_eval_count = 10
            mock_response.eval_count = 20
            mock_client.return_value.chat.return_value = mock_response

            model = OllamaModel(pull_on_init=False)
            model.generate("test prompt", think=True)

            call_args = mock_client.return_value.chat.call_args[1]
            assert call_args["think"] is True

            # Test reasoning parameter
            model.generate("test prompt", reasoning=True)
            call_args = mock_client.return_value.chat.call_args[1]
            assert call_args["think"] is True

    # -------------------------- Generate with Thought Tests --------------------------

    def test_generate_with_thought_basic(self):
        """Test basic thought generation."""
        with patch("novaeval.models.ollama.Client") as mock_client:
            mock_response = Mock()
            mock_response.message.content = "generated response"
            mock_response.thinking = "test thought"
            mock_response.total_duration = 1000000000
            mock_response.prompt_eval_count = 10
            mock_response.eval_count = 20
            mock_client.return_value.chat.return_value = mock_response

            model = OllamaModel(pull_on_init=False)
            response, thinking = model.generate_with_thought("test prompt")

            assert response == "generated response"
            assert thinking == "test thought"

    def test_generate_with_thought_non_streaming(self):
        """Test non-streaming with thought."""
        with patch("novaeval.models.ollama.Client") as mock_client:
            mock_response = Mock()
            mock_response.message.content = "generated response"
            mock_response.message.reasoning = "test reasoning"
            mock_response.total_duration = 1000000000
            mock_response.prompt_eval_count = 10
            mock_response.eval_count = 20
            mock_client.return_value.chat.return_value = mock_response

            model = OllamaModel(pull_on_init=False)
            response, thinking = model.generate_with_thought("test prompt")

            assert response == "generated response"
            assert thinking == "test reasoning"

    def test_generate_with_thought_extraction(self):
        """Test various thinking field formats."""
        with patch("novaeval.models.ollama.Client") as mock_client:
            # Test different thinking field names
            for field_name in ["thinking", "reasoning", "thought", "thoughts"]:
                mock_response = Mock()
                mock_response.message.content = "generated response"
                setattr(mock_response.message, field_name, f"test {field_name}")
                mock_response.total_duration = 1000000000
                mock_response.prompt_eval_count = 10
                mock_response.eval_count = 20
                mock_client.return_value.chat.return_value = mock_response

                model = OllamaModel(pull_on_init=False)
                _response, thinking = model.generate_with_thought("test prompt")

                assert thinking == f"test {field_name}"

    # -------------------------- Batch Generation Tests --------------------------

    def test_generate_batch_success(self):
        """Test successful batch processing."""
        with patch("novaeval.models.ollama.Client") as mock_client:
            mock_response = Mock()
            mock_response.message.content = "generated response"
            mock_response.total_duration = 1000000000
            mock_response.prompt_eval_count = 10
            mock_response.eval_count = 20
            mock_client.return_value.chat.return_value = mock_response

            model = OllamaModel(pull_on_init=False)
            prompts = ["prompt1", "prompt2", "prompt3"]
            responses = model.generate_batch(prompts)

            assert len(responses) == 3
            assert all(response == "generated response" for response in responses)
            assert mock_client.return_value.chat.call_count == 3

    def test_generate_batch_with_errors(self):
        """Test partial batch failures."""
        with patch("novaeval.models.ollama.Client") as mock_client:
            mock_client.return_value.chat.side_effect = [
                Mock(
                    message=Mock(content="success"),
                    total_duration=1000000000,
                    prompt_eval_count=10,
                    eval_count=20,
                ),
                Exception("API Error"),
                Mock(
                    message=Mock(content="success2"),
                    total_duration=1000000000,
                    prompt_eval_count=10,
                    eval_count=20,
                ),
            ]

            model = OllamaModel(pull_on_init=False)
            prompts = ["prompt1", "prompt2", "prompt3"]
            responses = model.generate_batch(prompts)

            assert responses == ["success", "", "success2"]
            assert (
                len(model.errors) == 2
            )  # One for the API error, one for the batch failure
            assert "Batch failure for: prompt2" in model.errors[1]

    def test_generate_batch_empty(self):
        """Test empty prompt list."""
        with patch("novaeval.models.ollama.Client"):
            model = OllamaModel(pull_on_init=False)
            responses = model.generate_batch([])

            assert responses == []

    # -------------------------- Cost Estimation Tests --------------------------

    def test_estimate_cost_with_gpu_cost(self):
        """Test GPU cost calculation."""
        with patch("novaeval.models.ollama.Client"):
            model = OllamaModel(gpu_cost_per_sec=0.001, pull_on_init=False)

        cost = model.estimate_cost("prompt", "response", total_duration_ns=1000000000)
        assert cost == 0.001  # 1 second * 0.001 per second

    def test_estimate_cost_without_gpu_cost(self):
        """Test no cost when gpu_cost_per_sec is None."""
        with patch("novaeval.models.ollama.Client"):
            model = OllamaModel(pull_on_init=False)

        cost = model.estimate_cost("prompt", "response")
        assert cost == 0.0

    def test_estimate_cost_with_duration_ns(self):
        """Test using total_duration_ns."""
        with patch("novaeval.models.ollama.Client"):
            model = OllamaModel(gpu_cost_per_sec=0.001, pull_on_init=False)

        cost = model.estimate_cost("prompt", "response", total_duration_ns=2000000000)
        assert cost == 0.002  # 2 seconds * 0.001 per second

    def test_estimate_cost_with_elapsed_seconds(self):
        """Test using elapsed_seconds fallback."""
        with patch("novaeval.models.ollama.Client"):
            model = OllamaModel(gpu_cost_per_sec=0.001, pull_on_init=False)

        cost = model.estimate_cost("prompt", "response", elapsed_seconds=3.0)
        assert cost == 0.003  # 3 seconds * 0.001 per second

    def test_estimate_cost_fallback(self):
        """Test no duration metrics available."""
        with patch("novaeval.models.ollama.Client"):
            model = OllamaModel(gpu_cost_per_sec=0.001, pull_on_init=False)

        cost = model.estimate_cost("prompt", "response")
        assert cost == 0.0

    # -------------------------- Connection and Validation Tests --------------------------

    def test_validate_connection_success(self):
        """Test successful connection test."""
        with patch("novaeval.models.ollama.Client") as mock_client:
            mock_response = Mock()
            mock_response.message.content = "Hello"
            mock_client.return_value.chat.return_value = mock_response

            model = OllamaModel(pull_on_init=False)
            result = model.validate_connection()

            assert result is True
            mock_client.return_value.chat.assert_called_once_with(
                model="llama3",
                messages=[{"role": "user", "content": "Hello"}],
                stream=False,
                options={"num_predict": 1},
            )

    def test_validate_connection_failure(self):
        """Test failed connection test."""
        with patch("novaeval.models.ollama.Client") as mock_client:
            mock_client.return_value.chat.side_effect = Exception("Connection failed")

            model = OllamaModel(pull_on_init=False)
            result = model.validate_connection()

            assert result is False
            assert len(model.errors) == 1
            assert "Ollama connection validation failed" in model.errors[0]

    def test_validate_connection_error_handling(self):
        """Test error during validation."""
        with patch("novaeval.models.ollama.Client") as mock_client:
            mock_client.return_value.chat.side_effect = Exception("API Error")

            model = OllamaModel(pull_on_init=False)
            result = model.validate_connection()

            assert result is False
            assert len(model.errors) == 1

    # -------------------------- Info and Provider Tests --------------------------

    def test_get_provider(self):
        """Test provider name."""
        with patch("novaeval.models.ollama.Client"):
            model = OllamaModel(pull_on_init=False)
            assert model.get_provider() == "ollama"

    def test_get_info(self):
        """Test model information including Ollama-specific fields."""
        with patch("novaeval.models.ollama.Client"):
            model = OllamaModel(
                model_name="test-model", gpu_cost_per_sec=0.001, pull_on_init=False
            )
            info = model.get_info()

            assert info["name"] == "ollama_test-model"
            assert info["model_name"] == "test-model"
            assert info["provider"] == "ollama"
            assert info["type"] == "OllamaModel"
            assert info["host"] == "http://localhost:11434"
            assert info["supports_batch"] is False
            assert info["pricing"] == (0.0, 0.0)
            assert info["gpu_cost_per_sec"] == 0.001

    def test_get_info_inheritance(self):
        """Test inherited base fields."""
        with patch("novaeval.models.ollama.Client"):
            model = OllamaModel(pull_on_init=False)
            info = model.get_info()

            # Base fields should be present
            assert "total_requests" in info
            assert "total_tokens" in info
            assert "total_cost" in info
            assert "error_count" in info

    # -------------------------- Error Handling Tests --------------------------

    def test_generate_api_error(self):
        """Test API call failures."""
        with patch("novaeval.models.ollama.Client") as mock_client:
            mock_client.return_value.chat.side_effect = Exception("API Error")

            model = OllamaModel(pull_on_init=False)

            with pytest.raises(Exception, match="API Error"):
                model.generate("test prompt")

            assert len(model.errors) == 1
            assert "Failed to generate text via Ollama chat API" in model.errors[0]

    def test_generate_with_thought_error(self):
        """Test thought generation errors."""
        with patch("novaeval.models.ollama.Client") as mock_client:
            mock_client.return_value.chat.side_effect = Exception("Thought Error")

            model = OllamaModel(pull_on_init=False)

            with pytest.raises(Exception, match="Thought Error"):
                model.generate_with_thought("test prompt")

    def test_batch_generation_error(self):
        """Test batch processing errors."""
        with patch("novaeval.models.ollama.Client") as mock_client:
            mock_client.return_value.chat.side_effect = Exception("Batch Error")

            model = OllamaModel(pull_on_init=False)
            responses = model.generate_batch(["prompt1", "prompt2"])

            assert responses == ["", ""]
            assert len(model.errors) == 4  # Two API errors + two batch failure errors

    # -------------------------- Edge Cases and Integration Tests --------------------------

    def test_response_format_variations(self):
        """Test different response object structures."""
        with patch("novaeval.models.ollama.Client") as mock_client:
            # Test attribute-style response
            mock_response = Mock()
            mock_response.message.content = "attribute content"
            mock_response.total_duration = 1000000000
            mock_response.prompt_eval_count = 10
            mock_response.eval_count = 20
            mock_client.return_value.chat.return_value = mock_response

            model = OllamaModel(pull_on_init=False)
            response = model.generate("test prompt")
            assert response == "attribute content"

            # Test mapping-style response
            mock_response = {"message": {"content": "mapping content"}}
            mock_client.return_value.chat.return_value = mock_response

            response = model.generate("test prompt")
            assert response == "mapping content"

    def test_metric_extraction_edge_cases(self):
        """Test missing or malformed metrics."""
        with patch("novaeval.models.ollama.Client") as mock_client:
            # Test missing metrics
            mock_response = Mock()
            mock_response.message.content = "content"
            # No metrics set
            mock_client.return_value.chat.return_value = mock_response

            model = OllamaModel(pull_on_init=False)
            response = model.generate("test prompt")

            # Should fall back to count_tokens method
            assert response == "content"

    def test_options_merging(self):
        """Test complex options merging scenarios."""
        with patch("novaeval.models.ollama.Client") as mock_client:
            mock_response = Mock()
            mock_response.message.content = "content"
            mock_response.total_duration = 1000000000
            mock_response.prompt_eval_count = 10
            mock_response.eval_count = 20
            mock_client.return_value.chat.return_value = mock_response

            model = OllamaModel(pull_on_init=False)
            model.generate(
                "test prompt", options={"existing": "value"}, top_p=0.9, num_ctx=2048
            )

            call_args = mock_client.return_value.chat.call_args[1]
            assert call_args["options"]["existing"] == "value"
            assert call_args["options"]["top_p"] == 0.9
            assert call_args["options"]["num_ctx"] == 2048

    def test_thinking_extraction_edge_cases(self):
        """Test various thinking field locations."""
        with patch("novaeval.models.ollama.Client") as mock_client:
            # Test thinking at different levels
            test_cases = [
                ({"thinking": "top"}, "top"),
                ({"message": {"reasoning": "nested"}}, "nested"),
                ({"message": {"thought": "deep"}}, "deep"),
                ({"message": {"thoughts": "plural"}}, "plural"),
            ]

            for response_data, expected_thinking in test_cases:
                mock_response = Mock()
                mock_response.message.content = "content"
                mock_response.total_duration = 1000000000
                mock_response.prompt_eval_count = 10
                mock_response.eval_count = 20

                # Convert to mock with proper structure
                if isinstance(response_data, dict):
                    for key, value in response_data.items():
                        if key == "message":
                            mock_message = Mock()
                            for msg_key, msg_value in value.items():
                                setattr(mock_message, msg_key, msg_value)
                            setattr(mock_response, key, mock_message)
                        else:
                            setattr(mock_response, key, value)

                mock_client.return_value.chat.return_value = mock_response

                model = OllamaModel(pull_on_init=False)
                _response, thinking = model.generate_with_thought("test prompt")
                assert thinking == expected_thinking

    # -------------------------- Performance and Metrics Tests --------------------------

    def test_token_counting_fallback(self):
        """Test token counting when metrics unavailable."""
        with patch("novaeval.models.ollama.Client") as mock_client:
            mock_response = Mock()
            mock_response.message.content = "generated response"
            # No metrics set
            mock_client.return_value.chat.return_value = mock_response

            model = OllamaModel(pull_on_init=False)
            response = model.generate("test prompt")

            # Should use count_tokens method for fallback
            assert response == "generated response"
            assert model.total_tokens > 0

    def test_duration_calculation(self):
        """Test duration calculation from various sources."""
        with patch("novaeval.models.ollama.Client") as mock_client:
            # Test with total_duration_ns
            mock_response = Mock()
            mock_response.message.content = "content"
            mock_response.total_duration = 1000000000
            mock_response.prompt_eval_count = 10
            mock_response.eval_count = 20
            mock_client.return_value.chat.return_value = mock_response

            model = OllamaModel(gpu_cost_per_sec=0.001, pull_on_init=False)
            model.generate("test prompt")

            # Cost should be calculated using total_duration_ns
            assert model.total_cost == 0.001

    def test_request_tracking(self):
        """Test statistics tracking during generation."""
        with patch("novaeval.models.ollama.Client") as mock_client:
            mock_response = Mock()
            mock_response.message.content = "generated response"
            mock_response.total_duration = 1000000000
            mock_response.prompt_eval_count = 10
            mock_response.eval_count = 20
            mock_client.return_value.chat.return_value = mock_response

            model = OllamaModel(pull_on_init=False)

            # Initial state
            assert model.total_requests == 0
            assert model.total_tokens == 0
            assert model.total_cost == 0.0

            model.generate("test prompt")

            # After generation
            assert model.total_requests == 1
            assert model.total_tokens > 0
            assert model.total_cost >= 0.0

    # -------------------------- Mock and Integration Tests --------------------------

    def test_client_initialization(self):
        """Test Ollama client setup."""
        with patch("novaeval.models.ollama.Client") as mock_client:
            mock_client.return_value.pull.return_value = iter([])
            OllamaModel(
                base_url="http://test-host:8080",
                headers={"Custom": "Header"},
                pull_on_init=False,
            )

            mock_client.assert_called_once_with(
                host="http://test-host:8080", headers={"Custom": "Header"}
            )

    def test_non_streaming_response_handling(self):
        """Test non-streaming response processing."""
        with patch("novaeval.models.ollama.Client") as mock_client:
            mock_response = Mock()
            mock_response.message.content = "complete response"
            mock_response.total_duration = 1000000000
            mock_response.prompt_eval_count = 10
            mock_response.eval_count = 20
            mock_client.return_value.chat.return_value = mock_response

            model = OllamaModel(pull_on_init=False)
            response = model.generate("test prompt")

            assert response == "complete response"

    # -------------------------- Additional Coverage Tests --------------------------

    def test_extract_metric_negative_values(self):
        """Test metric extraction with negative values."""
        mock_response = {"total_duration": -1000000000}
        metric = OllamaModel._extract_metric(mock_response, "total_duration")
        assert metric == -1000000000

    def test_extract_metric_malformed_values(self):
        """Test metric extraction with malformed values."""
        mock_response = {"total_duration": "not_a_number"}
        metric = OllamaModel._extract_metric(mock_response, "total_duration")
        assert metric is None

    def test_build_options_empty_options(self):
        """Test building options with empty provided options."""
        options = OllamaModel._build_options(
            {}, max_tokens=100, temperature=0.5, stop=None, extra_kwargs={}
        )
        assert options["num_predict"] == 100
        assert options["temperature"] == 0.5

    def test_build_options_none_options(self):
        """Test building options with None provided options."""
        # The method expects a dict, so None should raise an error
        with pytest.raises(TypeError, match="'NoneType' object is not iterable"):
            OllamaModel._build_options(
                None, max_tokens=100, temperature=0.5, stop=None, extra_kwargs={}
            )

    def test_build_options_existing_values_not_overwritten(self):
        """Test that existing options are not overwritten."""
        base_options = {"num_predict": 50, "temperature": 0.3}
        options = OllamaModel._build_options(
            base_options, max_tokens=100, temperature=0.5, stop=None, extra_kwargs={}
        )
        assert options["num_predict"] == 50  # Should not be overwritten
        assert options["temperature"] == 0.3  # Should not be overwritten

    def test_extract_content_empty_message(self):
        """Test content extraction with empty message."""
        mock_response = {"message": {}}
        content = OllamaModel._extract_content_from_response(mock_response)
        assert content == ""

    def test_extract_content_missing_message(self):
        """Test content extraction with missing message field."""
        mock_response = {"other_field": "value"}
        content = OllamaModel._extract_content_from_response(mock_response)
        assert content == ""

    def test_extract_content_non_string_content(self):
        """Test content extraction with non-string content types."""
        # Test with None content
        mock_response = {"message": {"content": None}}
        content = OllamaModel._extract_content_from_response(mock_response)
        assert content is None  # The method returns None for None content

        # Test with integer content
        mock_response = {"message": {"content": 123}}
        content = OllamaModel._extract_content_from_response(mock_response)
        assert content == 123  # The method returns the actual content value

        # Test with empty string content
        mock_response = {"message": {"content": ""}}
        content = OllamaModel._extract_content_from_response(mock_response)
        assert content == ""  # Empty string is returned as-is

    def test_extract_thinking_empty_strings(self):
        """Test thinking extraction with empty string values."""
        mock_response = {"thinking": ""}
        thinking = OllamaModel._extract_thinking_from_response(mock_response)
        assert thinking == ""

    def test_estimate_cost_negative_duration(self):
        """Test cost estimation with negative duration."""
        with patch("novaeval.models.ollama.Client"):
            model = OllamaModel(gpu_cost_per_sec=0.001, pull_on_init=False)

        cost = model.estimate_cost("prompt", "response", total_duration_ns=-1000000000)
        assert cost == 0.0  # The method returns 0.0 for negative duration

    def test_estimate_cost_zero_duration(self):
        """Test cost estimation with zero duration."""
        with patch("novaeval.models.ollama.Client"):
            model = OllamaModel(gpu_cost_per_sec=0.001, pull_on_init=False)

        cost = model.estimate_cost("prompt", "response", total_duration_ns=0)
        assert cost == 0.0

    def test_estimate_cost_very_small_duration(self):
        """Test cost estimation with very small duration."""
        with patch("novaeval.models.ollama.Client"):
            model = OllamaModel(gpu_cost_per_sec=0.001, pull_on_init=False)

        cost = model.estimate_cost(
            "prompt", "response", total_duration_ns=1000000
        )  # 1ms
        assert cost == 0.000001  # 0.001 seconds * 0.001 per second

    def test_validate_connection_empty_response(self):
        """Test connection validation with empty response."""
        with patch("novaeval.models.ollama.Client") as mock_client:
            mock_response = {"message": {"content": ""}}  # Empty content
            mock_client.return_value.chat.return_value = mock_response

            model = OllamaModel(pull_on_init=False)
            result = model.validate_connection()

            assert (
                result is True
            )  # The method returns True for any response with content field

    def test_validate_connection_missing_content(self):
        """Test connection validation with missing content."""
        with patch("novaeval.models.ollama.Client") as mock_client:
            mock_response = {"message": {}}  # No content field
            mock_client.return_value.chat.return_value = mock_response

            model = OllamaModel(pull_on_init=False)
            result = model.validate_connection()

            assert (
                result is True
            )  # The method returns True for any response with message field

    def test_generate_with_malformed_response(self):
        """Test generation with malformed response object."""
        with patch("novaeval.models.ollama.Client") as mock_client:
            # Response with no message field
            mock_response = {"other_field": "value"}
            mock_client.return_value.chat.return_value = mock_response

            model = OllamaModel(pull_on_init=False)
            response = model.generate("test prompt")

            assert response == ""  # Should handle malformed response gracefully

    def test_generate_with_thought_malformed_response(self):
        """Test thought generation with malformed response."""
        with patch("novaeval.models.ollama.Client") as mock_client:
            # Response with no message field
            mock_response = {"other_field": "value"}
            mock_client.return_value.chat.return_value = mock_response

            model = OllamaModel(pull_on_init=False)
            response, thinking = model.generate_with_thought("test prompt")

            assert response == ""
            assert thinking == ""

    def test_options_merging_complex_scenario(self):
        """Test complex options merging scenario."""
        with patch("novaeval.models.ollama.Client") as mock_client:
            mock_response = Mock()
            mock_response.message.content = "content"
            mock_response.total_duration = 1000000000
            mock_response.prompt_eval_count = 10
            mock_response.eval_count = 20
            mock_client.return_value.chat.return_value = mock_response

            model = OllamaModel(pull_on_init=False)
            model.generate(
                "test prompt",
                options={"existing": "value", "num_ctx": 1024},
                max_tokens=100,
                temperature=0.7,
                top_p=0.9,
                num_gpu=1,
                mirostat=2,
                mirostat_eta=0.1,
                mirostat_tau=5.0,
            )

            call_args = mock_client.return_value.chat.call_args[1]
            options = call_args["options"]
            assert options["existing"] == "value"
            assert options["num_predict"] == 100
            assert options["temperature"] == 0.7
            assert options["top_p"] == 0.9
            assert options["num_ctx"] == 1024
            assert options["num_gpu"] == 1
            assert options["mirostat"] == 2
            assert options["mirostat_eta"] == 0.1
            assert options["mirostat_tau"] == 5.0

    def test_generate_with_all_thinking_variants(self):
        """Test generation with all thinking field variants."""
        thinking_variants = [
            ("thinking", "test thinking"),
            ("reasoning", "test reasoning"),
            ("thought", "test thought"),
            ("thoughts", "test thoughts"),
        ]

        for field_name, field_value in thinking_variants:
            with patch("novaeval.models.ollama.Client") as mock_client:
                mock_response = Mock()
                mock_response.message.content = "content"
                setattr(mock_response.message, field_name, field_value)
                mock_response.total_duration = 1000000000
                mock_response.prompt_eval_count = 10
                mock_response.eval_count = 20
                mock_client.return_value.chat.return_value = mock_response

                model = OllamaModel(pull_on_init=False)
                _response, thinking = model.generate_with_thought("test prompt")

                assert thinking == field_value

    def test_generate_with_nested_thinking_variants(self):
        """Test generation with nested thinking field variants."""
        thinking_variants = [
            ("thinking", "nested thinking"),
            ("reasoning", "nested reasoning"),
            ("thought", "nested thought"),
            ("thoughts", "nested thoughts"),
        ]

        for field_name, field_value in thinking_variants:
            with patch("novaeval.models.ollama.Client") as mock_client:
                mock_response = {"message": {field_name: field_value}}
                mock_client.return_value.chat.return_value = mock_response

                model = OllamaModel(pull_on_init=False)
                _response, thinking = model.generate_with_thought("test prompt")

                assert thinking == field_value

    def test_generate_with_top_level_thinking_variants(self):
        """Test generation with top-level thinking field variants."""
        thinking_variants = [
            ("thinking", "top thinking"),
            ("reasoning", "top reasoning"),
            ("thought", "top thought"),
            ("thoughts", "top thoughts"),
        ]

        for field_name, field_value in thinking_variants:
            with patch("novaeval.models.ollama.Client") as mock_client:
                mock_response = {field_name: field_value}
                mock_client.return_value.chat.return_value = mock_response

                model = OllamaModel(pull_on_init=False)
                _response, thinking = model.generate_with_thought("test prompt")

                assert thinking == field_value

    def test_extract_content_from_response_exception_handling(self):
        """Test content extraction with exceptions during attribute access."""

        # Test exception during attribute access by creating an object that raises on attribute access
        class ExceptionObject:
            def __getattr__(self, name):
                raise Exception("Attribute error")

        mock_response = ExceptionObject()
        content = OllamaModel._extract_content_from_response(mock_response)
        assert content == ""

        # Test exception during getattr(msg, "content", None)
        class MessageExceptionObject:
            def __init__(self):
                self.message = ExceptionObject()

        mock_response = MessageExceptionObject()
        content = OllamaModel._extract_content_from_response(mock_response)
        assert content == ""

        # Test exception during response.get("message", {}).get("content", "")
        class GetExceptionObject:
            def get(self, key, default=None):
                raise Exception("Get method error")

        mock_response = GetExceptionObject()
        content = OllamaModel._extract_content_from_response(mock_response)
        assert content == ""

    def test_extract_metric_exception_handling(self):
        """Test metric extraction with exceptions during attribute access."""

        # Test exception during getattr(response_or_chunk, key, None)
        class ExceptionObject:
            def __getattr__(self, name):
                raise Exception("Attribute error")

        mock_response = ExceptionObject()
        metric = OllamaModel._extract_metric(mock_response, "total_duration")
        assert metric is None

        # Test exception during response_or_chunk.get(key)
        class GetExceptionObject:
            def get(self, key, default=None):
                raise Exception("Get method error")

        mock_response = GetExceptionObject()
        metric = OllamaModel._extract_metric(mock_response, "total_duration")
        assert metric is None

    def test_extract_thinking_from_response_exception_handling(self):
        """Test thinking extraction with exceptions during attribute access."""

        # Test exception during getattr(response, "thinking", None)
        class ExceptionObject:
            def __getattr__(self, name):
                raise Exception("Thinking attribute error")

        mock_response = ExceptionObject()
        thinking = OllamaModel._extract_thinking_from_response(mock_response)
        assert thinking == ""

        # Test exception during getattr(response, "message", None)
        class MessageExceptionObject:
            def __init__(self):
                self.message = ExceptionObject()

        mock_response = MessageExceptionObject()
        thinking = OllamaModel._extract_thinking_from_response(mock_response)
        assert thinking == ""

        # Test exception during getattr(msg, key, None) in the loop
        class MessageKeyExceptionObject:
            def __init__(self):
                self.thinking = ExceptionObject()

        class ResponseWithMessage:
            def __init__(self):
                self.message = MessageKeyExceptionObject()

        mock_response = ResponseWithMessage()
        thinking = OllamaModel._extract_thinking_from_response(mock_response)
        assert thinking == ""

        # Test exception during response.get(key)
        class GetExceptionObject:
            def get(self, key, default=None):
                raise Exception("Top-level get error")

        mock_response = GetExceptionObject()
        thinking = OllamaModel._extract_thinking_from_response(mock_response)
        assert thinking == ""

        # Test exception during response.get("message", {})
        class MessageGetExceptionObject:
            def get(self, key, default=None):
                if key == "message":
                    raise Exception("Message get error")
                return default

        mock_response = MessageGetExceptionObject()
        thinking = OllamaModel._extract_thinking_from_response(mock_response)
        assert thinking == ""

    def test_extract_thinking_from_response_non_string_values(self):
        """Test thinking extraction with non-string values."""
        # Test with None thinking
        mock_response = {"thinking": None}
        thinking = OllamaModel._extract_thinking_from_response(mock_response)
        assert thinking == ""

        # Test with empty string thinking
        mock_response = {"thinking": ""}
        thinking = OllamaModel._extract_thinking_from_response(mock_response)
        assert thinking == ""

        # Test with non-string thinking
        mock_response = {"thinking": 123}
        thinking = OllamaModel._extract_thinking_from_response(mock_response)
        assert thinking == ""

        # Test with boolean thinking
        mock_response = {"thinking": True}
        thinking = OllamaModel._extract_thinking_from_response(mock_response)
        assert thinking == ""

    def test_extract_thinking_from_response_malformed_message(self):
        """Test thinking extraction with malformed message structure."""
        # Test with message that is not a dict
        mock_response = {"message": "not_a_dict"}
        thinking = OllamaModel._extract_thinking_from_response(mock_response)
        assert thinking == ""

        # Test with message that is None
        mock_response = {"message": None}
        thinking = OllamaModel._extract_thinking_from_response(mock_response)
        assert thinking == ""

    def test_extract_metric_non_integer_values(self):
        """Test metric extraction with non-integer values."""
        # Test with string value
        mock_response = {"total_duration": "not_a_number"}
        metric = OllamaModel._extract_metric(mock_response, "total_duration")
        assert metric is None

        # Test with float value
        mock_response = {"total_duration": 1.5}
        metric = OllamaModel._extract_metric(mock_response, "total_duration")
        assert metric is None

        # Test with boolean value - in Python, bool is a subclass of int
        mock_response = {"total_duration": True}
        metric = OllamaModel._extract_metric(mock_response, "total_duration")
        assert metric is True  # True is considered an int in Python

    def test_extract_metric_attribute_access_failure(self):
        """Test metric extraction when attribute access fails."""
        # Test with object that doesn't support attribute access
        mock_response = 42  # Integer doesn't support getattr

        metric = OllamaModel._extract_metric(mock_response, "total_duration")
        assert metric is None

    def test_extract_content_attribute_access_failure(self):
        """Test content extraction when attribute access fails."""
        # Test with object that doesn't support attribute access
        mock_response = 42  # Integer doesn't support getattr

        content = OllamaModel._extract_content_from_response(mock_response)
        assert content == ""

    def test_extract_thinking_attribute_access_failure(self):
        """Test thinking extraction when attribute access fails."""
        # Test with object that doesn't support attribute access
        mock_response = 42  # Integer doesn't support getattr

        thinking = OllamaModel._extract_thinking_from_response(mock_response)
        assert thinking == ""
