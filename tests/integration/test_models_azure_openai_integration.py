"""
Integration tests for the AzureOpenAIModel implementation.

These tests validate the AzureOpenAIModel class against real Azure OpenAI endpoints,
verifying authentication, text generation, cost tracking, and framework integration.
"""

import os
import time

import pytest

from novaeval.models.azure_openai import AzureOpenAIModel

# Test markers
integration_test = pytest.mark.integration
smoke_test = pytest.mark.smoke
requires_api_key = pytest.mark.requires_api_key


@pytest.fixture(scope="session")
def azure_credentials():
    """Provides Azure credentials from environment variables."""
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    base_url = os.getenv("AZURE_OPENAI_BASE_URL")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION")
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
    if not all([api_key, base_url, api_version, deployment]):
        pytest.skip("Azure credentials must be set in environment variables.")
    return {
        "api_key": api_key,
        "base_url": base_url,
        "api_version": api_version,
        "deployment": deployment,
    }


@pytest.fixture
def azure_openai_model_factory(azure_credentials):
    """Factory to create AzureOpenAIModel instances."""

    def _create_model(model_name=None, **kwargs):
        return AzureOpenAIModel(
            model_name=model_name or azure_credentials["deployment"],
            api_key=azure_credentials["api_key"],
            base_url=azure_credentials["base_url"],
            api_version=azure_credentials["api_version"],
            **kwargs,
        )

    return _create_model


@pytest.fixture
def azure_openai_model(azure_openai_model_factory):
    """A default AzureOpenAIModel instance."""
    return azure_openai_model_factory()


@integration_test
@requires_api_key
class TestAzureOpenAIModelIntegration:
    """Core API functionality integration tests."""

    @smoke_test
    def test_model_initialization_with_real_api(self, azure_credentials):
        """Test model initialization connects to the real API."""
        model = AzureOpenAIModel(
            model_name=azure_credentials["deployment"],
            api_key=azure_credentials["api_key"],
            base_url=azure_credentials["base_url"],
            api_version=azure_credentials["api_version"],
        )
        assert model.name.startswith("azure_openai_")
        assert model.model_name == azure_credentials["deployment"]
        assert model.client is not None
        assert model.validate_connection() is True

    def test_text_generation(self, azure_openai_model):
        """Test basic text generation using the compatibility wrapper."""
        prompt = "What is the capital of France?"
        start_time = time.time()
        response = azure_openai_model.generate(prompt=prompt)
        duration = time.time() - start_time
        assert isinstance(response, str)
        assert "paris" in response.lower()
        assert duration < 15.0  # Should be reasonably fast
        assert azure_openai_model.total_requests == 1
        assert azure_openai_model.total_tokens > 0
        assert azure_openai_model.total_cost > 0.0

    def test_chat_generation(self, azure_openai_model):
        """Test chat-based generation."""
        messages = [{"role": "user", "content": "What is 2+2?"}]
        response = azure_openai_model.generate_chat(messages=messages)
        assert isinstance(response, str)
        assert "4" in response
        assert azure_openai_model.total_requests == 1  # Now tracked in generate_chat
        assert azure_openai_model.total_cost > 0.0

    def test_authentication_failure_scenarios(self, azure_credentials):
        """Test that authentication fails with invalid credentials."""
        with pytest.raises(Exception) as excinfo:
            model = AzureOpenAIModel(
                model_name=azure_credentials["deployment"],
                api_key="invalid_key",
                base_url=azure_credentials["base_url"],
                api_version=azure_credentials["api_version"],
            )
            model.generate(prompt="test")
        error_str = str(excinfo.value).lower()
        assert any(
            k in error_str
            for k in ["authenticationfailed", "invalid api key", "401", "access denied"]
        )

    def test_unsupported_model_handling(self, azure_openai_model_factory):
        """Test that using an unsupported model name is handled gracefully."""
        # Azure uses deployment name, so this tests our internal pricing/tracking
        model = azure_openai_model_factory(model_name="unsupported-model-name")
        cost = model.estimate_cost("prompt", "response")
        assert cost == 0.0
        info = model.get_info()
        assert info["pricing"] == (0.0, 0.0)

    def test_rate_limiting_and_retries(self, azure_openai_model):
        """
        This test is difficult to trigger reliably without a special setup.
        It serves as a placeholder for manual or targeted testing.
        """
        pytest.skip("Rate limit testing is environment-dependent and skipped.")


@integration_test
@requires_api_key
class TestAzureOpenAICostTracking:
    """Tests for cost and token tracking."""

    def test_token_counting_accuracy(self, azure_openai_model):
        """Validate token counting against expected values."""
        # Note: This uses a simple word-based approximation.
        test_cases = [
            ("Hello world", 4),
            ("A simple test.", 6),
            ("", 0),
        ]
        for text, expected_tokens in test_cases:
            assert azure_openai_model.count_tokens(text) == expected_tokens

    def test_cost_estimation_accuracy(
        self, azure_openai_model_factory, azure_credentials
    ):
        """Verify that cost estimation is reasonable."""
        model = azure_openai_model_factory(model_name=azure_credentials["deployment"])
        prompt = "Explain quantum computing in simple terms."
        response = model.generate(prompt=prompt, max_tokens=50)

        assert model.total_cost > 0.0
        info = model.get_info()
        in_rate, out_rate = info["pricing"]
        assert in_rate > 0 and out_rate > 0

        input_tokens = model.count_tokens(prompt)
        output_tokens = model.count_tokens(response)
        expected_cost = (input_tokens / 1_000_000) * in_rate + (
            output_tokens / 1_000_000
        ) * out_rate

        # Allow for a small discrepancy
        assert abs(model.total_cost - expected_cost) / expected_cost < 0.5


@integration_test
@requires_api_key
class TestAzureOpenAIAdvancedFeatures:
    """Tests for advanced model features."""

    def test_batch_generation(self, azure_openai_model):
        """Test generating multiple prompts in a batch."""
        prompts = ["1+1=", "2+2=", "3+3="]
        responses = azure_openai_model.generate_batch(prompts=prompts)
        assert len(responses) == 3
        assert "2" in responses[0]
        assert "4" in responses[1]
        assert "6" in responses[2]
        assert azure_openai_model.total_requests == 3

    def test_parameter_passing(self, azure_openai_model):
        """Test passing generation parameters like temperature and max_tokens."""
        response = azure_openai_model.generate(
            prompt="Once upon a time", max_tokens=5, temperature=0.0
        )
        assert len(response.split()) <= 10  # A bit of buffer

        # High temperature should yield more random output
        response1 = azure_openai_model.generate(
            prompt="The meaning of life is", temperature=0.9, max_tokens=10
        )
        response2 = azure_openai_model.generate(
            prompt="The meaning of life is", temperature=0.9, max_tokens=10
        )
        assert response1 != response2

    def test_stop_sequence(self, azure_openai_model):
        """Test that the stop sequence terminates generation."""
        prompt = "Count to 5: 1, 2, 3,"
        response = azure_openai_model.generate(prompt=prompt, max_tokens=20, stop=",")
        assert "," not in response

    def test_get_info_method_accuracy(self, azure_openai_model, azure_credentials):
        """Verify the get_info() method provides accurate data."""
        info = azure_openai_model.get_info()
        assert info["model_name"] == azure_credentials["deployment"]
        assert info["provider"] == "azure_openai"
        assert info["supports_batch"] is False
        assert "pricing" in info and isinstance(info["pricing"], tuple)
        assert "max_retries" in info
        assert "api_version" in info
