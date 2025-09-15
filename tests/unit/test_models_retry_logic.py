"""
Unit tests for retry logic functionality in models.
"""

from unittest.mock import Mock, patch

import pytest

from novaeval.models.base import BaseModel


class ConcreteModel(BaseModel):
    """Concrete implementation of BaseModel for testing."""

    def __init__(self, **kwargs):
        super().__init__(name="test_model", model_name="test-model-v1", **kwargs)
        # Set timeout and max_retries as instance attributes for retry logic
        self.timeout = kwargs.get("timeout", 60.0)
        self.max_retries = kwargs.get("max_retries", 3)

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


class TestRetryLogic:
    """Test cases for retry logic functionality."""

    def test_retry_with_exponential_backoff_success_first_attempt(self):
        """Test retry logic when function succeeds on first attempt."""
        model = ConcreteModel()

        def successful_func():
            return "success"

        result = model._retry_with_exponential_backoff(successful_func)
        assert result == "success"

    def test_retry_with_exponential_backoff_429_error_retries(self):
        """Test retry logic with 429 error that eventually succeeds."""
        model = ConcreteModel()
        call_count = 0

        def failing_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                error = Exception("Rate limit")
                error.status_code = 429
                raise error
            return "success"

        with patch("time.sleep") as mock_sleep:
            result = model._retry_with_exponential_backoff(failing_func)
            assert result == "success"
            assert call_count == 3
            assert mock_sleep.call_count == 2  # Should sleep twice before success

    def test_retry_with_exponential_backoff_429_error_max_retries_exceeded(self):
        """Test retry logic with 429 error that exceeds max retries."""
        model = ConcreteModel(max_retries=2)

        def always_failing_func():
            error = Exception("Rate limit")
            error.status_code = 429
            raise error

        with patch("time.sleep") as mock_sleep:
            result = model._retry_with_exponential_backoff(always_failing_func)
            assert result == ""  # Should return empty string when max retries exceeded
            assert mock_sleep.call_count == 2  # Should sleep twice (between attempts)

    def test_retry_with_exponential_backoff_non_429_error_immediate_raise(self):
        """Test retry logic with non-429 error raises immediately."""
        model = ConcreteModel()

        def failing_func():
            raise ValueError("Not a rate limit error")

        with pytest.raises(ValueError, match="Not a rate limit error"):
            model._retry_with_exponential_backoff(failing_func)

    def test_retry_with_exponential_backoff_exponential_backoff_timing(self):
        """Test that exponential backoff timing is correct."""
        model = ConcreteModel(timeout=1.0, max_retries=2)
        call_count = 0

        def failing_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                error = Exception("Rate limit")
                error.status_code = 429
                raise error
            return "success"

        with patch("time.sleep") as mock_sleep:
            model._retry_with_exponential_backoff(failing_func)

            # Check that sleep was called with increasing timeouts
            expected_sleeps = [1.0, 1.25]  # timeout, timeout * 1.25
            actual_sleeps = [call[0][0] for call in mock_sleep.call_args_list]
            assert actual_sleeps == expected_sleeps

    def test_extract_status_code_direct_attribute(self):
        """Test status code extraction from direct status_code attribute."""
        model = ConcreteModel()

        # Test with status_code attribute
        error = Exception("Test error")
        error.status_code = 429
        assert model._extract_status_code(error) == 429

        error.status_code = 500
        assert model._extract_status_code(error) == 500

    def test_extract_status_code_code_attribute(self):
        """Test status code extraction from code attribute (Google GenAI)."""
        model = ConcreteModel()

        # Test with code attribute
        error = Exception("Test error")
        error.code = 429
        assert model._extract_status_code(error) == 429

    def test_extract_status_code_response_object(self):
        """Test status code extraction from response object."""
        model = ConcreteModel()

        # Test with response.status_code
        error = Exception("Test error")
        response = Mock()
        response.status_code = 429
        error.response = response
        assert model._extract_status_code(error) == 429

        # Test with response.status
        error2 = Exception("Test error 2")
        response2 = Mock()
        response2.status = 500
        # Make sure status_code doesn't exist to test the status path
        del response2.status_code
        error2.response = response2
        assert model._extract_status_code(error2) == 500

        # Test with response.code
        error3 = Exception("Test error 3")
        response3 = Mock()
        response3.code = 403
        # Make sure status_code and status don't exist to test the code path
        del response3.status_code
        del response3.status
        error3.response = response3
        assert model._extract_status_code(error3) == 403

    def test_extract_status_code_args_string(self):
        """Test status code extraction from exception args containing '429'."""
        model = ConcreteModel()

        error = Exception("Rate limit error 429")
        assert model._extract_status_code(error) == 429

        error = Exception("Error with 429 in message")
        assert model._extract_status_code(error) == 429

    def test_extract_status_code_args_dict(self):
        """Test status code extraction from exception args containing dict."""
        model = ConcreteModel()

        error = Exception({"status_code": 429})
        assert model._extract_status_code(error) == 429

        error = Exception({"code": 500})
        assert model._extract_status_code(error) == 500

    def test_extract_status_code_exception_name_patterns(self):
        """Test status code extraction from exception name patterns."""
        model = ConcreteModel()

        # Test RateLimitError
        class RateLimitError(Exception):
            pass

        error = RateLimitError("Rate limit")
        assert model._extract_status_code(error) == 429

        # Test Rate_Limit_Error
        class Rate_Limit_Error(Exception):
            pass

        error = Rate_Limit_Error("Rate limit")
        assert model._extract_status_code(error) == 429

    def test_extract_status_code_response_json(self):
        """Test status code extraction from response_json (Google GenAI)."""
        model = ConcreteModel()

        error = Exception("Test error")
        error.response_json = {"error": {"code": 429}}
        assert model._extract_status_code(error) == 429

    def test_extract_status_code_no_match(self):
        """Test status code extraction when no pattern matches."""
        model = ConcreteModel()

        error = Exception("Some random error")
        assert model._extract_status_code(error) is None

        # Test with None
        assert model._extract_status_code(None) is None

    def test_retry_with_exponential_backoff_custom_max_retries(self):
        """Test retry logic with custom max_retries value."""
        model = ConcreteModel(max_retries=1)
        call_count = 0

        def failing_func():
            nonlocal call_count
            call_count += 1
            error = Exception("Rate limit")
            error.status_code = 429
            raise error

        with patch("time.sleep"):
            result = model._retry_with_exponential_backoff(failing_func)
            assert result == ""  # Should return empty string after 1 retry
            assert call_count == 2  # Initial call + 1 retry

    def test_retry_with_exponential_backoff_custom_timeout(self):
        """Test retry logic with custom timeout value."""
        model = ConcreteModel(timeout=2.0, max_retries=1)
        call_count = 0

        def failing_func():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                error = Exception("Rate limit")
                error.status_code = 429
                raise error
            return "success"

        with patch("time.sleep") as mock_sleep:
            model._retry_with_exponential_backoff(failing_func)
            # Should sleep with the custom timeout value
            assert mock_sleep.call_args[0][0] == 2.0

    def test_retry_with_exponential_backoff_logging(self):
        """Test that retry logic logs appropriate messages."""
        model = ConcreteModel(max_retries=1)

        def failing_func():
            error = Exception("Rate limit")
            error.status_code = 429
            raise error

        with patch("time.sleep"), patch("novaeval.models.base.logger") as mock_logger:
            model._retry_with_exponential_backoff(failing_func)

            # Should log warning for retry attempt
            mock_logger.warning.assert_called()
            warning_call = mock_logger.warning.call_args[0][0]
            assert "Rate limit hit (429)" in warning_call
            assert "Retrying in" in warning_call

            # Should log error when max retries exceeded
            mock_logger.error.assert_called()
            error_call = mock_logger.error.call_args[0][0]
            assert "Rate limit error (429) persisted" in error_call
            assert "Giving up on request" in error_call

    def test_retry_with_exponential_backoff_runtime_error_fallback(self):
        """Test that retry logic raises RuntimeError in unexpected cases."""
        model = ConcreteModel()

        # This should never happen in practice, but test the fallback
        # We need to mock the retry logic to reach the fallback case
        with patch.object(model, "_extract_status_code", return_value=None):

            def failing_func():
                error = Exception("Some error")
                error.status_code = 429  # This won't be detected due to mock
                raise error

            # The retry logic will raise the original exception since it's not a 429
            with pytest.raises(Exception, match="Some error"):
                model._retry_with_exponential_backoff(failing_func)

    def test_extract_status_code_nested_response_attributes(self):
        """Test status code extraction from nested response attributes."""
        model = ConcreteModel()

        # Test with nested response object - the current implementation doesn't handle
        # nested responses, so this test verifies the current behavior
        error = Exception("Test error")
        response = Mock()
        nested_response = Mock()
        nested_response.status_code = 429
        response.response = nested_response
        # Make sure the outer response doesn't have status_code, status, or code to test nested path
        del response.status_code
        del response.status
        del response.code
        error.response = response
        # The current implementation doesn't check nested responses, so this should return None
        assert model._extract_status_code(error) is None

    def test_extract_status_code_multiple_args(self):
        """Test status code extraction from multiple exception args."""
        model = ConcreteModel()

        # Test with multiple args, one containing 429
        error = Exception(
            "Error message with 429", {"status_code": 500}, "Another message"
        )
        assert model._extract_status_code(error) == 429

    def test_retry_with_exponential_backoff_zero_max_retries(self):
        """Test retry logic with zero max retries."""
        model = ConcreteModel(max_retries=0)

        def failing_func():
            error = Exception("Rate limit")
            error.status_code = 429
            raise error

        with patch("time.sleep"):
            result = model._retry_with_exponential_backoff(failing_func)
            assert result == ""  # Should return empty string immediately

    def test_retry_with_exponential_backoff_negative_timeout(self):
        """Test retry logic with negative timeout value."""
        model = ConcreteModel(timeout=-1.0, max_retries=1)

        def failing_func():
            error = Exception("Rate limit")
            error.status_code = 429
            raise error

        with patch("time.sleep") as mock_sleep:
            model._retry_with_exponential_backoff(failing_func)
            # Should still work with negative timeout (though not recommended)
            assert mock_sleep.call_count == 1
