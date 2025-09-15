"""
Pytest configuration for integration tests.

This module provides shared fixtures and configuration for integration tests,
including API key management, test markers, and common utilities.
"""

import os
import time
from typing import Optional

import pytest

from novaeval.models.gemini import GeminiModel


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "integration: mark test as an integration test")
    config.addinivalue_line(
        "markers", "requires_api_key: mark test as requiring API key"
    )
    config.addinivalue_line("markers", "gemini: mark test as Gemini-specific")
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line("markers", "smoke: mark test as a smoke test")
    config.addinivalue_line("markers", "stress: mark test as a stress test")


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically."""
    for item in items:
        # Add integration marker to all tests in integration directory
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)

        # Add gemini marker to Gemini-specific tests
        if "gemini" in str(item.fspath).lower():
            item.add_marker(pytest.mark.gemini)


@pytest.fixture(scope="session")
def gemini_api_key() -> Optional[str]:
    """
    Session-scoped fixture to provide Gemini API key.

    Returns:
        API key if available, None otherwise
    """
    return os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")


@pytest.fixture(scope="session")
def has_gemini_api_key(gemini_api_key) -> bool:
    """
    Session-scoped fixture to check if Gemini API key is available.

    Returns:
        True if API key is available
    """
    return gemini_api_key is not None


@pytest.fixture
def skip_if_no_gemini_key(has_gemini_api_key):
    """Fixture that skips test if no Gemini API key is available."""
    if not has_gemini_api_key:
        pytest.skip("GEMINI_API_KEY or GOOGLE_API_KEY environment variable not set")


def pytest_runtest_setup(item):
    """Setup hook to handle test skipping based on markers."""
    # Skip tests marked as requiring API key if no key is available
    if item.get_closest_marker("requires_api_key"):
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            pytest.skip("API key required but not available")


def pytest_report_header(config):
    """Add custom header information to pytest report."""
    api_key_status = (
        "Available"
        if (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))
        else "Not Available"
    )

    return [
        "Integration Tests Configuration:",
        f"  Gemini API Key: {api_key_status}",
        f"  Test Environment: {os.getenv('TEST_ENV', 'development')}",
    ]


@pytest.fixture
def gemini_model_factory(gemini_api_key):
    """
    Fixture to provide a factory for creating GeminiModel instances with different configurations.

    Args:
        gemini_api_key: API key from the gemini_api_key fixture

    Returns:
        Factory function that creates GeminiModel instances
    """

    def _create_model(model_name="gemini-2.5-flash", **kwargs):
        """Create a GeminiModel with the specified configuration."""
        if not gemini_api_key:
            pytest.skip("No API key available")

        return GeminiModel(model_name=model_name, api_key=gemini_api_key, **kwargs)

    return _create_model


@pytest.fixture
def gemini_model(gemini_model_factory):
    """
    Fixture to provide a default GeminiModel instance for testing.

    Returns:
        GeminiModel instance with default configuration
    """
    return gemini_model_factory()


@pytest.fixture
def gemini_test_config():
    """
    Fixture to provide test configuration for Gemini integration tests.

    Returns:
        Dictionary with test configuration
    """
    return {
        "model_variants": ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash"],
        "timeout": 30.0,
        "max_retries": 3,
        "generation_params": {
            "default": {"temperature": 0.0, "max_tokens": 100},
            "creative": {"temperature": 0.7, "max_tokens": 200},
            "concise": {"temperature": 0.3, "max_tokens": 50},
        },
    }


@pytest.fixture
def measure_execution_time():
    """
    Fixture to provide a function for measuring execution time.

    Returns:
        Function that measures execution time of a callable
    """

    def _measure(func, *args, **kwargs):
        """
        Measure execution time of a function.

        Args:
            func: Function to measure
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
            Tuple of (result, execution_time_seconds)
        """
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            end_time = time.time()
            return result, end_time - start_time
        except Exception as e:
            end_time = time.time()
            # Re-raise the exception but still return timing info
            raise e

    return _measure


@pytest.fixture
def test_prompts():
    """
    Fixture to provide a set of test prompts for integration tests.

    Returns:
        List of test prompts
    """
    from tests.integration.test_utils import TestDataGenerator

    return TestDataGenerator.get_basic_qa_prompts().prompts


@pytest.fixture
def expected_patterns():
    """
    Fixture to provide expected response patterns for test prompts.

    Returns:
        List of expected patterns
    """
    from tests.integration.test_utils import TestDataGenerator

    return TestDataGenerator.get_basic_qa_prompts().expected_patterns


@pytest.fixture
def mock_dataset():
    """
    Fixture to provide a mock dataset for testing.

    Returns:
        CustomDataset instance
    """
    from tests.integration.test_utils import TestDataGenerator

    prompt_set = TestDataGenerator.get_basic_qa_prompts()
    return TestDataGenerator.create_evaluation_dataset(prompt_set)


@pytest.fixture
def response_validator():
    """
    Fixture to provide a response validator.

    Returns:
        ResponseValidator instance
    """
    from tests.integration.test_utils import ResponseValidator

    return ResponseValidator()


@pytest.fixture
def gemini_test_helper():
    """
    Fixture to provide a GeminiTestHelper instance.

    Returns:
        GeminiTestHelper class for static method access
    """
    from tests.integration.test_utils import GeminiTestHelper

    return GeminiTestHelper


@pytest.fixture
def token_counting_test_data():
    """
    Fixture to provide test data for token counting tests.

    Returns:
        List of (text, expected_min_tokens) tuples
    """
    from tests.integration.test_utils import GeminiTestHelper

    return GeminiTestHelper.generate_test_data_for_token_counting()
