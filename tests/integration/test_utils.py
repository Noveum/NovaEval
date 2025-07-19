"""
Utility functions for integration tests.

This module provides helper functions for test data generation,
response validation, and common test operations for integration testing.
"""

import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from novaeval.datasets.custom import CustomDataset


@dataclass
class GeminiTestConfig:
    """Configuration for Gemini integration tests."""

    api_key: Optional[str] = None
    model_name: str = "gemini-2.5-flash"
    test_prompts: list[str] = field(default_factory=list)
    expected_patterns: list[str] = field(default_factory=list)
    timeout: float = 30.0
    max_retries: int = 3
    generation_params: dict[str, Any] = field(default_factory=dict)


@dataclass
class IntegrationTestResult:
    """Results tracking for integration tests."""

    test_name: str
    success: bool
    response_time: float
    tokens_used: int = 0
    cost_estimate: float = 0.0
    error_message: Optional[str] = None


@dataclass
class TestPromptSet:
    """A set of test prompts with expected response patterns."""

    name: str
    prompts: list[str]
    expected_patterns: list[str]
    description: str


class TestDataGenerator:
    """Generator for various types of test data."""

    @staticmethod
    def get_basic_qa_prompts() -> TestPromptSet:
        """Get basic question-answer test prompts."""
        return TestPromptSet(
            name="basic_qa",
            prompts=[
                "What is 2+2?",
                "What is the capital of France?",
                "What color is the sky?",
                "What is the largest planet in our solar system?",
                "Who wrote Romeo and Juliet?",
            ],
            expected_patterns=["4", "Paris", "blue", "Jupiter", "Shakespeare"],
            description="Basic factual questions with clear answers",
        )

    @staticmethod
    def get_math_prompts() -> TestPromptSet:
        """Get mathematical calculation prompts."""
        return TestPromptSet(
            name="math",
            prompts=[
                "Calculate 15 + 27",
                "What is 8 * 9?",
                "What is 100 divided by 4?",
                "What is the square root of 16?",
                "What is 2 to the power of 5?",
            ],
            expected_patterns=["42", "72", "25", "4", "32"],
            description="Mathematical calculations",
        )

    @staticmethod
    def get_science_prompts() -> TestPromptSet:
        """Get science-related prompts."""
        return TestPromptSet(
            name="science",
            prompts=[
                "What is the chemical symbol for water?",
                "How many bones are in the human body?",
                "What is the speed of light?",
                "What gas do plants absorb during photosynthesis?",
                "What is the smallest unit of matter?",
            ],
            expected_patterns=["H2O", "206", "299,792,458", "carbon dioxide", "atom"],
            description="Basic science questions",
        )

    @staticmethod
    def get_edge_case_prompts() -> TestPromptSet:
        """Get edge case prompts for testing robustness."""
        return TestPromptSet(
            name="edge_cases",
            prompts=[
                "",  # Empty prompt
                "A" * 1000,  # Very long prompt
                "What is 🌟 + 🌙?",  # Unicode characters
                "Tell me about\n\nmultiple\n\nline\n\nbreaks",  # Multiple line breaks
                "What is the answer to life, the universe, and everything?"
                * 10,  # Repetitive long prompt
            ],
            expected_patterns=[
                "",  # May return empty or error
                "A",  # Should handle long input
                "🌟",  # Should handle unicode
                "multiple",  # Should handle line breaks
                "42",  # Should handle repetitive content
            ],
            description="Edge cases and unusual inputs",
        )

    @staticmethod
    def get_all_prompt_sets() -> list[TestPromptSet]:
        """Get all available prompt sets."""
        return [
            TestDataGenerator.get_basic_qa_prompts(),
            TestDataGenerator.get_math_prompts(),
            TestDataGenerator.get_science_prompts(),
            TestDataGenerator.get_edge_case_prompts(),
        ]

    @staticmethod
    def create_evaluation_dataset(prompt_set: TestPromptSet) -> CustomDataset:
        """Create a CustomDataset from a TestPromptSet."""
        samples = [
            {"input": prompt, "expected": pattern}
            for prompt, pattern in zip(prompt_set.prompts, prompt_set.expected_patterns)
        ]

        return CustomDataset(
            data_source=samples, input_column="input", target_column="expected"
        )


class ResponseValidator:
    """Utilities for validating model responses."""

    @staticmethod
    def is_valid_response(response: str) -> bool:
        """
        Check if a response meets basic validity criteria.

        Args:
            response: The response to validate

        Returns:
            True if response is valid
        """
        if not isinstance(response, str):
            return False

        # Response should not be empty (unless intentionally so)
        if len(response.strip()) == 0:
            return False

        # Response should not be excessively long (likely an error)
        return not len(response) > 10000

    @staticmethod
    def contains_expected_content(
        response: str, expected: str, case_sensitive: bool = False
    ) -> bool:
        """
        Check if response contains expected content.

        Args:
            response: The response to check
            expected: The expected content
            case_sensitive: Whether to perform case-sensitive matching

        Returns:
            True if expected content is found
        """
        if not response or not expected:
            return False

        search_response = response if case_sensitive else response.lower()
        search_expected = expected if case_sensitive else expected.lower()

        return search_expected in search_response

    @staticmethod
    def validate_numeric_response(
        response: str, expected_number: float, tolerance: float = 0.01
    ) -> bool:
        """
        Validate numeric responses with tolerance.

        Args:
            response: The response containing a number
            expected_number: The expected numeric value
            tolerance: Acceptable tolerance for floating point comparison

        Returns:
            True if numeric value is within tolerance
        """

        # Extract numbers from response
        numbers = re.findall(r"-?\d+\.?\d*", response)

        if not numbers:
            return False

        try:
            # Check if any extracted number matches expected value
            for num_str in numbers:
                num_value = float(num_str)
                if abs(num_value - expected_number) <= tolerance:
                    return True
        except ValueError:
            pass

        return False

    @staticmethod
    def validate_response_length(
        response: str, min_length: int = 1, max_length: int = 1000
    ) -> bool:
        """
        Validate response length is within acceptable bounds.

        Args:
            response: The response to validate
            min_length: Minimum acceptable length
            max_length: Maximum acceptable length

        Returns:
            True if length is acceptable
        """
        if not isinstance(response, str):
            return False

        response_length = len(response.strip())
        return min_length <= response_length <= max_length


class PerformanceMeasurer:
    """Utilities for measuring test performance."""

    @staticmethod
    def measure_execution_time(func: Callable, *args, **kwargs) -> tuple[Any, float]:
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

    @staticmethod
    def measure_batch_performance(
        func: Callable, batch_args: list[tuple], **kwargs
    ) -> dict[str, Any]:
        """
        Measure performance of batch operations.

        Args:
            func: Function to measure
            batch_args: List of argument tuples for each call
            **kwargs: Common keyword arguments

        Returns:
            Dictionary with performance metrics
        """
        start_time = time.time()
        results = []
        individual_times = []
        errors = 0

        for args in batch_args:
            try:
                result, exec_time = PerformanceMeasurer.measure_execution_time(
                    func, *args, **kwargs
                )
                results.append(result)
                individual_times.append(exec_time)
            except Exception:
                errors += 1
                results.append(None)
                individual_times.append(0.0)

        total_time = time.time() - start_time

        return {
            "results": results,
            "total_time": total_time,
            "individual_times": individual_times,
            "average_time": (
                sum(individual_times) / len(individual_times) if individual_times else 0
            ),
            "min_time": min(individual_times) if individual_times else 0,
            "max_time": max(individual_times) if individual_times else 0,
            "success_count": len(batch_args) - errors,
            "error_count": errors,
            "success_rate": (
                (len(batch_args) - errors) / len(batch_args) if batch_args else 0
            ),
        }


class TestConfigManager:
    """Manager for test configurations and settings."""

    @staticmethod
    def get_test_model_configs() -> list[dict[str, Any]]:
        """Get configurations for different Gemini models to test."""
        return [
            {
                "model_name": "gemini-2.5-flash",
                "timeout": 30.0,
                "max_retries": 3,
                "description": "Latest fast model",
            },
            {
                "model_name": "gemini-2.0-flash",
                "timeout": 30.0,
                "max_retries": 3,
                "description": "Previous generation fast model",
            },
            {
                "model_name": "gemini-1.5-flash",
                "timeout": 30.0,
                "max_retries": 3,
                "description": "Stable fast model",
            },
        ]

    @staticmethod
    def get_generation_parameter_sets() -> list[dict[str, Any]]:
        """Get different parameter sets for generation testing."""
        return [
            {
                "temperature": 0.0,
                "max_tokens": 100,
                "description": "Deterministic generation",
            },
            {
                "temperature": 0.7,
                "max_tokens": 200,
                "description": "Balanced creativity",
            },
            {
                "temperature": 1.0,
                "max_tokens": 50,
                "description": "High creativity, short response",
            },
        ]

    @staticmethod
    def should_run_slow_tests() -> bool:
        """Check if slow tests should be run based on environment."""
        import os

        return os.getenv("RUN_SLOW_TESTS", "false").lower() in ("true", "1", "yes")

    @staticmethod
    def should_run_stress_tests() -> bool:
        """Check if stress tests should be run based on environment."""
        import os

        return os.getenv("RUN_STRESS_TESTS", "false").lower() in ("true", "1", "yes")


class MockDataHelper:
    """Helper for creating mock data for testing."""

    @staticmethod
    def create_mock_responses(count: int = 5) -> list[str]:
        """Create mock responses for testing."""
        return [
            "This is a mock response.",
            "Another mock response with different content.",
            "Mock response number three.",
            "Fourth mock response for testing.",
            "Final mock response in the set.",
        ][:count]

    @staticmethod
    def create_mock_evaluation_samples(count: int = 10) -> list[dict[str, str]]:
        """Create mock evaluation samples."""
        samples = [
            {
                "input": f"Test question {i + 1}?",
                "expected": f"Expected answer {i + 1}",
                "category": f"category_{i % 3}",
            }
            for i in range(count)
        ]

        return samples


class GeminiTestHelper:
    """Helper utilities specifically for Gemini model testing."""

    @staticmethod
    def create_test_config(
        model_name: str = "gemini-2.5-flash",
        api_key: Optional[str] = None,
        prompt_set: Optional[TestPromptSet] = None,
    ) -> GeminiTestConfig:
        """
        Create a test configuration for Gemini integration tests.

        Args:
            model_name: Name of the Gemini model to test
            api_key: API key for authentication
            prompt_set: Optional prompt set to use for testing

        Returns:
            GeminiTestConfig instance
        """
        if prompt_set is None:
            prompt_set = TestDataGenerator.get_basic_qa_prompts()

        return GeminiTestConfig(
            api_key=api_key,
            model_name=model_name,
            test_prompts=prompt_set.prompts,
            expected_patterns=prompt_set.expected_patterns,
            timeout=30.0,
            max_retries=3,
            generation_params={"temperature": 0.0, "max_tokens": 100},
        )

    @staticmethod
    def simulate_api_error(error_type: str = "authentication") -> Exception:
        """
        Create a simulated API error for testing error handling.

        Args:
            error_type: Type of error to simulate

        Returns:
            Exception instance
        """
        error_types = {
            "authentication": ValueError("Invalid API key"),
            "rate_limit": ValueError("Rate limit exceeded"),
            "timeout": TimeoutError("Request timed out"),
            "connection": ConnectionError("Failed to connect to API"),
            "invalid_request": ValueError("Invalid request parameters"),
            "server": Exception("Internal server error"),
        }

        return error_types.get(error_type, Exception(f"Simulated {error_type} error"))

    @staticmethod
    def generate_test_data_for_token_counting() -> list[tuple[str, int]]:
        """
        Generate test data for token counting tests.

        Returns:
            List of (text, expected_min_tokens) tuples
        """
        return [
            ("Hello world", 2),
            ("", 0),
            ("The quick brown fox jumps over the lazy dog", 9),
            ("A" * 100, 25),
            (
                "This is a longer sentence with multiple words that should be tokenized into individual tokens.",
                15,
            ),
            ("Special characters like !@#$%^&*() should also be counted properly.", 12),
            ("Numbers like 123456789 should be tokenized differently than words.", 10),
            ("Unicode characters like 🌟🌙🌎 should be handled correctly.", 10),
            (
                "Technical terms like API, JSON, and HTTP should be counted appropriately.",
                12,
            ),
            ("Very long text " + "word " * 100, 102),
            # Additional test cases for edge cases
            (" " * 20, 0),  # Just spaces
            ("\n\n\n\n", 0),  # Just newlines
            ("Line 1\nLine 2\nLine 3", 6),  # Text with newlines
            ("こんにちは 你好 안녕하세요", 3),  # Non-Latin characters
            ("I love 🌟 these emojis 🎉 so much! 🚀", 7),  # Emojis
            ("a", 1),  # Single character
            ("1234567890", 1),  # Just numbers
            ("https://www.example.com/path?query=value#fragment", 5),  # URL
            ("email@example.com", 3),  # Email address
            ("Short words: a an the of to", 6),  # Short words
            ("LongWordsThatMightGetTokenizedDifferently", 5),  # Long compound words
            ("Mixed CASE text with some ALL CAPS", 7),  # Mixed case
        ]

    @staticmethod
    def validate_gemini_response(
        response: str,
        expected_pattern: Optional[str] = None,
        min_length: int = 1,
        max_length: int = 10000,
    ) -> tuple[bool, Optional[str]]:
        """
        Validate a Gemini model response against multiple criteria.

        Args:
            response: The response to validate
            expected_pattern: Optional pattern to check for
            min_length: Minimum acceptable length
            max_length: Maximum acceptable length

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check if response is a string
        if not isinstance(response, str):
            return False, "Response is not a string"

        # Check if response is empty
        if len(response.strip()) == 0:
            return False, "Response is empty"

        # Check if response is too long
        if len(response) > max_length:
            return (
                False,
                f"Response exceeds maximum length ({len(response)} > {max_length})",
            )

        # Check if response is too short
        if len(response.strip()) < min_length:
            return (
                False,
                f"Response is too short ({len(response.strip())} < {min_length})",
            )

        # Check for expected pattern if provided
        if expected_pattern and expected_pattern.lower() not in response.lower():
            return False, f"Expected pattern '{expected_pattern}' not found in response"

        return True, None

    @staticmethod
    def track_test_result(
        test_name: str,
        success: bool,
        response_time: float,
        tokens_used: int = 0,
        cost_estimate: float = 0.0,
        error_message: Optional[str] = None,
    ) -> IntegrationTestResult:
        """
        Track the result of an integration test.

        Args:
            test_name: Name of the test
            success: Whether the test was successful
            response_time: Time taken for the response
            tokens_used: Number of tokens used
            cost_estimate: Estimated cost of the API call
            error_message: Optional error message if test failed

        Returns:
            IntegrationTestResult instance
        """
        return IntegrationTestResult(
            test_name=test_name,
            success=success,
            response_time=response_time,
            tokens_used=tokens_used,
            cost_estimate=cost_estimate,
            error_message=error_message,
        )

    @staticmethod
    def save_test_results(
        results: list[IntegrationTestResult], filename: str = "gemini_test_results.json"
    ) -> str:
        """
        Save test results to a JSON file.

        Args:
            results: List of test results
            filename: Name of the file to save results to

        Returns:
            Path to the saved file
        """
        # Convert results to dictionaries
        results_dict = [
            {
                "test_name": r.test_name,
                "success": r.success,
                "response_time": r.response_time,
                "tokens_used": r.tokens_used,
                "cost_estimate": r.cost_estimate,
                "error_message": r.error_message,
            }
            for r in results
        ]

        # Create results directory if it doesn't exist
        os.makedirs("test_results", exist_ok=True)

        # Save results to file
        filepath = os.path.join("test_results", filename)
        with open(filepath, "w") as f:
            json.dump(results_dict, f, indent=2)

        return filepath
