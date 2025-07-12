"""
Unit tests for logging utilities.
"""

import logging
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from novaeval.utils.logging import get_logger, setup_logging


class TestLoggingUtilities:
    """Test cases for logging utility functions."""

    def test_get_logger_default(self):
        """Test get_logger with default name."""
        logger = get_logger("test_logger")

        assert isinstance(logger, logging.Logger)
        assert logger.name == "test_logger"

    def test_get_logger_with_name(self):
        """Test getting a logger with custom name."""
        logger = get_logger("test_module")

        assert isinstance(logger, logging.Logger)
        assert logger.name == "test_module"

    def test_get_logger_returns_same_instance(self):
        """Test that get_logger returns the same instance for the same name."""
        logger1 = get_logger("same_name")
        logger2 = get_logger("same_name")

        assert logger1 is logger2

    def test_setup_logging_default(self):
        """Test setup_logging with default parameters."""
        logger = setup_logging()

        assert isinstance(logger, logging.Logger)
        assert logger.level == logging.INFO
        assert len(logger.handlers) >= 1

        # Test that console handler was added
        console_handlers = [
            h for h in logger.handlers if isinstance(h, logging.StreamHandler)
        ]
        assert len(console_handlers) >= 1

    def test_setup_logging_with_level(self):
        """Test setup_logging with custom level."""
        logger = setup_logging(level="DEBUG")

        assert logger.level == logging.DEBUG

    def test_setup_logging_with_file(self):
        """Test setup_logging with file output."""
        import os

        # Use temporary file instead of /tmp/test_log.log
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".log") as f:
            log_file = f.name

        try:
            logger = setup_logging(level="INFO", log_file=log_file)

            assert isinstance(logger, logging.Logger)
            assert logger.level == logging.INFO
            assert len(logger.handlers) >= 2  # Console + file handler

            # Check that file handler was added
            file_handlers = [
                h for h in logger.handlers if isinstance(h, logging.FileHandler)
            ]
            assert len(file_handlers) >= 1

            # Check that file was created
            assert os.path.exists(log_file)

        finally:
            if os.path.exists(log_file):
                os.unlink(log_file)

    def test_setup_logging_with_path_object(self):
        """Test setup_logging with Path object."""
        import os

        # Use temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = Path(temp_dir) / "test.log"

            logger = setup_logging(level="DEBUG", log_file=log_file)

            assert isinstance(logger, logging.Logger)
            assert logger.level == logging.DEBUG
            assert os.path.exists(log_file)

    def test_setup_logging_with_format_string(self):
        """Test setup_logging with custom format string."""
        logger = setup_logging(format_string="%(levelname)s: %(message)s")

        # Check that logger was configured (we can't easily test the format directly)
        assert isinstance(logger, logging.Logger)
        assert len(logger.handlers) >= 1

    def test_setup_logging_creates_parent_directory(self):
        """Test that setup_logging creates parent directories."""
        import os

        # Use temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = Path(temp_dir) / "subdir" / "test.log"

            logger = setup_logging(log_file=log_file)

            assert isinstance(logger, logging.Logger)
            assert os.path.exists(log_file)
            assert os.path.exists(log_file.parent)

    def test_setup_logging_invalid_level(self):
        """Test setup_logging with invalid log level."""
        with pytest.raises(AttributeError):
            setup_logging(level="INVALID_LEVEL")

    def test_setup_logging_numeric_level(self):
        """Test setup_logging with numeric log level."""
        with patch("novaeval.utils.logging.logging") as mock_logging:
            setup_logging(level=10)  # DEBUG level

            mock_logging.basicConfig.assert_called_once()
            call_args = mock_logging.basicConfig.call_args

            assert call_args[1]["level"] == 10


class TestLoggingIntegration:
    """Integration tests for logging functionality."""

    def test_actual_logging_to_file(self):
        """Test that messages are actually written to file."""
        import os

        # Use temporary file
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".log") as f:
            log_file = f.name

        try:
            # Setup logging
            setup_logging(level="INFO", log_file=log_file)

            # Get a specific logger
            test_logger = get_logger("test_integration")
            test_logger.info("This is a test log message")

            # Read file content
            with open(log_file) as f:
                content = f.read()

            # Check that message was written
            assert "This is a test log message" in content
            assert "test_integration" in content

        finally:
            if os.path.exists(log_file):
                os.unlink(log_file)

    def test_multiple_loggers_same_config(self):
        """Test that multiple loggers work with same configuration."""
        import os

        # Use temporary file
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".log") as f:
            log_file = f.name

        try:
            # Setup logging
            setup_logging(level="INFO", log_file=log_file)

            # Get multiple loggers
            logger1 = get_logger("module1")
            logger2 = get_logger("module2")

            # Log from both
            logger1.info("Message from module1")
            logger2.info("Message from module2")

            # Check file content
            with open(log_file) as f:
                content = f.read()

            assert "Message from module1" in content
            assert "Message from module2" in content

        finally:
            if os.path.exists(log_file):
                os.unlink(log_file)

    def test_log_level_filtering(self):
        """Test that log level filtering works correctly."""
        import os

        # Use temporary file
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".log") as f:
            log_file = f.name

        try:
            # Setup logging at WARNING level
            setup_logging(level="WARNING", log_file=log_file)

            # Get logger
            logger = get_logger("test_filtering")

            # Log at different levels
            logger.debug("Debug message - should NOT appear")
            logger.info("Info message - should NOT appear")
            logger.warning("Warning message - should appear")
            logger.error("Error message - should appear")

            # Check file content
            with open(log_file) as f:
                content = f.read()

            # Only WARNING and ERROR should appear
            assert "Debug message" not in content
            assert "Info message" not in content
            assert "Warning message - should appear" in content
            assert "Error message - should appear" in content

        finally:
            if os.path.exists(log_file):
                os.unlink(log_file)


class TestLoggingEdgeCases:
    """Test edge cases for logging utilities."""

    def test_setup_logging_called_multiple_times(self):
        """Test that setup_logging can be called multiple times."""
        import os

        # Use temporary file
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".log") as f:
            log_file = f.name

        try:
            # Setup logging multiple times
            logger1 = setup_logging(level="INFO", log_file=log_file)
            logger2 = setup_logging(level="WARNING", log_file=log_file)

            # Should return the same logger instance
            assert logger1 is logger2

            # Last configuration should win
            assert logger2.level == logging.WARNING

            # Test logging
            test_logger = get_logger("test_multiple_setup")
            test_logger.warning("Test message")

            # Check file content
            with open(log_file) as f:
                content = f.read()

            assert "Test message" in content

        finally:
            if os.path.exists(log_file):
                os.unlink(log_file)

    def test_get_logger_with_empty_name(self):
        """Test get_logger with empty string name."""
        logger = get_logger("")

        assert isinstance(logger, logging.Logger)
        assert logger.name == "root"  # Empty string returns root logger

    def test_get_logger_with_none_name(self):
        """Test get_logger with None name."""
        # logging.getLogger(None) actually returns the root logger, doesn't raise TypeError
        logger = get_logger(None)
        assert isinstance(logger, logging.Logger)
        assert logger.name == "root"  # None gets converted to root logger

    def test_setup_logging_permission_error(self):
        """Test setup_logging when log file cannot be created."""
        import os

        # Use temporary directory approach
        with tempfile.TemporaryDirectory() as temp_dir:
            non_existent_dir = os.path.join(
                temp_dir, "non_existent", "deeply", "nested"
            )
            log_file = os.path.join(non_existent_dir, "test.log")

            # This should handle the error gracefully by creating parent directories
            try:
                logger = setup_logging(log_file=log_file)
                # If it succeeds, check that the file was created
                assert os.path.exists(log_file)
                assert isinstance(logger, logging.Logger)
            except (OSError, PermissionError):
                # If permission error occurs, that's also acceptable
                pass


class TestLoggerMixin:
    """Test LoggerMixin functionality."""

    def test_logger_mixin_basic(self):
        """Test basic LoggerMixin functionality."""
        from novaeval.utils.logging import LoggerMixin

        class TestClass(LoggerMixin):
            pass

        obj = TestClass()

        # Test logger property
        logger = obj.logger
        assert isinstance(logger, logging.Logger)
        assert logger.name == "TestClass"

    def test_logger_mixin_log_methods(self):
        """Test LoggerMixin log methods."""
        import os

        from novaeval.utils.logging import LoggerMixin

        class TestClass(LoggerMixin):
            pass

        obj = TestClass()

        # Create temp log file
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".log") as f:
            log_file = f.name

        try:
            # Setup logging to capture messages
            setup_logging(level="DEBUG", log_file=log_file)

            # Test all log methods
            obj.log_info("Info message")
            obj.log_warning("Warning message")
            obj.log_error("Error message")
            obj.log_debug("Debug message")

            # Check that messages were written to file
            with open(log_file) as f:
                content = f.read()
                assert "Info message" in content
                assert "Warning message" in content
                assert "Error message" in content
                assert "Debug message" in content

        finally:
            if os.path.exists(log_file):
                os.unlink(log_file)


class TestThirdPartyLoggers:
    """Test third-party logger configuration."""

    def test_configure_third_party_loggers_default(self):
        """Test configure_third_party_loggers with default level."""
        from novaeval.utils.logging import configure_third_party_loggers

        configure_third_party_loggers()

        # Check that common noisy loggers are set to WARNING
        noisy_loggers = ["urllib3", "requests", "httpx", "openai", "anthropic"]
        for logger_name in noisy_loggers:
            logger = logging.getLogger(logger_name)
            assert logger.level == logging.WARNING

    def test_configure_third_party_loggers_custom_level(self):
        """Test configure_third_party_loggers with custom level."""
        from novaeval.utils.logging import configure_third_party_loggers

        configure_third_party_loggers(level="ERROR")

        # Check that loggers are set to ERROR
        logger = logging.getLogger("urllib3")
        assert logger.level == logging.ERROR

    def test_configure_third_party_loggers_string_level(self):
        """Test configure_third_party_loggers with string level."""
        from novaeval.utils.logging import configure_third_party_loggers

        configure_third_party_loggers(level="DEBUG")

        # Check that loggers are set to DEBUG
        logger = logging.getLogger("requests")
        assert logger.level == logging.DEBUG


class TestEvaluationLogging:
    """Test evaluation-specific logging functions."""

    def test_log_evaluation_start(self):
        """Test log_evaluation_start function."""
        import os

        from novaeval.utils.logging import log_evaluation_start

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".log") as f:
            log_file = f.name

        try:
            # Setup logging
            setup_logging(level="INFO", log_file=log_file)

            # Test the function
            log_evaluation_start(
                dataset_name="test_dataset",
                model_names=["gpt-4", "claude-3"],
                scorer_names=["accuracy", "f1"],
                num_samples=100,
            )

            # Check log content
            with open(log_file) as f:
                content = f.read()
                assert "Starting NovaEval evaluation" in content
                assert "test_dataset" in content
                assert "gpt-4, claude-3" in content
                assert "accuracy, f1" in content
                assert "100" in content

        finally:
            if os.path.exists(log_file):
                os.unlink(log_file)

    def test_log_evaluation_end(self):
        """Test log_evaluation_end function."""
        import os

        from novaeval.utils.logging import log_evaluation_end

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".log") as f:
            log_file = f.name

        try:
            # Setup logging
            setup_logging(level="INFO", log_file=log_file)

            # Test the function
            log_evaluation_end(
                duration=120.5, total_requests=50, total_tokens=10000, total_cost=5.25
            )

            # Check log content
            with open(log_file) as f:
                content = f.read()
                assert "NovaEval evaluation completed" in content
                assert "120.50 seconds" in content
                assert "50" in content
                assert "10,000" in content  # Numbers are formatted with commas
                assert "5.25" in content

        finally:
            if os.path.exists(log_file):
                os.unlink(log_file)

    def test_log_model_results(self):
        """Test log_model_results function."""
        import os

        from novaeval.utils.logging import log_model_results

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".log") as f:
            log_file = f.name

        try:
            # Setup logging
            setup_logging(level="INFO", log_file=log_file)

            # Test the function with correct results format
            results = {
                "scores": {"accuracy": {"mean": 0.85}, "f1": {"mean": 0.78}},
                "errors": [],
            }
            log_model_results("gpt-4", results)

            # Check log content
            with open(log_file) as f:
                content = f.read()
                assert "gpt-4" in content
                assert "accuracy" in content
                assert "f1" in content
                assert (
                    "0.8500" in content
                )  # Mean values are formatted with 4 decimal places
                assert "0.7800" in content

        finally:
            if os.path.exists(log_file):
                os.unlink(log_file)


class TestLoggingSetupVariations:
    """Test different logging setup variations."""

    def test_setup_logging_format_options(self):
        """Test setup_logging with different format options."""
        # Test with timestamp disabled
        logger = setup_logging(include_timestamp=False)
        assert isinstance(logger, logging.Logger)

        # Test with level disabled
        logger = setup_logging(include_level=False)
        assert isinstance(logger, logging.Logger)

        # Test with name disabled
        logger = setup_logging(include_name=False)
        assert isinstance(logger, logging.Logger)

        # Test with all disabled
        logger = setup_logging(
            include_timestamp=False, include_level=False, include_name=False
        )
        assert isinstance(logger, logging.Logger)

    def test_setup_logging_integer_level(self):
        """Test setup_logging with integer log level."""
        logger = setup_logging(level=logging.DEBUG)
        assert logger.level == logging.DEBUG

        logger = setup_logging(level=logging.ERROR)
        assert logger.level == logging.ERROR
