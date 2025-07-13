"""
Unit tests for CLI functionality.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import yaml
from click.testing import CliRunner

from novaeval.cli import _display_config_summary, _display_results_summary, cli, main
from novaeval.utils.config import Config


class TestCLI:
    """Test cases for CLI functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()

    def test_cli_group_help(self):
        """Test CLI group help command."""
        result = self.runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert (
            "NovaEval: A comprehensive AI model evaluation framework" in result.output
        )

    def test_cli_version(self):
        """Test CLI version command."""
        result = self.runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert "version" in result.output.lower()

    def test_cli_log_level_option(self):
        """Test CLI log level option."""
        with patch("novaeval.cli.setup_logging") as mock_setup:
            result = self.runner.invoke(cli, ["--log-level", "DEBUG", "list-datasets"])
            assert result.exit_code == 0
            mock_setup.assert_called_with(level="DEBUG", log_file=None)

    def test_cli_log_file_option(self):
        """Test CLI log file option."""
        with patch("novaeval.cli.setup_logging") as mock_setup:
            result = self.runner.invoke(
                cli, ["--log-file", "/tmp/test.log", "list-datasets"]
            )
            assert result.exit_code == 0
            mock_setup.assert_called_with(level="INFO", log_file="/tmp/test.log")

    def test_cli_invalid_log_level(self):
        """Test CLI with invalid log level."""
        result = self.runner.invoke(cli, ["--log-level", "INVALID"])
        assert result.exit_code != 0

    def test_main_function(self):
        """Test main function entry point."""
        with patch("novaeval.cli.cli") as mock_cli:
            main()
            mock_cli.assert_called_once()


class TestRunCommand:
    """Test cases for the run command."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()

    def test_run_with_valid_config(self):
        """Test run command with valid configuration."""
        config_data = {
            "dataset": {"type": "custom", "path": "test.jsonl"},
            "models": [{"type": "openai", "model_name": "gpt-3.5-turbo"}],
            "scorers": [{"type": "accuracy"}],
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name

        try:
            with (
                patch("novaeval.cli.Config.load") as mock_load,
                patch("novaeval.cli.Evaluator.from_config") as mock_evaluator,
            ):

                mock_config = Mock()
                mock_load.return_value = mock_config
                mock_evaluator.return_value.run.return_value = {
                    "model_results": {
                        "test_model": {"scores": {"accuracy": {"mean": 0.8}}}
                    },
                    "summary": {"total_samples": 10},
                }

                result = self.runner.invoke(cli, ["run", config_path])

                assert result.exit_code == 0
                assert "Evaluation completed successfully" in result.output
                mock_load.assert_called_once_with(config_path)
                mock_evaluator.assert_called_once_with(config_path)
        finally:
            Path(config_path).unlink()

    def test_run_with_output_dir(self):
        """Test run command with output directory option."""
        config_data = {"dataset": {"type": "custom"}}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name

        try:
            with (
                patch("novaeval.cli.Config.load") as mock_load,
                patch("novaeval.cli.Evaluator.from_config") as mock_evaluator,
            ):

                mock_config = Mock()
                mock_load.return_value = mock_config
                mock_evaluator.return_value.run.return_value = {"model_results": {}}

                result = self.runner.invoke(
                    cli, ["run", config_path, "--output-dir", "/tmp/output"]
                )

                assert result.exit_code == 0
                mock_config.set.assert_called_with("output.directory", "/tmp/output")
        finally:
            Path(config_path).unlink()

    def test_run_with_dry_run(self):
        """Test run command with dry run option."""
        config_data = {"dataset": {"type": "custom"}}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name

        try:
            with (
                patch("novaeval.cli.Config.load") as mock_load,
                patch("novaeval.cli._display_config_summary") as mock_display,
            ):

                mock_config = Mock()
                mock_load.return_value = mock_config

                result = self.runner.invoke(cli, ["run", config_path, "--dry-run"])

                assert result.exit_code == 0
                assert "Configuration is valid" in result.output
                mock_display.assert_called_once_with(mock_config)
        finally:
            Path(config_path).unlink()

    def test_run_with_nonexistent_config(self):
        """Test run command with nonexistent configuration file."""
        result = self.runner.invoke(cli, ["run", "/nonexistent/config.yaml"])
        assert result.exit_code != 0

    def test_run_with_exception(self):
        """Test run command with exception during evaluation."""
        config_data = {"dataset": {"type": "custom"}}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name

        try:
            with patch("novaeval.cli.Config.load") as mock_load:
                mock_load.side_effect = Exception("Test error")

                result = self.runner.invoke(cli, ["run", config_path])

                assert result.exit_code == 1
                assert "Test error" in result.output
        finally:
            Path(config_path).unlink()


class TestQuickCommand:
    """Test cases for the quick command."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()

    def test_quick_basic_usage(self):
        """Test quick command basic usage."""
        result = self.runner.invoke(
            cli,
            ["quick", "--dataset", "mmlu", "--model", "openai", "--scorer", "accuracy"],
        )

        assert result.exit_code == 0
        assert "Starting quick evaluation" in result.output
        assert "Dataset: mmlu" in result.output
        assert "Models: openai" in result.output
        assert "Scorers: accuracy" in result.output
        assert "Quick evaluation not yet implemented" in result.output

    def test_quick_multiple_models(self):
        """Test quick command with multiple models."""
        result = self.runner.invoke(
            cli,
            [
                "quick",
                "--dataset",
                "mmlu",
                "--model",
                "openai",
                "--model",
                "anthropic",
                "--scorer",
                "accuracy",
            ],
        )

        assert result.exit_code == 0
        assert "Models: openai, anthropic" in result.output

    def test_quick_multiple_scorers(self):
        """Test quick command with multiple scorers."""
        result = self.runner.invoke(
            cli,
            [
                "quick",
                "--dataset",
                "mmlu",
                "--model",
                "openai",
                "--scorer",
                "accuracy",
                "--scorer",
                "f1",
            ],
        )

        assert result.exit_code == 0
        assert "Scorers: accuracy, f1" in result.output

    def test_quick_with_num_samples(self):
        """Test quick command with number of samples."""
        result = self.runner.invoke(
            cli,
            ["quick", "--dataset", "mmlu", "--model", "openai", "--num-samples", "100"],
        )

        assert result.exit_code == 0

    def test_quick_with_output_dir(self):
        """Test quick command with output directory."""
        result = self.runner.invoke(
            cli,
            [
                "quick",
                "--dataset",
                "mmlu",
                "--model",
                "openai",
                "--output-dir",
                "/tmp/results",
            ],
        )

        assert result.exit_code == 0

    def test_quick_missing_dataset(self):
        """Test quick command with missing dataset."""
        result = self.runner.invoke(cli, ["quick", "--model", "openai"])

        assert result.exit_code != 0

    def test_quick_missing_model(self):
        """Test quick command with missing model."""
        result = self.runner.invoke(cli, ["quick", "--dataset", "mmlu"])

        assert result.exit_code != 0

    def test_quick_with_exception(self):
        """Test quick command with exception."""
        with patch("novaeval.cli.console.print", side_effect=Exception("Test error")):
            result = self.runner.invoke(
                cli, ["quick", "--dataset", "mmlu", "--model", "openai"]
            )

            assert result.exit_code == 1


class TestListCommands:
    """Test cases for list commands."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()

    def test_list_datasets(self):
        """Test list datasets command."""
        result = self.runner.invoke(cli, ["list-datasets"])

        assert result.exit_code == 0
        assert "Available Datasets" in result.output
        assert "mmlu" in result.output
        assert "hellaswag" in result.output
        assert "Massive Multitask Language Understanding" in result.output

    def test_list_models(self):
        """Test list models command."""
        result = self.runner.invoke(cli, ["list-models"])

        assert result.exit_code == 0
        assert "Available Model Providers" in result.output
        assert "openai" in result.output
        assert "anthropic" in result.output
        assert "OpenAI GPT models" in result.output

    def test_list_scorers(self):
        """Test list scorers command."""
        result = self.runner.invoke(cli, ["list-scorers"])

        assert result.exit_code == 0
        assert "Available Scorers" in result.output
        assert "accuracy" in result.output
        assert "exact_match" in result.output
        assert "Classification accuracy" in result.output


class TestGenerateConfigCommand:
    """Test cases for the generate-config command."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()

    def test_generate_config_yaml(self):
        """Test generate config command with YAML output."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            output_path = f.name

        try:
            with patch("novaeval.cli.Config") as mock_config:
                mock_config_instance = Mock()
                mock_config.return_value = mock_config_instance

                result = self.runner.invoke(cli, ["generate-config", output_path])

                assert result.exit_code == 0
                assert "Sample configuration saved to" in result.output
                assert Path(output_path).name in result.output
                mock_config_instance.save.assert_called_once_with(
                    output_path, format="yaml"
                )
        finally:
            Path(output_path).unlink(missing_ok=True)

    def test_generate_config_json(self):
        """Test generate config command with JSON output."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            output_path = f.name

        try:
            with patch("novaeval.cli.Config") as mock_config:
                mock_config_instance = Mock()
                mock_config.return_value = mock_config_instance

                result = self.runner.invoke(cli, ["generate-config", output_path])

                assert result.exit_code == 0
                assert "Sample configuration saved to" in result.output
                assert Path(output_path).name in result.output
                mock_config_instance.save.assert_called_once_with(
                    output_path, format="json"
                )
        finally:
            Path(output_path).unlink(missing_ok=True)

    def test_generate_config_default_format(self):
        """Test generate config command with default format."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            output_path = f.name

        try:
            with patch("novaeval.cli.Config") as mock_config:
                mock_config_instance = Mock()
                mock_config.return_value = mock_config_instance

                result = self.runner.invoke(cli, ["generate-config", output_path])

                assert result.exit_code == 0
                mock_config_instance.save.assert_called_once_with(
                    output_path, format="json"
                )
        finally:
            Path(output_path).unlink(missing_ok=True)

    def test_generate_config_with_expected_structure(self):
        """Test that generated config has expected structure."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            output_path = f.name

        try:
            result = self.runner.invoke(cli, ["generate-config", output_path])

            assert result.exit_code == 0

            # Verify the config file was created
            config_path = Path(output_path)
            assert config_path.exists()

            # Read and parse the generated config
            with open(config_path, encoding="utf-8") as file:
                if config_path.suffix.lower() in [".yaml", ".yml"]:
                    config_data = yaml.safe_load(file)
                else:
                    config_data = json.load(file)

            # Verify the expected structure
            assert "dataset" in config_data
            assert "models" in config_data
            assert "scorers" in config_data
            assert "output" in config_data
            assert "evaluation" in config_data

            # Verify dataset structure
            dataset = config_data["dataset"]
            assert "type" in dataset
            assert dataset["type"] == "mmlu"
            assert "subset" in dataset
            assert "num_samples" in dataset

            # Verify models structure
            models = config_data["models"]
            assert isinstance(models, list)
            assert len(models) > 0
            model = models[0]
            assert "type" in model
            assert "model_name" in model
            assert "temperature" in model

            # Verify scorers structure
            scorers = config_data["scorers"]
            assert isinstance(scorers, list)
            assert len(scorers) > 0
            scorer = scorers[0]
            assert "type" in scorer

            # Verify output structure
            output = config_data["output"]
            assert "directory" in output
            assert "formats" in output
            assert isinstance(output["formats"], list)

            # Verify evaluation structure
            evaluation = config_data["evaluation"]
            assert "max_workers" in evaluation
            assert "batch_size" in evaluation

        finally:
            Path(output_path).unlink(missing_ok=True)

    def test_generate_config_json_structure(self):
        """Test that generated JSON config has expected structure."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            output_path = f.name

        try:
            result = self.runner.invoke(cli, ["generate-config", output_path])

            assert result.exit_code == 0

            # Verify the config file was created
            config_path = Path(output_path)
            assert config_path.exists()

            # Read and parse the generated JSON config
            with open(config_path, encoding="utf-8") as file:
                config_data = json.load(file)

            # Verify the expected structure (same as YAML test)
            assert "dataset" in config_data
            assert "models" in config_data
            assert "scorers" in config_data
            assert "output" in config_data
            assert "evaluation" in config_data

            # Verify specific values match expected sample config
            assert config_data["dataset"]["type"] == "mmlu"
            assert config_data["dataset"]["subset"] == "abstract_algebra"
            assert config_data["dataset"]["num_samples"] == 100

            assert len(config_data["models"]) == 1
            assert config_data["models"][0]["type"] == "openai"
            assert config_data["models"][0]["model_name"] == "gpt-4"
            assert config_data["models"][0]["temperature"] == 0.0

            assert len(config_data["scorers"]) == 1
            assert config_data["scorers"][0]["type"] == "accuracy"

            assert config_data["output"]["directory"] == "./results"
            assert "json" in config_data["output"]["formats"]
            assert "csv" in config_data["output"]["formats"]
            assert "html" in config_data["output"]["formats"]

            assert config_data["evaluation"]["max_workers"] == 4
            assert config_data["evaluation"]["batch_size"] == 1

        finally:
            Path(output_path).unlink(missing_ok=True)


class TestDisplayFunctions:
    """Test cases for display helper functions."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()

    def test_display_config_summary(self):
        """Test display config summary function."""

        config = Config(
            {
                "dataset": {"type": "mmlu", "subset": "math"},
                "models": [
                    {"type": "openai", "model_name": "gpt-4"},
                    {"type": "anthropic", "model_name": "claude-3"},
                ],
                "scorers": [{"type": "accuracy"}, {"type": "f1"}],
                "output": {"directory": "/tmp/results"},
            }
        )

        with patch("novaeval.cli.console.print") as mock_print:
            _display_config_summary(config)

            # Verify that console.print was called with a table
            mock_print.assert_called_once()
            # The argument should be a Rich Table object
            args, kwargs = mock_print.call_args
            assert hasattr(args[0], "add_row")  # Rich Table has add_row method

    def test_display_results_summary(self):
        """Test display results summary function."""

        results = {
            "model_results": {
                "model1": {
                    "scores": {"accuracy": {"mean": 0.85, "std": 0.05}, "f1": 0.80}
                },
                "model2": {"scores": {"accuracy": {"mean": 0.92, "std": 0.03}}},
            }
        }

        with patch("novaeval.cli.console.print") as mock_print:
            _display_results_summary(results)

            # Verify that console.print was called with a table
            mock_print.assert_called_once()
            # The argument should be a Rich Table object
            args, kwargs = mock_print.call_args
            assert hasattr(args[0], "add_row")  # Rich Table has add_row method

    def test_display_results_summary_empty(self):
        """Test display results summary with empty results."""

        results = {"model_results": {}}

        with patch("novaeval.cli.console.print") as mock_print:
            _display_results_summary(results)

            # Should still create and print a table, even if empty
            mock_print.assert_called_once()

    def test_display_results_summary_string_scores(self):
        """Test display results summary with string scores."""

        results = {
            "model_results": {
                "model1": {"scores": {"accuracy": "0.85", "classification": "good"}}
            }
        }

        with patch("novaeval.cli.console.print") as mock_print:
            _display_results_summary(results)

            # Should handle string scores without error
            mock_print.assert_called_once()


class TestCLIErrorHandling:
    """Test cases for CLI error handling and edge cases."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()

    def test_cli_with_invalid_command(self):
        """Test CLI with invalid command."""
        result = self.runner.invoke(cli, ["invalid-command"])
        assert result.exit_code != 0
        assert "No such command" in result.output

    def test_cli_keyboard_interrupt(self):
        """Test CLI handling of keyboard interrupt."""
        with patch("novaeval.cli.setup_logging", side_effect=KeyboardInterrupt()):
            result = self.runner.invoke(cli, ["--log-level", "DEBUG"])
            # Click handles KeyboardInterrupt, so we expect non-zero exit
            assert result.exit_code != 0

    def test_run_config_load_failure(self):
        """Test run command when config loading fails."""
        config_data = {"dataset": {"type": "custom"}}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name

        try:
            with patch(
                "novaeval.cli.Config.load",
                side_effect=FileNotFoundError("File not found"),
            ):
                result = self.runner.invoke(cli, ["run", config_path])

                assert result.exit_code == 1
                assert "File not found" in result.output
        finally:
            Path(config_path).unlink()

    def test_run_evaluator_creation_failure(self):
        """Test run command when evaluator creation fails."""
        config_data = {"dataset": {"type": "custom"}}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name

        try:
            with (
                patch("novaeval.cli.Config.load") as mock_load,
                patch(
                    "novaeval.cli.Evaluator.from_config",
                    side_effect=ValueError("Invalid config"),
                ),
            ):
                mock_load.return_value = Mock()

                result = self.runner.invoke(cli, ["run", config_path])

                assert result.exit_code == 1
                assert "Invalid config" in result.output
        finally:
            Path(config_path).unlink()

    def test_run_evaluation_execution_failure(self):
        """Test run command when evaluation execution fails."""
        config_data = {"dataset": {"type": "custom"}}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name

        try:
            with (
                patch("novaeval.cli.Config.load") as mock_load,
                patch("novaeval.cli.Evaluator.from_config") as mock_evaluator,
            ):
                mock_load.return_value = Mock()
                mock_evaluator.return_value.run.side_effect = RuntimeError(
                    "Evaluation failed"
                )

                result = self.runner.invoke(cli, ["run", config_path])

                assert result.exit_code == 1
                assert "Evaluation failed" in result.output
        finally:
            Path(config_path).unlink()


class TestQuickCommandEnhanced:
    """Enhanced test cases for the quick command."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()

    def test_quick_with_all_options(self):
        """Test quick command with all available options."""
        result = self.runner.invoke(
            cli,
            [
                "quick",
                "--dataset",
                "mmlu",
                "--model",
                "openai",
                "--model",
                "anthropic",
                "--scorer",
                "accuracy",
                "--scorer",
                "f1",
                "--num-samples",
                "50",
                "--output-dir",
                "/custom/output",
            ],
        )

        assert result.exit_code == 0
        assert "Dataset: mmlu" in result.output
        assert "Models: openai, anthropic" in result.output
        assert "Scorers: accuracy, f1" in result.output

    def test_quick_default_scorer(self):
        """Test quick command uses default scorer when none specified."""
        result = self.runner.invoke(
            cli,
            ["quick", "--dataset", "mmlu", "--model", "openai"],
        )

        assert result.exit_code == 0
        assert "Scorers: accuracy" in result.output

    def test_quick_default_output_dir(self):
        """Test quick command uses default output directory."""
        result = self.runner.invoke(
            cli,
            ["quick", "--dataset", "mmlu", "--model", "openai"],
        )

        assert result.exit_code == 0
        # The default output dir is used internally, not displayed

    def test_quick_invalid_num_samples(self):
        """Test quick command with invalid number of samples."""
        result = self.runner.invoke(
            cli,
            [
                "quick",
                "--dataset",
                "mmlu",
                "--model",
                "openai",
                "--num-samples",
                "not-a-number",
            ],
        )

        assert result.exit_code != 0

    def test_quick_zero_num_samples(self):
        """Test quick command with zero samples."""
        result = self.runner.invoke(
            cli,
            ["quick", "--dataset", "mmlu", "--model", "openai", "--num-samples", "0"],
        )

        assert result.exit_code == 0
