"""
Unit tests for standard evaluator functionality.
"""

import json
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
import pytest

from novaeval.datasets.base import BaseDataset
from novaeval.evaluators.standard import Evaluator
from novaeval.models.base import BaseModel
from novaeval.scorers.base import BaseScorer


class MockDataset(BaseDataset):
    """Mock dataset for testing."""

    def __init__(self, samples=None, **kwargs):
        super().__init__(name="mock_dataset", **kwargs)
        self._test_samples = samples or [
            {"id": "1", "input": "test 1", "expected": "result 1"},
            {"id": "2", "input": "test 2", "expected": "result 2"},
            {"id": "3", "input": "test 3", "expected": "result 3"},
        ]

    def load_data(self):
        return self._test_samples

    def get_info(self):
        return {"name": self.name, "samples": len(self._test_samples)}


class MockModel(BaseModel):
    """Mock model for testing."""

    def __init__(self, name="test_model", responses=None, **kwargs):
        super().__init__(name=name, model_name="test-model", **kwargs)
        self.responses = responses or ["response 1", "response 2", "response 3"]
        self.call_count = 0

    def generate(self, prompt, **kwargs):
        if self.call_count < len(self.responses):
            response = self.responses[self.call_count]
            self.call_count += 1
            return response
        return "default response"

    def generate_batch(self, prompts, **kwargs):
        """Generate responses for a batch of prompts."""
        return [self.generate(prompt, **kwargs) for prompt in prompts]

    def get_provider(self) -> str:
        """Get the provider name."""
        return "test"

    def get_info(self):
        return {"name": self.name, "model_name": self.model_name, "provider": "test"}


class MockScorer(BaseScorer):
    """Mock scorer for testing."""

    def __init__(self, name="test_scorer", scores=None, **kwargs):
        super().__init__(name=name, description="Mock scorer", **kwargs)
        self.scores = scores or [0.8, 0.9, 0.7]
        self.call_count = 0

    def score(self, prediction, ground_truth, context=None):
        if self.call_count < len(self.scores):
            score = self.scores[self.call_count]
            self.call_count += 1
            return score
        return 0.5

    def get_info(self):
        return {"name": self.name, "description": self.description}


class TestEvaluator:
    """Test cases for Evaluator class."""

    def test_init_default(self):
        """Test initialization with default parameters."""
        dataset = MockDataset()
        models = [MockModel()]
        scorers = [MockScorer()]

        evaluator = Evaluator(dataset, models, scorers)

        assert evaluator.dataset == dataset
        assert evaluator.models == models
        assert evaluator.scorers == scorers
        assert evaluator.output_dir == Path("./results")
        assert evaluator.config == {}
        assert evaluator.max_workers == 4
        assert evaluator.batch_size == 1

    @patch("novaeval.evaluators.standard.setup_logging")
    def test_init_with_params(self, mock_setup_logging):
        """Test initialization with custom parameters."""
        dataset = MockDataset()
        models = [MockModel()]
        scorers = [MockScorer()]

        with tempfile.TemporaryDirectory() as temp_dir:
            config = {"param1": "value1", "log_level": "DEBUG"}

            evaluator = Evaluator(
                dataset=dataset,
                models=models,
                scorers=scorers,
                output_dir=temp_dir,
                config=config,
                max_workers=8,
                batch_size=2,
            )

            assert evaluator.dataset == dataset
            assert evaluator.models == models
            assert evaluator.scorers == scorers
            assert evaluator.output_dir == Path(temp_dir)
            assert evaluator.config == config
            assert evaluator.max_workers == 8
            assert evaluator.batch_size == 2

    @patch("novaeval.evaluators.standard.setup_logging")
    def test_init_logging_setup(self, mock_setup_logging):
        """Test that logging is setup correctly during initialization."""
        dataset = MockDataset()
        models = [MockModel()]
        scorers = [MockScorer()]

        evaluator = Evaluator(dataset, models, scorers)

        mock_setup_logging.assert_called_once_with(
            level="INFO",
            log_file=evaluator.output_dir / "evaluation.log",
        )

    @patch("novaeval.evaluators.standard.setup_logging")
    def test_run_complete_flow(self, mock_setup_logging):
        """Test complete evaluation run flow."""
        dataset = MockDataset()
        models = [MockModel()]
        scorers = [MockScorer()]

        with tempfile.TemporaryDirectory() as temp_dir:
            evaluator = Evaluator(dataset, models, scorers, output_dir=temp_dir)

            with patch.object(evaluator, "save_results") as mock_save:
                results = evaluator.run()

                # Verify results structure
                assert "metadata" in results
                assert "model_results" in results
                assert "summary" in results

                # Verify metadata
                assert "start_time" in results["metadata"]
                assert "end_time" in results["metadata"]
                assert "duration" in results["metadata"]
                assert "dataset" in results["metadata"]
                assert "models" in results["metadata"]
                assert "scorers" in results["metadata"]
                assert "config" in results["metadata"]

                # Verify model results
                assert models[0].name in results["model_results"]
                model_results = results["model_results"][models[0].name]
                assert "samples" in model_results
                assert "scores" in model_results
                assert "errors" in model_results

                # Verify summary
                assert "total_samples" in results["summary"]
                assert "total_models" in results["summary"]
                assert "total_scorers" in results["summary"]

                # Verify save_results was called
                mock_save.assert_called_once_with(results)

    def test_run_with_validation_error(self):
        """Test run with validation error."""
        dataset = MockDataset()
        models = []  # Empty models list should cause validation error
        scorers = [MockScorer()]

        evaluator = Evaluator(dataset, models, scorers)

        with pytest.raises(ValueError, match="At least one model is required"):
            evaluator.run()

    @patch("novaeval.evaluators.standard.tqdm")
    @patch("novaeval.evaluators.standard.as_completed")
    @patch("novaeval.evaluators.standard.ThreadPoolExecutor")
    def test_evaluate_model_threading(
        self, mock_executor, mock_as_completed, mock_tqdm
    ):
        """Test model evaluation with threading."""
        dataset = MockDataset()
        model = MockModel()
        scorers = [MockScorer()]

        evaluator = Evaluator(dataset, [model], scorers)

        # Mock the executor and futures
        mock_executor_instance = Mock()
        mock_executor.return_value.__enter__.return_value = mock_executor_instance
        mock_executor.return_value.__exit__.return_value = None

        # Mock futures for each sample
        mock_futures = []
        for i in range(3):  # 3 samples in MockDataset
            mock_future = Mock()
            mock_future.result.return_value = {
                "sample_id": f"{i+1}",
                "scores": {"test_scorer": 0.8},
                "prediction": f"test response {i+1}",
                "error": None,
            }
            mock_futures.append(mock_future)

        # Mock submit to return our mock futures
        mock_executor_instance.submit.side_effect = mock_futures

        # Mock as_completed to return the futures
        mock_as_completed.return_value = mock_futures

        # Mock tqdm to avoid progress bar issues
        mock_tqdm.return_value = mock_futures

        results = evaluator._evaluate_model(model)

        assert "samples" in results
        assert "scores" in results
        assert "errors" in results
        assert len(results["samples"]) == 3
        assert mock_executor_instance.submit.call_count == 3

    def test_evaluate_sample_success(self):
        """Test successful sample evaluation."""
        dataset = MockDataset()
        model = MockModel()
        scorers = [MockScorer()]

        evaluator = Evaluator(dataset, [model], scorers)

        sample = {"id": "1", "input": "test input", "expected": "expected output"}
        result = evaluator.evaluate_sample(sample, model, scorers)

        assert result["sample_id"] == "1"
        assert result["input"] == "test input"
        assert result["expected"] == "expected output"
        assert result["prediction"] == "response 1"
        assert result["scores"]["test_scorer"] == 0.8
        assert result["error"] is None

    def test_evaluate_sample_with_model_error(self):
        """Test sample evaluation with model error."""
        dataset = MockDataset()
        model = MockModel()
        scorers = [MockScorer()]

        evaluator = Evaluator(dataset, [model], scorers)

        # Mock model to raise exception
        model.generate = Mock(side_effect=Exception("Model error"))

        sample = {"id": "1", "input": "test input", "expected": "expected output"}
        result = evaluator.evaluate_sample(sample, model, scorers)

        assert result["sample_id"] == "1"
        assert result["prediction"] is None
        assert result["scores"] == {}
        assert "Model error" in result["error"]

    def test_evaluate_sample_with_scorer_error(self):
        """Test sample evaluation with scorer error."""
        dataset = MockDataset()
        model = MockModel()
        scorer = MockScorer()

        evaluator = Evaluator(dataset, [model], [scorer])

        # Mock scorer to raise exception
        scorer.score = Mock(side_effect=Exception("Scorer error"))

        sample = {"id": "1", "input": "test input", "expected": "expected output"}
        result = evaluator.evaluate_sample(sample, model, [scorer])

        assert result["sample_id"] == "1"
        assert result["prediction"] == "response 1"
        assert result["scores"] == {}
        assert "Scorer error" in result["error"]

    def test_aggregate_scores_float_values(self):
        """Test score aggregation with float values."""
        dataset = MockDataset()
        models = [MockModel()]
        scorers = [MockScorer()]

        evaluator = Evaluator(dataset, models, scorers)

        sample_results = [
            {"scores": {"accuracy": 0.8, "f1": 0.7}},
            {"scores": {"accuracy": 0.9, "f1": 0.8}},
            {"scores": {"accuracy": 0.7, "f1": 0.6}},
        ]

        aggregated = evaluator._aggregate_scores(sample_results)

        assert "accuracy" in aggregated
        assert "f1" in aggregated

        # Check accuracy stats
        acc_stats = aggregated["accuracy"]
        assert abs(acc_stats["mean"] - 0.8) < 0.001
        assert acc_stats["count"] == 3
        assert acc_stats["min"] == 0.7
        assert acc_stats["max"] == 0.9
        assert abs(acc_stats["std"] - 0.1) < 0.01

    def test_aggregate_scores_dict_values(self):
        """Test score aggregation with dict values."""
        dataset = MockDataset()
        models = [MockModel()]
        scorers = [MockScorer()]

        evaluator = Evaluator(dataset, models, scorers)

        sample_results = [
            {"scores": {"f1": {"score": 0.8, "precision": 0.7, "recall": 0.9}}},
            {"scores": {"f1": {"score": 0.9, "precision": 0.8, "recall": 0.95}}},
            {"scores": {"f1": {"score": 0.7, "precision": 0.6, "recall": 0.8}}},
        ]

        aggregated = evaluator._aggregate_scores(sample_results)

        assert "f1" in aggregated
        f1_stats = aggregated["f1"]
        assert abs(f1_stats["mean"] - 0.8) < 0.001
        assert f1_stats["count"] == 3

    def test_aggregate_scores_mixed_values(self):
        """Test score aggregation with mixed value types."""
        dataset = MockDataset()
        models = [MockModel()]
        scorers = [MockScorer()]

        evaluator = Evaluator(dataset, models, scorers)

        sample_results = [
            {"scores": {"accuracy": 0.8, "f1": {"score": 0.7}}},
            {"scores": {"accuracy": 0.9, "f1": {"score": 0.8}}},
            {"scores": {"accuracy": 0.7, "f1": {"score": 0.6}}},
        ]

        aggregated = evaluator._aggregate_scores(sample_results)

        assert "accuracy" in aggregated
        assert "f1" in aggregated
        assert abs(aggregated["accuracy"]["mean"] - 0.8) < 0.001
        assert abs(aggregated["f1"]["mean"] - 0.7) < 0.001

    def test_aggregate_scores_empty_results(self):
        """Test score aggregation with empty results."""
        dataset = MockDataset()
        models = [MockModel()]
        scorers = [MockScorer()]

        evaluator = Evaluator(dataset, models, scorers)

        sample_results = []
        aggregated = evaluator._aggregate_scores(sample_results)

        assert aggregated == {}

    def test_calculate_summary(self):
        """Test summary calculation."""
        dataset = MockDataset()
        models = [MockModel()]
        scorers = [MockScorer()]

        evaluator = Evaluator(dataset, models, scorers)

        model_results = {
            "model1": {
                "samples": [{"id": "1"}, {"id": "2"}],
                "scores": {"accuracy": {"mean": 0.8}},
                "errors": [],
            },
            "model2": {
                "samples": [{"id": "1"}, {"id": "2"}],
                "scores": {"accuracy": {"mean": 0.9}},
                "errors": ["error1"],
            },
        }

        summary = evaluator._calculate_summary(model_results)

        assert summary["total_samples"] == 2
        assert summary["total_models"] == 2
        assert summary["total_scorers"] == 1
        assert summary["total_errors"] == 1
        assert abs(summary["overall_accuracy"]["mean"] - 0.85) < 0.001

    def test_calculate_summary_empty_results(self):
        """Test summary calculation with empty results."""
        dataset = MockDataset()
        models = [MockModel()]
        scorers = [MockScorer()]

        evaluator = Evaluator(dataset, models, scorers)

        model_results = {}
        summary = evaluator._calculate_summary(model_results)

        assert summary["total_samples"] == 0
        assert summary["total_models"] == 0
        assert summary["total_scorers"] == 0
        assert summary["total_errors"] == 0

    @patch("novaeval.evaluators.standard.setup_logging")
    def test_save_results_json(self, mock_setup_logging):
        """Test saving results to JSON file."""
        dataset = MockDataset()
        models = [MockModel()]
        scorers = [MockScorer()]

        with tempfile.TemporaryDirectory() as temp_dir:
            evaluator = Evaluator(dataset, models, scorers, output_dir=temp_dir)

            results = {
                "metadata": {"start_time": time.time()},
                "model_results": {"test_model": {"scores": {"accuracy": 0.8}}},
                "summary": {"total_samples": 3},
            }

            evaluator.save_results(results)

            # Check that JSON file was created
            json_file = Path(temp_dir) / "results.json"
            assert json_file.exists()

            # Check content
            with open(json_file) as f:
                saved_data = json.load(f)
            assert saved_data["summary"]["total_samples"] == 3

    @patch("novaeval.evaluators.standard.setup_logging")
    def test_save_results_csv(self, mock_setup_logging):
        """Test saving results to CSV file."""
        dataset = MockDataset()
        models = [MockModel()]
        scorers = [MockScorer()]

        with tempfile.TemporaryDirectory() as temp_dir:
            evaluator = Evaluator(dataset, models, scorers, output_dir=temp_dir)

            results = {
                "model_results": {
                    "test_model": {
                        "samples": [
                            {
                                "sample_id": "1",
                                "prediction": "pred1",
                                "scores": {"accuracy": 0.8},
                            },
                            {
                                "sample_id": "2",
                                "prediction": "pred2",
                                "scores": {"accuracy": 0.9},
                            },
                        ]
                    }
                }
            }

            evaluator._save_csv_results(results)

            # Check that CSV file was created
            csv_file = Path(temp_dir) / "detailed_results.csv"
            assert csv_file.exists()

            # Check content
            df = pd.read_csv(csv_file)
            assert len(df) == 2
            assert "sample_id" in df.columns
            assert "prediction" in df.columns
            assert "score_accuracy" in df.columns

    @patch("novaeval.evaluators.standard.setup_logging")
    def test_save_results_csv_empty(self, mock_setup_logging):
        """Test saving empty results to CSV file."""
        dataset = MockDataset()
        models = [MockModel()]
        scorers = [MockScorer()]

        with tempfile.TemporaryDirectory() as temp_dir:
            evaluator = Evaluator(dataset, models, scorers, output_dir=temp_dir)

            results = {"model_results": {}}
            evaluator._save_csv_results(results)

            # CSV file should not be created for empty results
            csv_file = Path(temp_dir) / "detailed_results.csv"
            assert not csv_file.exists()

    @patch("novaeval.evaluators.standard.Config.load")
    def test_from_config_not_implemented(self, mock_config_load):
        """Test that from_config raises NotImplementedError."""
        mock_config_load.return_value = Mock()
        with pytest.raises(
            NotImplementedError,
            match="Configuration-based initialization not yet implemented",
        ):
            Evaluator.from_config("/path/to/config.yaml")

    @patch("novaeval.evaluators.standard.setup_logging")
    def test_run_with_multiple_models(self, mock_setup_logging):
        """Test evaluation with multiple models."""
        dataset = MockDataset()
        models = [
            MockModel(name="model1", responses=["resp1", "resp2", "resp3"]),
            MockModel(name="model2", responses=["resp4", "resp5", "resp6"]),
        ]
        scorers = [MockScorer()]

        with tempfile.TemporaryDirectory() as temp_dir:
            evaluator = Evaluator(dataset, models, scorers, output_dir=temp_dir)

            with patch.object(evaluator, "save_results"):
                results = evaluator.run()

                # Both models should be evaluated
                assert "model1" in results["model_results"]
                assert "model2" in results["model_results"]
                assert results["summary"]["total_models"] == 2

    @patch("novaeval.evaluators.standard.setup_logging")
    def test_run_with_multiple_scorers(self, mock_setup_logging):
        """Test evaluation with multiple scorers."""
        dataset = MockDataset()
        models = [MockModel()]
        scorers = [
            MockScorer(name="scorer1", scores=[0.8, 0.9, 0.7]),
            MockScorer(name="scorer2", scores=[0.6, 0.8, 0.5]),
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            evaluator = Evaluator(dataset, models, scorers, output_dir=temp_dir)

            with patch.object(evaluator, "save_results"):
                results = evaluator.run()

                # Both scorers should be used
                model_results = results["model_results"][models[0].name]
                assert "scorer1" in model_results["scores"]
                assert "scorer2" in model_results["scores"]
                assert results["summary"]["total_scorers"] == 2

    @patch("novaeval.evaluators.standard.setup_logging")
    def test_run_timing_metadata(self, mock_setup_logging):
        """Test that timing metadata is correctly set."""
        dataset = MockDataset()
        models = [MockModel()]
        scorers = [MockScorer()]

        evaluator = Evaluator(dataset, models, scorers)

        with patch.object(evaluator, "save_results"):
            start_time = time.time()
            results = evaluator.run()
            end_time = time.time()

            metadata = results["metadata"]
            assert metadata["start_time"] >= start_time
            assert metadata["end_time"] <= end_time
            assert metadata["duration"] >= 0  # Changed from > 0 to >= 0 for fast tests
            assert metadata["duration"] == metadata["end_time"] - metadata["start_time"]

    def test_max_workers_configuration(self):
        """Test that max_workers is properly configured."""
        dataset = MockDataset()
        models = [MockModel()]
        scorers = [MockScorer()]

        evaluator = Evaluator(dataset, models, scorers, max_workers=2)

        # Simply check that the max_workers attribute is set correctly
        assert evaluator.max_workers == 2

        # Test with different max_workers value
        evaluator2 = Evaluator(dataset, models, scorers, max_workers=8)
        assert evaluator2.max_workers == 8

    @patch("novaeval.evaluators.standard.tqdm")
    def test_progress_bar_integration(self, mock_tqdm):
        """Test that progress bar is properly integrated."""
        dataset = MockDataset()
        models = [MockModel()]
        scorers = [MockScorer()]

        evaluator = Evaluator(dataset, models, scorers)

        with patch("novaeval.evaluators.standard.ThreadPoolExecutor") as mock_executor:
            mock_executor_instance = Mock()
            mock_executor.return_value.__enter__.return_value = mock_executor_instance
            mock_executor.return_value.__exit__.return_value = None

            # Mock futures for each sample
            mock_futures = []
            for i in range(3):  # 3 samples in MockDataset
                mock_future = Mock()
                mock_future.result.return_value = {
                    "sample_id": f"{i+1}",
                    "scores": {"test_scorer": 0.8},
                    "prediction": f"test response {i+1}",
                    "error": None,
                }
                mock_futures.append(mock_future)

            mock_executor_instance.submit.side_effect = mock_futures

            with patch(
                "novaeval.evaluators.standard.as_completed"
            ) as mock_as_completed:
                mock_as_completed.return_value = mock_futures

                # Mock tqdm to return the futures (simulating the progress bar)
                mock_tqdm.return_value = mock_futures

                evaluator._evaluate_model(models[0])

                # Verify tqdm was called with correct parameters
                mock_tqdm.assert_called_once()
                args, kwargs = mock_tqdm.call_args
                assert kwargs.get("total") == 3
                assert kwargs.get("desc") == "Evaluating test_model"
