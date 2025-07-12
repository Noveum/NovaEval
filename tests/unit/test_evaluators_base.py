"""
Unit tests for base evaluator functionality.
"""

import tempfile
from pathlib import Path

import pytest

from novaeval.datasets.base import BaseDataset
from novaeval.evaluators.base import BaseEvaluator
from novaeval.models.base import BaseModel
from novaeval.scorers.base import BaseScorer


class MockDataset(BaseDataset):
    """Mock dataset for testing."""

    def __init__(self, data=None, **kwargs):
        super().__init__(name="mock_dataset", **kwargs)
        self._test_data = data or [
            {"id": "1", "input": "test 1", "expected": "result 1"},
            {"id": "2", "input": "test 2", "expected": "result 2"},
        ]

    def load_data(self):
        return self._test_data


class MockModel(BaseModel):
    """Mock model for testing."""

    def __init__(self, name="mock_model", **kwargs):
        super().__init__(name=name, model_name="mock-v1", **kwargs)

    def generate(self, prompt, **kwargs):
        return f"Generated: {prompt}"

    def generate_batch(self, prompts, **kwargs):
        return [f"Generated: {prompt}" for prompt in prompts]

    def get_provider(self):
        return "mock"


class MockScorer(BaseScorer):
    """Mock scorer for testing."""

    def __init__(self, name="mock_scorer", **kwargs):
        super().__init__(name=name, **kwargs)

    def score(self, prediction, ground_truth, context=None):
        return 0.8 if prediction == ground_truth else 0.5


class ConcreteEvaluator(BaseEvaluator):
    """Concrete implementation of BaseEvaluator for testing."""

    def run(self):
        """Mock implementation."""
        return {"status": "completed", "results": []}

    def evaluate_sample(self, sample, model, scorers):
        """Mock implementation."""
        prediction = model.generate(sample["input"])
        scores = {}
        for scorer in scorers:
            score = scorer.score(prediction, sample["expected"])
            scores[scorer.name] = score

        return {
            "sample_id": sample.get("id"),
            "prediction": prediction,
            "scores": scores,
        }

    def save_results(self, results):
        """Mock implementation."""
        output_file = self.output_dir / "results.json"
        import json

        with open(output_file, "w") as f:
            json.dump(results, f)


class TestBaseEvaluator:
    """Test cases for BaseEvaluator class."""

    def test_init_default(self):
        """Test initialization with default parameters."""
        dataset = MockDataset()
        models = [MockModel()]
        scorers = [MockScorer()]

        evaluator = ConcreteEvaluator(dataset, models, scorers)

        assert evaluator.dataset == dataset
        assert evaluator.models == models
        assert evaluator.scorers == scorers
        assert evaluator.output_dir == Path("./results")
        assert evaluator.config == {}
        assert evaluator.output_dir.exists()  # Should be created

    def test_init_with_params(self):
        """Test initialization with custom parameters."""
        dataset = MockDataset()
        models = [MockModel()]
        scorers = [MockScorer()]

        with tempfile.TemporaryDirectory() as temp_dir:
            config = {"param1": "value1", "param2": "value2"}

            evaluator = ConcreteEvaluator(
                dataset=dataset,
                models=models,
                scorers=scorers,
                output_dir=temp_dir,
                config=config,
            )

            assert evaluator.dataset == dataset
            assert evaluator.models == models
            assert evaluator.scorers == scorers
            assert evaluator.output_dir == Path(temp_dir)
            assert evaluator.config == config

    def test_validate_inputs_valid(self):
        """Test input validation with valid inputs."""
        dataset = MockDataset()
        models = [MockModel()]
        scorers = [MockScorer()]

        evaluator = ConcreteEvaluator(dataset, models, scorers)

        # Should not raise any exception
        evaluator.validate_inputs()

    def test_validate_inputs_no_dataset(self):
        """Test input validation with missing dataset."""
        models = [MockModel()]
        scorers = [MockScorer()]

        evaluator = ConcreteEvaluator(None, models, scorers)

        with pytest.raises(ValueError, match="Dataset is required"):
            evaluator.validate_inputs()

    def test_validate_inputs_no_models(self):
        """Test input validation with no models."""
        dataset = MockDataset()
        scorers = [MockScorer()]

        evaluator = ConcreteEvaluator(dataset, [], scorers)

        with pytest.raises(ValueError, match="At least one model is required"):
            evaluator.validate_inputs()

    def test_validate_inputs_no_scorers(self):
        """Test input validation with no scorers."""
        dataset = MockDataset()
        models = [MockModel()]

        evaluator = ConcreteEvaluator(dataset, models, [])

        with pytest.raises(ValueError, match="At least one scorer is required"):
            evaluator.validate_inputs()

    def test_evaluate_sample_implementation(self):
        """Test the concrete evaluate_sample implementation."""
        dataset = MockDataset()
        model = MockModel()
        scorers = [MockScorer()]

        evaluator = ConcreteEvaluator(dataset, [model], scorers)

        sample = {"id": "test", "input": "hello", "expected": "world"}
        result = evaluator.evaluate_sample(sample, model, scorers)

        assert result["sample_id"] == "test"
        assert result["prediction"] == "Generated: hello"
        assert "mock_scorer" in result["scores"]
        assert result["scores"]["mock_scorer"] == 0.5  # Different from expected

    def test_save_results_implementation(self):
        """Test the concrete save_results implementation."""
        dataset = MockDataset()
        models = [MockModel()]
        scorers = [MockScorer()]

        with tempfile.TemporaryDirectory() as temp_dir:
            evaluator = ConcreteEvaluator(dataset, models, scorers, output_dir=temp_dir)

            results = {"test": "data", "results": [1, 2, 3]}
            evaluator.save_results(results)

            # Check that file was created
            results_file = Path(temp_dir) / "results.json"
            assert results_file.exists()

            # Check content
            import json

            with open(results_file) as f:
                saved_data = json.load(f)
            assert saved_data == results

    def test_run_implementation(self):
        """Test the concrete run implementation."""
        dataset = MockDataset()
        models = [MockModel()]
        scorers = [MockScorer()]

        evaluator = ConcreteEvaluator(dataset, models, scorers)

        result = evaluator.run()
        assert result["status"] == "completed"
        assert "results" in result

    def test_from_config_not_implemented(self):
        """Test that from_config raises NotImplementedError in base class."""
        with pytest.raises(NotImplementedError):
            BaseEvaluator.from_config("dummy_config.yaml")

    def test_abstract_methods_not_implemented(self):
        """Test that abstract methods raise NotImplementedError."""
        with pytest.raises(TypeError):
            # Cannot instantiate abstract class
            BaseEvaluator(
                dataset=MockDataset(), models=[MockModel()], scorers=[MockScorer()]
            )


class TestEvaluatorEdgeCases:
    """Test edge cases for evaluator functionality."""

    def test_output_dir_creation(self):
        """Test that output directory is created if it doesn't exist."""
        dataset = MockDataset()
        models = [MockModel()]
        scorers = [MockScorer()]

        with tempfile.TemporaryDirectory() as temp_dir:
            nested_dir = Path(temp_dir) / "nested" / "output"

            evaluator = ConcreteEvaluator(
                dataset, models, scorers, output_dir=nested_dir
            )

            assert evaluator.output_dir.exists()
            assert evaluator.output_dir.is_dir()

    def test_string_output_dir(self):
        """Test initialization with string output directory."""
        dataset = MockDataset()
        models = [MockModel()]
        scorers = [MockScorer()]

        with tempfile.TemporaryDirectory() as temp_dir:
            evaluator = ConcreteEvaluator(
                dataset, models, scorers, output_dir=temp_dir  # String instead of Path
            )

            assert evaluator.output_dir == Path(temp_dir)
            assert evaluator.output_dir.exists()

    def test_multiple_models_and_scorers(self):
        """Test evaluation with multiple models and scorers."""
        dataset = MockDataset()
        models = [
            MockModel(name="model1"),
            MockModel(name="model2"),
        ]
        scorers = [
            MockScorer(name="scorer1"),
            MockScorer(name="scorer2"),
        ]

        evaluator = ConcreteEvaluator(dataset, models, scorers)

        sample = {"id": "test", "input": "hello", "expected": "world"}

        # Test with first model
        result1 = evaluator.evaluate_sample(sample, models[0], scorers)
        assert len(result1["scores"]) == 2
        assert "scorer1" in result1["scores"]
        assert "scorer2" in result1["scores"]

        # Test with second model
        result2 = evaluator.evaluate_sample(sample, models[1], scorers)
        assert len(result2["scores"]) == 2

    def test_empty_config(self):
        """Test evaluator with empty configuration."""
        dataset = MockDataset()
        models = [MockModel()]
        scorers = [MockScorer()]

        evaluator = ConcreteEvaluator(dataset, models, scorers, config={})

        assert evaluator.config == {}
        evaluator.validate_inputs()  # Should not raise

    def test_none_config(self):
        """Test evaluator with None configuration."""
        dataset = MockDataset()
        models = [MockModel()]
        scorers = [MockScorer()]

        evaluator = ConcreteEvaluator(dataset, models, scorers, config=None)

        assert evaluator.config == {}
        evaluator.validate_inputs()  # Should not raise
