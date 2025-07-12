"""Integration tests for the complete evaluation workflow."""

import json
import os
import tempfile
from pathlib import Path

from novaeval.datasets import BaseDataset
from novaeval.evaluators import BaseEvaluator
from novaeval.models import BaseModel
from novaeval.scorers import AccuracyScorer
from novaeval.utils.config import Config


class MockModel(BaseModel):
    """Mock model for testing."""

    def __init__(self, name="mock-model", responses=None):
        super().__init__(name=name, model_name=name)
        self.responses = responses or ["Mock response"] * 10
        self.response_index = 0

    def generate(self, prompt, **kwargs):
        """Generate a mock response."""
        if self.response_index < len(self.responses):
            response = self.responses[self.response_index]
            self.response_index += 1
            self._track_request(prompt, response)
            return response
        return "Default mock response"

    def generate_batch(self, prompts, **kwargs):
        """Generate responses for multiple prompts."""
        return [self.generate(prompt, **kwargs) for prompt in prompts]

    def get_provider(self):
        """Return provider name."""
        return "mock"


class MockDataset(BaseDataset):
    """Mock dataset for testing."""

    def __init__(self, samples=None, **kwargs):
        super().__init__(name="mock_dataset", **kwargs)
        if samples is None:
            self.samples = [
                {"input": "What is 2+2?", "expected": "4"},
                {"input": "What is the capital of France?", "expected": "Paris"},
                {"input": "What is 3*3?", "expected": "9"},
                {"input": "What color is the sky?", "expected": "blue"},
                {"input": "What is 5-1?", "expected": "4"},
            ]
        else:
            self.samples = samples

    def load_data(self):
        """Load the dataset."""
        return self.samples


class MockEvaluator(BaseEvaluator):
    """Mock evaluator for testing."""

    def __init__(self, dataset, models, scorers, **kwargs):
        super().__init__(dataset=dataset, models=models, scorers=scorers, **kwargs)
        self.results = []

    def run(self):
        """Run the evaluation."""
        results = {
            "summary": {
                "total_samples": len(list(self.dataset)),
                "total_models": len(self.models),
                "total_scorers": len(self.scorers),
            },
            "model_results": {},
            "detailed_results": [],
        }

        for model in self.models:
            model_results = {"model_name": model.name, "scores": {}, "statistics": {}}

            for sample in self.dataset:
                response = model.generate(sample["input"])

                for scorer in self.scorers:
                    score = scorer.score(
                        prediction=response, ground_truth=sample["expected"]
                    )

                    if scorer.name not in model_results["scores"]:
                        model_results["scores"][scorer.name] = []

                    model_results["scores"][scorer.name].append(score)

            # Calculate averages
            for scorer_name, scores in model_results["scores"].items():
                if scores:
                    model_results["statistics"][scorer_name] = {
                        "mean": self._calculate_mean_score(scores),
                        "count": len(scores),
                    }

            results["model_results"][model.name] = model_results

        return results

    def evaluate_sample(self, sample, model, scorers):
        """Evaluate a single sample."""
        response = model.generate(sample["input"])

        sample_result = {
            "input": sample["input"],
            "output": response,
            "expected": sample["expected"],
            "scores": {},
        }

        for scorer in scorers:
            score = scorer.score(prediction=response, ground_truth=sample["expected"])
            sample_result["scores"][scorer.name] = score

        return sample_result

    def save_results(self, results):
        """Save evaluation results."""
        # Mock implementation - just store in memory
        self.results = results

    def _calculate_mean_score(self, scores):
        """Calculate mean score with proper type handling."""
        numeric_scores = []
        for score in scores:
            if isinstance(score, (int, float)):
                numeric_scores.append(score)
            elif isinstance(score, dict) and "score" in score:
                numeric_scores.append(score["score"])
            else:
                # Log warning about unexpected score format and skip
                continue

        return sum(numeric_scores) / len(numeric_scores) if numeric_scores else 0.0


class TestEvaluationWorkflow:
    """Test the complete evaluation workflow."""

    def test_basic_evaluation_workflow(self):
        """Test basic end-to-end evaluation workflow."""
        # Create components
        dataset = MockDataset()
        model = MockModel(responses=["4", "Paris", "9", "blue", "4"])
        scorer = AccuracyScorer()
        evaluator = MockEvaluator(dataset, [model], [scorer])

        # Run evaluation
        results = evaluator.run()

        # Verify results structure
        assert "summary" in results
        assert "model_results" in results
        assert results["summary"]["total_samples"] == 5
        assert results["summary"]["total_models"] == 1
        assert results["summary"]["total_scorers"] == 1

        # Verify model results
        assert model.name in results["model_results"]
        model_results = results["model_results"][model.name]
        assert "scores" in model_results
        assert "statistics" in model_results

    def test_multi_model_evaluation(self):
        """Test evaluation with multiple models."""
        dataset = MockDataset()
        model1 = MockModel(name="model1", responses=["4", "Paris", "9", "blue", "4"])
        model2 = MockModel(name="model2", responses=["4", "London", "9", "blue", "4"])
        scorer = AccuracyScorer()
        evaluator = MockEvaluator(dataset, [model1, model2], [scorer])

        results = evaluator.run()

        assert len(results["model_results"]) == 2
        assert "model1" in results["model_results"]
        assert "model2" in results["model_results"]

    def test_multi_scorer_evaluation(self):
        """Test evaluation with multiple scorers."""
        dataset = MockDataset()
        model = MockModel(responses=["4", "Paris", "9", "blue", "4"])
        scorer1 = AccuracyScorer()
        scorer2 = AccuracyScorer()
        # Give them different names to distinguish them
        scorer2.name = "accuracy_2"
        evaluator = MockEvaluator(dataset, [model], [scorer1, scorer2])

        results = evaluator.run()

        model_results = results["model_results"][model.name]
        assert len(model_results["scores"]) == 2

    def test_error_handling_in_workflow(self):
        """Test error handling during evaluation."""
        dataset = MockDataset()
        model = MockModel()
        scorer = AccuracyScorer()

        # Mock the generate method to raise an error
        def error_generate(prompt, **kwargs):
            model._handle_error(Exception("Test error"), "Test context")
            return ""

        model.generate = error_generate
        evaluator = MockEvaluator(dataset, [model], [scorer])

        # Should handle errors gracefully
        results = evaluator.run()
        assert "model_results" in results

    def test_config_based_evaluation(self):
        """Test evaluation driven by configuration."""
        config_data = {
            "dataset": {
                "type": "custom",
                "samples": [
                    {"input": "What is 2+2?", "expected": "4"},
                    {"input": "What is 3+3?", "expected": "6"},
                ],
            },
            "models": [{"type": "mock", "name": "test-model", "responses": ["4", "6"]}],
            "scorers": [{"type": "accuracy"}],
        }

        config = Config(config_data)

        # Create components from config
        dataset = MockDataset(samples=config.get("dataset")["samples"])
        model = MockModel(
            name=config.get("models")[0]["name"],
            responses=config.get("models")[0]["responses"],
        )
        scorer = AccuracyScorer()
        evaluator = MockEvaluator(dataset, [model], [scorer])

        results = evaluator.run()
        assert results["summary"]["total_samples"] == 2

    def test_results_file_format(self):
        """Test results file format and saving."""
        dataset = MockDataset()
        model = MockModel(responses=["4", "Paris", "9", "blue", "4"])
        scorer = AccuracyScorer()
        evaluator = MockEvaluator(dataset, [model], [scorer])

        results = evaluator.run()

        # Test JSON serialization
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(results, f, default=str)
            temp_path = f.name

        try:
            # Verify file can be loaded
            with open(temp_path) as f:
                loaded_results = json.load(f)
                assert loaded_results["summary"]["total_samples"] == 5
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_empty_dataset_handling(self):
        """Test handling of empty datasets."""
        dataset = MockDataset(samples=[])
        model = MockModel()
        scorer = AccuracyScorer()
        evaluator = MockEvaluator(dataset, [model], [scorer])

        results = evaluator.run()
        assert results["summary"]["total_samples"] == 0

    def test_large_dataset_handling(self):
        """Test handling of larger datasets."""
        # Create a larger dataset
        large_samples = [
            {"input": f"Question {i}?", "expected": f"Answer {i}"} for i in range(100)
        ]

        dataset = MockDataset(samples=large_samples)
        model = MockModel(responses=[f"Answer {i}" for i in range(100)])
        scorer = AccuracyScorer()
        evaluator = MockEvaluator(dataset, [model], [scorer])

        results = evaluator.run()
        assert results["summary"]["total_samples"] == 100

    def test_scorer_statistics_integration(self):
        """Test scorer statistics integration."""
        dataset = MockDataset()
        model = MockModel(responses=["4", "Paris", "9", "blue", "4"])
        scorer = AccuracyScorer()
        evaluator = MockEvaluator(dataset, [model], [scorer])

        results = evaluator.run()

        model_results = results["model_results"][model.name]
        stats = model_results["statistics"]

        # Check that statistics were collected
        assert len(stats) > 0
        assert "mean" in next(iter(stats.values()))
        assert "count" in next(iter(stats.values()))


class TestRealWorldScenarios:
    """Test real-world evaluation scenarios."""

    def test_qa_evaluation_scenario(self):
        """Test question-answering evaluation scenario."""
        qa_samples = [
            {"input": "What is the capital of France?", "expected": "Paris"},
            {"input": "What is 2+2?", "expected": "4"},
            {"input": "What color is the sky?", "expected": "blue"},
        ]

        dataset = MockDataset(samples=qa_samples)
        model = MockModel(responses=["Paris", "4", "blue"])
        scorer = AccuracyScorer()
        evaluator = MockEvaluator(dataset, [model], [scorer])

        results = evaluator.run()

        # All responses should be correct
        model_results = results["model_results"][model.name]
        accuracy_stats = model_results["statistics"]["accuracy"]
        assert accuracy_stats["mean"] == 1.0  # Perfect accuracy

    def test_classification_evaluation_scenario(self):
        """Test classification evaluation scenario."""
        classification_samples = [
            {"input": "This movie is great!", "expected": "positive"},
            {"input": "This movie is terrible!", "expected": "negative"},
            {"input": "This movie is okay.", "expected": "neutral"},
        ]

        dataset = MockDataset(samples=classification_samples)
        model = MockModel(responses=["positive", "negative", "neutral"])
        scorer = AccuracyScorer()
        evaluator = MockEvaluator(dataset, [model], [scorer])

        results = evaluator.run()

        # Verify results structure
        assert "model_results" in results
        model_results = results["model_results"][model.name]
        assert "scores" in model_results

    def test_environment_variable_usage(self):
        """Test evaluation with environment variables."""
        # Set environment variable
        os.environ["TEST_MODEL_NAME"] = "test-model"

        try:
            dataset = MockDataset()
            model = MockModel(name=os.environ["TEST_MODEL_NAME"])
            scorer = AccuracyScorer()
            evaluator = MockEvaluator(dataset, [model], [scorer])

            results = evaluator.run()
            assert "test-model" in results["model_results"]
        finally:
            # Clean up
            del os.environ["TEST_MODEL_NAME"]


class TestErrorRecovery:
    """Test error recovery and resilience."""

    def test_partial_failure_handling(self):
        """Test handling partial failures during evaluation."""
        dataset = MockDataset()
        model = MockModel()
        scorer = AccuracyScorer()
        evaluator = MockEvaluator(dataset, [model], [scorer])

        # Mock some samples to fail
        original_generate = model.generate

        def conditional_generate(prompt, **kwargs):
            if "Bad question" in prompt:
                model._handle_error(Exception("API error"), "API context")
                return ""
            return original_generate(prompt, **kwargs)

        model.generate = conditional_generate

        # Should handle partial failures gracefully
        results = evaluator.run()
        assert "model_results" in results

    def test_scorer_error_handling(self):
        """Test scorer error handling."""
        dataset = MockDataset()
        model = MockModel()

        # Create a scorer that might fail
        scorer = AccuracyScorer()
        original_score = scorer.score

        def error_score(*args, **kwargs):
            try:
                return original_score(*args, **kwargs)
            except (ValueError, TypeError, KeyError, AttributeError) as e:
                # Log the error for debugging
                print(f"Scorer error (expected in test): {e}")
                return 0.0  # Return default score on error

        scorer.score = error_score
        evaluator = MockEvaluator(dataset, [model], [scorer])

        results = evaluator.run()
        assert "model_results" in results
