"""
Unit tests for the aggregation bug fix in NovaEval.

This module tests the fix for the aggregation bug that occurred when using
multiple scorers with different return types (float vs dict).
"""

from unittest.mock import Mock

from novaeval.evaluators.standard import Evaluator
from novaeval.scorers.accuracy import AccuracyScorer, ExactMatchScorer, F1Scorer
from novaeval.scorers.base import BaseScorer


class MockScorer(BaseScorer):
    """Mock scorer for testing purposes."""

    def __init__(self, name: str, return_type: str = "float", **kwargs):
        super().__init__(name=name, **kwargs)
        self.return_type = return_type

    def score(self, prediction: str, ground_truth: str, context=None) -> float:
        """Mock score method."""
        if self.return_type == "float":
            return 0.8
        elif self.return_type == "dict":
            return {"score": 0.8, "precision": 0.7, "recall": 0.9, "f1": 0.8}
        elif self.return_type == "dict_no_score":
            return {"precision": 0.7, "recall": 0.9, "f1": 0.8}
        elif self.return_type == "invalid":
            return "invalid_score"
        else:
            return 0.0


class TestAggregationBugFix:
    """Test cases for the aggregation bug fix."""

    def setup_method(self):
        """Setup test fixtures."""
        # Create mock dataset
        self.mock_dataset = Mock()
        self.mock_dataset.__iter__ = Mock(
            return_value=iter(
                [
                    {"id": "1", "input": "What is 2+2?", "expected": "4"},
                    {"id": "2", "input": "What color is the sky?", "expected": "blue"},
                ]
            )
        )

        # Create mock model
        self.mock_model = Mock()
        self.mock_model.name = "test_model"
        self.mock_model.generate.side_effect = ["4", "blue"]
        self.mock_model.get_info.return_value = {"name": "test_model", "type": "mock"}

    def test_single_float_scorer_aggregation(self):
        """Test aggregation with a single float-returning scorer."""
        scorer = AccuracyScorer()

        evaluator = Evaluator(
            dataset=self.mock_dataset,
            models=[self.mock_model],
            scorers=[scorer],
            output_dir="./test_output",
        )

        # Mock the sample results
        sample_results = [{"scores": {"accuracy": 1.0}}, {"scores": {"accuracy": 0.0}}]

        aggregated = evaluator._aggregate_scores(sample_results)

        assert "accuracy" in aggregated
        assert aggregated["accuracy"]["mean"] == 0.5
        assert aggregated["accuracy"]["count"] == 2
        assert aggregated["accuracy"]["min"] == 0.0
        assert aggregated["accuracy"]["max"] == 1.0

    def test_single_dict_scorer_aggregation(self):
        """Test aggregation with a single dict-returning scorer."""
        scorer = F1Scorer()

        evaluator = Evaluator(
            dataset=self.mock_dataset,
            models=[self.mock_model],
            scorers=[scorer],
            output_dir="./test_output",
        )

        # Mock the sample results
        sample_results = [
            {
                "scores": {
                    "f1": {"precision": 0.8, "recall": 0.9, "f1": 0.85, "score": 0.85}
                }
            },
            {
                "scores": {
                    "f1": {"precision": 0.6, "recall": 0.7, "f1": 0.65, "score": 0.65}
                }
            },
        ]

        aggregated = evaluator._aggregate_scores(sample_results)

        assert "f1" in aggregated
        assert aggregated["f1"]["mean"] == 0.75  # (0.85 + 0.65) / 2
        assert aggregated["f1"]["count"] == 2
        assert aggregated["f1"]["min"] == 0.65
        assert aggregated["f1"]["max"] == 0.85

        # Check detailed scores
        assert aggregated["f1"]["precision_mean"] == 0.7  # (0.8 + 0.6) / 2
        assert aggregated["f1"]["recall_mean"] == 0.8  # (0.9 + 0.7) / 2
        assert aggregated["f1"]["f1_mean"] == 0.75  # (0.85 + 0.65) / 2

    def test_multiple_mixed_scorers_aggregation(self):
        """Test aggregation with multiple scorers of different return types."""
        accuracy_scorer = AccuracyScorer()
        exact_match_scorer = ExactMatchScorer()
        f1_scorer = F1Scorer()

        evaluator = Evaluator(
            dataset=self.mock_dataset,
            models=[self.mock_model],
            scorers=[accuracy_scorer, exact_match_scorer, f1_scorer],
            output_dir="./test_output",
        )

        # Mock the sample results with mixed score types
        sample_results = [
            {
                "scores": {
                    "accuracy": 1.0,  # float
                    "exact_match": 0.0,  # float
                    "f1": {
                        "precision": 0.8,
                        "recall": 0.9,
                        "f1": 0.85,
                        "score": 0.85,
                    },  # dict
                }
            },
            {
                "scores": {
                    "accuracy": 0.0,  # float
                    "exact_match": 1.0,  # float
                    "f1": {
                        "precision": 0.6,
                        "recall": 0.7,
                        "f1": 0.65,
                        "score": 0.65,
                    },  # dict
                }
            },
        ]

        aggregated = evaluator._aggregate_scores(sample_results)

        # Check float scorers
        assert "accuracy" in aggregated
        assert aggregated["accuracy"]["mean"] == 0.5
        assert aggregated["accuracy"]["count"] == 2

        assert "exact_match" in aggregated
        assert aggregated["exact_match"]["mean"] == 0.5
        assert aggregated["exact_match"]["count"] == 2

        # Check dict scorer
        assert "f1" in aggregated
        assert aggregated["f1"]["mean"] == 0.75  # Uses "score" key for main aggregation
        assert aggregated["f1"]["count"] == 2
        assert aggregated["f1"]["precision_mean"] == 0.7
        assert aggregated["f1"]["recall_mean"] == 0.8
        assert aggregated["f1"]["f1_mean"] == 0.75

    def test_dict_scorer_without_score_key(self):
        """Test aggregation with dict scorer that doesn't have a 'score' key."""
        evaluator = Evaluator(
            dataset=self.mock_dataset,
            models=[self.mock_model],
            scorers=[MockScorer("test", "dict_no_score")],
            output_dir="./test_output",
        )

        sample_results = [
            {"scores": {"test": {"precision": 0.8, "recall": 0.9, "f1": 0.85}}},
            {"scores": {"test": {"precision": 0.6, "recall": 0.7, "f1": 0.65}}},
        ]

        aggregated = evaluator._aggregate_scores(sample_results)

        assert "test" in aggregated
        assert (
            aggregated["test"]["mean"] == 0.7
        )  # Uses first numeric value (precision): (0.8 + 0.6) / 2
        assert aggregated["test"]["count"] == 2

    def test_invalid_score_type_handling(self):
        """Test aggregation with invalid score types."""
        evaluator = Evaluator(
            dataset=self.mock_dataset,
            models=[self.mock_model],
            scorers=[MockScorer("test", "invalid")],
            output_dir="./test_output",
        )

        sample_results = [
            {"scores": {"test": "invalid_string"}},
            {"scores": {"test": None}},
        ]

        aggregated = evaluator._aggregate_scores(sample_results)

        # Should handle invalid scores gracefully
        assert "test" in aggregated
        assert aggregated["test"]["mean"] == 0.0  # Should convert to 0.0
        assert aggregated["test"]["count"] == 1  # None scores are filtered out

    def test_mixed_valid_invalid_scores(self):
        """Test aggregation with mix of valid and invalid scores."""
        evaluator = Evaluator(
            dataset=self.mock_dataset,
            models=[self.mock_model],
            scorers=[AccuracyScorer()],
            output_dir="./test_output",
        )

        sample_results = [
            {"scores": {"accuracy": 1.0}},
            {"scores": {"accuracy": None}},  # None score should be skipped
            {"scores": {"accuracy": 0.5}},
        ]

        aggregated = evaluator._aggregate_scores(sample_results)

        assert "accuracy" in aggregated
        assert aggregated["accuracy"]["mean"] == 0.75  # (1.0 + 0.5) / 2
        assert aggregated["accuracy"]["count"] == 2  # None was skipped

    def test_no_scores_handling(self):
        """Test aggregation when no scores are available."""
        evaluator = Evaluator(
            dataset=self.mock_dataset,
            models=[self.mock_model],
            scorers=[AccuracyScorer()],
            output_dir="./test_output",
        )

        sample_results = [{"scores": {}}, {"scores": {"accuracy": None}}]

        aggregated = evaluator._aggregate_scores(sample_results)

        # Should return empty dict when no valid scores
        assert aggregated == {}

    def test_empty_sample_results(self):
        """Test aggregation with empty sample results."""
        evaluator = Evaluator(
            dataset=self.mock_dataset,
            models=[self.mock_model],
            scorers=[AccuracyScorer()],
            output_dir="./test_output",
        )

        sample_results = []

        aggregated = evaluator._aggregate_scores(sample_results)

        assert aggregated == {}

    def test_real_scorer_instances(self):
        """Test aggregation with real scorer instances."""
        accuracy_scorer = AccuracyScorer()
        exact_match_scorer = ExactMatchScorer()
        f1_scorer = F1Scorer()

        evaluator = Evaluator(
            dataset=self.mock_dataset,
            models=[self.mock_model],
            scorers=[accuracy_scorer, exact_match_scorer, f1_scorer],
            output_dir="./test_output",
        )

        # Test with actual scorer outputs
        sample_results = []

        # Test predictions and ground truths
        test_cases = [
            ("4", "4"),  # Exact match
            ("blue", "blue"),  # Exact match
            ("The answer is 4", "4"),  # Partial match
            ("Blue", "blue"),  # Case difference
        ]

        for i, (prediction, ground_truth) in enumerate(test_cases):
            scores = {}

            # Get actual scores from each scorer
            scores[accuracy_scorer.name] = accuracy_scorer.score(
                prediction, ground_truth
            )
            scores[exact_match_scorer.name] = exact_match_scorer.score(
                prediction, ground_truth
            )
            scores[f1_scorer.name] = f1_scorer.score(prediction, ground_truth)

            sample_results.append({"sample_id": f"test_{i}", "scores": scores})

        # This should not raise any exceptions
        aggregated = evaluator._aggregate_scores(sample_results)

        # Check that all scorers have results
        assert "accuracy" in aggregated
        assert "exact_match" in aggregated
        assert "f1" in aggregated

        # Check that all results have expected structure
        for scorer_name in ["accuracy", "exact_match", "f1"]:
            assert "mean" in aggregated[scorer_name]
            assert "count" in aggregated[scorer_name]
            assert "min" in aggregated[scorer_name]
            assert "max" in aggregated[scorer_name]

        # F1 scorer should have additional detailed metrics
        assert "precision_mean" in aggregated["f1"]
        assert "recall_mean" in aggregated["f1"]
        assert "f1_mean" in aggregated["f1"]

    def test_integration_with_full_evaluation(self):
        """Integration test with full evaluation workflow."""
        # Create a mock dataset with proper iteration
        test_data = [
            {"id": "1", "input": "What is 2+2?", "expected": "4"},
            {"id": "2", "input": "What color is the sky?", "expected": "blue"},
        ]

        mock_dataset = Mock()
        mock_dataset.__iter__ = Mock(return_value=iter(test_data))
        mock_dataset.get_info.return_value = {"name": "test_dataset", "size": 2}

        # Create a mock model
        mock_model = Mock()
        mock_model.name = "test_model"
        mock_model.generate.side_effect = ["4", "blue"]
        mock_model.get_info.return_value = {"name": "test_model", "type": "mock"}

        # Create evaluator with multiple scorers
        evaluator = Evaluator(
            dataset=mock_dataset,
            models=[mock_model],
            scorers=[AccuracyScorer(), ExactMatchScorer(), F1Scorer()],
            output_dir="./test_output",
        )

        # Mock the _evaluate_model method to test aggregation
        sample_results = [
            {
                "sample_id": "1",
                "scores": {
                    "accuracy": 1.0,
                    "exact_match": 1.0,
                    "f1": {"precision": 1.0, "recall": 1.0, "f1": 1.0, "score": 1.0},
                },
            },
            {
                "sample_id": "2",
                "scores": {
                    "accuracy": 1.0,
                    "exact_match": 1.0,
                    "f1": {"precision": 1.0, "recall": 1.0, "f1": 1.0, "score": 1.0},
                },
            },
        ]

        # Test aggregation directly
        aggregated = evaluator._aggregate_scores(sample_results)

        # Should not raise any exceptions and should return valid results
        assert isinstance(aggregated, dict)
        assert len(aggregated) == 3  # Three scorers

        for scorer_name in ["accuracy", "exact_match", "f1"]:
            assert scorer_name in aggregated
            assert "mean" in aggregated[scorer_name]
            assert aggregated[scorer_name]["mean"] == 1.0  # All perfect scores
