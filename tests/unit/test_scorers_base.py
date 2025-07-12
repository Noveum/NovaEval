"""
Unit tests for base scorer functionality.
"""

import pytest

from novaeval.scorers.base import BaseScorer, ScoreResult


class ConcreteScorer(BaseScorer):
    """Concrete implementation of BaseScorer for testing."""

    def __init__(self, name: str = "test_scorer", **kwargs):
        # Remove name from kwargs to avoid duplicate argument
        kwargs.pop("name", None)
        description = kwargs.pop("description", "Test scorer for unit tests")
        super().__init__(name, description=description, **kwargs)

    def score(self, prediction, ground_truth, context=None):
        """Mock scoring implementation."""
        # Simple exact match scoring with statistics tracking
        score = 1.0 if prediction == ground_truth else 0.0

        # Track the score for statistics
        self._track_score(score)

        return score


class TestScoreResult:
    """Test cases for ScoreResult class."""

    def test_score_result_creation(self):
        """Test creating a ScoreResult object."""
        result = ScoreResult(
            score=0.85,
            passed=True,
            reasoning="Good match between prediction and ground truth",
            metadata={"confidence": 0.9},
        )

        assert result.score == 0.85
        assert result.passed is True
        assert result.reasoning == "Good match between prediction and ground truth"
        assert result.metadata["confidence"] == 0.9

    def test_score_result_default_metadata(self):
        """Test ScoreResult with default metadata."""
        result = ScoreResult(score=0.5, passed=False, reasoning="Partial match")

        assert result.score == 0.5
        assert result.passed is False
        assert result.reasoning == "Partial match"
        assert result.metadata == {}

    def test_score_result_validation(self):
        """Test ScoreResult validation."""
        # Valid score
        result = ScoreResult(score=0.5, passed=True, reasoning="Valid")
        assert result.score == 0.5

        # Test that pydantic validates types
        with pytest.raises(ValueError):
            ScoreResult(score="invalid", passed=True, reasoning="Invalid score type")


class TestBaseScorer:
    """Test cases for BaseScorer class."""

    def test_init_default(self):
        """Test initialization with default parameters."""
        scorer = ConcreteScorer()

        assert scorer.name == "test_scorer"
        assert scorer.description == "Test scorer for unit tests"
        assert scorer.total_scores == 0
        assert scorer.score_sum == 0.0
        assert scorer.scores_history == []

    def test_init_with_params(self):
        """Test initialization with custom parameters."""
        scorer = ConcreteScorer(
            name="custom_scorer",
            description="Custom test scorer",
            threshold=0.8,
            custom_param="custom_value",
        )

        assert scorer.name == "custom_scorer"
        assert scorer.description == "Custom test scorer"
        assert scorer.kwargs["threshold"] == 0.8
        assert scorer.kwargs["custom_param"] == "custom_value"

    def test_score_implementation(self):
        """Test the concrete score implementation."""
        scorer = ConcreteScorer()

        # Test exact match
        score = scorer.score("hello", "hello")
        assert score == 1.0

        # Test non-match
        score = scorer.score("hello", "hello world")
        assert score == 0.0

    def test_abstract_score_method(self):
        """Test that abstract score method cannot be called directly."""
        with pytest.raises(TypeError):
            # Cannot instantiate abstract class
            BaseScorer("test")

    def test_validate_inputs_valid(self):
        """Test input validation with valid inputs."""
        scorer = ConcreteScorer()

        assert scorer.validate_inputs("prediction", "ground_truth") is True
        assert (
            scorer.validate_inputs("prediction", "ground_truth", {"key": "value"})
            is True
        )

    def test_validate_inputs_invalid(self):
        """Test input validation with invalid inputs."""
        scorer = ConcreteScorer()

        # None prediction
        assert scorer.validate_inputs(None, "ground_truth") is False

        # None ground truth
        assert scorer.validate_inputs("prediction", None) is False

        # Both None
        assert scorer.validate_inputs(None, None) is False

        # Empty strings should be valid
        assert scorer.validate_inputs("", "") is True

    def test_get_info(self):
        """Test scorer info retrieval."""
        scorer = ConcreteScorer(threshold=0.8, custom_param="value")

        info = scorer.get_info()

        assert info["name"] == "test_scorer"
        assert info["description"] == "Test scorer for unit tests"
        assert info["type"] == "ConcreteScorer"
        assert info["total_scores"] == 0
        assert info["average_score"] == 0.0
        assert info["config"]["threshold"] == 0.8
        assert info["config"]["custom_param"] == "value"

    def test_scoring_statistics_tracking(self):
        """Test that scoring statistics are tracked correctly."""
        scorer = ConcreteScorer()

        # Score some predictions
        scores = [
            scorer.score("test1", "test1"),  # 1.0
            scorer.score("test2", "test2"),  # 1.0
            scorer.score("test3", "different"),  # 0.0
        ]

        assert scores == [1.0, 1.0, 0.0]

        # Check that statistics were tracked
        assert scorer.total_scores == 3
        assert (
            abs(scorer.score_sum - 2.0) < 1e-10
        )  # Use small epsilon for floating point comparison
        assert len(scorer.scores_history) == 3

        # Check average score
        info = scorer.get_info()
        expected_avg = 2.0 / 3
        assert abs(info["average_score"] - expected_avg) < 1e-10

    def test_get_stats_empty(self):
        """Test get_stats with no scores."""
        scorer = ConcreteScorer()

        stats = scorer.get_stats()

        assert stats["count"] == 0
        assert stats["mean"] == 0.0
        assert stats["min"] == 0.0
        assert stats["max"] == 0.0
        assert stats["std"] == 0.0

    def test_get_stats_with_scores(self):
        """Test get_stats with tracked scores."""
        scorer = ConcreteScorer()

        # Track some scores
        scores = [0.8, 0.9, 0.7, 0.6, 0.85]
        for score in scores:
            scorer._track_score(score)

        stats = scorer.get_stats()

        assert stats["count"] == 5
        assert abs(stats["mean"] - 0.77) < 0.01  # (0.8+0.9+0.7+0.6+0.85)/5 = 0.77
        assert stats["min"] == 0.6
        assert stats["max"] == 0.9
        assert stats["std"] > 0  # Should have some standard deviation

    def test_reset_stats(self):
        """Test resetting statistics."""
        scorer = ConcreteScorer()

        # Track some scores
        scorer._track_score(0.8)
        scorer._track_score(0.9)

        assert scorer.total_scores == 2
        assert len(scorer.scores_history) == 2

        # Reset
        scorer.reset_stats()

        assert scorer.total_scores == 0
        assert scorer.score_sum == 0.0
        assert scorer.scores_history == []

    def test_str_representation(self):
        """Test string representation."""
        scorer = ConcreteScorer()
        str_repr = str(scorer)
        assert "ConcreteScorer" in str_repr
        assert "test_scorer" in str_repr

    def test_repr_representation(self):
        """Test detailed string representation."""
        scorer = ConcreteScorer()
        repr_str = repr(scorer)
        assert "ConcreteScorer" in repr_str
        assert "test_scorer" in repr_str

    def test_score_with_context(self):
        """Test scoring with context parameter."""
        scorer = ConcreteScorer()

        context = {"sample_id": "test_001", "metadata": {"type": "test"}}
        score = scorer.score("hello", "hello world", context)

        # The score should be 0.0 since strings don't match exactly
        assert score == 0.0

        # Test exact match with context
        score = scorer.score("hello", "hello", context)
        assert score == 1.0

    def test_different_score_types(self):
        """Test handling different types of predictions and ground truth."""
        scorer = ConcreteScorer()

        # String inputs
        score1 = scorer.score("test", "testing")
        assert isinstance(score1, float)
        assert score1 == 0.0  # Not exact match

        # Different string inputs
        score2 = scorer.score("short", "much longer text")
        assert isinstance(score2, float)
        assert score2 == 0.0  # Not exact match

        # Empty strings
        score3 = scorer.score("", "")
        assert score3 == 1.0  # Exact match

    def test_score_batch(self):
        """Test batch scoring functionality."""
        scorer = ConcreteScorer()

        predictions = ["hello", "world", "test"]
        ground_truths = ["hello", "word", "testing"]

        scores = scorer.score_batch(predictions, ground_truths)

        assert len(scores) == 3
        assert scores[0] == 1.0  # "hello" vs "hello" - exact match
        assert scores[1] == 0.0  # "world" vs "word" - no match
        assert scores[2] == 0.0  # "test" vs "testing" - no match

    def test_score_batch_with_contexts(self):
        """Test batch scoring with contexts."""
        scorer = ConcreteScorer()

        predictions = ["hello", "world"]
        ground_truths = ["hello", "word"]
        contexts = [{"id": "1"}, {"id": "2"}]

        scores = scorer.score_batch(predictions, ground_truths, contexts)

        assert len(scores) == 2
        assert scores[0] == 1.0  # "hello" vs "hello" - exact match
        assert scores[1] == 0.0  # "world" vs "word" - no match

    def test_score_batch_with_error(self):
        """Test batch scoring with scoring errors."""

        # Create a scorer that will fail on certain inputs
        class FailingScorer(BaseScorer):
            def score(self, prediction, ground_truth, context=None):
                if prediction == "fail":
                    raise ValueError("Intentional failure")
                return 0.5

        scorer = FailingScorer("failing_scorer")

        predictions = ["good", "fail", "good"]
        ground_truths = ["good", "fail", "good"]

        scores = scorer.score_batch(predictions, ground_truths)

        assert len(scores) == 3
        assert scores[0] == 0.5
        assert scores[1] == 0.0  # Failed scoring returns 0.0
        assert scores[2] == 0.5

    def test_track_score_with_dict(self):
        """Test tracking scores that are dictionaries."""
        scorer = ConcreteScorer()

        # Track a dictionary score
        dict_score = {"score": 0.85, "confidence": 0.9}
        scorer._track_score(dict_score)

        assert scorer.total_scores == 1
        assert scorer.score_sum == 0.85  # Should extract the "score" value
        assert scorer.scores_history[0] == dict_score

    def test_track_score_with_dict_no_score_key(self):
        """Test tracking dict scores without 'score' key."""
        scorer = ConcreteScorer()

        # Track a dictionary score without 'score' key
        dict_score = {"accuracy": 0.75, "precision": 0.8}
        scorer._track_score(dict_score)

        assert scorer.total_scores == 1
        assert scorer.score_sum == 0.75  # Should use first value
        assert scorer.scores_history[0] == dict_score

    def test_from_config_classmethod(self):
        """Test creating scorer from configuration."""
        config = {
            "name": "config_scorer",
            "description": "Scorer from config",
            "threshold": 0.8,
        }

        scorer = ConcreteScorer.from_config(config)

        assert scorer.name == "config_scorer"
        assert scorer.description == "Scorer from config"
        assert scorer.kwargs["threshold"] == 0.8


class TestScorerEdgeCases:
    """Test edge cases for scorer functionality."""

    def test_scorer_with_minimal_params(self):
        """Test scorer with minimal initialization."""
        scorer = ConcreteScorer(name="minimal")

        assert scorer.name == "minimal"
        assert scorer.description == "Test scorer for unit tests"  # Uses init default

    def test_score_with_empty_strings(self):
        """Test scoring with empty strings."""
        scorer = ConcreteScorer()

        score = scorer.score("", "")
        assert score == 1.0  # Exact match

    def test_score_with_very_long_strings(self):
        """Test scoring with very long strings."""
        scorer = ConcreteScorer()

        long_string1 = "a" * 1000
        long_string2 = "b" * 1100

        score = scorer.score(long_string1, long_string2)
        assert score == 0.0  # No match

        # Test exact match with long strings
        score = scorer.score(long_string1, long_string1)
        assert score == 1.0  # Exact match
