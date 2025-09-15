"""
Extended tests for panel judge functionality to improve coverage.
"""

import pytest

from novaeval.models.base import BaseModel
from novaeval.scorers.panel_judge import (
    AggregationMethod,
    JudgeConfig,
    PanelOfJudgesScorer,
)


class MockLLMModel(BaseModel):
    """Mock LLM model for testing."""

    def __init__(self, name="MockModel", model_name="mock-model"):
        super().__init__(name=name, model_name=model_name)

    async def generate(self, prompt: str, **kwargs) -> str:
        return "Mock response"

    def generate_batch(self, prompts: list[str], **kwargs) -> list[str]:
        return ["Mock response"] * len(prompts)

    def get_provider(self) -> str:
        return "mock"

    def validate_connection(self) -> bool:
        return True


class TestPanelJudgeCoverage:
    """Additional tests to improve panel_judge.py coverage."""

    def test_judge_config_validation_negative_weight(self):
        """Test JudgeConfig validation with negative weight."""
        mock_model = MockLLMModel()

        with pytest.raises(ValueError, match="Judge weight must be non-negative"):
            JudgeConfig(model=mock_model, weight=-1.0)

    def test_panel_scorer_initialization_no_judges(self):
        """Test PanelOfJudgesScorer initialization with no judges."""
        with pytest.raises(ValueError, match="At least one judge must be provided"):
            PanelOfJudgesScorer(judges=[])

    def test_panel_scorer_weight_normalization(self):
        """Test weight normalization for weighted mean aggregation."""
        mock_model1 = MockLLMModel()
        mock_model2 = MockLLMModel()

        judge1 = JudgeConfig(model=mock_model1, weight=2.0)
        judge2 = JudgeConfig(model=mock_model2, weight=3.0)

        scorer = PanelOfJudgesScorer(
            judges=[judge1, judge2], aggregation_method=AggregationMethod.WEIGHTED_MEAN
        )

        # Weights should be normalized (2.0/5.0 = 0.4, 3.0/5.0 = 0.6)
        assert abs(scorer.judges[0].weight - 0.4) < 0.01
        assert abs(scorer.judges[1].weight - 0.6) < 0.01

    def test_panel_scorer_zero_total_weight(self):
        """Test initialization when total weight is zero."""
        mock_model1 = MockLLMModel()
        mock_model2 = MockLLMModel()

        judge1 = JudgeConfig(model=mock_model1, weight=0.0)
        judge2 = JudgeConfig(model=mock_model2, weight=0.0)

        # Should not crash when total weight is zero
        scorer = PanelOfJudgesScorer(
            judges=[judge1, judge2], aggregation_method=AggregationMethod.WEIGHTED_MEAN
        )

        assert len(scorer.judges) == 2

    def test_aggregate_scores_all_methods(self):
        """Test all aggregation methods."""
        mock_model = MockLLMModel()
        judge = JudgeConfig(model=mock_model, weight=1.0)
        scorer = PanelOfJudgesScorer(judges=[judge])

        scores = [0.8, 0.9, 0.7]
        weights = [1.0, 1.0, 1.0]

        # Test MEAN
        result = scorer._aggregate_scores(scores, weights, AggregationMethod.MEAN)
        assert abs(result - 0.8) < 0.01

        # Test MEDIAN
        result = scorer._aggregate_scores(scores, weights, AggregationMethod.MEDIAN)
        assert result == 0.8

        # Test WEIGHTED_MEAN (now properly normalized)
        result = scorer._aggregate_scores(
            scores, weights, AggregationMethod.WEIGHTED_MEAN
        )
        assert abs(result - 0.8) < 0.01

        # Test MIN
        result = scorer._aggregate_scores(scores, weights, AggregationMethod.MIN)
        assert result == 0.7  # min([0.8, 0.9, 0.7]) = 0.7

        # Test MAX
        result = scorer._aggregate_scores(scores, weights, AggregationMethod.MAX)
        assert result == 0.9  # max([0.8, 0.9, 0.7]) = 0.9

    def test_aggregate_scores_majority_vote(self):
        """Test majority vote aggregation."""
        mock_model = MockLLMModel()
        judge = JudgeConfig(model=mock_model, weight=1.0)
        scorer = PanelOfJudgesScorer(judges=[judge], threshold=0.7)

        scores = [0.8, 0.9, 0.6]  # 2 pass, 1 fail
        weights = [1.0, 1.0, 1.0]

        result = scorer._aggregate_scores(
            scores, weights, AggregationMethod.MAJORITY_VOTE
        )
        assert result == 1.0  # Majority pass

    def test_aggregate_scores_consensus(self):
        """Test consensus aggregation."""
        mock_model = MockLLMModel()
        judge = JudgeConfig(model=mock_model, weight=1.0)
        scorer = PanelOfJudgesScorer(judges=[judge])

        scores = [0.8, 0.9, 0.7]
        weights = [1.0, 1.0, 1.0]

        result = scorer._aggregate_scores(scores, weights, AggregationMethod.CONSENSUS)
        assert 0.0 <= result <= 1.0

    def test_aggregate_scores_unknown_method(self):
        """Test aggregation with unknown method."""
        mock_model = MockLLMModel()
        judge = JudgeConfig(model=mock_model, weight=1.0)
        scorer = PanelOfJudgesScorer(judges=[judge])

        scores = [0.8, 0.9, 0.7]
        weights = [1.0, 1.0, 1.0]

        # Use a string that's not in the enum
        result = scorer._aggregate_scores(scores, weights, "unknown_method")
        # Should fallback to mean
        assert abs(result - 0.8) < 0.01

    def test_calculate_consensus_single_score(self):
        """Test consensus calculation with single score."""
        mock_model = MockLLMModel()
        judge = JudgeConfig(model=mock_model, weight=1.0)
        scorer = PanelOfJudgesScorer(judges=[judge])

        consensus = scorer._calculate_consensus([0.8])
        assert consensus == 1.0

    def test_calculate_consensus_multiple_scores(self):
        """Test consensus calculation with multiple scores."""
        mock_model = MockLLMModel()
        judge = JudgeConfig(model=mock_model, weight=1.0)
        scorer = PanelOfJudgesScorer(judges=[judge])

        consensus = scorer._calculate_consensus([0.8, 0.9, 0.7])
        assert 0.0 <= consensus <= 1.0

    def test_build_evaluation_prompt_variations(self):
        """Test evaluation prompt building with different parameters."""
        mock_model = MockLLMModel()
        judge = JudgeConfig(model=mock_model, weight=1.0)
        scorer = PanelOfJudgesScorer(judges=[judge])

        # Test with all parameters
        prompt = scorer._build_evaluation_prompt(
            input_text="What is AI?",
            output_text="AI is artificial intelligence",
            expected_output="Artificial intelligence is...",
            context="AI context",
        )
        assert "What is AI?" in prompt
        assert "AI is artificial intelligence" in prompt
        assert "Artificial intelligence is..." in prompt
        assert "AI context" in prompt

        # Test without expected output
        prompt = scorer._build_evaluation_prompt(
            input_text="What is AI?",
            output_text="AI is artificial intelligence",
            expected_output=None,
            context="AI context",
        )
        assert "Expected/Reference Answer" not in prompt

        # Test without context
        prompt = scorer._build_evaluation_prompt(
            input_text="What is AI?",
            output_text="AI is artificial intelligence",
            expected_output="Artificial intelligence is...",
            context=None,
        )
        assert "Additional Context" not in prompt

    def test_panel_scorer_process_results_empty(self):
        """Test processing results with empty list."""
        mock_model = MockLLMModel()
        judge = JudgeConfig(model=mock_model, weight=1.0)
        scorer = PanelOfJudgesScorer(judges=[judge])

        result = scorer._process_judge_results_sync([])
        assert result == 0.0

    def test_panel_scorer_process_results_invalid(self):
        """Test processing results with invalid data."""
        mock_model = MockLLMModel()
        judge = JudgeConfig(model=mock_model, weight=1.0)
        scorer = PanelOfJudgesScorer(judges=[judge])

        # Mock judge with invalid result
        results = [(judge, {"invalid": "data"})]
        result = scorer._process_judge_results_sync(results)
        assert result == 0.0

    def test_aggregate_scores_mismatched_lengths(self):
        """Test aggregation with mismatched score and weight lengths."""
        mock_model = MockLLMModel()
        judge = JudgeConfig(model=mock_model, weight=1.0)
        scorer = PanelOfJudgesScorer(judges=[judge])

        scores = [0.8, 0.9, 0.7]
        weights = [0.5, 0.3]  # Mismatched length

        result = scorer._aggregate_scores(
            scores, weights, AggregationMethod.WEIGHTED_MEAN
        )
        # Should fallback to mean
        assert abs(result - 0.8) < 0.01
