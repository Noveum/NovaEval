"""
Additional tests for PanelOfJudgesScorer to increase coverage.
"""

from unittest.mock import AsyncMock, patch

import pytest

from novaeval.models.base import BaseModel
from novaeval.scorers.panel_judge import (
    AggregationMethod,
    JudgeConfig,
    PanelOfJudgesScorer,
    PanelResult,
)

pytestmark = pytest.mark.unit


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
    """Test cases to increase coverage of PanelOfJudgesScorer."""

    def test_init_with_weighted_mean_normalization(self):
        """Test weight normalization during initialization."""
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

    def test_init_with_zero_total_weight(self):
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
        assert scorer.judges[0].weight == 0.0
        assert scorer.judges[1].weight == 0.0

    @pytest.mark.asyncio
    async def test_evaluate_all_judges_fail(self):
        """Test evaluation when all judges fail."""
        mock_model = MockLLMModel()
        judge = JudgeConfig(model=mock_model, weight=1.0)
        scorer = PanelOfJudgesScorer(judges=[judge])

        # Mock the _evaluate_with_judge method to raise exception
        with patch.object(
            scorer,
            "_evaluate_with_judge",
            new_callable=AsyncMock,
            side_effect=Exception("Model error"),
        ):
            result = await scorer.evaluate(
                input_text="test input", output_text="test output"
            )

        assert result.score == 0.0
        assert not result.passed
        assert "All judges failed to evaluate" in result.reasoning
        assert "failed_judges" in result.metadata

    @pytest.mark.asyncio
    async def test_evaluate_with_invalid_results(self):
        """Test evaluation with invalid judge results."""
        mock_model = MockLLMModel()
        mock_model.generate = AsyncMock(return_value="invalid response")

        judge = JudgeConfig(model=mock_model, weight=1.0)
        scorer = PanelOfJudgesScorer(judges=[judge])

        # Mock the result processing to return invalid data
        with patch.object(scorer, "_process_judge_results_sync", return_value=0.0):
            result = await scorer.evaluate(
                input_text="test input", output_text="test output"
            )

        assert result.score == 0.0

    def test_aggregate_scores_consensus_method(self):
        """Test consensus aggregation method."""
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
        assert result == 0.8  # Should fallback to mean (0.8)

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
        """Test prompt building with different parameter combinations."""
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

    def test_process_judge_results_empty(self):
        """Test processing empty judge results."""
        mock_model = MockLLMModel()
        judge = JudgeConfig(model=mock_model, weight=1.0)
        scorer = PanelOfJudgesScorer(judges=[judge])

        result = scorer._process_judge_results_sync([])
        assert result == 0.0

    def test_process_judge_results_invalid(self):
        """Test processing invalid judge results."""
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

    def test_score_method_sync_wrapper(self):
        """Test the sync wrapper for score method."""
        mock_model = MockLLMModel()
        judge = JudgeConfig(model=mock_model, weight=1.0)
        scorer = PanelOfJudgesScorer(judges=[judge])

        # Mock the _evaluate_with_judge_sync method to return a valid result
        with patch.object(
            scorer,
            "_evaluate_with_judge_sync",
            return_value={"score": 0.8, "reasoning": "test"},
        ):
            result = scorer.score("test input", "test output")

        assert result == 0.8

    def test_score_method_with_exception(self):
        """Test score method with exception."""
        mock_model = MockLLMModel()
        judge = JudgeConfig(model=mock_model, weight=1.0)
        scorer = PanelOfJudgesScorer(judges=[judge])

        # Mock the synchronous helper method that's called by ThreadPoolExecutor
        with patch.object(
            scorer, "_evaluate_with_judge_sync", side_effect=Exception("Test error")
        ):
            result = scorer.score("test input", "test output")

        assert result == 0.0

    def test_panel_result_creation(self):
        """Test PanelResult creation and validation."""
        # Test PanelResult creation
        panel_result = PanelResult(
            aggregated_score=0.8,
            individual_scores=[0.8, 0.9, 0.7],
            individual_reasonings=["good", "excellent", "fair"],
            judge_names=["Judge1", "Judge2", "Judge3"],
            aggregation_method=AggregationMethod.MEAN,
            consensus_level=0.8,
        )

        assert panel_result.aggregated_score == 0.8
        assert len(panel_result.individual_scores) == 3
        assert panel_result.aggregation_method == AggregationMethod.MEAN
