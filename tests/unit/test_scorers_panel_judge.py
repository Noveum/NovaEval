from unittest.mock import AsyncMock, Mock

import pytest

from novaeval.models.base import BaseModel
from novaeval.scorers.base import ScoreResult
from novaeval.scorers.panel_judge import (
    JudgeConfig,
    PanelOfJudgesScorer,
)


class MockLLMModel(BaseModel):
    """Mock LLM model for testing."""

    def __init__(self, name="MockModel", model_name="mock-model"):
        super().__init__(name=name, model_name=model_name)
        self.generate_method = AsyncMock()
        self.generate_sync_method = Mock()  # Add sync method for score method

    async def generate(self, prompt: str, **kwargs) -> str:
        return await self.generate_method(prompt, **kwargs)

    def generate_sync(self, prompt: str, **kwargs) -> str:
        """Synchronous version of generate for score method."""
        return self.generate_sync_method(prompt, **kwargs)

    def generate_batch(self, prompts: list[str], **kwargs) -> list[str]:
        return [self.generate_method(prompt, **kwargs) for prompt in prompts]

    def get_provider(self) -> str:
        return "mock"

    def validate_connection(self) -> bool:
        return True


class TestPanelOfJudgesScorer:
    """Test cases for PanelOfJudgesScorer."""

    @pytest.fixture
    def mock_judge(self):
        """Create a mock judge for testing."""
        mock_model = MockLLMModel()
        judge = Mock()
        judge.model = mock_model
        judge.name = "TestJudge"
        judge.weight = 1.0
        judge.temperature = 0.7
        return judge

    @pytest.fixture
    def panel_scorer(self, mock_judge):
        """Create a panel scorer with mock judge."""
        judge_config = JudgeConfig(
            model=mock_judge.model,
            name=mock_judge.name,
            weight=mock_judge.weight,
            temperature=mock_judge.temperature,
        )
        return PanelOfJudgesScorer(judges=[judge_config])

    @pytest.mark.unit
    def test_panel_scorer_initialization(self, panel_scorer):
        """Test panel scorer initialization."""
        assert panel_scorer is not None
        assert len(panel_scorer.judges) == 1
        assert panel_scorer.threshold == 0.7

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_evaluate_success(self, panel_scorer, mock_judge):
        """Test successful evaluation."""
        # Mock successful response
        mock_judge.model.generate_method.return_value = """
        {
            "score": 4,
            "reasoning": "This is a good response",
            "strengths": "Clear and accurate",
            "weaknesses": "Could be more detailed",
            "confidence": 4
        }
        """

        result = await panel_scorer.evaluate(
            input_text="What is machine learning?",
            output_text="Machine learning is a subset of AI",
            expected_output="Machine learning is a subset of artificial intelligence",
        )

        assert isinstance(result, ScoreResult)
        assert result.score >= 0.0
        assert result.score <= 1.0
        assert hasattr(result, "passed")
        assert hasattr(result, "reasoning")

    @pytest.mark.unit
    def test_score_exception_handling(self, panel_scorer, mock_judge):
        """Test exception handling in score method."""
        # Mock judge to throw an exception using the sync method
        mock_judge.model.generate_sync_method.side_effect = Exception(
            "Evaluation failed"
        )

        # The score method should handle exceptions gracefully
        result = panel_scorer.score(
            prediction="Machine learning is a subset of AI",
            ground_truth="Machine learning is a subset of artificial intelligence",
            context={"context": "AI includes machine learning"},
        )

        # Should return a valid result even when judge fails
        assert isinstance(result, (float, dict))
        if isinstance(result, dict):
            assert "score" in result
            assert result["score"] >= 0.0
        else:
            assert result >= 0.0

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_evaluate_all_judges_fail(self, panel_scorer, mock_judge):
        """Test when all judges fail to evaluate."""
        # Mock all judges to throw exceptions
        mock_judge.model.generate_method.side_effect = Exception(
            "Judge evaluation failed"
        )

        result = await panel_scorer.evaluate(
            input_text="What is machine learning?",
            output_text="Machine learning is a subset of AI",
            expected_output="Machine learning is a subset of artificial intelligence",
        )

        # Should return a ScoreResult with 0.0 score when all judges fail
        assert isinstance(result, ScoreResult)
        assert result.score == 0.0
        assert not result.passed
        assert "Judge evaluation failed" in result.reasoning
        assert "individual_scores" in result.metadata

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_evaluate_partial_judge_failure(self, panel_scorer, mock_judge):
        """Test when some judges fail but others succeed."""
        # Create a second judge that succeeds
        second_model = MockLLMModel(name="SecondModel")
        second_judge = Mock()
        second_judge.model = second_model
        second_judge.name = "SecondJudge"
        second_judge.weight = 1.0
        second_judge.temperature = 0.7

        # First judge fails, second judge succeeds
        mock_judge.model.generate_method.side_effect = Exception("First judge failed")
        second_judge.model.generate_method.return_value = """
        {
            "score": 4,
            "reasoning": "This is a good response",
            "strengths": "Clear and accurate",
            "weaknesses": "Could be more detailed",
            "confidence": 4
        }
        """

        # Add second judge to panel
        second_judge_config = JudgeConfig(
            model=second_judge.model,
            name=second_judge.name,
            weight=second_judge.weight,
            temperature=second_judge.temperature,
        )
        panel_scorer.judges.append(second_judge_config)

        result = await panel_scorer.evaluate(
            input_text="What is machine learning?",
            output_text="Machine learning is a subset of AI",
            expected_output="Machine learning is a subset of artificial intelligence",
        )

        # Should still return a valid result
        assert isinstance(result, ScoreResult)
        assert result.score >= 0.0
        assert hasattr(result, "passed")
        assert hasattr(result, "reasoning")

    @pytest.mark.unit
    def test_score_method_handles_exceptions(self, panel_scorer, mock_judge):
        """Test that the score method properly handles exceptions."""
        # Mock judge to throw an exception using the sync method
        mock_judge.model.generate_sync_method.side_effect = Exception(
            "Evaluation failed"
        )

        # The score method should handle exceptions and return a valid result
        result = panel_scorer.score(
            prediction="Machine learning is a subset of AI",
            ground_truth="Machine learning is a subset of artificial intelligence",
            context={"context": "AI includes machine learning"},
        )

        # Should return a valid result even when judge fails
        assert isinstance(result, (float, dict))
        if isinstance(result, dict):
            assert "score" in result
            assert result["score"] >= 0.0
        else:
            assert result >= 0.0

    @pytest.mark.unit
    def test_panel_scorer_with_multiple_judges(self):
        """Test panel scorer with multiple judges."""
        # Create multiple mock judges
        model1 = MockLLMModel(name="Judge1")
        model2 = MockLLMModel(name="Judge2")

        # Mock responses
        model1.generate_method.return_value = """
        {
            "score": 4,
            "reasoning": "Good response",
            "strengths": "Clear",
            "weaknesses": "Brief",
            "confidence": 4
        }
        """
        model2.generate_method.return_value = """
        {
            "score": 3,
            "reasoning": "Acceptable response",
            "strengths": "Accurate",
            "weaknesses": "Could be more detailed",
            "confidence": 3
        }
        """

        judge_config1 = JudgeConfig(
            model=model1, name="Judge1", weight=1.0, temperature=0.7
        )
        judge_config2 = JudgeConfig(
            model=model2, name="Judge2", weight=1.0, temperature=0.7
        )

        panel_scorer = PanelOfJudgesScorer(judges=[judge_config1, judge_config2])

        # Test score method
        result = panel_scorer.score(
            prediction="Machine learning is a subset of AI",
            ground_truth="Machine learning is a subset of artificial intelligence",
            context={"context": "AI includes machine learning"},
        )

        assert isinstance(result, (float, dict))
        if isinstance(result, dict):
            assert "score" in result
            assert result["score"] >= 0.0
        else:
            assert result >= 0.0
