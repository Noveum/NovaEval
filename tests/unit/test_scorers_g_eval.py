"""Tests for G-Eval scorer."""

import inspect
from unittest.mock import AsyncMock, Mock, patch

import pytest

from novaeval.scorers.g_eval import CommonGEvalCriteria, GEvalCriteria, GEvalScorer


@pytest.mark.unit
class TestBasicImport:
    """Basic test to verify imports work."""

    def test_can_import_classes(self):
        """Test that we can import the classes."""
        assert inspect.isclass(GEvalScorer)
        assert inspect.isclass(GEvalCriteria)
        assert inspect.isclass(CommonGEvalCriteria)


@pytest.mark.unit
class TestGEvalCriteriaModel:
    """Test the GEvalCriteria Pydantic model."""

    def test_create_criteria_with_all_fields(self):
        """Test creating GEvalCriteria - this exercises the model."""
        criteria = GEvalCriteria(
            name="Test Criteria",
            description="Test description",
            steps=["Step 1", "Step 2"],
            score_range=(1, 10),
        )

        assert criteria.name == "Test Criteria"
        assert criteria.description == "Test description"
        assert criteria.steps == ["Step 1", "Step 2"]
        assert criteria.score_range == (1, 10)

    def test_create_criteria_with_defaults(self):
        """Test creating GEvalCriteria with default score range."""
        criteria = GEvalCriteria(
            name="Default Test", description="Test with defaults", steps=["Step 1"]
        )

        assert criteria.score_range == (1, 5)  # Default


@pytest.mark.unit
class TestCommonGEvalCriteria:
    """Test CommonGEvalCriteria static methods."""

    def test_correctness(self):
        """Test correctness criteria."""
        criteria = CommonGEvalCriteria.correctness()

        # These assertions force the method to execute
        assert criteria.name == "Correctness"
        assert "factually correct" in criteria.description
        assert len(criteria.steps) == 4
        assert criteria.score_range == (1, 5)
        assert "factual errors" in criteria.steps[0]

    def test_relevance(self):
        """Test relevance criteria."""
        criteria = CommonGEvalCriteria.relevance()

        assert criteria.name == "Relevance"
        assert "relevant" in criteria.description
        assert len(criteria.steps) == 4
        assert "main topic" in criteria.steps[0]

    def test_coherence(self):
        """Test coherence criteria."""
        criteria = CommonGEvalCriteria.coherence()

        assert criteria.name == "Coherence"
        assert "logical flow" in criteria.description
        assert len(criteria.steps) == 4
        assert "clear structure" in criteria.steps[0]

    def test_helpfulness(self):
        """Test helpfulness criteria."""
        criteria = CommonGEvalCriteria.helpfulness()

        assert criteria.name == "Helpfulness"
        assert "helpful" in criteria.description
        assert len(criteria.steps) == 4
        assert "actionable information" in criteria.steps[0]


@pytest.mark.unit
class TestGEvalScorerBasics:
    """Test basic GEvalScorer functionality."""

    @patch.multiple(GEvalScorer, __abstractmethods__=set())
    def test_scorer_initialization_with_string(self):
        """Test GEvalScorer initialization with string criteria."""
        mock_model = Mock()

        scorer = GEvalScorer(
            model=mock_model, criteria="Test evaluation criteria", threshold=0.7
        )

        # This exercises the __init__ method lines 37-66
        assert scorer.model == mock_model
        assert scorer.threshold == 0.7
        assert scorer.criteria.name == "Custom Evaluation"
        assert scorer.criteria.description == "Test evaluation criteria"
        assert len(scorer.criteria.steps) == 3

    @patch.multiple(GEvalScorer, __abstractmethods__=set())
    def test_scorer_initialization_with_criteria_object(self):
        """Test GEvalScorer initialization with GEvalCriteria object."""
        mock_model = Mock()
        criteria = GEvalCriteria(
            name="Test",
            description="Test description",
            steps=["Step 1", "Step 2"],
            score_range=(1, 5),
        )

        scorer = GEvalScorer(
            model=mock_model,
            criteria=criteria,
            threshold=0.8,
            use_cot=False,
            num_iterations=2,
        )

        assert scorer.criteria == criteria
        assert scorer.threshold == 0.8
        assert scorer.use_cot is False
        assert scorer.num_iterations == 2


@pytest.mark.unit
class TestGEvalScorerMethods:
    """Test GEvalScorer methods that need to be covered."""

    @patch.multiple(GEvalScorer, __abstractmethods__=set())
    def test_build_prompt_with_cot(self):
        """Test _build_prompt method with CoT - covers lines 72-123."""
        mock_model = Mock()
        criteria = GEvalCriteria(
            name="Test Evaluation",
            description="Test evaluation criteria",
            steps=["Read input", "Analyze output", "Provide score"],
            score_range=(1, 5),
        )

        scorer = GEvalScorer(model=mock_model, criteria=criteria, use_cot=True)
        prompt = scorer._build_prompt("Test input", "Test output")

        # Verify prompt structure - this executes lines 72-123
        assert "# Evaluation Task: Test Evaluation" in prompt
        assert "## Criteria:\nTest evaluation criteria" in prompt
        assert "## Score Range: 1 to 5" in prompt
        assert "## Evaluation Steps:" in prompt
        assert "1. Read input" in prompt
        assert "## Input:\nTest input" in prompt
        assert "## Output to Evaluate:\nTest output" in prompt
        assert "**Step-by-step Analysis:**" in prompt
        assert "**Final Score:**" in prompt

    @patch.multiple(GEvalScorer, __abstractmethods__=set())
    def test_build_prompt_without_cot(self):
        """Test _build_prompt method without CoT."""
        mock_model = Mock()
        criteria = GEvalCriteria(
            name="Test Evaluation",
            description="Test evaluation criteria",
            steps=["Read input", "Analyze output", "Provide score"],
            score_range=(1, 5),
        )

        scorer = GEvalScorer(model=mock_model, criteria=criteria, use_cot=False)
        prompt = scorer._build_prompt("Test input", "Test output")

        # Verify non-CoT structure
        assert "# Evaluation Task: Test Evaluation" in prompt
        assert "## Evaluation Steps:" not in prompt
        assert "**Step-by-step Analysis:**" not in prompt
        assert "**Score:**" in prompt
        assert "**Final Score:**" not in prompt

    @patch.multiple(GEvalScorer, __abstractmethods__=set())
    def test_build_prompt_with_context(self):
        """Test _build_prompt method with context."""
        mock_model = Mock()
        criteria = GEvalCriteria(
            name="Test Evaluation",
            description="Test evaluation criteria",
            steps=["Read input", "Analyze output", "Provide score"],
            score_range=(1, 5),
        )

        scorer = GEvalScorer(model=mock_model, criteria=criteria)
        prompt = scorer._build_prompt("Input", "Output", "Context info")

        assert "## Context:\nContext info" in prompt

    @patch.multiple(GEvalScorer, __abstractmethods__=set())
    def test_parse_response_final_score_format(self):
        """Test _parse_response method - covers lines 127-170."""
        mock_model = Mock()
        criteria = GEvalCriteria(
            name="Test", description="Test", steps=["Step 1"], score_range=(1, 5)
        )

        scorer = GEvalScorer(model=mock_model, criteria=criteria)
        response = "**Final Score:** 4\n**Reasoning:** Good response"

        score, reasoning = scorer._parse_response(response)

        assert score == 0.75  # (4-1)/(5-1) = 0.75
        assert "Good response" in reasoning  # Changed from exact match to contains

    @patch.multiple(GEvalScorer, __abstractmethods__=set())
    def test_parse_response_score_format(self):
        """Test _parse_response method with **Score:** format."""
        mock_model = Mock()
        criteria = GEvalCriteria(
            name="Test", description="Test", steps=["Step 1"], score_range=(1, 5)
        )

        scorer = GEvalScorer(model=mock_model, criteria=criteria)
        response = "**Score:** 3\n**Reasoning:** Average"

        score, reasoning = scorer._parse_response(response)

        assert score == 0.5  # (3-1)/(5-1) = 0.5
        assert "Average" in reasoning  # Changed from exact match to contains

    @patch.multiple(GEvalScorer, __abstractmethods__=set())
    def test_parse_response_no_score(self):
        """Test _parse_response when no score found."""
        mock_model = Mock()
        criteria = GEvalCriteria(
            name="Test", description="Test", steps=["Step 1"], score_range=(1, 5)
        )

        scorer = GEvalScorer(model=mock_model, criteria=criteria)
        response = "No score in this response"

        score, reasoning = scorer._parse_response(response)

        assert score == 0.0
        assert "Failed to parse score" in reasoning

    @patch.multiple(GEvalScorer, __abstractmethods__=set())
    @pytest.mark.asyncio
    async def test_evaluate_single_success(self):
        """Test _evaluate_single method - covers lines 176-183."""
        mock_model = Mock()
        mock_model.generate = AsyncMock(
            return_value="**Final Score:** 4\n**Reasoning:** Good"
        )

        criteria = GEvalCriteria(
            name="Test", description="Test", steps=["Step 1"], score_range=(1, 5)
        )

        scorer = GEvalScorer(model=mock_model, criteria=criteria)

        score, reasoning = await scorer._evaluate_single("input", "output")

        assert score == 0.75
        assert "Good" in reasoning  # Changed from exact match to contains
        mock_model.generate.assert_called_once()

    @patch.multiple(GEvalScorer, __abstractmethods__=set())
    @pytest.mark.asyncio
    async def test_evaluate_single_with_context(self):
        """Test _evaluate_single method with context."""
        mock_model = Mock()
        mock_model.generate = AsyncMock(
            return_value="**Final Score:** 4\n**Reasoning:** Good"
        )

        criteria = GEvalCriteria(
            name="Test", description="Test", steps=["Step 1"], score_range=(1, 5)
        )

        scorer = GEvalScorer(model=mock_model, criteria=criteria)

        await scorer._evaluate_single("input", "output", "context")

        # Verify context was passed to prompt
        call_args = mock_model.generate.call_args[0][0]
        assert "context" in call_args

    @patch.multiple(GEvalScorer, __abstractmethods__=set())
    @pytest.mark.asyncio
    async def test_evaluate_single_error(self):
        """Test _evaluate_single error handling."""
        mock_model = Mock()
        mock_model.generate = AsyncMock(side_effect=Exception("Model error"))

        criteria = GEvalCriteria(
            name="Test", description="Test", steps=["Step 1"], score_range=(1, 5)
        )

        scorer = GEvalScorer(model=mock_model, criteria=criteria)

        score, reasoning = await scorer._evaluate_single("input", "output")

        assert score == 0.0
        assert "Evaluation failed: Model error" in reasoning

    @patch.multiple(GEvalScorer, __abstractmethods__=set())
    @pytest.mark.asyncio
    async def test_evaluate_single_iteration(self):
        """Test main evaluate method with single iteration - covers lines 205-231."""
        mock_model = Mock()
        mock_model.generate = AsyncMock(
            return_value="**Final Score:** 4\n**Reasoning:** Good response"
        )

        criteria = GEvalCriteria(
            name="Test Evaluation",
            description="Test description",
            steps=["Step 1", "Step 2"],
            score_range=(1, 5),
        )

        scorer = GEvalScorer(model=mock_model, criteria=criteria, num_iterations=1)

        result = await scorer.evaluate(
            input_text="What is 2+2?", output_text="The answer is 4."
        )

        # Check ScoreResult structure
        assert hasattr(result, "score")
        assert hasattr(result, "passed")
        assert hasattr(result, "reasoning")
        assert hasattr(result, "metadata")

        assert result.score == 0.75  # (4-1)/(5-1) = 0.75
        assert result.passed is True  # Above 0.5 threshold
        assert "Iteration 1:" in result.reasoning
        assert "Good response" in result.reasoning

        # Check metadata
        assert result.metadata["criteria"] == "Test Evaluation"
        assert result.metadata["iterations"] == 1
        assert result.metadata["individual_scores"] == [0.75]
        assert result.metadata["confidence"] == 0.8  # Default for single iteration
        assert result.metadata["use_cot"] is True
        assert result.metadata["score_range"] == (1, 5)

    @patch.multiple(GEvalScorer, __abstractmethods__=set())
    @pytest.mark.asyncio
    async def test_evaluate_multiple_iterations(self):
        """Test main evaluate method with multiple iterations."""
        mock_model = Mock()
        mock_model.generate = AsyncMock(
            side_effect=[
                "**Final Score:** 4\n**Reasoning:** Good response",
                "**Final Score:** 5\n**Reasoning:** Excellent response",
                "**Final Score:** 3\n**Reasoning:** Average response",
            ]
        )

        criteria = GEvalCriteria(
            name="Multi Test",
            description="Multi iteration test",
            steps=["Step 1"],
            score_range=(1, 5),
        )

        scorer = GEvalScorer(model=mock_model, criteria=criteria, num_iterations=3)

        result = await scorer.evaluate(
            input_text="Test question", output_text="Test answer"
        )

        # Should average the scores: (0.75 + 1.0 + 0.5) / 3 = 0.75
        assert result.score == 0.75
        assert result.passed is True

        # Check all iterations are in reasoning
        assert "Iteration 1:" in result.reasoning
        assert "Iteration 2:" in result.reasoning
        assert "Iteration 3:" in result.reasoning
        assert "Final Score (average): 0.750" in result.reasoning

        # Check metadata
        assert result.metadata["iterations"] == 3
        assert result.metadata["individual_scores"] == [0.75, 1.0, 0.5]

        # Confidence should be calculated based on variance
        expected_mean = 0.75
        variance = sum((s - expected_mean) ** 2 for s in [0.75, 1.0, 0.5]) / 3
        expected_confidence = max(0.0, 1.0 - variance)
        assert abs(result.metadata["confidence"] - expected_confidence) < 0.01

    @patch.multiple(GEvalScorer, __abstractmethods__=set())
    @pytest.mark.asyncio
    async def test_evaluate_with_expected_output_and_context(self):
        """Test evaluate method with expected output and context parameters."""
        mock_model = Mock()
        mock_model.generate = AsyncMock(
            return_value="**Final Score:** 3\n**Reasoning:** Average"
        )

        criteria = GEvalCriteria(
            name="Context Test",
            description="Test with context",
            steps=["Step 1"],
            score_range=(1, 5),
        )

        scorer = GEvalScorer(model=mock_model, criteria=criteria)

        result = await scorer.evaluate(
            input_text="Question",
            output_text="Answer",
            expected_output="Expected answer",
            context="Additional context",
        )

        assert isinstance(result.score, float)
        assert isinstance(result.passed, bool)
        assert isinstance(result.reasoning, str)
        assert isinstance(result.metadata, dict)

    @patch.multiple(GEvalScorer, __abstractmethods__=set())
    @pytest.mark.asyncio
    async def test_evaluate_threshold_behavior(self):
        """Test different threshold values in evaluate method."""
        mock_model = Mock()
        mock_model.generate = AsyncMock(
            return_value="**Final Score:** 4\n**Reasoning:** Good"
        )

        criteria = GEvalCriteria(
            name="Threshold Test",
            description="Test thresholds",
            steps=["Step 1"],
            score_range=(1, 5),
        )

        # Test high threshold (should fail)
        scorer_high = GEvalScorer(model=mock_model, criteria=criteria, threshold=0.9)
        result_high = await scorer_high.evaluate("input", "output")

        assert result_high.score == 0.75  # (4-1)/(5-1) = 0.75
        assert result_high.passed is False  # 0.75 < 0.9

        # Test low threshold (should pass)
        scorer_low = GEvalScorer(model=mock_model, criteria=criteria, threshold=0.3)
        result_low = await scorer_low.evaluate("input", "output")

        assert result_low.score == 0.75
        assert result_low.passed is True  # 0.75 >= 0.3
