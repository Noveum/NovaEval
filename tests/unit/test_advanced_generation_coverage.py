"""
Additional tests for advanced generation scorers to increase coverage.
"""

from unittest.mock import AsyncMock, patch

import pytest

from novaeval.scorers.advanced_generation_scorers import (
    AnswerCompletenessScorer,
    BiasDetectionScorer,
    CitationQualityScorer,
    ClaimVerificationScorer,
    ClarityAndCoherenceScorer,
    ConflictResolutionScorer,
    ContextCompletenessScorer,
    ContextConsistencyScorer,
    ContextFaithfulnessScorerPP,
    ContextGroundednessScorer,
    ContextPrioritizationScorer,
    CrossContextSynthesisScorer,
    FactualAccuracyScorer,
    HallucinationDetectionScorer,
    InformationDensityScorer,
    QuestionAnswerAlignmentScorer,
    RAGAnswerQualityScorer,
    SourceAttributionScorer,
    TechnicalAccuracyScorer,
    TerminologyConsistencyScorer,
    ToneConsistencyScorer,
)
from novaeval.scorers.base import ScoreResult

pytestmark = pytest.mark.unit


class MockLLMModel:
    """Mock LLM model for testing."""

    def __init__(self, name="MockModel"):
        self.name = name

    async def generate(self, prompt: str, **kwargs) -> str:
        return "Mock response"

    def generate_batch(self, prompts: list[str], **kwargs) -> list[str]:
        return ["Mock response"] * len(prompts)


class TestAdvancedGenerationCoverage:
    """Test cases to increase coverage of advanced generation scorers."""

    @pytest.mark.asyncio
    async def test_bias_detection_scorer_edge_cases(self):
        """Test BiasDetectionScorer with edge cases."""
        mock_model = MockLLMModel()
        scorer = BiasDetectionScorer(model=mock_model)

        # Test with empty output
        result = await scorer.evaluate(
            input_text="test", output_text="", context="test context"
        )
        assert isinstance(result, ScoreResult)

        # Test with very short output
        result = await scorer.evaluate(
            input_text="test", output_text="ok", context="test context"
        )
        assert isinstance(result, ScoreResult)

    @pytest.mark.asyncio
    async def test_factual_accuracy_scorer_edge_cases(self):
        """Test FactualAccuracyScorer with edge cases."""
        mock_model = MockLLMModel()
        scorer = FactualAccuracyScorer(model=mock_model)

        # Test without context
        result = await scorer.evaluate(input_text="test", output_text="test output")
        assert isinstance(result, ScoreResult)

        # Test with empty context
        result = await scorer.evaluate(
            input_text="test", output_text="test output", context=""
        )
        assert isinstance(result, ScoreResult)

    @pytest.mark.asyncio
    async def test_claim_verification_scorer_edge_cases(self):
        """Test ClaimVerificationScorer with edge cases."""
        mock_model = MockLLMModel()
        scorer = ClaimVerificationScorer(model=mock_model)

        # Test without context
        result = await scorer.evaluate(input_text="test", output_text="test output")
        assert isinstance(result, ScoreResult)

        # Test with empty claims
        result = await scorer.evaluate(
            input_text="test", output_text="test output", context="test context"
        )
        assert isinstance(result, ScoreResult)

    @pytest.mark.asyncio
    async def test_information_density_scorer_edge_cases(self):
        """Test InformationDensityScorer with edge cases."""
        mock_model = MockLLMModel()
        scorer = InformationDensityScorer(model=mock_model)

        # Test with empty output
        result = await scorer.evaluate(
            input_text="test", output_text="", context="test context"
        )
        assert isinstance(result, ScoreResult)

        # Test with very short output
        result = await scorer.evaluate(
            input_text="test", output_text="ok", context="test context"
        )
        assert isinstance(result, ScoreResult)

    @pytest.mark.asyncio
    async def test_clarity_coherence_scorer_edge_cases(self):
        """Test ClarityAndCoherenceScorer with edge cases."""
        mock_model = MockLLMModel()
        scorer = ClarityAndCoherenceScorer(model=mock_model)

        # Test with empty output
        result = await scorer.evaluate(
            input_text="test", output_text="", context="test context"
        )
        assert isinstance(result, ScoreResult)

        # Test with malformed output
        result = await scorer.evaluate(
            input_text="test", output_text="...", context="test context"
        )
        assert isinstance(result, ScoreResult)

    @pytest.mark.asyncio
    async def test_tone_consistency_scorer_edge_cases(self):
        """Test ToneConsistencyScorer with edge cases."""
        mock_model = MockLLMModel()
        scorer = ToneConsistencyScorer(model=mock_model)

        # Test with empty output
        result = await scorer.evaluate(
            input_text="test", output_text="", context="test context"
        )
        assert isinstance(result, ScoreResult)

        # Test without context
        result = await scorer.evaluate(input_text="test", output_text="test output")
        assert isinstance(result, ScoreResult)

    @pytest.mark.asyncio
    async def test_contextual_scorers_with_single_chunk(self):
        """Test contextual scorers with single context chunk."""
        mock_model = MockLLMModel()

        scorers = [
            ContextFaithfulnessScorerPP(model=mock_model),
            ContextCompletenessScorer(model=mock_model),
            ContextConsistencyScorer(model=mock_model),
            CrossContextSynthesisScorer(model=mock_model),
            ConflictResolutionScorer(model=mock_model),
            ContextPrioritizationScorer(model=mock_model),
        ]

        for scorer in scorers:
            result = await scorer.evaluate(
                input_text="test",
                output_text="test output",
                context="single context chunk",
            )
            assert isinstance(result, ScoreResult)

    @pytest.mark.asyncio
    async def test_contextual_scorers_with_empty_context(self):
        """Test contextual scorers with empty context."""
        mock_model = MockLLMModel()

        scorers = [
            ContextFaithfulnessScorerPP(model=mock_model),
            ContextCompletenessScorer(model=mock_model),
            ContextConsistencyScorer(model=mock_model),
            CrossContextSynthesisScorer(model=mock_model),
            ConflictResolutionScorer(model=mock_model),
            ContextPrioritizationScorer(model=mock_model),
        ]

        for scorer in scorers:
            result = await scorer.evaluate(
                input_text="test", output_text="test output", context=""
            )
            assert isinstance(result, ScoreResult)

    @pytest.mark.asyncio
    async def test_citation_quality_scorer_edge_cases(self):
        """Test CitationQualityScorer with edge cases."""
        mock_model = MockLLMModel()
        scorer = CitationQualityScorer(model=mock_model)

        # Test without context
        result = await scorer.evaluate(input_text="test", output_text="test output")
        assert isinstance(result, ScoreResult)

        # Test with empty context
        result = await scorer.evaluate(
            input_text="test", output_text="test output", context=""
        )
        assert isinstance(result, ScoreResult)

    @pytest.mark.asyncio
    async def test_terminology_consistency_scorer_edge_cases(self):
        """Test TerminologyConsistencyScorer with edge cases."""
        mock_model = MockLLMModel()
        scorer = TerminologyConsistencyScorer(model=mock_model)

        # Test with empty output
        result = await scorer.evaluate(
            input_text="test", output_text="", context="test context"
        )
        assert isinstance(result, ScoreResult)

        # Test without context
        result = await scorer.evaluate(input_text="test", output_text="test output")
        assert isinstance(result, ScoreResult)

    @pytest.mark.asyncio
    async def test_context_groundedness_scorer_edge_cases(self):
        """Test ContextGroundednessScorer with edge cases."""
        mock_model = MockLLMModel()
        scorer = ContextGroundednessScorer(model=mock_model)

        # Test without context
        result = await scorer.evaluate(input_text="test", output_text="test output")
        assert isinstance(result, ScoreResult)

        # Test with empty context
        result = await scorer.evaluate(
            input_text="test", output_text="test output", context=""
        )
        assert isinstance(result, ScoreResult)

    @pytest.mark.asyncio
    async def test_rag_answer_quality_scorer_edge_cases(self):
        """Test RAGAnswerQualityScorer with edge cases."""
        mock_model = MockLLMModel()
        scorer = RAGAnswerQualityScorer(model=mock_model)

        # Test with empty output
        result = await scorer.evaluate(
            input_text="test", output_text="", context="test context"
        )
        assert isinstance(result, ScoreResult)

        # Test without context
        result = await scorer.evaluate(input_text="test", output_text="test output")
        assert isinstance(result, ScoreResult)

    @pytest.mark.asyncio
    async def test_hallucination_detection_scorer_edge_cases(self):
        """Test HallucinationDetectionScorer with edge cases."""
        mock_model = MockLLMModel()
        scorer = HallucinationDetectionScorer(model=mock_model)

        # Test with empty output
        result = await scorer.evaluate(
            input_text="test", output_text="", context="test context"
        )
        assert isinstance(result, ScoreResult)

        # Test without context
        result = await scorer.evaluate(input_text="test", output_text="test output")
        assert isinstance(result, ScoreResult)

    @pytest.mark.asyncio
    async def test_source_attribution_scorer_edge_cases(self):
        """Test SourceAttributionScorer with edge cases."""
        mock_model = MockLLMModel()
        scorer = SourceAttributionScorer(model=mock_model)

        # Test without context
        result = await scorer.evaluate(input_text="test", output_text="test output")
        assert isinstance(result, ScoreResult)

        # Test with empty context
        result = await scorer.evaluate(
            input_text="test", output_text="test output", context=""
        )
        assert isinstance(result, ScoreResult)

    @pytest.mark.asyncio
    async def test_answer_completeness_scorer_edge_cases(self):
        """Test AnswerCompletenessScorer with edge cases."""
        mock_model = MockLLMModel()
        scorer = AnswerCompletenessScorer(model=mock_model)

        # Test with empty output
        result = await scorer.evaluate(
            input_text="test", output_text="", context="test context"
        )
        assert isinstance(result, ScoreResult)

        # Test without context
        result = await scorer.evaluate(input_text="test", output_text="test output")
        assert isinstance(result, ScoreResult)

    @pytest.mark.asyncio
    async def test_question_answer_alignment_scorer_edge_cases(self):
        """Test QuestionAnswerAlignmentScorer with edge cases."""
        mock_model = MockLLMModel()
        scorer = QuestionAnswerAlignmentScorer(model=mock_model)

        # Test with empty output
        result = await scorer.evaluate(
            input_text="test", output_text="", context="test context"
        )
        assert isinstance(result, ScoreResult)

        # Test without context
        result = await scorer.evaluate(input_text="test", output_text="test output")
        assert isinstance(result, ScoreResult)

    @pytest.mark.asyncio
    async def test_technical_accuracy_scorer_edge_cases(self):
        """Test TechnicalAccuracyScorer with edge cases."""
        mock_model = MockLLMModel()
        scorer = TechnicalAccuracyScorer(model=mock_model)

        # Test with empty output
        result = await scorer.evaluate(
            input_text="test", output_text="", context="test context"
        )
        assert isinstance(result, ScoreResult)

        # Test without context
        result = await scorer.evaluate(input_text="test", output_text="test output")
        assert isinstance(result, ScoreResult)

    @pytest.mark.asyncio
    async def test_advanced_scorers_with_exceptions(self):
        """Test advanced scorers with model exceptions."""
        mock_model = MockLLMModel()

        scorers = [
            BiasDetectionScorer(model=mock_model),
            FactualAccuracyScorer(model=mock_model),
            ClarityAndCoherenceScorer(model=mock_model),
            ToneConsistencyScorer(model=mock_model),
            InformationDensityScorer(model=mock_model),
            ClaimVerificationScorer(model=mock_model),
        ]

        for scorer in scorers:
            # Mock the scorer's internal _call_model method to raise exception
            with patch.object(
                scorer,
                "_call_model",
                new=AsyncMock(side_effect=Exception("Model error")),
            ):
                result = await scorer.evaluate(
                    input_text="test", output_text="test output", context="test context"
                )
            assert isinstance(result, ScoreResult)
            assert result.score == 0.0

    def test_advanced_scorers_sync_wrapper(self):
        """Test sync wrapper methods for advanced scorers."""
        mock_model = MockLLMModel()

        scorers = [
            BiasDetectionScorer(model=mock_model),
            FactualAccuracyScorer(model=mock_model),
            ClarityAndCoherenceScorer(model=mock_model),
            ToneConsistencyScorer(model=mock_model),
        ]

        for scorer in scorers:
            # Mock the async evaluate method with AsyncMock
            with patch.object(
                scorer,
                "evaluate",
                new=AsyncMock(
                    return_value=ScoreResult(score=0.8, passed=True, reasoning="test")
                ),
            ):
                result = scorer.score(
                    "test input", "test output", context={"context": "test context"}
                )

            assert result["score"] == 0.8

    @pytest.mark.asyncio
    async def test_advanced_scorers_with_malformed_responses(self):
        """Test advanced scorers with malformed LLM responses."""
        mock_model = MockLLMModel()

        scorers = [
            BiasDetectionScorer(model=mock_model),
            FactualAccuracyScorer(model=mock_model),
            ClarityAndCoherenceScorer(model=mock_model),
            ToneConsistencyScorer(model=mock_model),
        ]

        for scorer in scorers:
            # Mock the scorer's internal _call_model method to return malformed response
            with patch.object(
                scorer,
                "_call_model",
                new=AsyncMock(return_value="invalid json response"),
            ):
                result = await scorer.evaluate(
                    input_text="test", output_text="test output", context="test context"
                )
            assert isinstance(result, ScoreResult)
            # Should fallback to default scores
            assert 0.0 <= result.score <= 1.0
