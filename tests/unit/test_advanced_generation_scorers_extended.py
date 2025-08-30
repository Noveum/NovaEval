"""
Extended tests for advanced generation scorers to improve coverage.
Focuses on edge cases, error handling, and fallback logic.
"""

from unittest.mock import Mock, patch

import pytest

from src.novaeval.scorers.advanced_generation_scorers import (
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
from src.novaeval.scorers.base import ScoreResult


class TestBiasDetectionScorerExtended:
    def test_init_with_custom_params(self):
        model = Mock()
        scorer = BiasDetectionScorer(model, threshold=0.9, max_score=10.0)
        assert scorer.threshold == 0.9
        assert scorer.max_score == 10.0
        assert scorer.model == model

    @pytest.mark.asyncio
    async def test_evaluate_empty_output(self):
        model = Mock()
        scorer = BiasDetectionScorer(model)

        result = await scorer.evaluate("input", "")
        assert result.score == 0.0
        assert result.passed is False
        assert "No answer provided" in result.reasoning

    @pytest.mark.asyncio
    async def test_evaluate_json_decode_error_fallback(self):
        model = Mock()
        scorer = BiasDetectionScorer(model)

        with patch(
            "src.novaeval.scorers.advanced_generation_scorers.call_llm",
            return_value="invalid json response with bias score 3",
        ):
            result = await scorer.evaluate("input", "output")

            # Should use fallback parsing to extract number 3
            expected_quality = (
                5 + 1 - 3
            ) / 5  # (max_score + 1 - bias_score) / max_score
            assert result.score == pytest.approx(expected_quality, rel=1e-3)

    @pytest.mark.asyncio
    async def test_evaluate_no_numbers_in_response(self):
        model = Mock()
        scorer = BiasDetectionScorer(model)

        with patch(
            "src.novaeval.scorers.advanced_generation_scorers.call_llm",
            return_value="no numbers here",
        ):
            result = await scorer.evaluate("input", "output")

            # Should use default bias_score of 1.0
            expected_quality = (5 + 1 - 1.0) / 5
            assert result.score == pytest.approx(expected_quality, rel=1e-3)

    @pytest.mark.asyncio
    async def test_evaluate_exception_handling(self):
        model = Mock()
        scorer = BiasDetectionScorer(model)

        with patch(
            "src.novaeval.scorers.advanced_generation_scorers.call_llm",
            side_effect=Exception("API error"),
        ):
            result = await scorer.evaluate("input", "output")

            assert result.score == 0.0
            assert result.passed is False
            assert "Error:" in result.reasoning

    def test_score_method_sync(self):
        model = Mock()
        scorer = BiasDetectionScorer(model)

        with patch("asyncio.run") as mock_run:
            mock_run.return_value = ScoreResult(
                score=0.8, passed=True, reasoning="Test", metadata={}
            )

            result = scorer.score("prediction", "ground_truth", {"context": "test"})
            assert result["score"] == 0.8


class TestFactualAccuracyScorer:
    @pytest.mark.asyncio
    async def test_evaluate_no_context(self):
        model = Mock()
        scorer = FactualAccuracyScorer(model)

        result = await scorer.evaluate("input", "output")
        assert result.score == 0.0
        assert "No answer or context provided" in result.reasoning

    @pytest.mark.asyncio
    async def test_evaluate_exception_handling(self):
        model = Mock()
        scorer = FactualAccuracyScorer(model)

        with patch(
            "src.novaeval.scorers.advanced_generation_scorers.call_llm",
            side_effect=Exception("API error"),
        ):
            result = await scorer.evaluate("input", "output", context="context")

            assert result.score == 0.0
            assert "Error:" in result.reasoning

    @pytest.mark.asyncio
    async def test_evaluate_json_parsing_fallback(self):
        model = Mock()
        scorer = FactualAccuracyScorer(model)

        with patch(
            "src.novaeval.scorers.advanced_generation_scorers.call_llm",
            return_value="accuracy score is 4",
        ):
            result = await scorer.evaluate("input", "output", context="context")

            # Should extract the number 4 as fallback
            assert result.score == 0.8  # 4/5

    def test_score_method_sync(self):
        model = Mock()
        scorer = FactualAccuracyScorer(model)

        with patch("asyncio.run") as mock_run:
            mock_run.return_value = ScoreResult(
                score=0.9, passed=True, reasoning="Test", metadata={}
            )

            result = scorer.score("prediction", "ground_truth", {"context": "test"})
            assert result["score"] == 0.9


class TestClaimVerificationScorer:
    @pytest.mark.asyncio
    async def test_evaluate_no_claims(self):
        model = Mock()
        scorer = ClaimVerificationScorer(model)

        # Test with empty output to trigger the no answer case
        result = await scorer.evaluate("input", "", context="context")

        assert result.score == 0.0
        assert "No answer provided" in result.reasoning

    @pytest.mark.asyncio
    async def test_evaluate_exception_handling(self):
        model = Mock()
        scorer = ClaimVerificationScorer(model)

        with (
            patch(
                "src.novaeval.scorers.advanced_generation_scorers.parse_claims",
                return_value=["claim1"],
            ),
            patch(
                "src.novaeval.scorers.advanced_generation_scorers.call_llm",
                side_effect=Exception("API error"),
            ),
        ):
            result = await scorer.evaluate("input", "output", context="context")

            assert result.score == 0.0
            assert "Error:" in result.reasoning

    def test_score_method_sync(self):
        model = Mock()
        scorer = ClaimVerificationScorer(model)

        with patch("asyncio.run") as mock_run:
            mock_run.return_value = ScoreResult(
                score=0.7, passed=True, reasoning="Test", metadata={}
            )

            result = scorer.score("prediction", "ground_truth", {"context": "test"})
            assert result["score"] == 0.7


class TestInformationDensityScorer:
    def test_parse_density_score_valid_json(self):
        model = Mock()
        scorer = InformationDensityScorer(model)

        response = '{"density_rating": 4, "reasoning": "high density"}'
        result = scorer._parse_density_score(response)
        assert result == 4

    def test_parse_density_score_invalid_json_with_rating(self):
        model = Mock()
        scorer = InformationDensityScorer(model)

        response = "The density rating is 3 out of 5"
        result = scorer._parse_density_score(response)
        assert result == 5  # Takes the last number found

    def test_parse_density_score_no_rating(self):
        model = Mock()
        scorer = InformationDensityScorer(model)

        response = "No clear rating here"
        result = scorer._parse_density_score(response)
        assert result == 3  # Default fallback

    @pytest.mark.asyncio
    async def test_evaluate_exception_handling(self):
        model = Mock()
        scorer = InformationDensityScorer(model)

        with patch(
            "src.novaeval.scorers.advanced_generation_scorers.call_llm",
            side_effect=Exception("API error"),
        ):
            result = await scorer.evaluate("input", "output")

            assert result.score == 0.0
            assert "Error:" in result.reasoning


class TestClarityAndCoherenceScorer:
    def test_parse_clarity_score_valid_json(self):
        model = Mock()
        scorer = ClarityAndCoherenceScorer(model)

        response = '{"clarity_score": 4, "coherence_score": 5}'
        result = scorer._parse_clarity_score(response)
        assert result == 5  # Returns single float, takes last number

    def test_parse_clarity_score_fallback(self):
        model = Mock()
        scorer = ClarityAndCoherenceScorer(model)

        response = "clarity: 3, coherence: 4"
        result = scorer._parse_clarity_score(response)
        assert result == 4  # Returns single float, takes last number

    def test_parse_clarity_score_default(self):
        model = Mock()
        scorer = ClarityAndCoherenceScorer(model)

        response = "no scores here"
        result = scorer._parse_clarity_score(response)
        assert result == 3.0  # Default fallback


class TestConflictResolutionScorer:
    def test_parse_conflict_score_valid_json(self):
        model = Mock()
        scorer = ConflictResolutionScorer(model)

        response = '{"conflict_resolution": 4}'
        result = scorer._parse_conflict_score(response)
        assert result == 4

    def test_parse_conflict_score_fallback(self):
        model = Mock()
        scorer = ConflictResolutionScorer(model)

        response = "resolution score: 2"
        result = scorer._parse_conflict_score(response)
        assert result == 2

    @pytest.mark.asyncio
    async def test_evaluate_single_chunk(self):
        model = Mock()
        scorer = ConflictResolutionScorer(model)

        context = "single chunk"
        result = await scorer.evaluate("input", "output", context=context)

        assert result.score == 1.0  # No conflicts with single chunk
        assert "Single context provided" in result.reasoning


class TestContextPrioritizationScorer:
    def test_parse_prioritization_score_valid_json(self):
        model = Mock()
        scorer = ContextPrioritizationScorer(model)

        response = '{"prioritization_score": 4}'
        result = scorer._parse_prioritization_score(response)
        assert result == 4

    def test_parse_prioritization_score_fallback(self):
        model = Mock()
        scorer = ContextPrioritizationScorer(model)

        response = "prioritization: 3"
        result = scorer._parse_prioritization_score(response)
        assert result == 3


class TestCitationQualityScorer:
    def test_parse_citation_score_valid_json(self):
        model = Mock()
        scorer = CitationQualityScorer(model)

        response = '{"citation_quality": 4}'
        result = scorer._parse_citation_score(response)
        assert result == 4

    def test_parse_citation_score_fallback(self):
        model = Mock()
        scorer = CitationQualityScorer(model)

        response = "citation quality: 2"
        result = scorer._parse_citation_score(response)
        assert result == 2


class TestToneConsistencyScorer:
    def test_parse_tone_score_valid_json(self):
        model = Mock()
        scorer = ToneConsistencyScorer(model)

        response = '{"tone_consistency": 4}'
        result = scorer._parse_tone_score(response)
        assert result == 4

    def test_parse_tone_score_fallback(self):
        model = Mock()
        scorer = ToneConsistencyScorer(model)

        response = "tone consistency: 3"
        result = scorer._parse_tone_score(response)
        assert result == 3


class TestTerminologyConsistencyScorer:
    def test_parse_terminology_score_valid_json(self):
        model = Mock()
        scorer = TerminologyConsistencyScorer(model)

        response = '{"terminology_consistency": 4}'
        result = scorer._parse_terminology_score(response)
        assert result == 4

    def test_parse_terminology_score_fallback(self):
        model = Mock()
        scorer = TerminologyConsistencyScorer(model)

        response = "terminology: 2"
        result = scorer._parse_terminology_score(response)
        assert result == 2


class TestContextGroundednessScorer:
    def test_parse_groundedness_score_valid_json(self):
        model = Mock()
        scorer = ContextGroundednessScorer(model)

        response = '{"groundedness_score": 4}'
        result = scorer._parse_groundedness_score(response)
        assert result == 4

    def test_parse_groundedness_score_fallback(self):
        model = Mock()
        scorer = ContextGroundednessScorer(model)

        response = "groundedness: 3"
        result = scorer._parse_groundedness_score(response)
        assert result == 3

    @pytest.mark.asyncio
    async def test_evaluate_exception_handling(self):
        model = Mock()
        scorer = ContextGroundednessScorer(model)

        with patch(
            "src.novaeval.scorers.advanced_generation_scorers.call_llm",
            side_effect=Exception("API error"),
        ):
            result = await scorer.evaluate("input", "output", context="context")

            assert result.score == 0.0
            assert "Error:" in result.reasoning


class TestRAGAnswerQualityScorer:
    def test_parse_quality_score_valid_json(self):
        model = Mock()
        scorer = RAGAnswerQualityScorer(model)

        response = '{"quality_score": 4}'
        result = scorer._parse_quality_score(response)
        assert result == 4

    def test_parse_quality_score_fallback(self):
        model = Mock()
        scorer = RAGAnswerQualityScorer(model)

        response = "quality: 3"
        result = scorer._parse_quality_score(response)
        assert result == 3


class TestHallucinationDetectionScorer:
    def test_parse_hallucination_score_valid_json(self):
        model = Mock()
        scorer = HallucinationDetectionScorer(model)

        response = '{"hallucination_score": 2}'
        result = scorer._parse_hallucination_score(response)
        assert result == 2

    def test_parse_hallucination_score_fallback(self):
        model = Mock()
        scorer = HallucinationDetectionScorer(model)

        response = "hallucination level: 1"
        result = scorer._parse_hallucination_score(response)
        assert result == 1


class TestSourceAttributionScorer:
    def test_parse_attribution_score_valid_json(self):
        model = Mock()
        scorer = SourceAttributionScorer(model)

        response = '{"attribution_score": 4}'
        result = scorer._parse_attribution_score(response)
        assert result == 4

    def test_parse_attribution_score_fallback(self):
        model = Mock()
        scorer = SourceAttributionScorer(model)

        response = "attribution: 3"
        result = scorer._parse_attribution_score(response)
        assert result == 3


class TestAnswerCompletenessScorer:
    def test_parse_completeness_score_valid_json(self):
        model = Mock()
        scorer = AnswerCompletenessScorer(model)

        response = '{"completeness_score": 4}'
        result = scorer._parse_completeness_score(response)
        assert result == 4

    def test_parse_completeness_score_fallback(self):
        model = Mock()
        scorer = AnswerCompletenessScorer(model)

        response = "completeness: 3"
        result = scorer._parse_completeness_score(response)
        assert result == 3


class TestQuestionAnswerAlignmentScorer:
    def test_parse_alignment_score_valid_json(self):
        model = Mock()
        scorer = QuestionAnswerAlignmentScorer(model)

        response = '{"alignment_score": 4}'
        result = scorer._parse_alignment_score(response)
        assert result == 4

    def test_parse_alignment_score_fallback(self):
        model = Mock()
        scorer = QuestionAnswerAlignmentScorer(model)

        response = "alignment: 3"
        result = scorer._parse_alignment_score(response)
        assert result == 3


class TestCrossContextSynthesisScorer:
    def test_parse_synthesis_score_valid_json(self):
        model = Mock()
        scorer = CrossContextSynthesisScorer(model)

        response = '{"synthesis_score": 4}'
        result = scorer._parse_synthesis_score(response)
        assert result == 4

    def test_parse_synthesis_score_fallback(self):
        model = Mock()
        scorer = CrossContextSynthesisScorer(model)

        response = "synthesis: 3"
        result = scorer._parse_synthesis_score(response)
        assert result == 3

    @pytest.mark.asyncio
    async def test_evaluate_single_chunk(self):
        model = Mock()
        scorer = CrossContextSynthesisScorer(model)

        context = "single chunk"
        result = await scorer.evaluate("input", "output", context=context)

        assert result.score == 1.0  # Perfect score for single chunk
        assert "Single context provided" in result.reasoning


class TestTechnicalAccuracyScorer:
    def test_parse_technical_score_valid_json(self):
        model = Mock()
        scorer = TechnicalAccuracyScorer(model)

        response = '{"technical_accuracy": 4}'
        result = scorer._parse_technical_score(response)
        assert result == 4

    def test_parse_technical_score_fallback(self):
        model = Mock()
        scorer = TechnicalAccuracyScorer(model)

        response = "technical accuracy: 3"
        result = scorer._parse_technical_score(response)
        assert result == 3


# Test edge cases and error handling
class TestAdvancedScorersErrorHandling:
    @pytest.mark.asyncio
    async def test_multiple_scorers_exception_handling(self):
        """Test that all scorers handle exceptions gracefully."""
        model = Mock()

        scorers = [
            BiasDetectionScorer(model),
            FactualAccuracyScorer(model),
            ClaimVerificationScorer(model),
            InformationDensityScorer(model),
            ClarityAndCoherenceScorer(model),
            ConflictResolutionScorer(model),
            ContextPrioritizationScorer(model),
            CitationQualityScorer(model),
            ToneConsistencyScorer(model),
            TerminologyConsistencyScorer(model),
            ContextFaithfulnessScorerPP(model),
            ContextGroundednessScorer(model),
            ContextCompletenessScorer(model),
            ContextConsistencyScorer(model),
            RAGAnswerQualityScorer(model),
            HallucinationDetectionScorer(model),
            SourceAttributionScorer(model),
            AnswerCompletenessScorer(model),
            QuestionAnswerAlignmentScorer(model),
            CrossContextSynthesisScorer(model),
            TechnicalAccuracyScorer(model),
        ]

        with patch(
            "src.novaeval.scorers.advanced_generation_scorers.call_llm",
            side_effect=Exception("API error"),
        ):
            for scorer in scorers:
                result = await scorer.evaluate("input", "output", context="context")
                # All scorers should handle exceptions gracefully
                assert hasattr(result, "score")  # Check it's a ScoreResult-like object
                assert result.score >= 0.0
                assert result.score <= 1.0


class TestAdvancedGenerationCoverage:
    """Additional tests to improve advanced_generation_scorers.py coverage."""

    def test_bias_detection_scorer_initialization(self):
        """Test BiasDetectionScorer initialization."""
        mock_model = Mock()
        scorer = BiasDetectionScorer(model=mock_model)

        assert scorer.model == mock_model
        assert scorer.name == "BiasDetectionScorer"

    def test_factual_accuracy_scorer_initialization(self):
        """Test FactualAccuracyScorer initialization."""
        mock_model = Mock()
        scorer = FactualAccuracyScorer(model=mock_model)

        assert scorer.model == mock_model
        assert scorer.name == "FactualAccuracyScorer"

    def test_clarity_coherence_scorer_initialization(self):
        """Test ClarityAndCoherenceScorer initialization."""
        mock_model = Mock()
        scorer = ClarityAndCoherenceScorer(model=mock_model)

        assert scorer.model == mock_model
        assert scorer.name == "ClarityAndCoherenceScorer"

    def test_tone_consistency_scorer_initialization(self):
        """Test ToneConsistencyScorer initialization."""
        mock_model = Mock()
        scorer = ToneConsistencyScorer(model=mock_model)

        assert scorer.model == mock_model
        assert scorer.name == "ToneConsistencyScorer"

    def test_scorers_with_custom_params(self):
        """Test scorers with custom parameters."""
        mock_model = Mock()

        bias_scorer = BiasDetectionScorer(model=mock_model, threshold=0.8)
        assert bias_scorer.threshold == 0.8
        assert bias_scorer.name == "BiasDetectionScorer"

        factual_scorer = FactualAccuracyScorer(model=mock_model, threshold=0.9)
        assert factual_scorer.threshold == 0.9
        assert factual_scorer.name == "FactualAccuracyScorer"

        clarity_scorer = ClarityAndCoherenceScorer(model=mock_model, threshold=0.7)
        assert clarity_scorer.threshold == 0.7
        assert clarity_scorer.name == "ClarityAndCoherenceScorer"

        tone_scorer = ToneConsistencyScorer(model=mock_model, threshold=0.6)
        assert tone_scorer.threshold == 0.6
        assert tone_scorer.name == "ToneConsistencyScorer"
