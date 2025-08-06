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
from novaeval.scorers.basic_rag_scorers import (
    AggregateRAGScorer,
    ContextualPrecisionScorerPP,
    ContextualRecallScorerPP,
    RetrievalDiversityScorer,
    RetrievalRankingScorer,
    SemanticSimilarityScorer,
)

# Test fixtures are automatically available from conftest.py


@pytest.fixture
def sample_context():
    return "This is a sample context about machine learning. Machine learning is a subset of artificial intelligence."


# Test Context-Aware Generation Scorers
@pytest.mark.unit
@pytest.mark.asyncio
async def test_context_faithfulness_scorer_pp(mock_llm):
    scorer = ContextFaithfulnessScorerPP(mock_llm, threshold=0.8)
    result = await scorer.evaluate(
        "What is ML?", "Machine learning is AI", context="ML is AI subset"
    )
    assert isinstance(result, ScoreResult)
    assert hasattr(result, "score")
    assert hasattr(result, "passed")


@pytest.mark.unit
@pytest.mark.asyncio
async def test_context_groundedness_scorer(mock_llm):
    scorer = ContextGroundednessScorer(mock_llm, threshold=0.7)
    result = await scorer.evaluate(
        "What is ML?", "Machine learning is AI", context="ML is AI subset"
    )
    assert isinstance(result, ScoreResult)
    assert hasattr(result, "score")
    assert hasattr(result, "passed")


@pytest.mark.unit
@pytest.mark.asyncio
async def test_context_completeness_scorer(mock_llm):
    scorer = ContextCompletenessScorer(mock_llm, threshold=0.6)
    result = await scorer.evaluate(
        "What is ML?", "Machine learning is AI", context="ML is AI subset"
    )
    assert isinstance(result, ScoreResult)
    assert hasattr(result, "score")
    assert hasattr(result, "passed")


@pytest.mark.unit
@pytest.mark.asyncio
async def test_context_consistency_scorer(mock_llm):
    scorer = ContextConsistencyScorer(mock_llm, threshold=0.7)
    result = await scorer.evaluate(
        "What is ML?",
        "Machine learning is AI",
        context="ML is AI subset\n\nAI includes ML",
    )
    assert isinstance(result, ScoreResult)
    assert hasattr(result, "score")
    assert hasattr(result, "passed")


# Test Answer Quality Enhancement Scorers
@pytest.mark.unit
@pytest.mark.asyncio
async def test_rag_answer_quality_scorer(mock_llm):
    scorer = RAGAnswerQualityScorer(mock_llm, threshold=0.7)
    result = await scorer.evaluate(
        "What is ML?", "Machine learning is AI", context="ML is AI subset"
    )
    assert isinstance(result, ScoreResult)
    assert hasattr(result, "score")
    assert hasattr(result, "passed")


# Test Hallucination Detection Scorers
@pytest.mark.unit
@pytest.mark.asyncio
async def test_hallucination_detection_scorer(mock_llm):
    scorer = HallucinationDetectionScorer(mock_llm, threshold=0.8)
    result = await scorer.evaluate(
        "What is ML?", "Machine learning is AI", context="ML is AI subset"
    )
    assert isinstance(result, ScoreResult)
    assert hasattr(result, "score")
    assert hasattr(result, "passed")


@pytest.mark.unit
@pytest.mark.asyncio
async def test_source_attribution_scorer(mock_llm):
    scorer = SourceAttributionScorer(mock_llm, threshold=0.6)
    result = await scorer.evaluate(
        "What is ML?", "Machine learning is AI", context="ML is AI subset"
    )
    assert isinstance(result, ScoreResult)
    assert hasattr(result, "score")
    assert hasattr(result, "passed")


@pytest.mark.unit
@pytest.mark.asyncio
async def test_factual_accuracy_scorer(mock_llm):
    scorer = FactualAccuracyScorer(mock_llm, threshold=0.8)
    result = await scorer.evaluate(
        "What is ML?", "Machine learning is AI", context="ML is AI subset"
    )
    assert isinstance(result, ScoreResult)
    assert hasattr(result, "score")
    assert hasattr(result, "passed")


@pytest.mark.unit
@pytest.mark.asyncio
async def test_claim_verification_scorer(mock_llm):
    scorer = ClaimVerificationScorer(mock_llm, threshold=0.7)
    result = await scorer.evaluate(
        "What is ML?", "Machine learning is AI", context="ML is AI subset"
    )
    assert isinstance(result, ScoreResult)
    assert hasattr(result, "score")
    assert hasattr(result, "passed")


# Test Answer Completeness and Relevance Scorers
@pytest.mark.unit
@pytest.mark.asyncio
async def test_answer_completeness_scorer(mock_llm):
    scorer = AnswerCompletenessScorer(mock_llm, threshold=0.7)
    result = await scorer.evaluate(
        "What is ML?", "Machine learning is AI", context="ML is AI subset"
    )
    assert isinstance(result, ScoreResult)
    assert hasattr(result, "score")
    assert hasattr(result, "passed")


@pytest.mark.unit
@pytest.mark.asyncio
async def test_question_answer_alignment_scorer(mock_llm):
    scorer = QuestionAnswerAlignmentScorer(mock_llm, threshold=0.7)
    result = await scorer.evaluate(
        "What is ML?", "Machine learning is AI", context="ML is AI subset"
    )
    assert isinstance(result, ScoreResult)
    assert hasattr(result, "score")
    assert hasattr(result, "passed")


@pytest.mark.unit
@pytest.mark.asyncio
async def test_information_density_scorer(mock_llm):
    scorer = InformationDensityScorer(mock_llm, threshold=0.6)
    result = await scorer.evaluate(
        "What is ML?", "Machine learning is AI", context="ML is AI subset"
    )
    assert isinstance(result, ScoreResult)
    assert hasattr(result, "score")
    assert hasattr(result, "passed")


@pytest.mark.unit
@pytest.mark.asyncio
async def test_clarity_coherence_scorer(mock_llm):
    scorer = ClarityAndCoherenceScorer(mock_llm, threshold=0.7)
    result = await scorer.evaluate(
        "What is ML?", "Machine learning is AI", context="ML is AI subset"
    )
    assert isinstance(result, ScoreResult)
    assert hasattr(result, "score")
    assert hasattr(result, "passed")


# Test Multi-Context Integration Scorers
@pytest.mark.unit
@pytest.mark.asyncio
async def test_cross_context_synthesis_scorer(mock_llm):
    scorer = CrossContextSynthesisScorer(mock_llm, threshold=0.7)
    result = await scorer.evaluate(
        "What is ML?",
        "Machine learning is AI",
        context="ML is AI subset\n\nAI includes ML",
    )
    assert isinstance(result, ScoreResult)
    assert hasattr(result, "score")
    assert hasattr(result, "passed")


@pytest.mark.unit
@pytest.mark.asyncio
async def test_conflict_resolution_scorer(mock_llm):
    scorer = ConflictResolutionScorer(mock_llm, threshold=0.7)
    result = await scorer.evaluate(
        "What is ML?",
        "Machine learning is AI",
        context="ML is AI subset\n\nAI includes ML",
    )
    assert isinstance(result, ScoreResult)
    assert hasattr(result, "score")
    assert hasattr(result, "passed")


@pytest.mark.unit
@pytest.mark.asyncio
async def test_context_prioritization_scorer(mock_llm):
    scorer = ContextPrioritizationScorer(mock_llm, threshold=0.6)
    result = await scorer.evaluate(
        "What is ML?", "Machine learning is AI", context="ML is AI subset"
    )
    assert isinstance(result, ScoreResult)
    assert hasattr(result, "score")
    assert hasattr(result, "passed")


@pytest.mark.unit
@pytest.mark.asyncio
async def test_citation_quality_scorer(mock_llm):
    scorer = CitationQualityScorer(mock_llm, threshold=0.6)
    result = await scorer.evaluate(
        "What is ML?", "Machine learning is AI", context="ML is AI subset"
    )
    assert isinstance(result, ScoreResult)
    assert hasattr(result, "score")
    assert hasattr(result, "passed")


# Test Domain-Specific Evaluation Scorers
@pytest.mark.unit
@pytest.mark.asyncio
async def test_technical_accuracy_scorer(mock_llm):
    scorer = TechnicalAccuracyScorer(mock_llm, threshold=0.8)
    result = await scorer.evaluate(
        "What is ML?", "Machine learning is AI", context="ML is AI subset"
    )
    assert isinstance(result, ScoreResult)
    assert hasattr(result, "score")
    assert hasattr(result, "passed")


@pytest.mark.unit
@pytest.mark.asyncio
async def test_bias_detection_scorer(mock_llm):
    scorer = BiasDetectionScorer(mock_llm, threshold=0.8)
    result = await scorer.evaluate(
        "What is ML?", "Machine learning is AI", context="ML is AI subset"
    )
    assert isinstance(result, ScoreResult)
    assert hasattr(result, "score")
    assert hasattr(result, "passed")


@pytest.mark.unit
@pytest.mark.asyncio
async def test_tone_consistency_scorer(mock_llm):
    scorer = ToneConsistencyScorer(mock_llm, threshold=0.7)
    result = await scorer.evaluate(
        "What is ML?", "Machine learning is AI", context="ML is AI subset"
    )
    assert isinstance(result, ScoreResult)
    assert hasattr(result, "score")
    assert hasattr(result, "passed")


@pytest.mark.unit
@pytest.mark.asyncio
async def test_terminology_consistency_scorer(mock_llm):
    scorer = TerminologyConsistencyScorer(mock_llm, threshold=0.7)
    result = await scorer.evaluate(
        "What is ML?", "Machine learning is AI", context="ML is AI subset"
    )
    assert isinstance(result, ScoreResult)
    assert hasattr(result, "score")
    assert hasattr(result, "passed")


# Test existing scorers (keeping original tests)
@pytest.mark.unit
def test_contextual_precision_scorer_pp(mock_llm):
    scorer = ContextualPrecisionScorerPP(mock_llm, threshold=0.7)
    result = scorer.score(
        "Machine learning is AI", "What is ML?", {"context": "ML is AI subset"}
    )
    assert isinstance(result, (float, dict))
    if isinstance(result, dict):
        assert "score" in result
        assert result["score"] >= 0.0
    else:
        assert result >= 0.0


@pytest.mark.unit
def test_contextual_recall_scorer_pp(mock_llm):
    scorer = ContextualRecallScorerPP(mock_llm, threshold=0.7)
    result = scorer.score(
        "Machine learning is AI", "What is ML?", {"context": "ML is AI subset"}
    )
    assert isinstance(result, (float, dict))
    if isinstance(result, dict):
        assert "score" in result
        assert result["score"] >= 0.0
    else:
        assert result >= 0.0

@pytest.mark.unit
def test_retrieval_ranking_scorer():
    scorer = RetrievalRankingScorer(threshold=0.5)
    result = scorer.score(
        "Machine learning is AI", "What is ML?", {"context": "ML is AI subset"}
    )
    assert isinstance(result, (float, dict))
    if isinstance(result, dict):
        assert "score" in result
        assert result["score"] >= 0.0
    else:
        assert result >= 0.0


@pytest.mark.unit
def test_semantic_similarity_scorer():
    scorer = SemanticSimilarityScorer(threshold=0.7)
    result = scorer.score(
        "Machine learning is AI", "What is ML?", {"context": "ML is AI subset"}
    )
    assert isinstance(result, (float, dict))
    if isinstance(result, dict):
        assert "similarity" in result
        assert result["similarity"] >= 0.0
    else:
        assert result >= 0.0


@pytest.mark.unit
def test_retrieval_diversity_scorer():
    scorer = RetrievalDiversityScorer()
    result = scorer.score(
        "Machine learning is AI",
        "What is ML?",
        {"context": "ML is AI subset\n\nAI includes ML"},
    )
    assert isinstance(result, (float, dict))
    if isinstance(result, dict):
        assert "score" in result
        assert result["score"] >= 0.0
    else:
        assert result >= 0.0


@pytest.mark.unit
def test_aggregate_rag_scorer(mock_llm):
    scorers = {
        "precision": ContextualPrecisionScorerPP(mock_llm, threshold=0.7),
        "recall": ContextualRecallScorerPP(mock_llm, threshold=0.7),
    }
    weights = {"precision": 0.5, "recall": 0.5}
    scorer = AggregateRAGScorer(scorers, weights)
    result = scorer.score(
        "Machine learning is AI", "What is ML?", {"context": "ML is AI subset"}
    )
    assert isinstance(result, (float, dict))
    if isinstance(result, dict):
        assert "aggregate" in result
        assert result["aggregate"] >= 0.0
    else:
        assert result >= 0.0


# Test G-Eval scorers
@pytest.mark.unit
@pytest.mark.asyncio
async def test_g_eval_helpfulness_scorer(mock_llm):
    from novaeval.scorers.g_eval import CommonGEvalCriteria, GEvalScorer

    scorer = GEvalScorer(mock_llm, criteria=CommonGEvalCriteria.helpfulness())
    result = await scorer.evaluate(
        "What is ML?", "Machine learning is AI", context="ML is AI subset"
    )
    assert isinstance(result, ScoreResult)
    assert hasattr(result, "score")
    assert hasattr(result, "passed")


@pytest.mark.unit
@pytest.mark.asyncio
async def test_g_eval_correctness_scorer(mock_llm):
    from novaeval.scorers.g_eval import CommonGEvalCriteria, GEvalScorer

    scorer = GEvalScorer(mock_llm, criteria=CommonGEvalCriteria.correctness())
    result = await scorer.evaluate(
        "What is ML?", "Machine learning is AI", context="ML is AI subset"
    )
    assert isinstance(result, ScoreResult)
    assert hasattr(result, "score")
    assert hasattr(result, "passed")


@pytest.mark.unit
@pytest.mark.asyncio
async def test_g_eval_coherence_scorer(mock_llm):
    from novaeval.scorers.g_eval import CommonGEvalCriteria, GEvalScorer

    scorer = GEvalScorer(mock_llm, criteria=CommonGEvalCriteria.coherence())
    result = await scorer.evaluate(
        "What is ML?", "Machine learning is AI", context="ML is AI subset"
    )
    assert isinstance(result, ScoreResult)
    assert hasattr(result, "score")
    assert hasattr(result, "passed")


@pytest.mark.unit
@pytest.mark.asyncio
async def test_g_eval_relevance_scorer(mock_llm):
    from novaeval.scorers.g_eval import CommonGEvalCriteria, GEvalScorer

    scorer = GEvalScorer(mock_llm, criteria=CommonGEvalCriteria.relevance())
    result = await scorer.evaluate(
        "What is ML?", "Machine learning is AI", context="ML is AI subset"
    )
    assert isinstance(result, ScoreResult)
    assert hasattr(result, "score")
    assert hasattr(result, "passed")


@pytest.mark.unit
def test_g_eval_custom_criteria(mock_llm):
    from novaeval.scorers.g_eval import GEvalCriteria, GEvalScorer

    custom_criteria = GEvalCriteria(
        name="custom",
        criteria="Custom evaluation criteria",
        description="A custom evaluation criteria for testing",
        steps=["Step 1: Evaluate the answer", "Step 2: Rate from 1-5"],
        score_mapping={1: "Poor", 2: "Fair", 3: "Good", 4: "Very Good", 5: "Excellent"},
    )
    scorer = GEvalScorer(mock_llm, criteria=custom_criteria)
    result = scorer.score(
        "Machine learning is AI", "What is ML?", {"context": "ML is AI subset"}
    )
    assert isinstance(result, ScoreResult)
    assert hasattr(result, "score")
    assert hasattr(result, "passed")


@pytest.mark.unit
@pytest.mark.asyncio
async def test_g_eval_multiple_iterations(mock_llm):
    from novaeval.scorers.g_eval import CommonGEvalCriteria, GEvalScorer

    scorer = GEvalScorer(
        mock_llm, criteria=CommonGEvalCriteria.helpfulness(), iterations=3
    )
    result = await scorer.evaluate(
        "What is ML?", "Machine learning is AI", context="ML is AI subset"
    )
    assert isinstance(result, ScoreResult)
    assert hasattr(result, "score")
    assert hasattr(result, "passed")


# Note: RAGAssessmentEngine integration tests are covered in panel judge tests
# to avoid duplication and reduce maintenance overhead.

if __name__ == "__main__":
    pytest.main([__file__])
