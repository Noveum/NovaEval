import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import pytest
from unittest.mock import Mock, patch
from novaeval.scorers.basic_rag_scorers import (
    ContextualPrecisionScorerPP, ContextualRecallScorerPP, ContextualF1Scorer,
    RetrievalRankingScorer, SemanticSimilarityScorer, RetrievalDiversityScorer,
    AggregateRetrievalScorer
)
from novaeval.scorers.advanced_generation_scorers import (
    ContextFaithfulnessScorerPP, ContextGroundednessScorer,
    ContextCompletenessScorer, ContextConsistencyScorer, RAGAnswerQualityScorer,
    HallucinationDetectionScorer, SourceAttributionScorer, FactualAccuracyScorer,
    ClaimVerificationScorer, AnswerCompletenessScorer, QuestionAnswerAlignmentScorer,
    InformationDensityScorer, ClarityAndCoherenceScorer, CrossContextSynthesisScorer,
    ConflictResolutionScorer, ContextPrioritizationScorer, CitationQualityScorer,
    TechnicalAccuracyScorer, BiasDetectionScorer, ToneConsistencyScorer,
    TerminologyConsistencyScorer
)
from novaeval.scorers.rag_assessment import RAGAssessmentEngine, AgentData
from novaeval.scorers.base import ScoreResult

class MockLLM:
    def __init__(self):
        self.responses = {
            "Rate relevance 1-5:": "Rating: 4",
            "Extract key facts from:": "1. Fact 1\n2. Fact 2\n3. Fact 3",
            "Extract all factual claims from this answer": "1. Claim 1\n2. Claim 2",
            "Can this claim be verified from the provided context": "Rating: 4",
            "Evaluate how well this answer is grounded": "Rating: 4",
            "Evaluate if the provided context is complete": "Rating: 4",
            "Evaluate the overall quality of this RAG-generated answer": "Rating: 4",
            "Detect any hallucinations": "Rating: 2",
            "Evaluate the quality of source attribution": "Rating: 4",
            "Verify the factual accuracy": "Rating: 4",
            "Extract all specific claims from this answer": "1. Specific claim 1\n2. Specific claim 2",
            "Can this specific claim be verified": "Rating: 4",
            "Evaluate the completeness of this answer": "Rating: 4",
            "Evaluate how well this answer directly addresses": "Rating: 4",
            "Evaluate the information density of this answer": "Rating: 4",
            "Evaluate the clarity and coherence of this answer": "Rating: 4",
            "Evaluate how well this answer synthesizes information": "Rating: 4",
            "Evaluate how well this answer handles potential conflicts": "Rating: 4",
            "Evaluate how well this answer prioritizes": "Rating: 4",
            "Evaluate the quality of citations": "Rating: 4",
            "Evaluate the technical accuracy": "Rating: 4",
            "Detect any bias in this answer": "Rating: 2",
            "Evaluate the appropriateness and consistency of tone": "Rating: 4",
            "Evaluate the consistency of terminology": "Rating: 4",
            "Evaluate if this answer is consistent with this specific context chunk": "Rating: 4",
        }
    
    def __call__(self, prompt):
        for key, response in self.responses.items():
            if key in prompt:
                return response
        return "Rating: 3"  # Default response

@pytest.fixture
def mock_llm():
    return MockLLM()

@pytest.fixture
def sample_context():
    return "This is a sample context about machine learning. Machine learning is a subset of artificial intelligence."

@pytest.fixture
def sample_agent_data():
    return AgentData(
        ground_truth="What is machine learning?",
        agent_response="Machine learning is a subset of artificial intelligence that enables computers to learn from data.",
        retrieved_context="Machine learning is a subset of artificial intelligence. It involves training algorithms on data to make predictions or decisions."
    )

# Test Context-Aware Generation Scorers
@pytest.mark.asyncio
async def test_context_faithfulness_scorer_pp(mock_llm):
    scorer = ContextFaithfulnessScorerPP(mock_llm, threshold=0.8)
    result = await scorer.evaluate("What is ML?", "Machine learning is AI", context="ML is AI subset")
    assert isinstance(result, ScoreResult)
    assert hasattr(result, 'score')
    assert hasattr(result, 'passed')

@pytest.mark.asyncio
async def test_context_groundedness_scorer(mock_llm):
    scorer = ContextGroundednessScorer(mock_llm, threshold=0.7)
    result = await scorer.evaluate("What is ML?", "Machine learning is AI", context="ML is AI subset")
    assert isinstance(result, ScoreResult)
    assert hasattr(result, 'score')
    assert hasattr(result, 'passed')

@pytest.mark.asyncio
async def test_context_completeness_scorer(mock_llm):
    scorer = ContextCompletenessScorer(mock_llm, threshold=0.6)
    result = await scorer.evaluate("What is ML?", "Machine learning is AI", context="ML is AI subset")
    assert isinstance(result, ScoreResult)
    assert hasattr(result, 'score')
    assert hasattr(result, 'passed')

@pytest.mark.asyncio
async def test_context_consistency_scorer(mock_llm):
    scorer = ContextConsistencyScorer(mock_llm, threshold=0.7)
    result = await scorer.evaluate("What is ML?", "Machine learning is AI", context="ML is AI subset\n\nAI includes ML")
    assert isinstance(result, ScoreResult)
    assert hasattr(result, 'score')
    assert hasattr(result, 'passed')

# Test Answer Quality Enhancement Scorers
@pytest.mark.asyncio
async def test_rag_answer_quality_scorer(mock_llm):
    scorer = RAGAnswerQualityScorer(mock_llm, threshold=0.7)
    result = await scorer.evaluate("What is ML?", "Machine learning is AI", context="ML is AI subset")
    assert isinstance(result, ScoreResult)
    assert hasattr(result, 'score')
    assert hasattr(result, 'passed')

# Test Hallucination Detection Scorers
@pytest.mark.asyncio
async def test_hallucination_detection_scorer(mock_llm):
    scorer = HallucinationDetectionScorer(mock_llm, threshold=0.8)
    result = await scorer.evaluate("What is ML?", "Machine learning is AI", context="ML is AI subset")
    assert isinstance(result, ScoreResult)
    assert hasattr(result, 'score')
    assert hasattr(result, 'passed')

@pytest.mark.asyncio
async def test_source_attribution_scorer(mock_llm):
    scorer = SourceAttributionScorer(mock_llm, threshold=0.6)
    result = await scorer.evaluate("What is ML?", "Machine learning is AI", context="ML is AI subset")
    assert isinstance(result, ScoreResult)
    assert hasattr(result, 'score')
    assert hasattr(result, 'passed')

@pytest.mark.asyncio
async def test_factual_accuracy_scorer(mock_llm):
    scorer = FactualAccuracyScorer(mock_llm, threshold=0.8)
    result = await scorer.evaluate("What is ML?", "Machine learning is AI", context="ML is AI subset")
    assert isinstance(result, ScoreResult)
    assert hasattr(result, 'score')
    assert hasattr(result, 'passed')

@pytest.mark.asyncio
async def test_claim_verification_scorer(mock_llm):
    scorer = ClaimVerificationScorer(mock_llm, threshold=0.7)
    result = await scorer.evaluate("What is ML?", "Machine learning is AI", context="ML is AI subset")
    assert isinstance(result, ScoreResult)
    assert hasattr(result, 'score')
    assert hasattr(result, 'passed')

# Test Answer Completeness and Relevance Scorers
@pytest.mark.asyncio
async def test_answer_completeness_scorer(mock_llm):
    scorer = AnswerCompletenessScorer(mock_llm, threshold=0.7)
    result = await scorer.evaluate("What is ML?", "Machine learning is AI", context="ML is AI subset")
    assert isinstance(result, ScoreResult)
    assert hasattr(result, 'score')
    assert hasattr(result, 'passed')

@pytest.mark.asyncio
async def test_question_answer_alignment_scorer(mock_llm):
    scorer = QuestionAnswerAlignmentScorer(mock_llm, threshold=0.7)
    result = await scorer.evaluate("What is ML?", "Machine learning is AI", context="ML is AI subset")
    assert isinstance(result, ScoreResult)
    assert hasattr(result, 'score')
    assert hasattr(result, 'passed')

@pytest.mark.asyncio
async def test_information_density_scorer(mock_llm):
    scorer = InformationDensityScorer(mock_llm, threshold=0.6)
    result = await scorer.evaluate("What is ML?", "Machine learning is AI", context="ML is AI subset")
    assert isinstance(result, ScoreResult)
    assert hasattr(result, 'score')
    assert hasattr(result, 'passed')

@pytest.mark.asyncio
async def test_clarity_coherence_scorer(mock_llm):
    scorer = ClarityAndCoherenceScorer(mock_llm, threshold=0.7)
    result = await scorer.evaluate("What is ML?", "Machine learning is AI", context="ML is AI subset")
    assert isinstance(result, ScoreResult)
    assert hasattr(result, 'score')
    assert hasattr(result, 'passed')

# Test Multi-Context Integration Scorers
@pytest.mark.asyncio
async def test_cross_context_synthesis_scorer(mock_llm):
    scorer = CrossContextSynthesisScorer(mock_llm, threshold=0.7)
    result = await scorer.evaluate("What is ML?", "Machine learning is AI", context="ML is AI subset\n\nAI includes ML")
    assert isinstance(result, ScoreResult)
    assert hasattr(result, 'score')
    assert hasattr(result, 'passed')

@pytest.mark.asyncio
async def test_conflict_resolution_scorer(mock_llm):
    scorer = ConflictResolutionScorer(mock_llm, threshold=0.7)
    result = await scorer.evaluate("What is ML?", "Machine learning is AI", context="ML is AI subset\n\nAI includes ML")
    assert isinstance(result, ScoreResult)
    assert hasattr(result, 'score')
    assert hasattr(result, 'passed')

@pytest.mark.asyncio
async def test_context_prioritization_scorer(mock_llm):
    scorer = ContextPrioritizationScorer(mock_llm, threshold=0.6)
    result = await scorer.evaluate("What is ML?", "Machine learning is AI", context="ML is AI subset")
    assert isinstance(result, ScoreResult)
    assert hasattr(result, 'score')
    assert hasattr(result, 'passed')

@pytest.mark.asyncio
async def test_citation_quality_scorer(mock_llm):
    scorer = CitationQualityScorer(mock_llm, threshold=0.6)
    result = await scorer.evaluate("What is ML?", "Machine learning is AI", context="ML is AI subset")
    assert isinstance(result, ScoreResult)
    assert hasattr(result, 'score')
    assert hasattr(result, 'passed')

# Test Domain-Specific Evaluation Scorers
@pytest.mark.asyncio
async def test_technical_accuracy_scorer(mock_llm):
    scorer = TechnicalAccuracyScorer(mock_llm, threshold=0.8)
    result = await scorer.evaluate("What is ML?", "Machine learning is AI", context="ML is AI subset")
    assert isinstance(result, ScoreResult)
    assert hasattr(result, 'score')
    assert hasattr(result, 'passed')

@pytest.mark.asyncio
async def test_bias_detection_scorer(mock_llm):
    scorer = BiasDetectionScorer(mock_llm, threshold=0.8)
    result = await scorer.evaluate("What is ML?", "Machine learning is AI", context="ML is AI subset")
    assert isinstance(result, ScoreResult)
    assert hasattr(result, 'score')
    assert hasattr(result, 'passed')

@pytest.mark.asyncio
async def test_tone_consistency_scorer(mock_llm):
    scorer = ToneConsistencyScorer(mock_llm, threshold=0.7)
    result = await scorer.evaluate("What is ML?", "Machine learning is AI", context="ML is AI subset")
    assert isinstance(result, ScoreResult)
    assert hasattr(result, 'score')
    assert hasattr(result, 'passed')

@pytest.mark.asyncio
async def test_terminology_consistency_scorer(mock_llm):
    scorer = TerminologyConsistencyScorer(mock_llm, threshold=0.7)
    result = await scorer.evaluate("What is ML?", "Machine learning is AI", context="ML is AI subset")
    assert isinstance(result, ScoreResult)
    assert hasattr(result, 'score')
    assert hasattr(result, 'passed')

# Test existing scorers (keeping original tests)
@pytest.mark.asyncio
async def test_contextual_precision_scorer_pp(mock_llm):
    scorer = ContextualPrecisionScorerPP(mock_llm, threshold=0.7)
    result = await scorer.evaluate("What is ML?", "Machine learning is AI", context="ML is AI subset")
    assert isinstance(result, ScoreResult)
    assert hasattr(result, 'score')
    assert hasattr(result, 'passed')

@pytest.mark.asyncio
async def test_contextual_recall_scorer_pp(mock_llm):
    scorer = ContextualRecallScorerPP(mock_llm, threshold=0.7)
    result = await scorer.evaluate("What is ML?", "Machine learning is AI", context="ML is AI subset")
    assert isinstance(result, ScoreResult)
    assert hasattr(result, 'score')
    assert hasattr(result, 'passed')

def test_contextual_f1_scorer(mock_llm):
    precision_scorer = ContextualPrecisionScorerPP(mock_llm, threshold=0.7)
    recall_scorer = ContextualRecallScorerPP(mock_llm, threshold=0.7)
    scorer = ContextualF1Scorer(precision_scorer, recall_scorer, threshold=0.5)
    result = scorer.score("Machine learning is AI", "What is ML?", {"context": "ML is AI subset"})
    assert isinstance(result, ScoreResult)
    assert hasattr(result, 'score')
    assert hasattr(result, 'passed')

def test_retrieval_ranking_scorer():
    scorer = RetrievalRankingScorer(threshold=0.5)
    result = scorer.score("Machine learning is AI", "What is ML?", {"context": "ML is AI subset"})
    assert isinstance(result, ScoreResult)
    assert hasattr(result, 'score')
    assert hasattr(result, 'passed')

def test_semantic_similarity_scorer():
    scorer = SemanticSimilarityScorer(threshold=0.7)
    result = scorer.score("Machine learning is AI", "What is ML?", {"context": "ML is AI subset"})
    assert isinstance(result, ScoreResult)
    assert hasattr(result, 'score')
    assert hasattr(result, 'passed')

def test_retrieval_diversity_scorer():
    scorer = RetrievalDiversityScorer()
    result = scorer.score("Machine learning is AI", "What is ML?", {"context": "ML is AI subset\n\nAI includes ML"})
    assert isinstance(result, ScoreResult)
    assert hasattr(result, 'score')
    assert hasattr(result, 'passed')

def test_aggregate_retrieval_scorer(mock_llm):
    scorers = {
        "precision": ContextualPrecisionScorerPP(mock_llm, threshold=0.7),
        "recall": ContextualRecallScorerPP(mock_llm, threshold=0.7)
    }
    weights = {"precision": 0.5, "recall": 0.5}
    scorer = AggregateRetrievalScorer(scorers, weights)
    result = scorer.score("Machine learning is AI", "What is ML?", {"context": "ML is AI subset"})
    assert isinstance(result, ScoreResult)
    assert hasattr(result, 'score')
    assert hasattr(result, 'passed')

# Test G-Eval scorers
@pytest.mark.asyncio
async def test_g_eval_helpfulness_scorer(mock_llm):
    from novaeval.scorers.g_eval import GEvalScorer, CommonGEvalCriteria
    scorer = GEvalScorer(mock_llm, criteria=CommonGEvalCriteria.helpfulness())
    result = await scorer.evaluate("What is ML?", "Machine learning is AI", context="ML is AI subset")
    assert isinstance(result, ScoreResult)
    assert hasattr(result, 'score')
    assert hasattr(result, 'passed')

@pytest.mark.asyncio
async def test_g_eval_correctness_scorer(mock_llm):
    from novaeval.scorers.g_eval import GEvalScorer, CommonGEvalCriteria
    scorer = GEvalScorer(mock_llm, criteria=CommonGEvalCriteria.correctness())
    result = await scorer.evaluate("What is ML?", "Machine learning is AI", context="ML is AI subset")
    assert isinstance(result, ScoreResult)
    assert hasattr(result, 'score')
    assert hasattr(result, 'passed')

@pytest.mark.asyncio
async def test_g_eval_coherence_scorer(mock_llm):
    from novaeval.scorers.g_eval import GEvalScorer, CommonGEvalCriteria
    scorer = GEvalScorer(mock_llm, criteria=CommonGEvalCriteria.coherence())
    result = await scorer.evaluate("What is ML?", "Machine learning is AI", context="ML is AI subset")
    assert isinstance(result, ScoreResult)
    assert hasattr(result, 'score')
    assert hasattr(result, 'passed')

@pytest.mark.asyncio
async def test_g_eval_relevance_scorer(mock_llm):
    from novaeval.scorers.g_eval import GEvalScorer, CommonGEvalCriteria
    scorer = GEvalScorer(mock_llm, criteria=CommonGEvalCriteria.relevance())
    result = await scorer.evaluate("What is ML?", "Machine learning is AI", context="ML is AI subset")
    assert isinstance(result, ScoreResult)
    assert hasattr(result, 'score')
    assert hasattr(result, 'passed')

def test_g_eval_custom_criteria(mock_llm):
    from novaeval.scorers.g_eval import GEvalScorer, GEvalCriteria
    custom_criteria = GEvalCriteria(
        name="custom",
        criteria="Custom evaluation criteria",
        description="A custom evaluation criteria for testing",
        steps=["Step 1: Evaluate the answer", "Step 2: Rate from 1-5"],
        score_mapping={1: "Poor", 2: "Fair", 3: "Good", 4: "Very Good", 5: "Excellent"}
    )
    scorer = GEvalScorer(mock_llm, criteria=custom_criteria)
    result = scorer.score("Machine learning is AI", "What is ML?", {"context": "ML is AI subset"})
    assert isinstance(result, ScoreResult)
    assert hasattr(result, 'score')
    assert hasattr(result, 'passed')

@pytest.mark.asyncio
async def test_g_eval_multiple_iterations(mock_llm):
    from novaeval.scorers.g_eval import GEvalScorer, CommonGEvalCriteria
    scorer = GEvalScorer(mock_llm, criteria=CommonGEvalCriteria.helpfulness(), iterations=3)
    result = await scorer.evaluate("What is ML?", "Machine learning is AI", context="ML is AI subset")
    assert isinstance(result, ScoreResult)
    assert hasattr(result, 'score')
    assert hasattr(result, 'passed')

# Test integration with RAGAssessmentEngine
@pytest.mark.asyncio
async def test_integration_with_rag_assessment_engine(mock_llm, sample_agent_data):
    engine = RAGAssessmentEngine(mock_llm, threshold=0.7)
    
    # Test single evaluation
    result = await engine.evaluate(sample_agent_data)
    assert isinstance(result, dict)
    
    # Check that all expected metrics are present
    expected_metrics = [
        "answer_relevancy", "faithfulness", "contextual_precision", "contextual_recall",
        "contextual_precision_pp", "contextual_recall_pp", "contextual_f1", "retrieval_ranking",
        "semantic_similarity", "retrieval_diversity", "helpfulness", "correctness",
        "context_faithfulness_pp", "context_groundedness", "context_completeness", "context_consistency",
        "rag_answer_quality", "hallucination_detection", "source_attribution", "factual_accuracy",
        "claim_verification", "answer_completeness", "question_answer_alignment", "information_density",
        "clarity_coherence", "cross_context_synthesis", "conflict_resolution", "context_prioritization",
        "citation_quality", "technical_accuracy", "bias_detection", "tone_consistency",
        "terminology_consistency", "aggregate"
    ]
    
    for metric in expected_metrics:
        assert metric in result, f"Metric {metric} not found in results"
        assert isinstance(result[metric], ScoreResult), f"Metric {metric} should be ScoreResult"

@pytest.mark.asyncio
async def test_comprehensive_evaluation(mock_llm, sample_agent_data):
    engine = RAGAssessmentEngine(mock_llm, threshold=0.7)
    result = await engine.evaluate(sample_agent_data)
    
    # Verify all metric categories are present
    categories = {
        "Basic RAG": ["answer_relevancy", "faithfulness", "contextual_precision", "contextual_recall"],
        "Advanced Retrieval": ["contextual_precision_pp", "contextual_recall_pp", "semantic_similarity", "retrieval_diversity"],
        "G-Eval": ["helpfulness", "correctness"],
        "Context-Aware Generation": ["context_faithfulness_pp", "context_groundedness", "context_completeness", "context_consistency"],
        "Answer Quality": ["rag_answer_quality"],
        "Hallucination Detection": ["hallucination_detection", "source_attribution", "factual_accuracy", "claim_verification"],
        "Answer Completeness": ["answer_completeness", "question_answer_alignment", "information_density", "clarity_coherence"],
        "Multi-Context": ["cross_context_synthesis", "conflict_resolution", "context_prioritization", "citation_quality"],
        "Domain-Specific": ["technical_accuracy", "bias_detection", "tone_consistency", "terminology_consistency"]
    }
    
    for category, metrics in categories.items():
        for metric in metrics:
            assert metric in result, f"Category {category} metric {metric} not found"
            print(f"Category {category} - {metric}: PASSED")

if __name__ == "__main__":
    pytest.main([__file__]) 