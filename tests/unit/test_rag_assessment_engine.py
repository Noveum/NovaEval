import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import pytest
from unittest.mock import Mock, patch
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
            "Evaluate the faithfulness": "Rating: 4",
            "Evaluate the answer relevancy": "Rating: 4",
            "Evaluate the contextual precision": "Rating: 4",
            "Evaluate the contextual recall": "Rating: 4",
            "Evaluate the helpfulness": "Rating: 4",
            "Evaluate the correctness": "Rating: 4",
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
def sample_agent_data():
    return AgentData(
        ground_truth="What is machine learning?",
        agent_response="Machine learning is a subset of artificial intelligence that enables computers to learn from data.",
        retrieved_context="Machine learning is a subset of artificial intelligence. It involves training algorithms on data to make predictions or decisions."
    )

@pytest.fixture
def sample_agent_data_list():
    return [
        AgentData(
            ground_truth="What is machine learning?",
            agent_response="Machine learning is a subset of artificial intelligence that enables computers to learn from data.",
            retrieved_context="Machine learning is a subset of artificial intelligence. It involves training algorithms on data to make predictions or decisions."
        ),
        AgentData(
            ground_truth="What is deep learning?",
            agent_response="Deep learning is a subset of machine learning that uses neural networks with multiple layers.",
            retrieved_context="Deep learning is a subset of machine learning. It uses artificial neural networks with multiple layers to process data."
        )
    ]

def test_rag_assessment_engine_initialization(mock_llm):
    engine = RAGAssessmentEngine(mock_llm, threshold=0.7)
    assert engine.model == mock_llm
    assert engine.threshold == 0.7
    
    # Check that all scorers are initialized
    assert hasattr(engine, 'answer_relevancy_scorer')
    assert hasattr(engine, 'faithfulness_scorer')
    assert hasattr(engine, 'contextual_precision_scorer')
    assert hasattr(engine, 'contextual_recall_scorer')
    assert hasattr(engine, 'ragas_scorer')
    assert hasattr(engine, 'contextual_precision_pp')
    assert hasattr(engine, 'contextual_recall_pp')
    assert hasattr(engine, 'retrieval_ranking_scorer')
    assert hasattr(engine, 'semantic_similarity_scorer')
    assert hasattr(engine, 'retrieval_diversity_scorer')
    assert hasattr(engine, 'helpfulness_scorer')
    assert hasattr(engine, 'correctness_scorer')
    assert hasattr(engine, 'context_faithfulness_pp')
    assert hasattr(engine, 'context_groundedness_scorer')
    assert hasattr(engine, 'context_completeness_scorer')
    assert hasattr(engine, 'context_consistency_scorer')
    assert hasattr(engine, 'rag_answer_quality_scorer')
    assert hasattr(engine, 'hallucination_detection_scorer')
    assert hasattr(engine, 'source_attribution_scorer')
    assert hasattr(engine, 'factual_accuracy_scorer')
    assert hasattr(engine, 'claim_verification_scorer')
    assert hasattr(engine, 'answer_completeness_scorer')
    assert hasattr(engine, 'question_answer_alignment_scorer')
    assert hasattr(engine, 'information_density_scorer')
    assert hasattr(engine, 'clarity_coherence_scorer')
    assert hasattr(engine, 'cross_context_synthesis_scorer')
    assert hasattr(engine, 'conflict_resolution_scorer')
    assert hasattr(engine, 'context_prioritization_scorer')
    assert hasattr(engine, 'citation_quality_scorer')
    assert hasattr(engine, 'technical_accuracy_scorer')
    assert hasattr(engine, 'bias_detection_scorer')
    assert hasattr(engine, 'tone_consistency_scorer')
    assert hasattr(engine, 'terminology_consistency_scorer')
    assert hasattr(engine, 'aggregate_scorer')

@pytest.mark.asyncio
async def test_evaluate_single(mock_llm, sample_agent_data):
    engine = RAGAssessmentEngine(mock_llm, threshold=0.7)
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
async def test_evaluate_batch(mock_llm, sample_agent_data_list):
    engine = RAGAssessmentEngine(mock_llm, threshold=0.7)
    results = await engine.evaluate_batch(sample_agent_data_list)
    
    assert isinstance(results, list)
    assert len(results) == 2
    
    for result in results:
        assert isinstance(result, dict)
        assert "answer_relevancy" in result
        assert "faithfulness" in result
        assert "aggregate" in result

@pytest.mark.asyncio
async def test_evaluate_no_context(mock_llm):
    engine = RAGAssessmentEngine(mock_llm, threshold=0.7)
    agent_data = AgentData(
        ground_truth="What is machine learning?",
        agent_response="Machine learning is a subset of artificial intelligence.",
        retrieved_context=None
    )
    
    result = await engine.evaluate(agent_data)
    assert isinstance(result, dict)
    assert "error" in result
    assert "No retrieved context available for evaluation" in result["error"]

@pytest.mark.asyncio
async def test_g_eval_integration(mock_llm, sample_agent_data):
    engine = RAGAssessmentEngine(mock_llm, threshold=0.7)
    result = await engine.evaluate(sample_agent_data)
    
    # Check G-Eval scorers are properly integrated
    assert "helpfulness" in result
    assert "correctness" in result
    assert isinstance(result["helpfulness"], ScoreResult)
    assert isinstance(result["correctness"], ScoreResult)

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

def test_aggregate_scorer_weights(mock_llm):
    engine = RAGAssessmentEngine(mock_llm, threshold=0.7)
    
    # Check that aggregate scorer includes all important scorers
    aggregate_scorers = engine.aggregate_scorer.scorers
    aggregate_weights = engine.aggregate_scorer.weights
    
    important_scorers = [
        "answer_relevancy", "faithfulness", "contextual_precision", "contextual_recall",
        "semantic_similarity", "retrieval_diversity", "context_faithfulness_pp", "rag_answer_quality",
        "hallucination_detection", "answer_completeness", "technical_accuracy", "bias_detection"
    ]
    
    for scorer_name in important_scorers:
        assert scorer_name in aggregate_scorers, f"Scorer {scorer_name} not in aggregate"
        assert scorer_name in aggregate_weights, f"Weight for {scorer_name} not in aggregate"
    
    # Check that weights sum to approximately 1.0
    total_weight = sum(aggregate_weights.values())
    assert abs(total_weight - 1.0) < 0.01, f"Weights should sum to 1.0, got {total_weight}"

@pytest.mark.asyncio
async def test_error_handling(mock_llm):
    engine = RAGAssessmentEngine(mock_llm, threshold=0.7)
    
    # Test with malformed data
    malformed_data = AgentData(
        ground_truth=None,
        agent_response=None,
        retrieved_context="Some context"
    )
    
    result = await engine.evaluate(malformed_data)
    assert isinstance(result, dict)
    # Should still return results even with None values
    assert "answer_relevancy" in result
    assert "faithfulness" in result

@pytest.mark.asyncio
async def test_threshold_configuration(mock_llm):
    # Test with different thresholds
    engine_low = RAGAssessmentEngine(mock_llm, threshold=0.3)
    engine_high = RAGAssessmentEngine(mock_llm, threshold=0.9)
    
    agent_data = AgentData(
        ground_truth="What is machine learning?",
        agent_response="Machine learning is a subset of artificial intelligence.",
        retrieved_context="Machine learning is a subset of artificial intelligence."
    )
    
    result_low = await engine_low.evaluate(agent_data)
    result_high = await engine_high.evaluate(agent_data)
    
    # Both should return results, but with different pass/fail outcomes
    assert isinstance(result_low, dict)
    assert isinstance(result_high, dict)
    assert "answer_relevancy" in result_low
    assert "answer_relevancy" in result_high

@pytest.mark.asyncio
async def test_all_scorers_count(mock_llm):
    engine = RAGAssessmentEngine(mock_llm, threshold=0.7)
    agent_data = AgentData(
        ground_truth="What is machine learning?",
        agent_response="Machine learning is a subset of artificial intelligence.",
        retrieved_context="Machine learning is a subset of artificial intelligence."
    )
    
    result = await engine.evaluate(agent_data)
    
    # Count all scorers (excluding aggregate which is a composite)
    scorer_metrics = [k for k in result.keys() if k != "aggregate"]
    assert len(scorer_metrics) >= 32, f"Expected at least 32 individual scorers, got {len(scorer_metrics)}"
    
    print(f"Total individual scorers tested: {len(scorer_metrics)}")
    print("All scorers:")
    for metric in sorted(scorer_metrics):
        print(f"  - {metric}")

if __name__ == "__main__":
    pytest.main([__file__]) 