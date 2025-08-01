import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import pytest
from unittest.mock import Mock, patch
from novaeval.scorers.rag_assessment import RAGAssessmentEngine, AgentData
from novaeval.scorers.base import ScoreResult

# Import shared test utilities
from test_utils import mock_llm, sample_agent_data, sample_agent_data_list

def test_rag_assessment_engine_initialization(mock_llm):
    engine = RAGAssessmentEngine(mock_llm, threshold=0.7)
    assert engine.model == mock_llm
    assert engine.threshold == 0.7
    
    # Check that enabled scorers are initialized
    enabled_scorers = engine.get_enabled_scorers()
    assert len(enabled_scorers) > 0, "At least some scorers should be enabled"
    
    # Check that aggregate scorer is initialized
    assert hasattr(engine, 'aggregate_scorer')
    
    # Check that configuration is properly set
    assert engine.config is not None
    assert len(engine.config.scorers) > 0

@pytest.mark.asyncio
async def test_evaluate_single(mock_llm, sample_agent_data):
    engine = RAGAssessmentEngine(mock_llm, threshold=0.7)
    result = await engine.evaluate(sample_agent_data)
    
    assert isinstance(result, dict)
    
    # Check that enabled metrics are present
    enabled_scorers = engine.get_enabled_scorers()
    for scorer_name in enabled_scorers:
        assert scorer_name in result, f"Enabled scorer {scorer_name} not found in results"
        assert isinstance(result[scorer_name], ScoreResult), f"Metric {scorer_name} should be ScoreResult"
    
    # Check that aggregate score is present
    assert "aggregate" in result
    assert isinstance(result["aggregate"], ScoreResult)

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