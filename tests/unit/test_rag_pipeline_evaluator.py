"""
Minimal tests for RAG Pipeline Evaluator to improve code coverage.
"""

from unittest.mock import Mock

import pytest

from novaeval.scorers.rag_pipeline_evaluator import (
    AgentData,
    ErrorPropagationScorer,
    LatencyAnalysisScorer,
    PipelineCoordinationScorer,
    QueryProcessingEvaluator,
    RAGContext,
    RAGPipelineEvaluator,
    RAGSample,
    RerankingEvaluator,
    ResourceUtilizationScorer,
    RetrievalStageEvaluator,
    StageMetrics,
)
from tests.unit.test_utils import MockLLM


@pytest.mark.unit
def test_rag_pipeline_evaluator_initialization():
    """Test basic initialization of RAGPipelineEvaluator."""
    mock_llm = Mock()
    evaluator = RAGPipelineEvaluator(llm=mock_llm)

    assert evaluator is not None
    assert hasattr(evaluator, "evaluate_rag_pipeline")


@pytest.mark.unit
def test_rag_pipeline_evaluator_with_minimal_data():
    """Test RAGPipelineEvaluator with minimal data to exercise type conversion paths."""
    mock_llm = MockLLM()
    evaluator = RAGPipelineEvaluator(llm=mock_llm)

    # Create minimal test data
    retrieved_contexts = [
        RAGContext(
            content="Machine learning is AI", source="test_source", relevance_score=0.8
        )
    ]

    generated_answer = "Machine learning is a branch of artificial intelligence"

    rag_sample = RAGSample(
        query="What is machine learning?",
        ground_truth="Machine learning is a subset of AI",
        generated_answer=generated_answer,
        retrieved_contexts=retrieved_contexts,
    )

    # Test the evaluation - this should exercise the type conversion paths
    result = evaluator.evaluate_rag_pipeline(
        rag_sample=rag_sample,
        retrieved_contexts=retrieved_contexts,
        generated_answer=generated_answer,
    )

    # Basic validation that the result structure is correct
    assert hasattr(result, "overall_score")
    assert hasattr(result, "detailed_scores")
    assert isinstance(result.overall_score, float)
    assert isinstance(result.detailed_scores, dict)

    # Verify that any numeric values in detailed_scores are Python native types
    for _key, value in result.detailed_scores.items():
        if isinstance(value, dict) and "score" in value:
            assert isinstance(value["score"], (int, float))


@pytest.mark.unit
def test_rag_evaluation_result_type_safety():
    """Test that RAGEvaluationResult properly handles type conversions."""
    from novaeval.scorers.rag_pipeline_evaluator import RAGEvaluationResult

    # Test with mixed types to ensure proper conversion
    result = RAGEvaluationResult(
        overall_score=0.85,
        stage_metrics={},
        retrieval_score=0.8,
        generation_score=0.9,
        pipeline_coordination_score=0.85,
        latency_analysis={},
        resource_utilization={},
        error_propagation_score=0.0,
        detailed_scores={"test_score": {"score": 0.5}},
        recommendations=["Test recommendation"],
    )
    assert isinstance(result.overall_score, float)
    assert isinstance(result.detailed_scores, dict)


def test_evaluate_query_clarity_methods():
    """Test the query evaluation methods for coverage."""
    evaluator = QueryProcessingEvaluator(llm=MockLLM())

    # Test _evaluate_query_clarity
    assert evaluator._evaluate_query_clarity("short") == 0.3  # Too short
    assert evaluator._evaluate_query_clarity("moderate length query") == 0.6  # Moderate
    assert (
        evaluator._evaluate_query_clarity(
            "very detailed query with many words that exceeds twenty words count and this is a very long sentence that should definitely have more than twenty words in total"
        )
        == 0.8
    )  # Detailed (25+ words)

    # Test _evaluate_intent_detection
    assert (
        evaluator._evaluate_intent_detection("what is machine learning") == 0.9
    )  # Has question word and specific terms
    assert (
        evaluator._evaluate_intent_detection("machine learning algorithm") == 0.6
    )  # Has specific terms only
    assert (
        evaluator._evaluate_intent_detection("what is it") == 0.6
    )  # Has question word only
    assert evaluator._evaluate_intent_detection("short") == 0.3  # Neither

    # Test _evaluate_preprocessing
    assert evaluator._evaluate_preprocessing("clean query") == 0.8  # Clean
    assert evaluator._evaluate_preprocessing("  messy  query  ") == 0.5  # Messy
    assert evaluator._evaluate_preprocessing("") == 0.5  # Empty

    # Test _evaluate_specificity
    assert evaluator._evaluate_specificity("specific query") == 0.7  # Has specific term
    assert evaluator._evaluate_specificity("query with 123") == 0.65  # Has numbers
    assert (
        evaluator._evaluate_specificity("query about Paris") == 0.65
    )  # Has proper noun
    assert evaluator._evaluate_specificity("basic query") == 0.5  # Base score only

    # Test _evaluate_complexity
    assert evaluator._evaluate_complexity("") == 0.0  # Empty
    assert (
        evaluator._evaluate_complexity("simple words") == 0.7
    )  # Medium complexity (unique/total > 0.6, avg > 5)
    assert evaluator._evaluate_complexity("basic query") == 0.5  # Low complexity
    assert (
        evaluator._evaluate_complexity(
            "extraordinarily sophisticated vocabulary with exceptionally diverse terminology"
        )
        == 0.9
    )  # High complexity

    # Test _evaluate_ambiguity
    assert evaluator._evaluate_ambiguity("clear query") == 1.0  # No ambiguous words
    assert (
        evaluator._evaluate_ambiguity("it is this thing") == 0.0
    )  # Many ambiguous words
    assert (
        abs(evaluator._evaluate_ambiguity("what is it") - 0.67) < 0.01
    )  # Some ambiguous words


def test_enhanced_query_processing_evaluation_exception_handling():
    """Test enhanced query processing evaluation with exception handling."""
    evaluator = QueryProcessingEvaluator(llm=MockLLM())

    # Mock the evaluation methods to raise exceptions
    evaluator._evaluate_query_clarity = Mock(side_effect=Exception("Clarity error"))

    result = evaluator.score("test query", "ground truth")

    assert result["score"] == 0.0
    assert "Error in enhanced query processing evaluation" in result["reasoning"]
    assert "Clarity error" in result["reasoning"]
    assert "error" in result["details"]


def test_enhanced_query_processing_evaluation_success():
    """Test enhanced query processing evaluation success case."""
    evaluator = QueryProcessingEvaluator(llm=MockLLM())

    # Mock the evaluation methods to return predictable results
    evaluator._evaluate_query_clarity = Mock(return_value=0.8)
    evaluator._evaluate_intent_detection = Mock(return_value=0.9)
    evaluator._evaluate_preprocessing = Mock(return_value=0.7)
    evaluator._evaluate_specificity = Mock(return_value=0.6)
    evaluator._evaluate_complexity = Mock(return_value=0.5)
    evaluator._evaluate_ambiguity = Mock(return_value=0.8)

    result = evaluator.score("test query", "ground truth")

    assert result["score"] > 0.0
    assert "score" in result
    assert "reasoning" in result
    assert "details" in result
    assert "clarity_score" in result["details"]
    assert "intent_score" in result["details"]
    assert "preprocessing_score" in result["details"]
    assert "specificity_score" in result["details"]
    assert "complexity_score" in result["details"]
    assert "ambiguity_score" in result["details"]
    assert "weights_used" in result["details"]


def test_enhanced_query_processing_with_custom_weights():
    """Test enhanced query processing evaluation with custom weights."""
    evaluator = QueryProcessingEvaluator(llm=MockLLM())

    # Mock the evaluation methods to return predictable results
    evaluator._evaluate_query_clarity = Mock(return_value=0.8)
    evaluator._evaluate_intent_detection = Mock(return_value=0.9)
    evaluator._evaluate_preprocessing = Mock(return_value=0.7)
    evaluator._evaluate_specificity = Mock(return_value=0.6)
    evaluator._evaluate_complexity = Mock(return_value=0.5)
    evaluator._evaluate_ambiguity = Mock(return_value=0.8)

    custom_weights = {
        "clarity": 0.3,
        "intent": 0.2,
        "preprocessing": 0.1,
        "specificity": 0.2,
        "complexity": 0.1,
        "ambiguity": 0.1,
    }

    result = evaluator.score("test query", "ground truth", weights=custom_weights)

    assert result["score"] > 0.0
    assert result["details"]["weights_used"] == custom_weights


# ============================================================================
# RAGContext and RAGSample Tests
# ============================================================================


def test_rag_context_creation():
    """Test RAGContext creation and attributes."""
    context = RAGContext(
        content="Test content",
        source="test_source",
        relevance_score=0.8,
        rank=1,
        metadata={"key": "value"},
    )

    assert context.content == "Test content"
    assert context.source == "test_source"
    assert context.relevance_score == 0.8
    assert context.rank == 1
    assert context.metadata == {"key": "value"}


def test_rag_context_defaults():
    """Test RAGContext with default values."""
    context = RAGContext(content="Test content", source="test_source")

    assert context.content == "Test content"
    assert context.source == "test_source"
    assert context.relevance_score == 0.0
    assert context.rank == 0
    assert context.metadata == {}


def test_rag_sample_creation():
    """Test RAGSample creation and attributes."""
    contexts = [
        RAGContext(content="Context 1", source="source1"),
        RAGContext(content="Context 2", source="source2"),
    ]

    sample = RAGSample(
        query="Test query",
        ground_truth="Test ground truth",
        generated_answer="Test answer",
        retrieved_contexts=contexts,
        pipeline_metadata={"test": "metadata"},
    )

    assert sample.query == "Test query"
    assert sample.ground_truth == "Test ground truth"
    assert sample.generated_answer == "Test answer"
    assert len(sample.retrieved_contexts) == 2
    assert sample.pipeline_metadata == {"test": "metadata"}


def test_rag_sample_defaults():
    """Test RAGSample with default values."""
    sample = RAGSample(
        query="Test query",
        ground_truth="Test ground truth",
        generated_answer="Test answer",
        retrieved_contexts=[],
    )

    assert sample.query == "Test query"
    assert sample.ground_truth == "Test ground truth"
    assert sample.generated_answer == "Test answer"
    assert sample.retrieved_contexts == []
    assert sample.pipeline_metadata == {}


# ============================================================================
# StageMetrics Tests
# ============================================================================


def test_stage_metrics_creation():
    """Test StageMetrics creation and attributes."""
    metrics = StageMetrics(
        stage_name="test_stage",
        latency_ms=100.5,
        success=True,
        error_message=None,
        metrics={"score": 0.8},
        resource_usage={"cpu": 0.5, "memory": 0.3},
    )

    assert metrics.stage_name == "test_stage"
    assert metrics.latency_ms == 100.5
    assert metrics.success is True
    assert metrics.error_message is None
    assert metrics.metrics == {"score": 0.8}
    assert metrics.resource_usage == {"cpu": 0.5, "memory": 0.3}


def test_stage_metrics_defaults():
    """Test StageMetrics with default values."""
    metrics = StageMetrics(
        stage_name="test_stage",
        latency_ms=50.0,
        success=False,
        error_message="Test error",
    )

    assert metrics.stage_name == "test_stage"
    assert metrics.latency_ms == 50.0
    assert metrics.success is False
    assert metrics.error_message == "Test error"
    assert metrics.metrics == {}
    assert metrics.resource_usage == {}


# ============================================================================
# RetrievalStageEvaluator Tests
# ============================================================================


def test_retrieval_stage_evaluator_initialization():
    """Test RetrievalStageEvaluator initialization."""
    evaluator = RetrievalStageEvaluator(llm=MockLLM())

    assert evaluator is not None
    assert hasattr(evaluator, "score")
    assert hasattr(evaluator, "precision_scorer")
    assert hasattr(evaluator, "recall_scorer")
    assert hasattr(evaluator, "ranking_scorer")
    assert hasattr(evaluator, "similarity_scorer")
    assert hasattr(evaluator, "diversity_scorer")


def test_retrieval_stage_evaluator_no_context():
    """Test RetrievalStageEvaluator with no context."""
    evaluator = RetrievalStageEvaluator(llm=MockLLM())

    result = evaluator.score("", "ground truth", context=None)

    assert result["score"] == 0.0
    assert "No retrieved contexts provided" in result["reasoning"]
    assert "error" in result["details"]


def test_retrieval_stage_evaluator_with_context():
    """Test RetrievalStageEvaluator with valid context."""
    evaluator = RetrievalStageEvaluator(llm=MockLLM())

    contexts = [
        RAGContext(content="Context 1", source="source1", relevance_score=0.8),
        RAGContext(content="Context 2", source="source2", relevance_score=0.6),
    ]

    result = evaluator.score(
        "", "ground truth", context={"retrieved_contexts": contexts}
    )

    assert "score" in result
    assert "reasoning" in result
    assert "details" in result
    assert "precision" in result["details"]
    assert "recall" in result["details"]
    assert "f1" in result["details"]
    assert "ranking" in result["details"]
    assert "similarity" in result["details"]
    assert "diversity" in result["details"]


def test_retrieval_stage_evaluator_calculate_diversity():
    """Test RetrievalStageEvaluator diversity calculation."""
    evaluator = RetrievalStageEvaluator(llm=MockLLM())

    # Test with single context
    contexts = [RAGContext(content="Context 1", source="source1")]
    diversity = evaluator._calculate_diversity(contexts)
    assert diversity == 0.0

    # Test with multiple contexts from same source
    contexts = [
        RAGContext(content="Context 1", source="source1"),
        RAGContext(content="Context 2", source="source1"),
    ]
    diversity = evaluator._calculate_diversity(contexts)
    assert diversity == 0.5  # 1 source / 2 contexts

    # Test with multiple contexts from different sources
    contexts = [
        RAGContext(content="Context 1", source="source1"),
        RAGContext(content="Context 2", source="source2"),
    ]
    diversity = evaluator._calculate_diversity(contexts)
    assert diversity == 1.0  # 2 sources / 2 contexts


# ============================================================================
# RerankingEvaluator Tests
# ============================================================================


def test_reranking_evaluator_initialization():
    """Test RerankingEvaluator initialization."""
    evaluator = RerankingEvaluator(llm=MockLLM())

    assert evaluator is not None
    assert hasattr(evaluator, "score")
    assert hasattr(evaluator, "bias_scorer")
    assert hasattr(evaluator, "factual_scorer")
    assert hasattr(evaluator, "claim_scorer")


def test_reranking_evaluator_no_generated_answer():
    """Test RerankingEvaluator with no generated answer."""
    evaluator = RerankingEvaluator(llm=MockLLM())

    result = evaluator.score("", "ground truth", context=None)

    assert result["score"] == 0.0
    assert "No generated answer provided" in result["reasoning"]
    assert "error" in result["details"]


def test_reranking_evaluator_with_generated_answer():
    """Test RerankingEvaluator with valid generated answer."""
    evaluator = RerankingEvaluator(llm=MockLLM())

    contexts = [
        RAGContext(content="Context 1", source="source1"),
        RAGContext(content="Context 2", source="source2"),
    ]

    context_dict = {
        "generated_answer": "This is a generated answer",
        "retrieved_contexts": contexts,
    }

    result = evaluator.score("", "ground truth", context=context_dict)

    assert "score" in result
    assert "reasoning" in result
    assert "details" in result
    assert "quality" in result["details"]
    assert "faithfulness" in result["details"]
    assert "groundedness" in result["details"]
    assert "factual" in result["details"]
    assert "clarity" in result["details"]


# ============================================================================
# PipelineCoordinationScorer Tests
# ============================================================================


def test_pipeline_coordination_scorer_initialization():
    """Test PipelineCoordinationScorer initialization."""
    scorer = PipelineCoordinationScorer()

    assert scorer is not None
    assert hasattr(scorer, "score")


def test_pipeline_coordination_scorer_no_metrics():
    """Test PipelineCoordinationScorer with no stage metrics."""
    scorer = PipelineCoordinationScorer()

    result = scorer.score("", "", context=None)

    assert result == 0.0


def test_pipeline_coordination_scorer_with_metrics():
    """Test PipelineCoordinationScorer with stage metrics."""
    scorer = PipelineCoordinationScorer()

    stage_metrics = {
        "stage1": StageMetrics("stage1", 100.0, True),
        "stage2": StageMetrics("stage2", 200.0, True),
        "stage3": StageMetrics("stage3", 150.0, False, "Test error"),
    }

    result = scorer.score("", "", context={"stage_metrics": stage_metrics})

    assert "score" in result
    assert "reasoning" in result
    assert "details" in result
    assert "success_rate" in result["details"]
    assert "latency_consistency" in result["details"]
    assert "data_flow_quality" in result["details"]


def test_pipeline_coordination_scorer_calculate_success_rate():
    """Test PipelineCoordinationScorer success rate calculation."""
    scorer = PipelineCoordinationScorer()

    stage_metrics = {
        "stage1": StageMetrics("stage1", 100.0, True),
        "stage2": StageMetrics("stage2", 200.0, False),
        "stage3": StageMetrics("stage3", 150.0, True),
    }

    success_rate = scorer._calculate_success_rate(stage_metrics)
    assert success_rate == 2 / 3  # 2 out of 3 stages successful


def test_pipeline_coordination_scorer_calculate_latency_consistency():
    """Test PipelineCoordinationScorer latency consistency calculation."""
    scorer = PipelineCoordinationScorer()

    stage_metrics = {
        "stage1": StageMetrics("stage1", 100.0, True),
        "stage2": StageMetrics("stage2", 200.0, True),
        "stage3": StageMetrics("stage3", 150.0, True),
    }

    consistency = scorer._calculate_latency_consistency(stage_metrics)
    assert isinstance(consistency, float)
    assert 0.0 <= consistency <= 1.0


def test_pipeline_coordination_scorer_calculate_data_flow_quality():
    """Test PipelineCoordinationScorer data flow quality calculation."""
    scorer = PipelineCoordinationScorer()

    stage_metrics = {
        "stage1": StageMetrics("stage1", 100.0, True),
        "stage2": StageMetrics("stage2", 200.0, True),
        "stage3": StageMetrics("stage3", 150.0, True),
    }

    quality = scorer._calculate_data_flow_quality(stage_metrics)
    assert isinstance(quality, float)
    assert 0.0 <= quality <= 1.0


# ============================================================================
# LatencyAnalysisScorer Tests
# ============================================================================


def test_latency_analysis_scorer_initialization():
    """Test LatencyAnalysisScorer initialization."""
    scorer = LatencyAnalysisScorer()

    assert scorer is not None
    assert hasattr(scorer, "score")


def test_latency_analysis_scorer_no_metrics():
    """Test LatencyAnalysisScorer with no stage metrics."""
    scorer = LatencyAnalysisScorer()

    result = scorer.score("", "", context=None)

    assert result["score"] == 0.0
    assert "No stage metrics provided" in result["reasoning"]


def test_latency_analysis_scorer_with_metrics():
    """Test LatencyAnalysisScorer with stage metrics."""
    scorer = LatencyAnalysisScorer()

    stage_metrics = {
        "stage1": StageMetrics("stage1", 100.0, True),
        "stage2": StageMetrics("stage2", 200.0, True),
        "stage3": StageMetrics("stage3", 150.0, True),
    }

    result = scorer.score("", "", context={"stage_metrics": stage_metrics})

    assert "score" in result
    assert "reasoning" in result
    assert "details" in result
    assert "total_latency" in result["details"]
    assert "avg_latency" in result["details"]
    assert "max_latency" in result["details"]
    assert "latency_distribution" in result["details"]


def test_latency_analysis_scorer_calculate_latency_distribution():
    """Test LatencyAnalysisScorer latency distribution calculation."""
    scorer = LatencyAnalysisScorer()

    stage_metrics = {
        "stage1": StageMetrics("stage1", 100.0, True),
        "stage2": StageMetrics("stage2", 200.0, True),
        "stage3": StageMetrics("stage3", 150.0, True),
    }

    distribution = scorer._calculate_latency_distribution(stage_metrics)

    assert distribution == {"stage1": 100.0, "stage2": 200.0, "stage3": 150.0}


# ============================================================================
# ResourceUtilizationScorer Tests
# ============================================================================


def test_resource_utilization_scorer_initialization():
    """Test ResourceUtilizationScorer initialization."""
    scorer = ResourceUtilizationScorer()

    assert scorer is not None
    assert hasattr(scorer, "score")


def test_resource_utilization_scorer_no_metrics():
    """Test ResourceUtilizationScorer with no stage metrics."""
    scorer = ResourceUtilizationScorer()

    result = scorer.score("", "", context=None)

    assert result == 0.0


def test_resource_utilization_scorer_with_metrics():
    """Test ResourceUtilizationScorer with stage metrics."""
    scorer = ResourceUtilizationScorer()

    stage_metrics = {
        "stage1": StageMetrics("stage1", 100.0, True),
        "stage2": StageMetrics("stage2", 200.0, True),
        "stage3": StageMetrics("stage3", 150.0, False),
    }

    result = scorer.score("", "", context={"stage_metrics": stage_metrics})

    assert "score" in result
    assert "reasoning" in result
    assert "details" in result
    assert "cpu_usage" in result["details"]
    assert "memory_usage" in result["details"]
    assert "resource_efficiency" in result["details"]


def test_resource_utilization_scorer_calculate_cpu_usage():
    """Test ResourceUtilizationScorer CPU usage calculation."""
    scorer = ResourceUtilizationScorer()

    stage_metrics = {
        "stage1": StageMetrics("stage1", 100.0, True),
        "stage2": StageMetrics("stage2", 2000.0, True),  # > 1000ms
        "stage3": StageMetrics("stage3", 150.0, False),
    }

    cpu_usage = scorer._calculate_cpu_usage(stage_metrics)
    assert isinstance(cpu_usage, float)
    assert 0.0 <= cpu_usage <= 1.0


def test_resource_utilization_scorer_calculate_memory_usage():
    """Test ResourceUtilizationScorer memory usage calculation."""
    scorer = ResourceUtilizationScorer()

    stage_metrics = {
        "stage1": StageMetrics("stage1", 100.0, True),
        "stage2": StageMetrics("stage2", 200.0, True),
        "stage3": StageMetrics("stage3", 150.0, False),
    }

    memory_usage = scorer._calculate_memory_usage(stage_metrics)
    assert memory_usage == 2 / 3  # 2 out of 3 stages successful


def test_resource_utilization_scorer_calculate_resource_efficiency():
    """Test ResourceUtilizationScorer resource efficiency calculation."""
    scorer = ResourceUtilizationScorer()

    stage_metrics = {
        "stage1": StageMetrics("stage1", 100.0, True),
        "stage2": StageMetrics("stage2", 200.0, True),
        "stage3": StageMetrics("stage3", 150.0, False),
    }

    efficiency = scorer._calculate_resource_efficiency(stage_metrics)
    assert isinstance(efficiency, float)
    assert 0.0 <= efficiency <= 1.0


# ============================================================================
# ErrorPropagationScorer Tests
# ============================================================================


def test_error_propagation_scorer_initialization():
    """Test ErrorPropagationScorer initialization."""
    scorer = ErrorPropagationScorer()

    assert scorer is not None
    assert hasattr(scorer, "score")


def test_error_propagation_scorer_no_metrics():
    """Test ErrorPropagationScorer with no stage metrics."""
    scorer = ErrorPropagationScorer()

    result = scorer.score("", "", context=None)

    assert result["score"] == 0.0
    assert "No stage metrics provided" in result["reasoning"]


def test_error_propagation_scorer_with_metrics():
    """Test ErrorPropagationScorer with stage metrics."""
    scorer = ErrorPropagationScorer()

    stage_metrics = {
        "stage1": StageMetrics("stage1", 100.0, True),
        "stage2": StageMetrics("stage2", 200.0, False),
        "stage3": StageMetrics("stage3", 150.0, True),
    }

    result = scorer.score("", "", context={"stage_metrics": stage_metrics})

    assert "score" in result
    assert "reasoning" in result
    assert "details" in result
    assert "error_rate" in result["details"]
    assert "error_isolation" in result["details"]
    assert "recovery_effectiveness" in result["details"]


def test_error_propagation_scorer_calculate_error_rate():
    """Test ErrorPropagationScorer error rate calculation."""
    scorer = ErrorPropagationScorer()

    stage_metrics = {
        "stage1": StageMetrics("stage1", 100.0, True),
        "stage2": StageMetrics("stage2", 200.0, False),
        "stage3": StageMetrics("stage3", 150.0, False),
    }

    error_rate = scorer._calculate_error_rate(stage_metrics)
    assert error_rate == 2 / 3  # 2 out of 3 stages failed


def test_error_propagation_scorer_calculate_error_isolation():
    """Test ErrorPropagationScorer error isolation calculation."""
    scorer = ErrorPropagationScorer()

    # Test with no errors
    stage_metrics = {
        "stage1": StageMetrics("stage1", 100.0, True),
        "stage2": StageMetrics("stage2", 200.0, True),
        "stage3": StageMetrics("stage3", 150.0, True),
    }

    isolation = scorer._calculate_error_isolation(stage_metrics)
    assert isolation == 1.0  # No errors

    # Test with some errors
    stage_metrics = {
        "stage1": StageMetrics("stage1", 100.0, True),
        "stage2": StageMetrics("stage2", 200.0, False),
        "stage3": StageMetrics("stage3", 150.0, True),
    }

    isolation = scorer._calculate_error_isolation(stage_metrics)
    assert isinstance(isolation, float)
    assert 0.0 <= isolation <= 1.0


def test_error_propagation_scorer_calculate_recovery_effectiveness():
    """Test ErrorPropagationScorer recovery effectiveness calculation."""
    scorer = ErrorPropagationScorer()

    # Test with no critical failures
    stage_metrics = {
        "query_processing": StageMetrics("query_processing", 100.0, True),
        "retrieval": StageMetrics("retrieval", 200.0, True),
        "generation": StageMetrics("generation", 150.0, True),
    }

    recovery = scorer._calculate_recovery_effectiveness(stage_metrics)
    assert recovery == 1.0  # No critical failures

    # Test with some critical failures
    stage_metrics = {
        "query_processing": StageMetrics("query_processing", 100.0, False),
        "retrieval": StageMetrics("retrieval", 200.0, True),
        "generation": StageMetrics("generation", 150.0, True),
    }

    recovery = scorer._calculate_recovery_effectiveness(stage_metrics)
    assert isinstance(recovery, float)
    assert 0.0 <= recovery <= 1.0


# ============================================================================
# AgentData Tests
# ============================================================================


def test_agent_data_creation():
    """Test AgentData creation and attributes."""
    agent_data = AgentData(
        user_id="user123",
        task_id="task456",
        ground_truth="Test ground truth",
        agent_response="Test response",
        retrieval_query="Test query",
        retrieved_context="Test context",
        metadata="Test metadata",
    )

    assert agent_data.user_id == "user123"
    assert agent_data.task_id == "task456"
    assert agent_data.ground_truth == "Test ground truth"
    assert agent_data.agent_response == "Test response"
    assert agent_data.retrieval_query == "Test query"
    assert agent_data.retrieved_context == "Test context"
    assert agent_data.metadata == "Test metadata"


def test_agent_data_defaults():
    """Test AgentData with default values."""
    agent_data = AgentData()

    assert agent_data.user_id is None
    assert agent_data.task_id is None
    assert agent_data.ground_truth is None
    assert agent_data.agent_response is None
    assert agent_data.retrieval_query is None
    assert agent_data.retrieved_context is None
    assert agent_data.metadata is None


# ============================================================================
# RAGPipelineEvaluator Additional Tests
# ============================================================================


def test_rag_pipeline_evaluator_convert_agent_data():
    """Test RAGPipelineEvaluator agent data conversion."""
    mock_llm = MockLLM()
    evaluator = RAGPipelineEvaluator(llm=mock_llm)

    agent_data = AgentData(
        user_id="user123",
        task_id="task456",
        ground_truth="Test ground truth",
        agent_response="Test response",
        retrieval_query="Test query",
        retrieved_context="Context chunk 1\n\nContext chunk 2\n\nContext chunk 3",
        metadata="Test metadata",
    )

    rag_sample = evaluator._convert_agent_data_to_rag_sample(agent_data)

    assert rag_sample.query == "Test query"
    assert rag_sample.ground_truth == "Test ground truth"
    assert rag_sample.generated_answer == "Test response"
    assert len(rag_sample.retrieved_contexts) == 3  # Split by double newlines
    assert rag_sample.pipeline_metadata["user_id"] == "user123"
    assert rag_sample.pipeline_metadata["task_id"] == "task456"
    assert rag_sample.pipeline_metadata["metadata"] == "Test metadata"


def test_rag_pipeline_evaluator_evaluate_from_agent_data():
    """Test RAGPipelineEvaluator evaluation from agent data."""
    mock_llm = MockLLM()
    evaluator = RAGPipelineEvaluator(llm=mock_llm)

    agent_data = AgentData(
        user_id="user123",
        task_id="task456",
        ground_truth="Machine learning is a subset of AI",
        agent_response="Machine learning is a branch of artificial intelligence",
        retrieval_query="What is machine learning?",
        retrieved_context="Machine learning is AI\n\nAI includes machine learning",
        metadata="Test metadata",
    )

    result = evaluator.evaluate_from_agent_data(agent_data)

    assert hasattr(result, "overall_score")
    assert hasattr(result, "stage_metrics")
    assert hasattr(result, "detailed_scores")
    assert hasattr(result, "recommendations")
    assert isinstance(result.overall_score, float)
    assert isinstance(result.stage_metrics, dict)
    assert isinstance(result.detailed_scores, dict)
    assert isinstance(result.recommendations, list)


def test_rag_pipeline_evaluator_test_pipeline_compatibility():
    """Test RAGPipelineEvaluator pipeline compatibility testing."""
    mock_llm = MockLLM()
    evaluator = RAGPipelineEvaluator(llm=mock_llm)

    agent_data = AgentData(
        user_id="user123",
        task_id="task456",
        ground_truth="Test ground truth",
        agent_response="Test response",
        retrieval_query="Test query",
        retrieved_context="Test context",
        metadata="Test metadata",
    )

    result = evaluator.test_pipeline_compatibility(agent_data)

    assert "success" in result
    assert "conversion_works" in result
    assert "evaluation_works" in result
    assert "overall_score" in result
    assert "stage_count" in result
    assert "recommendations" in result
    assert "error" in result


def test_rag_pipeline_evaluator_with_custom_weights():
    """Test RAGPipelineEvaluator with custom weights."""
    mock_llm = MockLLM()
    evaluator = RAGPipelineEvaluator(llm=mock_llm)

    contexts = [
        RAGContext(
            content="Machine learning is AI", source="source1", relevance_score=0.8
        )
    ]

    rag_sample = RAGSample(
        query="What is machine learning?",
        ground_truth="Machine learning is a subset of AI",
        generated_answer="Machine learning is a branch of artificial intelligence",
        retrieved_contexts=contexts,
    )

    custom_weights = {
        "query_weights": {"clarity": 0.3, "intent": 0.2},
        "retrieval_weights": {"precision": 0.3, "recall": 0.2},
        "generation_weights": {"quality": 0.2, "faithfulness": 0.1},
        "pipeline_weights": {"retrieval": 0.4, "generation": 0.3},
    }

    result = evaluator.evaluate_rag_pipeline(
        rag_sample=rag_sample,
        retrieved_contexts=contexts,
        generated_answer="Machine learning is a branch of artificial intelligence",
        weights=custom_weights,
    )

    assert hasattr(result, "overall_score")
    assert hasattr(result, "stage_metrics")
    assert hasattr(result, "detailed_scores")
    assert isinstance(result.overall_score, float)
    assert isinstance(result.stage_metrics, dict)
    assert isinstance(result.detailed_scores, dict)
