"""
Unit tests for enhanced RAG evaluation system.
"""

import os

# Import the RAG scorers (assuming they are in rag_comprehensive.py)
import sys

import pytest

from novaeval.models.base import BaseModel as LLMModel
from novaeval.scorers.base import ScoreResult

sys.path.append(os.path.join(os.path.dirname(__file__), "../../src"))

from novaeval.scorers.rag_comprehensive import (
    AnswerCorrectnessScorer,
    AnswerRelevancyScorer,
    AnswerSimilarityScorer,
    ContextEntityRecallScorer,
    ContextPrecisionScorer,
    ContextRecallScorer,
    ContextRelevancyScorer,
    EnhancedFaithfulnessScorer,
    EnhancedRAGASScorer,
    RAGEvaluationConfig,
    RAGEvaluationSuite,
    RAGTriadScorer,
    create_rag_scorer,
    get_default_rag_config,
    get_optimized_rag_config,
)


class MockLLMModel(LLMModel):
    """Mock LLM model for testing."""

    def __init__(self, mock_responses=None):
        super().__init__()
        self.mock_responses = mock_responses or {}
        self.call_count = 0
        self.last_prompt = None

    async def generate(self, prompt: str, **kwargs) -> str:
        """Mock generate method."""
        self.last_prompt = prompt
        self.call_count += 1

        # Return specific responses based on prompt content
        if "relevance_score" in prompt.lower():
            return '{"relevance_score": 4, "reasoning": "Highly relevant content"}'
        elif "questions" in prompt.lower():
            return '{"questions": ["What is the main topic?", "How does it work?", "What are the benefits?"]}'
        elif "structural_similarity" in prompt.lower():
            return '{"structural_similarity": 0.8, "reasoning": "Similar structure and organization"}'
        elif "classification" in prompt.lower():
            return '{"classification": "TRUE_POSITIVE", "confidence": 0.9, "reasoning": "Statement is factually correct"}'
        elif "presence_status" in prompt.lower():
            return '{"presence_status": "FULLY_PRESENT", "confidence": 0.9, "supporting_evidence": "Found in context", "reasoning": "Information clearly present"}'
        elif "supported" in prompt.lower():
            return '{"status": "SUPPORTED", "confidence": 0.9, "category": "factual", "supporting_evidence": "Direct evidence found", "reasoning": "Claim is supported by context"}'
        elif "extract" in prompt.lower() and "claims" in prompt.lower():
            return '{"factual_claims": ["Claim 1", "Claim 2"], "numerical_claims": ["Number fact"], "temporal_claims": [], "relational_claims": [], "opinion_claims": []}'
        elif "extract" in prompt.lower() and "statements" in prompt.lower():
            return '{"statements": ["Statement 1", "Statement 2", "Statement 3"]}'
        elif "extract" in prompt.lower() and (
            "key" in prompt.lower() or "information" in prompt.lower()
        ):
            return '{"key_facts": ["Fact 1", "Fact 2"], "key_concepts": ["Concept 1"], "key_details": ["Detail 1"], "essential_information": ["Info 1"]}'
        elif "entities" in prompt.lower():
            return '{"persons": ["John Doe"], "organizations": ["ACME Corp"], "locations": ["New York"], "dates": ["2024"], "numbers": ["100"], "concepts": ["AI"], "other_entities": ["Technology"]}'
        elif "yes" in prompt.lower() or "no" in prompt.lower():
            return "YES"
        else:
            return "Default response for testing"


class TestRAGEvaluationConfig:
    """Test RAG evaluation configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = RAGEvaluationConfig()

        assert config.embedding_model == "all-MiniLM-L6-v2"
        assert config.similarity_threshold == 0.7
        assert config.faithfulness_threshold == 0.8
        assert config.relevancy_threshold == 0.7
        assert config.precision_threshold == 0.7
        assert config.recall_threshold == 0.7
        assert config.answer_correctness_threshold == 0.8

        # Check default weights
        assert len(config.ragas_weights) == 8
        assert sum(config.ragas_weights.values()) == 1.0

    def test_custom_config(self):
        """Test custom configuration values."""
        custom_weights = {
            "context_precision": 0.3,
            "context_relevancy": 0.2,
            "context_recall": 0.2,
            "context_entity_recall": 0.1,
            "answer_relevancy": 0.1,
            "answer_similarity": 0.05,
            "answer_correctness": 0.05,
            "faithfulness": 0.2,
        }

        config = RAGEvaluationConfig(
            embedding_model="custom-model",
            similarity_threshold=0.8,
            ragas_weights=custom_weights,
        )

        assert config.embedding_model == "custom-model"
        assert config.similarity_threshold == 0.8
        assert config.ragas_weights == custom_weights


class TestContextPrecisionScorer:
    """Test Context Precision Scorer."""

    @pytest.fixture
    def scorer(self):
        """Create a context precision scorer for testing."""
        mock_model = MockLLMModel()
        return ContextPrecisionScorer(mock_model)

    @pytest.mark.asyncio
    async def test_evaluate_with_context(self, scorer):
        """Test evaluation with valid context."""
        input_text = "What is machine learning?"
        output_text = "Machine learning is a subset of AI."
        context = [
            "Machine learning is a method of data analysis.",
            "It automates analytical model building.",
        ]

        result = await scorer.evaluate(input_text, output_text, context=context)

        assert isinstance(result, ScoreResult)
        assert 0.0 <= result.score <= 1.0
        assert isinstance(result.passed, bool)
        assert len(result.reasoning) > 0
        assert "context_chunks_count" in result.metadata

    @pytest.mark.asyncio
    async def test_evaluate_no_context(self, scorer):
        """Test evaluation without context."""
        input_text = "What is machine learning?"
        output_text = "Machine learning is a subset of AI."

        result = await scorer.evaluate(input_text, output_text)

        assert result.score == 0.0
        assert result.passed is False
        assert "no_context" in result.metadata["error"]

    @pytest.mark.asyncio
    async def test_evaluate_string_context(self, scorer):
        """Test evaluation with string context."""
        input_text = "What is machine learning?"
        output_text = "Machine learning is a subset of AI."
        context = "Machine learning is a method of data analysis. It automates analytical model building."

        result = await scorer.evaluate(input_text, output_text, context=context)

        assert isinstance(result, ScoreResult)
        assert 0.0 <= result.score <= 1.0
        assert "context_chunks_count" in result.metadata


class TestContextRelevancyScorer:
    """Test Context Relevancy Scorer."""

    @pytest.fixture
    def scorer(self):
        """Create a context relevancy scorer for testing."""
        mock_model = MockLLMModel()
        return ContextRelevancyScorer(mock_model)

    @pytest.mark.asyncio
    async def test_evaluate_with_context(self, scorer):
        """Test evaluation with valid context."""
        input_text = "What is machine learning?"
        output_text = "Machine learning is a subset of AI."
        context = "Machine learning is a method of data analysis that automates analytical model building."

        result = await scorer.evaluate(input_text, output_text, context=context)

        assert isinstance(result, ScoreResult)
        assert 0.0 <= result.score <= 1.0
        assert "context_length" in result.metadata
        assert "context_chunks_count" in result.metadata

    @pytest.mark.asyncio
    async def test_evaluate_no_context(self, scorer):
        """Test evaluation without context."""
        input_text = "What is machine learning?"
        output_text = "Machine learning is a subset of AI."

        result = await scorer.evaluate(input_text, output_text)

        assert result.score == 0.0
        assert result.passed is False
        assert "no_context" in result.metadata["error"]


class TestContextRecallScorer:
    """Test Context Recall Scorer."""

    @pytest.fixture
    def scorer(self):
        """Create a context recall scorer for testing."""
        mock_model = MockLLMModel()
        return ContextRecallScorer(mock_model)

    @pytest.mark.asyncio
    async def test_evaluate_with_context_and_expected(self, scorer):
        """Test evaluation with context and expected output."""
        input_text = "What is machine learning?"
        output_text = "Machine learning is a subset of AI."
        expected_output = "Machine learning is a method of data analysis."
        context = "Machine learning is a method of data analysis that automates analytical model building."

        result = await scorer.evaluate(
            input_text, output_text, expected_output, context
        )

        assert isinstance(result, ScoreResult)
        assert 0.0 <= result.score <= 1.0
        assert "total_key_info" in result.metadata
        assert "recall_evaluations" in result.metadata

    @pytest.mark.asyncio
    async def test_evaluate_missing_inputs(self, scorer):
        """Test evaluation with missing required inputs."""
        input_text = "What is machine learning?"
        output_text = "Machine learning is a subset of AI."

        result = await scorer.evaluate(input_text, output_text)

        assert result.score == 0.0
        assert result.passed is False
        assert "missing_inputs" in result.metadata["error"]


class TestContextEntityRecallScorer:
    """Test Context Entity Recall Scorer."""

    @pytest.fixture
    def scorer(self):
        """Create a context entity recall scorer for testing."""
        mock_model = MockLLMModel()
        return ContextEntityRecallScorer(mock_model)

    @pytest.mark.asyncio
    async def test_evaluate_with_entities(self, scorer):
        """Test evaluation with entities in expected output."""
        input_text = "Who founded Microsoft?"
        output_text = "Bill Gates founded Microsoft."
        expected_output = "Microsoft was founded by Bill Gates in 1975."
        context = "Bill Gates and Paul Allen founded Microsoft Corporation in 1975."

        result = await scorer.evaluate(
            input_text, output_text, expected_output, context
        )

        assert isinstance(result, ScoreResult)
        assert 0.0 <= result.score <= 1.0
        assert "total_entities" in result.metadata
        assert "entity_evaluations" in result.metadata

    @pytest.mark.asyncio
    async def test_evaluate_no_entities(self, scorer):
        """Test evaluation when no entities are found."""
        input_text = "What is the weather?"
        output_text = "It is sunny."
        expected_output = "The weather is nice."
        context = "Today is a beautiful day."

        # Mock the model to return empty entities
        scorer.model.mock_responses = {
            "entities": '{"persons": [], "organizations": [], "locations": [], "dates": [], "numbers": [], "concepts": [], "other_entities": []}'
        }

        result = await scorer.evaluate(
            input_text, output_text, expected_output, context
        )

        assert result.score == 1.0  # No entities means perfect recall
        assert result.passed is True


class TestAnswerRelevancyScorer:
    """Test Answer Relevancy Scorer."""

    @pytest.fixture
    def scorer(self):
        """Create an answer relevancy scorer for testing."""
        mock_model = MockLLMModel()
        return AnswerRelevancyScorer(mock_model)

    @pytest.mark.asyncio
    async def test_evaluate_multi_approach(self, scorer):
        """Test multi-approach evaluation."""
        input_text = "What is machine learning?"
        output_text = "Machine learning is a subset of artificial intelligence that enables computers to learn without being explicitly programmed."

        result = await scorer.evaluate(input_text, output_text)

        assert isinstance(result, ScoreResult)
        assert 0.0 <= result.score <= 1.0
        assert "question_generation_score" in result.metadata
        assert "direct_relevance_score" in result.metadata
        assert "semantic_similarity_score" in result.metadata
        assert "approach" in result.metadata
        assert result.metadata["approach"] == "multi_method"


class TestAnswerSimilarityScorer:
    """Test Answer Similarity Scorer."""

    @pytest.fixture
    def scorer(self):
        """Create an answer similarity scorer for testing."""
        mock_model = MockLLMModel()
        return AnswerSimilarityScorer(mock_model)

    @pytest.mark.asyncio
    async def test_evaluate_with_expected_output(self, scorer):
        """Test evaluation with expected output."""
        input_text = "What is machine learning?"
        output_text = (
            "Machine learning is a subset of AI that enables computers to learn."
        )
        expected_output = "Machine learning is an AI technique that allows computers to learn from data."

        result = await scorer.evaluate(input_text, output_text, expected_output)

        assert isinstance(result, ScoreResult)
        assert 0.0 <= result.score <= 1.0
        assert "semantic_similarity" in result.metadata
        assert "lexical_similarity" in result.metadata
        assert "structural_similarity" in result.metadata

    @pytest.mark.asyncio
    async def test_evaluate_no_expected_output(self, scorer):
        """Test evaluation without expected output."""
        input_text = "What is machine learning?"
        output_text = "Machine learning is a subset of AI."

        result = await scorer.evaluate(input_text, output_text)

        assert result.score == 0.0
        assert result.passed is False
        assert "no_expected_output" in result.metadata["error"]


class TestAnswerCorrectnessScorer:
    """Test Answer Correctness Scorer."""

    @pytest.fixture
    def scorer(self):
        """Create an answer correctness scorer for testing."""
        mock_model = MockLLMModel()
        return AnswerCorrectnessScorer(mock_model)

    @pytest.mark.asyncio
    async def test_evaluate_with_statements(self, scorer):
        """Test evaluation with factual statements."""
        input_text = "What is the capital of France?"
        output_text = (
            "The capital of France is Paris. It is located in northern France."
        )
        expected_output = "Paris is the capital and largest city of France."

        result = await scorer.evaluate(input_text, output_text, expected_output)

        assert isinstance(result, ScoreResult)
        assert 0.0 <= result.score <= 1.0
        assert "true_positives" in result.metadata
        assert "false_positives" in result.metadata
        assert "false_negatives" in result.metadata
        assert "precision" in result.metadata
        assert "recall" in result.metadata
        assert "f1_score" in result.metadata


class TestEnhancedFaithfulnessScorer:
    """Test Enhanced Faithfulness Scorer."""

    @pytest.fixture
    def scorer(self):
        """Create an enhanced faithfulness scorer for testing."""
        mock_model = MockLLMModel()
        return EnhancedFaithfulnessScorer(mock_model)

    @pytest.mark.asyncio
    async def test_evaluate_with_context(self, scorer):
        """Test evaluation with context."""
        input_text = "What is machine learning?"
        output_text = "Machine learning is a subset of AI that enables computers to learn from data."
        context = "Machine learning is a method of data analysis that automates analytical model building using algorithms."

        result = await scorer.evaluate(input_text, output_text, context=context)

        assert isinstance(result, ScoreResult)
        assert 0.0 <= result.score <= 1.0
        assert "total_claims" in result.metadata
        assert "claims_by_category" in result.metadata
        assert "verification_results" in result.metadata

    @pytest.mark.asyncio
    async def test_evaluate_no_context(self, scorer):
        """Test evaluation without context."""
        input_text = "What is machine learning?"
        output_text = "Machine learning is a subset of AI."

        result = await scorer.evaluate(input_text, output_text)

        assert result.score == 0.0
        assert result.passed is False
        assert "no_context" in result.metadata["error"]


class TestEnhancedRAGASScorer:
    """Test Enhanced RAGAS Scorer."""

    @pytest.fixture
    def scorer(self):
        """Create an enhanced RAGAS scorer for testing."""
        mock_model = MockLLMModel()
        return EnhancedRAGASScorer(mock_model)

    @pytest.mark.asyncio
    async def test_evaluate_comprehensive(self, scorer):
        """Test comprehensive RAGAS evaluation."""
        input_text = "What is machine learning?"
        output_text = "Machine learning is a subset of AI that enables computers to learn from data."
        expected_output = "Machine learning is an AI technique for automated learning."
        context = "Machine learning is a method of data analysis that automates analytical model building."

        result = await scorer.evaluate(
            input_text, output_text, expected_output, context
        )

        assert isinstance(result, ScoreResult)
        assert 0.0 <= result.score <= 1.0
        assert "component_scores" in result.metadata
        assert "retrieval_score" in result.metadata
        assert "generation_score" in result.metadata
        assert "weights" in result.metadata
        assert "passed_components" in result.metadata
        assert "total_components" in result.metadata

        # Check that all expected components are present
        expected_components = [
            "context_precision",
            "context_relevancy",
            "context_recall",
            "context_entity_recall",
            "answer_relevancy",
            "answer_similarity",
            "answer_correctness",
            "faithfulness",
        ]
        for component in expected_components:
            assert component in result.metadata["component_scores"]


class TestRAGTriadScorer:
    """Test RAG Triad Scorer."""

    @pytest.fixture
    def scorer(self):
        """Create a RAG triad scorer for testing."""
        mock_model = MockLLMModel()
        return RAGTriadScorer(mock_model)

    @pytest.mark.asyncio
    async def test_evaluate_triad(self, scorer):
        """Test RAG triad evaluation."""
        input_text = "What is machine learning?"
        output_text = "Machine learning is a subset of AI that enables computers to learn from data."
        expected_output = "Machine learning is an AI technique for automated learning."
        context = "Machine learning is a method of data analysis that automates analytical model building."

        result = await scorer.evaluate(
            input_text, output_text, expected_output, context
        )

        assert isinstance(result, ScoreResult)
        assert 0.0 <= result.score <= 1.0
        assert "context_relevance" in result.metadata
        assert "groundedness" in result.metadata
        assert "answer_relevance" in result.metadata
        assert "triad_methodology" in result.metadata
        assert result.metadata["triad_methodology"] is True

        # Check that each triad component has score and passed status
        for component in ["context_relevance", "groundedness", "answer_relevance"]:
            assert "score" in result.metadata[component]
            assert "passed" in result.metadata[component]


class TestRAGEvaluationSuite:
    """Test RAG Evaluation Suite."""

    @pytest.fixture
    def suite(self):
        """Create a RAG evaluation suite for testing."""
        mock_model = MockLLMModel()
        return RAGEvaluationSuite(mock_model)

    def test_get_available_metrics(self, suite):
        """Test getting available metrics."""
        metrics = suite.get_available_metrics()

        expected_metrics = [
            "context_precision",
            "context_relevancy",
            "context_recall",
            "context_entity_recall",
            "answer_relevancy",
            "answer_similarity",
            "answer_correctness",
            "faithfulness",
            "ragas",
            "rag_triad",
        ]

        for metric in expected_metrics:
            assert metric in metrics

    def test_get_metric_info(self, suite):
        """Test getting metric information."""
        info = suite.get_metric_info()

        assert isinstance(info, dict)
        assert len(info) > 0

        for metric_name, description in info.items():
            assert isinstance(metric_name, str)
            assert isinstance(description, str)
            assert len(description) > 0

    @pytest.mark.asyncio
    async def test_evaluate_single_metric(self, suite):
        """Test evaluating a single metric."""
        input_text = "What is machine learning?"
        output_text = "Machine learning is a subset of AI."
        context = "Machine learning is a method of data analysis."

        result = await suite.evaluate_single_metric(
            "context_relevancy", input_text, output_text, context=context
        )

        assert isinstance(result, ScoreResult)
        assert 0.0 <= result.score <= 1.0

    @pytest.mark.asyncio
    async def test_evaluate_unknown_metric(self, suite):
        """Test evaluating an unknown metric."""
        input_text = "What is machine learning?"
        output_text = "Machine learning is a subset of AI."

        result = await suite.evaluate_single_metric(
            "unknown_metric", input_text, output_text
        )

        assert result.score == 0.0
        assert result.passed is False
        assert "unknown_metric" in result.metadata["error"]

    @pytest.mark.asyncio
    async def test_evaluate_retrieval_pipeline(self, suite):
        """Test evaluating retrieval pipeline."""
        input_text = "What is machine learning?"
        output_text = "Machine learning is a subset of AI."
        expected_output = "Machine learning is an AI technique."
        context = "Machine learning is a method of data analysis."

        results = await suite.evaluate_retrieval_pipeline(
            input_text, output_text, expected_output, context
        )

        expected_metrics = [
            "context_precision",
            "context_relevancy",
            "context_recall",
            "context_entity_recall",
        ]

        assert isinstance(results, dict)
        assert len(results) == len(expected_metrics)

        for metric in expected_metrics:
            assert metric in results
            assert isinstance(results[metric], ScoreResult)

    @pytest.mark.asyncio
    async def test_evaluate_generation_pipeline(self, suite):
        """Test evaluating generation pipeline."""
        input_text = "What is machine learning?"
        output_text = "Machine learning is a subset of AI."
        expected_output = "Machine learning is an AI technique."
        context = "Machine learning is a method of data analysis."

        results = await suite.evaluate_generation_pipeline(
            input_text, output_text, expected_output, context
        )

        expected_metrics = [
            "answer_relevancy",
            "answer_similarity",
            "answer_correctness",
            "faithfulness",
        ]

        assert isinstance(results, dict)
        assert len(results) == len(expected_metrics)

        for metric in expected_metrics:
            assert metric in results
            assert isinstance(results[metric], ScoreResult)

    @pytest.mark.asyncio
    async def test_evaluate_comprehensive(self, suite):
        """Test comprehensive evaluation."""
        input_text = "What is machine learning?"
        output_text = "Machine learning is a subset of AI."
        expected_output = "Machine learning is an AI technique."
        context = "Machine learning is a method of data analysis."

        results = await suite.evaluate_comprehensive(
            input_text, output_text, expected_output, context, include_individual=True
        )

        # Should include all individual metrics plus composite metrics
        expected_count = 10  # 8 individual + 2 composite
        assert len(results) == expected_count

        # Check composite metrics are present
        assert "ragas" in results
        assert "rag_triad" in results

        for result in results.values():
            assert isinstance(result, ScoreResult)


class TestUtilityFunctions:
    """Test utility functions."""

    def test_create_rag_scorer(self):
        """Test creating RAG scorers using factory function."""
        mock_model = MockLLMModel()

        # Test creating different scorer types
        scorer_types = [
            "context_precision",
            "context_relevancy",
            "context_recall",
            "context_entity_recall",
            "answer_relevancy",
            "answer_similarity",
            "answer_correctness",
            "faithfulness",
            "ragas",
            "rag_triad",
        ]

        for scorer_type in scorer_types:
            scorer = create_rag_scorer(scorer_type, mock_model)
            assert scorer is not None
            assert hasattr(scorer, "evaluate")

    def test_create_rag_scorer_unknown_type(self):
        """Test creating RAG scorer with unknown type."""
        mock_model = MockLLMModel()

        with pytest.raises(ValueError) as exc_info:
            create_rag_scorer("unknown_scorer", mock_model)

        assert "Unknown RAG scorer type" in str(exc_info.value)

    def test_get_default_rag_config(self):
        """Test getting default RAG configuration."""
        config = get_default_rag_config()

        assert isinstance(config, RAGEvaluationConfig)
        assert config.embedding_model == "all-MiniLM-L6-v2"
        assert config.similarity_threshold == 0.7

    def test_get_optimized_rag_config_precision(self):
        """Test getting precision-optimized RAG configuration."""
        config = get_optimized_rag_config("precision")

        assert isinstance(config, RAGEvaluationConfig)
        assert config.faithfulness_threshold == 0.9
        assert config.answer_correctness_threshold == 0.9
        assert (
            config.ragas_weights["faithfulness"] == 0.3
        )  # Higher weight for faithfulness

    def test_get_optimized_rag_config_recall(self):
        """Test getting recall-optimized RAG configuration."""
        config = get_optimized_rag_config("recall")

        assert isinstance(config, RAGEvaluationConfig)
        assert config.recall_threshold == 0.8
        assert (
            config.ragas_weights["context_recall"] == 0.25
        )  # Higher weight for recall

    def test_get_optimized_rag_config_speed(self):
        """Test getting speed-optimized RAG configuration."""
        config = get_optimized_rag_config("speed")

        assert isinstance(config, RAGEvaluationConfig)
        assert config.ragas_weights["context_recall"] == 0.0  # Skip slower metrics
        assert config.ragas_weights["context_entity_recall"] == 0.0

    def test_get_optimized_rag_config_balanced(self):
        """Test getting balanced RAG configuration."""
        config = get_optimized_rag_config("balanced")

        assert isinstance(config, RAGEvaluationConfig)
        # Should be same as default
        default_config = get_default_rag_config()
        assert config.similarity_threshold == default_config.similarity_threshold


class TestErrorHandling:
    """Test error handling in RAG evaluation system."""

    @pytest.fixture
    def failing_model(self):
        """Create a model that fails on generate."""

        class FailingModel(LLMModel):
            async def generate(self, prompt: str, **kwargs) -> str:
                raise Exception("Model generation failed")

        return FailingModel()

    @pytest.mark.asyncio
    async def test_scorer_handles_model_failure(self, failing_model):
        """Test that scorers handle model failures gracefully."""
        scorer = ContextRelevancyScorer(failing_model)

        result = await scorer.evaluate(
            "What is AI?",
            "AI is artificial intelligence.",
            context="Artificial intelligence is a field of computer science.",
        )

        assert result.score == 0.0
        assert result.passed is False
        assert "error" in result.metadata

    @pytest.mark.asyncio
    async def test_ragas_scorer_handles_component_failures(self, failing_model):
        """Test that RAGAS scorer handles component failures."""
        scorer = EnhancedRAGASScorer(failing_model)

        result = await scorer.evaluate(
            "What is AI?",
            "AI is artificial intelligence.",
            expected_output="Artificial intelligence is machine intelligence.",
            context="AI is a field of computer science.",
        )

        assert isinstance(result, ScoreResult)
        assert "failed_components" in result.metadata
        # Should still return a result even with failures


class TestIntegrationScenarios:
    """Test realistic integration scenarios."""

    @pytest.fixture
    def realistic_model(self):
        """Create a model with realistic responses."""
        responses = {
            "relevance": '{"relevance_score": 4, "reasoning": "Content is highly relevant to the question"}',
            "similarity": '{"structural_similarity": 0.85, "reasoning": "Similar organization and flow"}',
            "questions": '{"questions": ["What is the definition?", "How does it work?", "What are applications?"]}',
            "claims": '{"factual_claims": ["AI is a field of computer science", "Machine learning is a subset of AI"], "numerical_claims": [], "temporal_claims": [], "relational_claims": ["ML is part of AI"], "opinion_claims": []}',
            "statements": '{"statements": ["AI enables machines to think", "Machine learning uses algorithms", "Deep learning uses neural networks"]}',
            "entities": '{"persons": ["Alan Turing"], "organizations": ["MIT", "Stanford"], "locations": ["Silicon Valley"], "dates": ["1950"], "numbers": ["70%"], "concepts": ["Intelligence", "Learning"], "other_entities": ["Computer"]}',
        }

        class RealisticModel(LLMModel):
            async def generate(self, prompt: str, **kwargs) -> str:
                prompt_lower = prompt.lower()
                if "relevance" in prompt_lower:
                    return responses["relevance"]
                elif "similarity" in prompt_lower:
                    return responses["similarity"]
                elif "questions" in prompt_lower:
                    return responses["questions"]
                elif "claims" in prompt_lower:
                    return responses["claims"]
                elif "statements" in prompt_lower:
                    return responses["statements"]
                elif "entities" in prompt_lower:
                    return responses["entities"]
                elif "supported" in prompt_lower:
                    return '{"status": "SUPPORTED", "confidence": 0.9, "category": "factual", "supporting_evidence": "Information found in context", "reasoning": "Claim is well supported"}'
                elif "present" in prompt_lower:
                    return '{"presence_status": "FULLY_PRESENT", "confidence": 0.95, "supporting_evidence": "Direct match found", "reasoning": "Information clearly present in context"}'
                elif "classification" in prompt_lower:
                    return '{"classification": "TRUE_POSITIVE", "confidence": 0.9, "reasoning": "Statement is factually accurate"}'
                else:
                    return "Comprehensive analysis completed successfully"

        return RealisticModel()

    @pytest.mark.asyncio
    async def test_realistic_rag_evaluation(self, realistic_model):
        """Test a realistic RAG evaluation scenario."""
        suite = RAGEvaluationSuite(realistic_model)

        # Realistic RAG scenario
        question = "What is artificial intelligence and how does machine learning relate to it?"

        generated_answer = """
        Artificial intelligence (AI) is a field of computer science that focuses on creating
        systems capable of performing tasks that typically require human intelligence.
        Machine learning is a subset of AI that enables computers to learn and improve
        from experience without being explicitly programmed. Deep learning, a subset of
        machine learning, uses neural networks with multiple layers to process data and
        make decisions.
        """

        expected_answer = """
        Artificial intelligence is the simulation of human intelligence in machines.
        Machine learning is an important branch of AI that allows systems to automatically
        learn and improve from experience. It uses algorithms to analyze data, identify
        patterns, and make predictions or decisions.
        """

        context = """
        Artificial intelligence (AI) is a branch of computer science that aims to create
        intelligent machines that can think and act like humans. The field was founded
        in 1956 by Alan Turing and other pioneers. Machine learning is a core component
        of AI that focuses on algorithms that can learn from and make predictions on data.
        Deep learning is a subset of machine learning that uses artificial neural networks
        with multiple layers. Major AI research centers include MIT, Stanford, and companies
        in Silicon Valley. Today, over 70% of businesses use some form of AI technology.
        """

        # Run comprehensive evaluation
        results = await suite.evaluate_comprehensive(
            question,
            generated_answer,
            expected_answer,
            context,
            include_individual=True,
        )

        # Verify all metrics were evaluated
        assert len(results) == 10  # 8 individual + 2 composite

        # Check that all results are valid
        for _metric_name, result in results.items():
            assert isinstance(result, ScoreResult)
            assert 0.0 <= result.score <= 1.0
            assert isinstance(result.passed, bool)
            assert len(result.reasoning) > 0
            assert isinstance(result.metadata, dict)

        # Check specific metrics have expected metadata
        assert "component_scores" in results["ragas"].metadata
        assert "triad_methodology" in results["rag_triad"].metadata

        # Verify retrieval metrics
        retrieval_metrics = [
            "context_precision",
            "context_relevancy",
            "context_recall",
            "context_entity_recall",
        ]
        for metric in retrieval_metrics:
            assert metric in results
            # These should have context-related metadata
            assert any(
                key in results[metric].metadata
                for key in [
                    "context_chunks_count",
                    "context_length",
                    "total_key_info",
                    "total_entities",
                ]
            )

        # Verify generation metrics
        generation_metrics = [
            "answer_relevancy",
            "answer_similarity",
            "answer_correctness",
            "faithfulness",
        ]
        for metric in generation_metrics:
            assert metric in results
            # These should have answer-related metadata
            if metric == "answer_relevancy":
                assert "approach" in results[metric].metadata
            elif metric == "answer_similarity":
                assert "semantic_similarity" in results[metric].metadata
            elif metric == "answer_correctness":
                assert "precision" in results[metric].metadata
            elif metric == "faithfulness":
                assert "total_claims" in results[metric].metadata


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])
