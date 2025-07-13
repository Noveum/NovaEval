"""
Comprehensive unit tests for RAG evaluation system.

This test suite provides thorough coverage of all RAG evaluation components
including edge cases, error handling, and integration scenarios.
"""

import asyncio
import os
import sys
from typing import Optional
from unittest.mock import AsyncMock, Mock, patch

import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), "../../src"))

from novaeval.scorers.base import ScoreResult
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


class MockLLMModel:
    """Mock LLM model for testing."""

    def __init__(
        self, response: Optional[str] = None, responses: Optional[list[str]] = None
    ):
        self.response = response or '{"score": 0.8, "reasoning": "Test reasoning"}'
        self.responses = responses or [self.response]
        self.call_count = 0

    async def generate(self, prompt: str, **kwargs) -> str:
        response = self.responses[self.call_count % len(self.responses)]
        self.call_count += 1
        return response


class TestRAGEvaluationConfig:
    """Test RAG evaluation configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = RAGEvaluationConfig()

        assert config.embedding_model == "all-MiniLM-L6-v2"
        assert config.similarity_threshold == 0.7
        assert config.faithfulness_threshold == 0.8
        assert config.relevancy_threshold == 0.7
        assert isinstance(config.ragas_weights, dict)
        assert len(config.ragas_weights) == 8

    def test_custom_config(self):
        """Test custom configuration values."""
        custom_weights = {
            "context_precision": 0.4,
            "context_recall": 0.3,
            "answer_relevancy": 0.2,
            "faithfulness": 0.1,
        }

        config = RAGEvaluationConfig(
            similarity_threshold=0.9,
            faithfulness_threshold=0.95,
            relevancy_threshold=0.85,
            ragas_weights=custom_weights,
        )

        assert config.similarity_threshold == 0.9
        assert config.faithfulness_threshold == 0.95
        assert config.relevancy_threshold == 0.85
        assert config.ragas_weights == custom_weights

    def test_config_validation(self):
        """Test configuration validation."""
        # Note: Current implementation doesn't have validation
        # This test is a placeholder for future validation logic
        pass


class TestContextEvaluationScorers:
    """Test context evaluation scorers."""

    @pytest.fixture
    def mock_model(self):
        return MockLLMModel()

    @pytest.fixture
    def config(self):
        return RAGEvaluationConfig()

    @pytest.mark.asyncio
    async def test_context_precision_scorer(self, mock_model, config):
        """Test context precision scorer."""
        scorer = ContextPrecisionScorer(mock_model, config)

        # Test normal evaluation
        result = await scorer.evaluate(
            input_text="What is machine learning?",
            output_text="ML is a subset of AI",
            context="Machine learning is a field of artificial intelligence",
        )

        assert isinstance(result, ScoreResult)
        assert 0.0 <= result.score <= 1.0
        assert isinstance(result.passed, bool)
        assert isinstance(result.reasoning, str)
        assert isinstance(result.metadata, dict)

    @pytest.mark.asyncio
    async def test_context_precision_edge_cases(self, config):
        """Test context precision scorer edge cases."""
        # Test with empty context
        mock_model = MockLLMModel('{"relevance_score": 0, "reasoning": "No context"}')
        scorer = ContextPrecisionScorer(mock_model, config)

        result = await scorer.evaluate(
            input_text="What is AI?",
            output_text="AI is artificial intelligence",
            context="",
        )

        assert result.score == 0.0
        assert not result.passed

    @pytest.mark.asyncio
    async def test_context_relevancy_scorer(self, mock_model, config):
        """Test context relevancy scorer."""
        scorer = ContextRelevancyScorer(mock_model, config)

        result = await scorer.evaluate(
            input_text="Explain photosynthesis",
            output_text="Photosynthesis converts light to energy",
            context="Plants use sunlight to create glucose through photosynthesis",
        )

        assert isinstance(result, ScoreResult)
        assert 0.0 <= result.score <= 1.0
        assert "irrelevant_percentage" in result.metadata

    @pytest.mark.asyncio
    async def test_context_recall_scorer(self, config):
        """Test context recall scorer."""
        # Mock model that returns structured response
        mock_responses = [
            '{"key_information": ["fact1", "fact2"], "coverage_score": 0.8}',
            '{"extracted_info": ["fact1"], "coverage": 0.5}',
        ]
        mock_model = MockLLMModel(responses=mock_responses)
        scorer = ContextRecallScorer(mock_model, config)

        result = await scorer.evaluate(
            input_text="What are renewable energy benefits?",
            output_text="Renewable energy reduces emissions",
            expected_output="Benefits include emission reduction and sustainability",
            context="Renewable energy provides environmental and economic benefits",
        )

        assert isinstance(result, ScoreResult)
        assert 0.0 <= result.score <= 1.0
        assert "key_information" in result.metadata

    @pytest.mark.asyncio
    async def test_context_entity_recall_scorer(self, config):
        """Test context entity recall scorer."""
        mock_model = MockLLMModel(
            '{"entities": ["Paris", "France"], "entity_types": {"Paris": "LOCATION", "France": "COUNTRY"}}'
        )
        scorer = ContextEntityRecallScorer(mock_model, config)

        result = await scorer.evaluate(
            input_text="Tell me about Paris",
            output_text="Paris is the capital of France",
            expected_output="Paris is France's capital city",
            context="Paris is the capital and largest city of France",
        )

        assert isinstance(result, ScoreResult)
        assert 0.0 <= result.score <= 1.0
        assert "all_entities" in result.metadata
        assert "total_entities" in result.metadata


class TestAnswerEvaluationScorers:
    """Test answer evaluation scorers."""

    @pytest.fixture
    def mock_model(self):
        return MockLLMModel()

    @pytest.fixture
    def config(self):
        return RAGEvaluationConfig()

    @pytest.mark.asyncio
    async def test_answer_relevancy_scorer(self, mock_model, config):
        """Test answer relevancy scorer."""
        scorer = AnswerRelevancyScorer(mock_model, config)

        result = await scorer.evaluate(
            input_text="How does photosynthesis work?",
            output_text="Photosynthesis converts sunlight into chemical energy using chlorophyll",
            context="Plants use photosynthesis to create glucose from CO2 and water",
        )

        assert isinstance(result, ScoreResult)
        assert 0.0 <= result.score <= 1.0
        assert "approach" in result.metadata

    @pytest.mark.asyncio
    async def test_answer_similarity_scorer(self, config):
        """Test answer similarity scorer."""
        # Mock embedding model
        with patch("sentence_transformers.SentenceTransformer") as mock_st:
            mock_st.return_value.encode.return_value = [
                [0.1, 0.2, 0.3],
                [0.15, 0.25, 0.35],
            ]

            scorer = AnswerSimilarityScorer(MockLLMModel(), config)

            result = await scorer.evaluate(
                input_text="What is AI?",
                output_text="AI is artificial intelligence",
                expected_output="Artificial intelligence is AI",
            )

            assert isinstance(result, ScoreResult)
            assert 0.0 <= result.score <= 1.0
            assert "semantic_similarity" in result.metadata
            assert "lexical_similarity" in result.metadata

    @pytest.mark.asyncio
    async def test_answer_correctness_scorer(self, config):
        """Test answer correctness scorer."""
        mock_responses = [
            '{"statements": ["AI is intelligence", "AI uses computers"], "accuracy": 0.9}',
            '{"verified_statements": 2, "total_statements": 2, "accuracy_score": 1.0}',
        ]
        mock_model = MockLLMModel(responses=mock_responses)
        scorer = AnswerCorrectnessScorer(mock_model, config)

        result = await scorer.evaluate(
            input_text="What is artificial intelligence?",
            output_text="AI is machine intelligence used in computers",
            expected_output="Artificial intelligence is intelligence demonstrated by machines",
            context="AI refers to intelligence exhibited by machines",
        )

        assert isinstance(result, ScoreResult)
        assert 0.0 <= result.score <= 1.0
        assert "f1_score" in result.metadata

    @pytest.mark.asyncio
    async def test_enhanced_faithfulness_scorer(self, config):
        """Test enhanced faithfulness scorer."""
        mock_responses = [
            '{"claims": [{"claim": "AI uses computers", "supported": true, "category": "factual"}], "faithfulness_score": 0.95}',
            '{"verification_results": [{"claim": "test", "is_supported": true}], "overall_faithfulness": 0.9}',
        ]
        mock_model = MockLLMModel(responses=mock_responses)
        scorer = EnhancedFaithfulnessScorer(mock_model, config)

        result = await scorer.evaluate(
            input_text="Explain AI",
            output_text="AI involves computer systems that can perform tasks requiring intelligence",
            context="Artificial intelligence systems use computers to simulate human intelligence",
        )

        assert isinstance(result, ScoreResult)
        assert 0.0 <= result.score <= 1.0
        assert "claims" in result.metadata


class TestCompositeScorers:
    """Test composite scorers."""

    @pytest.fixture
    def mock_model(self):
        return MockLLMModel()

    @pytest.fixture
    def config(self):
        return RAGEvaluationConfig()

    @pytest.mark.asyncio
    async def test_enhanced_ragas_scorer(self, config):
        """Test enhanced RAGAS scorer."""
        # Mock individual scorer results
        mock_responses = [
            '{"precision_score": 0.8, "reasoning": "Good precision"}',
            '{"recall_score": 0.7, "reasoning": "Good recall"}',
            '{"relevancy_score": 0.9, "reasoning": "Highly relevant"}',
            '{"faithfulness_score": 0.85, "reasoning": "Mostly faithful"}',
        ]
        mock_model = MockLLMModel(responses=mock_responses)
        scorer = EnhancedRAGASScorer(mock_model, config)

        result = await scorer.evaluate(
            input_text="What is machine learning?",
            output_text="Machine learning is a subset of AI that learns from data",
            expected_output="ML is AI that learns patterns from data",
            context="Machine learning uses algorithms to learn from data without explicit programming",
        )

        assert isinstance(result, ScoreResult)
        assert 0.0 <= result.score <= 1.0
        assert "component_scores" in result.metadata
        assert "weighted_average" in result.metadata
        assert len(result.metadata["component_scores"]) == 4

    @pytest.mark.asyncio
    async def test_rag_triad_scorer(self, config):
        """Test RAG triad scorer."""
        mock_responses = [
            '{"relevancy": 0.8}',
            '{"faithfulness": 0.9}',
            '{"answer_relevancy": 0.85}',
        ]
        mock_model = MockLLMModel(responses=mock_responses)
        scorer = RAGTriadScorer(mock_model, config)

        result = await scorer.evaluate(
            input_text="Explain photosynthesis",
            output_text="Photosynthesis converts light energy to chemical energy in plants",
            expected_output="Plants convert sunlight to energy through photosynthesis",
            context="Photosynthesis is the process plants use to convert light into glucose",
        )

        assert isinstance(result, ScoreResult)
        assert 0.0 <= result.score <= 1.0
        assert "triad_scores" in result.metadata
        assert len(result.metadata["triad_scores"]) == 3


class TestRAGEvaluationSuite:
    """Test RAG evaluation suite."""

    @pytest.fixture
    def mock_model(self):
        return MockLLMModel()

    @pytest.fixture
    def config(self):
        return RAGEvaluationConfig()

    @pytest.fixture
    def suite(self, mock_model, config):
        return RAGEvaluationSuite(mock_model, config)

    @pytest.mark.asyncio
    async def test_evaluate_single_metric(self, suite):
        """Test single metric evaluation."""
        result = await suite.evaluate_single_metric(
            "answer_relevancy",
            "What is AI?",
            "AI is artificial intelligence",
            context="Artificial intelligence refers to machine intelligence",
        )

        assert isinstance(result, ScoreResult)
        assert 0.0 <= result.score <= 1.0

    @pytest.mark.asyncio
    async def test_evaluate_retrieval_pipeline(self, suite):
        """Test retrieval pipeline evaluation."""
        results = await suite.evaluate_retrieval_pipeline(
            "What is machine learning?",
            "ML is a subset of AI",
            "Machine learning uses algorithms to learn from data",
        )

        assert isinstance(results, dict)
        assert len(results) >= 3  # Should have context metrics

        # Check that all results are ScoreResult objects
        for _metric_name, result in results.items():
            assert isinstance(result, ScoreResult)
            assert 0.0 <= result.score <= 1.0

    @pytest.mark.asyncio
    async def test_evaluate_generation_pipeline(self, suite):
        """Test generation pipeline evaluation."""
        results = await suite.evaluate_generation_pipeline(
            "Explain photosynthesis",
            "Photosynthesis converts light to energy",
            "Plants convert sunlight to glucose through photosynthesis",
        )

        assert isinstance(results, dict)
        assert len(results) >= 3  # Should have answer metrics

        for _metric_name, result in results.items():
            assert isinstance(result, ScoreResult)
            assert 0.0 <= result.score <= 1.0

    @pytest.mark.asyncio
    async def test_evaluate_comprehensive(self, suite):
        """Test comprehensive evaluation."""
        results = await suite.evaluate_comprehensive(
            "What are the benefits of renewable energy?",
            "Renewable energy reduces emissions and provides sustainability",
            "Benefits include environmental protection and energy independence",
            "Renewable energy sources like solar and wind provide clean power",
        )

        assert isinstance(results, dict)
        assert len(results) >= 6  # Should have both context and answer metrics

        # Check for expected metrics
        expected_metrics = [
            "context_relevancy",
            "answer_relevancy",
            "faithfulness",
            "answer_correctness",
            "ragas",
        ]

        for metric in expected_metrics:
            assert metric in results
            assert isinstance(results[metric], ScoreResult)

    @pytest.mark.asyncio
    async def test_get_all_available_metrics(self, suite):
        """Test getting all available metrics."""
        metrics = suite.get_all_available_metrics()

        assert isinstance(metrics, list)
        assert len(metrics) > 0

        # Check for core metrics
        expected_core_metrics = [
            "context_precision",
            "context_relevancy",
            "context_recall",
            "answer_relevancy",
            "answer_similarity",
            "faithfulness",
        ]

        for metric in expected_core_metrics:
            assert metric in metrics

    def test_suite_initialization_with_defaults(self, mock_model):
        """Test suite initialization with default config."""
        suite = RAGEvaluationSuite(mock_model)

        assert suite.model == mock_model
        assert isinstance(suite.config, RAGEvaluationConfig)
        assert suite.config.similarity_threshold == 0.7  # Default value


class TestConfigurationHelpers:
    """Test configuration helper functions."""

    def test_get_rag_config_balanced(self):
        """Test balanced configuration."""
        config = get_default_rag_config()

        assert isinstance(config, RAGEvaluationConfig)
        assert config.similarity_threshold == 0.7
        assert config.faithfulness_threshold == 0.8
        assert config.relevancy_threshold == 0.7

    def test_get_rag_config_precision(self):
        """Test precision-focused configuration."""
        config = get_optimized_rag_config("precision")

        assert isinstance(config, RAGEvaluationConfig)
        assert config.faithfulness_threshold >= 0.8
        assert config.answer_correctness_threshold >= 0.8
        assert config.precision_threshold >= 0.7

    def test_get_rag_config_recall(self):
        """Test recall-focused configuration."""
        config = get_optimized_rag_config("recall")

        assert isinstance(config, RAGEvaluationConfig)
        assert config.recall_threshold >= 0.7
        assert config.relevancy_threshold <= 0.7

    def test_get_rag_config_speed(self):
        """Test speed-optimized configuration."""
        config = get_optimized_rag_config("speed")

        assert isinstance(config, RAGEvaluationConfig)
        # Speed config should use faster embedding model
        assert config.embedding_model == "all-MiniLM-L6-v2"

    def test_get_rag_config_invalid(self):
        """Test invalid configuration name."""
        with pytest.raises(ValueError):
            get_optimized_rag_config("invalid_config")


class TestScorerFactory:
    """Test scorer factory functions."""

    @pytest.fixture
    def mock_model(self):
        return MockLLMModel()

    @pytest.fixture
    def config(self):
        return RAGEvaluationConfig()

    def test_create_rag_scorer_valid_types(self, mock_model, config):
        """Test creating valid scorer types."""
        valid_types = [
            "context_precision",
            "context_relevancy",
            "context_recall",
            "answer_relevancy",
            "answer_similarity",
            "faithfulness",
            "answer_correctness",
            "ragas",
            "rag_triad",
        ]

        for scorer_type in valid_types:
            scorer = create_rag_scorer(scorer_type, mock_model, config)
            assert scorer is not None
            assert hasattr(scorer, "evaluate")

    def test_create_rag_scorer_invalid_type(self, mock_model, config):
        """Test creating invalid scorer type."""
        with pytest.raises(ValueError):
            create_rag_scorer("invalid_scorer", mock_model, config)

    def test_create_rag_scorer_with_defaults(self, mock_model):
        """Test creating scorer with default config."""
        scorer = create_rag_scorer("answer_relevancy", mock_model)
        assert scorer is not None
        assert hasattr(scorer, "config")
        assert isinstance(scorer.config, RAGEvaluationConfig)


class TestErrorHandling:
    """Test error handling and edge cases."""

    @pytest.fixture
    def mock_model(self):
        return MockLLMModel()

    @pytest.fixture
    def config(self):
        return RAGEvaluationConfig()

    @pytest.mark.asyncio
    async def test_empty_inputs(self, mock_model, config):
        """Test handling of empty inputs."""
        scorer = AnswerRelevancyScorer(mock_model, config)

        # Test empty question
        result = await scorer.evaluate("", "Some answer", context="Some context")
        assert isinstance(result, ScoreResult)
        assert result.score <= 0.5  # Should be low for empty question

        # Test empty answer
        result = await scorer.evaluate("Some question", "", context="Some context")
        assert isinstance(result, ScoreResult)
        assert result.score <= 0.5  # Should be low for empty answer

    @pytest.mark.asyncio
    async def test_very_long_inputs(self, mock_model, config):
        """Test handling of very long inputs."""
        scorer = AnswerRelevancyScorer(mock_model, config)

        long_text = "This is a very long text. " * 1000  # Very long input

        result = await scorer.evaluate(
            "What is this about?", long_text, context="Some context"
        )

        assert isinstance(result, ScoreResult)
        assert 0.0 <= result.score <= 1.0

    @pytest.mark.asyncio
    async def test_special_characters(self, mock_model, config):
        """Test handling of special characters and unicode."""
        scorer = AnswerRelevancyScorer(mock_model, config)

        special_text = "Test with émojis 🚀, symbols ∑∆, and quotes 'smart quotes'"

        result = await scorer.evaluate(special_text, special_text, context=special_text)

        assert isinstance(result, ScoreResult)
        assert 0.0 <= result.score <= 1.0

    @pytest.mark.asyncio
    async def test_model_error_handling(self, config):
        """Test handling of model errors."""
        # Mock model that raises an exception
        error_model = Mock()
        error_model.generate = AsyncMock(side_effect=Exception("Model error"))

        scorer = AnswerRelevancyScorer(error_model, config)

        result = await scorer.evaluate(
            "Test question", "Test answer", context="Test context"
        )

        # Should handle error gracefully
        assert isinstance(result, ScoreResult)
        assert result.score == 0.0
        assert not result.passed
        assert "error" in result.reasoning.lower()

    @pytest.mark.asyncio
    async def test_invalid_json_response(self, config):
        """Test handling of invalid JSON responses from model."""
        invalid_json_model = MockLLMModel("This is not valid JSON")
        scorer = AnswerRelevancyScorer(invalid_json_model, config)

        result = await scorer.evaluate(
            "Test question", "Test answer", context="Test context"
        )

        # Should handle invalid JSON gracefully
        assert isinstance(result, ScoreResult)
        assert 0.0 <= result.score <= 1.0


class TestPerformanceAndConcurrency:
    """Test performance and concurrency aspects."""

    @pytest.fixture
    def mock_model(self):
        return MockLLMModel()

    @pytest.fixture
    def suite(self, mock_model):
        return RAGEvaluationSuite(mock_model)

    @pytest.mark.asyncio
    async def test_concurrent_evaluations(self, suite):
        """Test concurrent evaluation execution."""
        # Create multiple evaluation tasks
        tasks = []
        for i in range(5):
            task = suite.evaluate_single_metric(
                "answer_relevancy",
                f"Question {i}",
                f"Answer {i}",
                context=f"Context {i}",
            )
            tasks.append(task)

        # Execute concurrently
        results = await asyncio.gather(*tasks)

        # Verify all results
        assert len(results) == 5
        for result in results:
            assert isinstance(result, ScoreResult)
            assert 0.0 <= result.score <= 1.0

    @pytest.mark.asyncio
    async def test_batch_evaluation_performance(self, suite):
        """Test batch evaluation performance."""
        # Create test data
        test_cases = [
            {
                "question": f"Question {i}",
                "answer": f"Answer {i}",
                "expected": f"Expected {i}",
                "context": f"Context {i}",
            }
            for i in range(10)
        ]

        # Measure sequential vs parallel execution
        import time

        # Sequential execution
        start_time = time.time()
        sequential_results = []
        for case in test_cases:
            result = await suite.evaluate_comprehensive(
                case["question"], case["answer"], case["expected"], case["context"]
            )
            sequential_results.append(result)
        sequential_time = time.time() - start_time

        # Parallel execution
        start_time = time.time()
        parallel_tasks = [
            suite.evaluate_comprehensive(
                case["question"], case["answer"], case["expected"], case["context"]
            )
            for case in test_cases
        ]
        parallel_results = await asyncio.gather(*parallel_tasks)
        parallel_time = time.time() - start_time

        # Verify results are equivalent
        assert len(sequential_results) == len(parallel_results)

        # Parallel should be faster (or at least not significantly slower)
        # Allow some tolerance for test environment variations
        assert parallel_time <= sequential_time * 1.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
