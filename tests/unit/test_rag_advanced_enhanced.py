"""
Comprehensive unit tests for advanced RAG evaluation system.

This test suite covers all advanced safety metrics, conversational evaluation,
and comprehensive evaluation functionality with thorough edge case testing.
"""

import asyncio
import os
import sys
from typing import List
from unittest.mock import AsyncMock, Mock

import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), "../../src"))

from novaeval.scorers.base import ScoreResult
from novaeval.scorers.rag_advanced import (
    AdvancedRAGConfig,
    BiasDetectionScorer,
    ComprehensiveRAGEvaluationSuite,
    ConversationCoherenceScorer,
    HallucinationDetectionScorer,
    ToxicityDetectionScorer,
    create_comprehensive_rag_scorer,
    get_advanced_rag_config,
)
from novaeval.scorers.rag_comprehensive import (
    get_default_rag_config,
)


class MockLLMModel:
    """Mock LLM model for testing advanced features."""

    def __init__(self, response: str = None, responses: List[str] = None):
        self.response = (
            response
            or '{"score": 0.8, "reasoning": "Test reasoning", "confidence": 0.9}'
        )
        self.responses = responses or [self.response]
        self.call_count = 0

    async def generate(self, prompt: str, **kwargs) -> str:
        response = self.responses[self.call_count % len(self.responses)]
        self.call_count += 1
        return response


class TestAdvancedRAGConfig:
    """Test advanced RAG configuration."""

    def test_default_advanced_config(self):
        """Test default advanced configuration values."""
        config = AdvancedRAGConfig()

        assert config.hallucination_threshold == 0.8
        assert config.bias_threshold == 0.3
        assert config.toxicity_threshold == 0.2
        assert isinstance(config.bias_categories, list)
        assert len(config.bias_categories) >= 5
        assert isinstance(config.toxicity_severity_levels, list)
        assert len(config.toxicity_severity_levels) >= 3

    def test_custom_advanced_config(self):
        """Test custom advanced configuration values."""
        custom_bias_categories = ["gender", "race", "age"]
        custom_toxicity_levels = ["mild", "moderate", "severe"]

        config = AdvancedRAGConfig(
            hallucination_threshold=0.95,
            bias_threshold=0.1,
            toxicity_threshold=0.05,
            bias_categories=custom_bias_categories,
            toxicity_severity_levels=custom_toxicity_levels,
        )

        assert config.hallucination_threshold == 0.95
        assert config.bias_threshold == 0.1
        assert config.toxicity_threshold == 0.05
        assert config.bias_categories == custom_bias_categories
        assert config.toxicity_severity_levels == custom_toxicity_levels

    def test_advanced_config_validation(self):
        """Test advanced configuration validation."""
        # Test invalid threshold values
        with pytest.raises(ValueError):
            AdvancedRAGConfig(hallucination_threshold=1.5)

        with pytest.raises(ValueError):
            AdvancedRAGConfig(bias_threshold=-0.1)

        with pytest.raises(ValueError):
            AdvancedRAGConfig(toxicity_threshold=2.0)

        # Test empty categories
        with pytest.raises(ValueError):
            AdvancedRAGConfig(bias_categories=[])

        with pytest.raises(ValueError):
            AdvancedRAGConfig(toxicity_severity_levels=[])


class TestHallucinationDetectionScorer:
    """Test hallucination detection scorer."""

    @pytest.fixture
    def mock_model(self):
        return MockLLMModel()

    @pytest.fixture
    def config(self):
        return AdvancedRAGConfig()

    @pytest.mark.asyncio
    async def test_hallucination_detection_basic(self, mock_model, config):
        """Test basic hallucination detection."""
        scorer = HallucinationDetectionScorer(mock_model, config)

        result = await scorer.evaluate(
            input_text="What is the population of Paris?",
            output_text="Paris has a population of 2.1 million people.",
            context="Paris is the capital of France with approximately 2.1 million residents.",
        )

        assert isinstance(result, ScoreResult)
        assert 0.0 <= result.score <= 1.0
        assert isinstance(result.passed, bool)
        assert isinstance(result.reasoning, str)
        assert "confidence" in result.metadata

    @pytest.mark.asyncio
    async def test_hallucination_detection_with_hallucinations(self, config):
        """Test detection of actual hallucinations."""
        # Mock model that detects hallucinations
        halluc_response = """
        {
            "verification_results": [
                {"claim": "Paris has 50 million people", "is_hallucination": true, "category": "numerical", "confidence": 0.95},
                {"claim": "Paris was founded in 1850", "is_hallucination": true, "category": "temporal", "confidence": 0.9}
            ],
            "hallucination_score": 0.1,
            "confidence": 0.92,
            "hallucination_count": 2
        }
        """
        mock_model = MockLLMModel(halluc_response)
        scorer = HallucinationDetectionScorer(mock_model, config)

        result = await scorer.evaluate(
            input_text="Tell me about Paris.",
            output_text="Paris has 50 million people and was founded in 1850.",
            context="Paris is the capital of France with about 2.1 million residents, founded in the 3rd century BC.",
        )

        assert result.score <= config.hallucination_threshold
        assert not result.passed
        assert "verification_results" in result.metadata
        assert result.metadata["hallucination_count"] == 2

    @pytest.mark.asyncio
    async def test_hallucination_categories(self, config):
        """Test different hallucination categories."""
        categories_response = """
        {
            "verification_results": [
                {"claim": "Test factual", "is_hallucination": false, "category": "factual"},
                {"claim": "Test numerical", "is_hallucination": true, "category": "numerical"},
                {"claim": "Test temporal", "is_hallucination": true, "category": "temporal"},
                {"claim": "Test entity", "is_hallucination": false, "category": "entity"},
                {"claim": "Test relational", "is_hallucination": true, "category": "relational"}
            ],
            "hallucination_score": 0.4,
            "confidence": 0.85,
            "category_breakdown": {
                "factual": {"total": 1, "hallucinations": 0},
                "numerical": {"total": 1, "hallucinations": 1},
                "temporal": {"total": 1, "hallucinations": 1},
                "entity": {"total": 1, "hallucinations": 0},
                "relational": {"total": 1, "hallucinations": 1}
            }
        }
        """
        mock_model = MockLLMModel(categories_response)
        scorer = HallucinationDetectionScorer(mock_model, config)

        result = await scorer.evaluate(
            "Test question", "Test answer with various claims", context="Test context"
        )

        assert "category_breakdown" in result.metadata
        assert "numerical" in result.metadata["category_breakdown"]
        assert "temporal" in result.metadata["category_breakdown"]

    @pytest.mark.asyncio
    async def test_hallucination_no_context(self, mock_model, config):
        """Test hallucination detection without context."""
        scorer = HallucinationDetectionScorer(mock_model, config)

        result = await scorer.evaluate(
            input_text="What is AI?",
            output_text="AI is artificial intelligence.",
            context="",  # No context provided
        )

        # Should handle gracefully but may have lower confidence
        assert isinstance(result, ScoreResult)
        assert 0.0 <= result.score <= 1.0


class TestBiasDetectionScorer:
    """Test bias detection scorer."""

    @pytest.fixture
    def mock_model(self):
        return MockLLMModel()

    @pytest.fixture
    def config(self):
        return AdvancedRAGConfig()

    @pytest.mark.asyncio
    async def test_bias_detection_basic(self, mock_model, config):
        """Test basic bias detection."""
        scorer = BiasDetectionScorer(mock_model, config)

        result = await scorer.evaluate(
            input_text="What makes a good employee?",
            output_text="Good employees are skilled, reliable, and communicate effectively.",
        )

        assert isinstance(result, ScoreResult)
        assert 0.0 <= result.score <= 1.0
        assert isinstance(result.passed, bool)
        assert "confidence" in result.metadata

    @pytest.mark.asyncio
    async def test_bias_detection_with_bias(self, config):
        """Test detection of actual bias."""
        bias_response = """
        {
            "detected_biases": [
                {"category": "gender", "evidence": "men are better at math", "severity": "moderate", "confidence": 0.9},
                {"category": "age", "evidence": "young people are more innovative", "severity": "mild", "confidence": 0.7}
            ],
            "bias_score": 0.2,
            "overall_confidence": 0.85,
            "bias_categories_found": ["gender", "age"]
        }
        """
        mock_model = MockLLMModel(bias_response)
        scorer = BiasDetectionScorer(mock_model, config)

        result = await scorer.evaluate(
            input_text="Who is better at math?",
            output_text="Men are naturally better at math, and young people are more innovative.",
        )

        assert result.score <= config.bias_threshold
        assert not result.passed
        assert "detected_biases" in result.metadata
        assert len(result.metadata["detected_biases"]) == 2
        assert "gender" in result.metadata["bias_categories_found"]

    @pytest.mark.asyncio
    async def test_bias_categories_coverage(self, config):
        """Test coverage of different bias categories."""
        comprehensive_bias_response = """
        {
            "detected_biases": [
                {"category": "gender", "evidence": "test", "severity": "mild"},
                {"category": "race", "evidence": "test", "severity": "moderate"},
                {"category": "religion", "evidence": "test", "severity": "mild"},
                {"category": "political", "evidence": "test", "severity": "severe"},
                {"category": "age", "evidence": "test", "severity": "mild"},
                {"category": "disability", "evidence": "test", "severity": "moderate"},
                {"category": "socioeconomic", "evidence": "test", "severity": "mild"},
                {"category": "nationality", "evidence": "test", "severity": "mild"},
                {"category": "appearance", "evidence": "test", "severity": "mild"}
            ],
            "bias_score": 0.1,
            "overall_confidence": 0.8,
            "category_analysis": {
                "total_categories_checked": 9,
                "categories_with_bias": 9,
                "severity_distribution": {"mild": 6, "moderate": 2, "severe": 1}
            }
        }
        """
        mock_model = MockLLMModel(comprehensive_bias_response)
        scorer = BiasDetectionScorer(mock_model, config)

        result = await scorer.evaluate(
            "Test question about groups", "Test answer with various biases"
        )

        assert "category_analysis" in result.metadata
        assert result.metadata["category_analysis"]["total_categories_checked"] == 9
        assert "severity_distribution" in result.metadata["category_analysis"]

    @pytest.mark.asyncio
    async def test_bias_severity_levels(self, config):
        """Test different bias severity levels."""
        severity_response = """
        {
            "detected_biases": [
                {"category": "gender", "evidence": "subtle bias", "severity": "mild", "confidence": 0.6},
                {"category": "race", "evidence": "clear bias", "severity": "moderate", "confidence": 0.8},
                {"category": "religion", "evidence": "strong bias", "severity": "severe", "confidence": 0.95}
            ],
            "bias_score": 0.05,
            "severity_impact": {"mild": 0.1, "moderate": 0.3, "severe": 0.6}
        }
        """
        mock_model = MockLLMModel(severity_response)
        scorer = BiasDetectionScorer(mock_model, config)

        result = await scorer.evaluate(
            "Test question", "Test answer with varying bias severities"
        )

        assert "severity_impact" in result.metadata
        assert (
            result.metadata["severity_impact"]["severe"]
            > result.metadata["severity_impact"]["mild"]
        )


class TestToxicityDetectionScorer:
    """Test toxicity detection scorer."""

    @pytest.fixture
    def mock_model(self):
        return MockLLMModel()

    @pytest.fixture
    def config(self):
        return AdvancedRAGConfig()

    @pytest.mark.asyncio
    async def test_toxicity_detection_basic(self, mock_model, config):
        """Test basic toxicity detection."""
        scorer = ToxicityDetectionScorer(mock_model, config)

        result = await scorer.evaluate(
            input_text="How do I stay healthy?",
            output_text="Eat well, exercise regularly, and get enough sleep.",
        )

        assert isinstance(result, ScoreResult)
        assert 0.0 <= result.score <= 1.0
        assert isinstance(result.passed, bool)
        assert "confidence" in result.metadata

    @pytest.mark.asyncio
    async def test_toxicity_detection_with_toxicity(self, config):
        """Test detection of toxic content."""
        toxic_response = """
        {
            "detected_issues": [
                {"category": "offensive_language", "evidence": "inappropriate words", "severity": "moderate", "confidence": 0.9},
                {"category": "harassment", "evidence": "threatening language", "severity": "severe", "confidence": 0.95}
            ],
            "toxicity_score": 0.1,
            "overall_confidence": 0.92,
            "severity_breakdown": {"moderate": 1, "severe": 1}
        }
        """
        mock_model = MockLLMModel(toxic_response)
        scorer = ToxicityDetectionScorer(mock_model, config)

        result = await scorer.evaluate(
            input_text="Test question",
            output_text="Test answer with inappropriate content",
        )

        assert result.score <= config.toxicity_threshold
        assert not result.passed
        assert "detected_issues" in result.metadata
        assert len(result.metadata["detected_issues"]) == 2

    @pytest.mark.asyncio
    async def test_toxicity_categories(self, config):
        """Test different toxicity categories."""
        categories_response = """
        {
            "detected_issues": [
                {"category": "offensive_language", "severity": "mild"},
                {"category": "hate_speech", "severity": "severe"},
                {"category": "harassment", "severity": "moderate"},
                {"category": "threats", "severity": "severe"},
                {"category": "discrimination", "severity": "moderate"},
                {"category": "explicit_content", "severity": "mild"},
                {"category": "misinformation", "severity": "moderate"}
            ],
            "toxicity_score": 0.05,
            "category_coverage": {
                "total_categories": 7,
                "categories_detected": 7,
                "highest_severity": "severe"
            }
        }
        """
        mock_model = MockLLMModel(categories_response)
        scorer = ToxicityDetectionScorer(mock_model, config)

        result = await scorer.evaluate(
            "Test question", "Test answer with various toxic elements"
        )

        assert "category_coverage" in result.metadata
        assert result.metadata["category_coverage"]["total_categories"] == 7
        assert result.metadata["category_coverage"]["highest_severity"] == "severe"


class TestConversationCoherenceScorer:
    """Test conversation coherence scorer."""

    @pytest.fixture
    def mock_model(self):
        return MockLLMModel()

    @pytest.fixture
    def config(self):
        return AdvancedRAGConfig()

    @pytest.mark.asyncio
    async def test_conversation_coherence_basic(self, mock_model, config):
        """Test basic conversation coherence."""
        scorer = ConversationCoherenceScorer(mock_model, config)

        conversation_history = [
            {"role": "user", "content": "What is machine learning?"},
            {
                "role": "assistant",
                "content": "ML is a subset of AI that learns from data.",
            },
        ]

        result = await scorer.evaluate(
            input_text="How does it work?",
            output_text="ML algorithms identify patterns in data to make predictions.",
            conversation_history=conversation_history,
        )

        assert isinstance(result, ScoreResult)
        assert 0.0 <= result.score <= 1.0
        assert "conversation_length" in result.metadata

    @pytest.mark.asyncio
    async def test_conversation_coherence_with_context_switch(self, config):
        """Test coherence with context switching."""
        context_switch_response = """
        {
            "coherence_score": 0.3,
            "context_switch_detected": true,
            "topic_consistency": 0.2,
            "context_maintenance": 0.4,
            "transition_quality": 0.1,
            "analysis": "Abrupt topic change detected"
        }
        """
        mock_model = MockLLMModel(context_switch_response)
        scorer = ConversationCoherenceScorer(mock_model, config)

        conversation_history = [
            {"role": "user", "content": "Tell me about machine learning."},
            {
                "role": "assistant",
                "content": "ML is about algorithms learning from data.",
            },
            {"role": "user", "content": "Actually, let's talk about cooking pasta."},
        ]

        result = await scorer.evaluate(
            input_text="Actually, let's talk about cooking pasta.",
            output_text="To cook pasta, boil water and add the pasta.",
            conversation_history=conversation_history,
        )

        assert result.metadata["context_switch_detected"] == True
        assert result.metadata["topic_consistency"] < 0.5
        assert "transition_quality" in result.metadata

    @pytest.mark.asyncio
    async def test_conversation_coherence_long_conversation(self, config):
        """Test coherence in long conversations."""
        long_conv_response = """
        {
            "coherence_score": 0.85,
            "conversation_length": 10,
            "topic_consistency": 0.9,
            "context_maintenance": 0.8,
            "coherence_trend": "stable",
            "turn_by_turn_analysis": [0.9, 0.85, 0.8, 0.85, 0.9]
        }
        """
        mock_model = MockLLMModel(long_conv_response)
        scorer = ConversationCoherenceScorer(mock_model, config)

        # Create long conversation history
        conversation_history = []
        for i in range(10):
            conversation_history.extend(
                [
                    {"role": "user", "content": f"Question {i} about AI"},
                    {"role": "assistant", "content": f"Answer {i} about AI"},
                ]
            )

        result = await scorer.evaluate(
            input_text="Final question about AI",
            output_text="Final comprehensive answer about AI",
            conversation_history=conversation_history,
        )

        assert result.metadata["conversation_length"] == 10
        assert "coherence_trend" in result.metadata
        assert "turn_by_turn_analysis" in result.metadata

    @pytest.mark.asyncio
    async def test_conversation_coherence_no_history(self, mock_model, config):
        """Test coherence with no conversation history."""
        scorer = ConversationCoherenceScorer(mock_model, config)

        result = await scorer.evaluate(
            input_text="What is AI?",
            output_text="AI is artificial intelligence.",
            conversation_history=[],
        )

        # Should handle gracefully
        assert isinstance(result, ScoreResult)
        assert result.metadata["conversation_length"] == 0


class TestComprehensiveRAGEvaluationSuite:
    """Test comprehensive RAG evaluation suite."""

    @pytest.fixture
    def mock_model(self):
        return MockLLMModel()

    @pytest.fixture
    def rag_config(self):
        return get_default_rag_config()

    @pytest.fixture
    def advanced_config(self):
        return get_advanced_rag_config("balanced")

    @pytest.fixture
    def suite(self, mock_model, rag_config, advanced_config):
        return ComprehensiveRAGEvaluationSuite(mock_model, rag_config, advanced_config)

    @pytest.mark.asyncio
    async def test_evaluate_comprehensive_plus_core_only(self, suite):
        """Test comprehensive evaluation with core metrics only."""
        results = await suite.evaluate_comprehensive_plus(
            "What is AI?",
            "AI is artificial intelligence",
            expected_output="Artificial intelligence is AI",
            context="AI refers to machine intelligence",
            include_safety_metrics=False,
            include_conversational_metrics=False,
        )

        assert isinstance(results, dict)
        assert len(results) >= 4  # Should have core RAG metrics

        # Should not have safety or conversational metrics
        safety_metrics = [
            "hallucination_detection",
            "bias_detection",
            "toxicity_detection",
        ]
        conv_metrics = ["conversation_coherence"]

        for metric in safety_metrics + conv_metrics:
            assert metric not in results

    @pytest.mark.asyncio
    async def test_evaluate_comprehensive_plus_with_safety(self, suite):
        """Test comprehensive evaluation with safety metrics."""
        results = await suite.evaluate_comprehensive_plus(
            "What is machine learning?",
            "ML is a subset of AI that learns from data",
            context="Machine learning uses algorithms to learn patterns",
            include_safety_metrics=True,
            include_conversational_metrics=False,
        )

        assert isinstance(results, dict)

        # Should have both core and safety metrics
        expected_safety_metrics = [
            "hallucination_detection",
            "bias_detection",
            "toxicity_detection",
        ]
        for metric in expected_safety_metrics:
            assert metric in results
            assert isinstance(results[metric], ScoreResult)

    @pytest.mark.asyncio
    async def test_evaluate_comprehensive_plus_with_conversation(self, suite):
        """Test comprehensive evaluation with conversational metrics."""
        conversation_history = [
            {"role": "user", "content": "What is AI?"},
            {"role": "assistant", "content": "AI is artificial intelligence"},
        ]

        results = await suite.evaluate_comprehensive_plus(
            "How does AI work?",
            "AI works by processing data and making decisions",
            context="AI systems process information to make intelligent decisions",
            include_safety_metrics=False,
            include_conversational_metrics=True,
            conversation_history=conversation_history,
        )

        assert isinstance(results, dict)

        # Should have conversational metrics
        assert "conversation_coherence" in results
        assert isinstance(results["conversation_coherence"], ScoreResult)

    @pytest.mark.asyncio
    async def test_evaluate_comprehensive_plus_full(self, suite):
        """Test full comprehensive evaluation with all metrics."""
        conversation_history = [
            {"role": "user", "content": "Tell me about renewable energy"},
            {
                "role": "assistant",
                "content": "Renewable energy comes from natural sources",
            },
        ]

        results = await suite.evaluate_comprehensive_plus(
            "What are the benefits?",
            "Benefits include environmental protection and energy independence",
            expected_output="Renewable energy provides environmental and economic benefits",
            context="Renewable energy offers sustainability and reduces emissions",
            include_safety_metrics=True,
            include_conversational_metrics=True,
            conversation_history=conversation_history,
        )

        assert isinstance(results, dict)
        assert len(results) >= 8  # Should have core + safety + conversational metrics

        # Check for all expected metric categories
        core_metrics = ["answer_relevancy", "faithfulness"]
        safety_metrics = [
            "hallucination_detection",
            "bias_detection",
            "toxicity_detection",
        ]
        conv_metrics = ["conversation_coherence"]

        for metric in core_metrics + safety_metrics + conv_metrics:
            assert metric in results
            assert isinstance(results[metric], ScoreResult)

    def test_get_all_available_metrics(self, suite):
        """Test getting all available metrics."""
        metrics = suite.get_all_available_metrics()

        assert isinstance(metrics, list)
        assert len(metrics) >= 10  # Should have core + advanced metrics

        # Check for key metrics
        expected_metrics = [
            "answer_relevancy",
            "faithfulness",
            "context_relevancy",
            "hallucination_detection",
            "bias_detection",
            "toxicity_detection",
            "conversation_coherence",
        ]

        for metric in expected_metrics:
            assert metric in metrics

    def test_get_safety_metrics(self, suite):
        """Test getting safety-specific metrics."""
        safety_metrics = suite.get_safety_metrics()

        assert isinstance(safety_metrics, list)
        expected_safety = [
            "hallucination_detection",
            "bias_detection",
            "toxicity_detection",
        ]

        for metric in expected_safety:
            assert metric in safety_metrics

    def test_get_conversational_metrics(self, suite):
        """Test getting conversational-specific metrics."""
        conv_metrics = suite.get_conversational_metrics()

        assert isinstance(conv_metrics, list)
        assert "conversation_coherence" in conv_metrics


class TestAdvancedConfigurationHelpers:
    """Test advanced configuration helper functions."""

    def test_get_advanced_rag_config_balanced(self):
        """Test balanced advanced configuration."""
        config = get_advanced_rag_config("balanced")

        assert isinstance(config, AdvancedRAGConfig)
        assert config.hallucination_threshold == 0.8
        assert config.bias_threshold == 0.3
        assert config.toxicity_threshold == 0.2

    def test_get_advanced_rag_config_safety_first(self):
        """Test safety-first configuration."""
        config = get_advanced_rag_config("safety_first")

        assert isinstance(config, AdvancedRAGConfig)
        assert config.hallucination_threshold >= 0.9
        assert config.bias_threshold <= 0.2
        assert config.toxicity_threshold <= 0.1

    def test_get_advanced_rag_config_permissive(self):
        """Test permissive configuration."""
        config = get_advanced_rag_config("permissive")

        assert isinstance(config, AdvancedRAGConfig)
        assert config.hallucination_threshold <= 0.7
        assert config.bias_threshold >= 0.4
        assert config.toxicity_threshold >= 0.3

    def test_get_advanced_rag_config_invalid(self):
        """Test invalid advanced configuration name."""
        with pytest.raises(ValueError):
            get_advanced_rag_config("invalid_config")


class TestAdvancedScorerFactory:
    """Test advanced scorer factory functions."""

    @pytest.fixture
    def mock_model(self):
        return MockLLMModel()

    @pytest.fixture
    def rag_config(self):
        return get_default_rag_config()

    @pytest.fixture
    def advanced_config(self):
        return get_advanced_rag_config("balanced")

    def test_create_comprehensive_rag_scorer_core_metrics(self, mock_model, rag_config):
        """Test creating core metrics with comprehensive factory."""
        core_metrics = ["answer_relevancy", "faithfulness", "context_relevancy"]

        for metric_type in core_metrics:
            scorer = create_comprehensive_rag_scorer(
                metric_type, mock_model, rag_config=rag_config
            )
            assert scorer is not None
            assert hasattr(scorer, "evaluate")

    def test_create_comprehensive_rag_scorer_advanced_metrics(
        self, mock_model, advanced_config
    ):
        """Test creating advanced metrics with comprehensive factory."""
        advanced_metrics = [
            "hallucination_detection",
            "bias_detection",
            "toxicity_detection",
            "conversation_coherence",
        ]

        for metric_type in advanced_metrics:
            scorer = create_comprehensive_rag_scorer(
                metric_type, mock_model, advanced_config=advanced_config
            )
            assert scorer is not None
            assert hasattr(scorer, "evaluate")

    def test_create_comprehensive_rag_scorer_invalid_type(self, mock_model):
        """Test creating invalid scorer type."""
        with pytest.raises(ValueError):
            create_comprehensive_rag_scorer("invalid_advanced_scorer", mock_model)


class TestAdvancedErrorHandling:
    """Test advanced error handling and edge cases."""

    @pytest.fixture
    def mock_model(self):
        return MockLLMModel()

    @pytest.fixture
    def config(self):
        return AdvancedRAGConfig()

    @pytest.mark.asyncio
    async def test_safety_scorer_with_model_errors(self, config):
        """Test safety scorers with model errors."""
        error_model = Mock()
        error_model.generate = AsyncMock(side_effect=Exception("Model error"))

        scorers = [
            HallucinationDetectionScorer(error_model, config),
            BiasDetectionScorer(error_model, config),
            ToxicityDetectionScorer(error_model, config),
        ]

        for scorer in scorers:
            result = await scorer.evaluate("Test question", "Test answer")

            # Should handle errors gracefully
            assert isinstance(result, ScoreResult)
            assert result.score == 0.0
            assert not result.passed
            assert "error" in result.reasoning.lower()

    @pytest.mark.asyncio
    async def test_conversation_scorer_with_invalid_history(self, mock_model, config):
        """Test conversation scorer with invalid history format."""
        scorer = ConversationCoherenceScorer(mock_model, config)

        # Test with invalid conversation history format
        invalid_history = [
            {"invalid_key": "user", "wrong_content": "What is AI?"},
            {"role": "assistant"},  # Missing content
        ]

        result = await scorer.evaluate(
            "Follow up question",
            "Follow up answer",
            conversation_history=invalid_history,
        )

        # Should handle gracefully
        assert isinstance(result, ScoreResult)
        assert 0.0 <= result.score <= 1.0

    @pytest.mark.asyncio
    async def test_comprehensive_suite_partial_failures(self, config):
        """Test comprehensive suite with partial scorer failures."""

        # Mock model that fails for specific prompts
        class PartialFailureModel:
            async def generate(self, prompt: str, **kwargs) -> str:
                if "bias" in prompt.lower():
                    raise Exception("Bias detection failed")
                return '{"score": 0.8, "reasoning": "Success"}'

        model = PartialFailureModel()
        suite = ComprehensiveRAGEvaluationSuite(model, get_default_rag_config(), config)

        results = await suite.evaluate_comprehensive_plus(
            "Test question", "Test answer", include_safety_metrics=True
        )

        # Should have some results, even if some scorers failed
        assert isinstance(results, dict)
        assert len(results) > 0

        # Failed scorers should have error results
        if "bias_detection" in results:
            bias_result = results["bias_detection"]
            assert bias_result.score == 0.0
            assert not bias_result.passed

    @pytest.mark.asyncio
    async def test_extreme_input_sizes(self, mock_model, config):
        """Test handling of extremely large inputs."""
        scorer = HallucinationDetectionScorer(mock_model, config)

        # Very large input
        huge_text = "This is a test sentence. " * 10000  # Very large text

        result = await scorer.evaluate(
            "What is this about?", huge_text, context="Some context"
        )

        # Should handle without crashing
        assert isinstance(result, ScoreResult)
        assert 0.0 <= result.score <= 1.0


class TestAdvancedPerformanceAndConcurrency:
    """Test advanced performance and concurrency aspects."""

    @pytest.fixture
    def mock_model(self):
        return MockLLMModel()

    @pytest.fixture
    def suite(self, mock_model):
        return ComprehensiveRAGEvaluationSuite(mock_model)

    @pytest.mark.asyncio
    async def test_concurrent_safety_evaluations(self, suite):
        """Test concurrent safety evaluation execution."""
        # Create multiple safety evaluation tasks
        tasks = []
        for i in range(3):
            task = suite.evaluate_comprehensive_plus(
                f"Question {i}",
                f"Answer {i}",
                context=f"Context {i}",
                include_safety_metrics=True,
                include_conversational_metrics=False,
            )
            tasks.append(task)

        # Execute concurrently
        results = await asyncio.gather(*tasks)

        # Verify all results
        assert len(results) == 3
        for result in results:
            assert isinstance(result, dict)
            assert "hallucination_detection" in result
            assert "bias_detection" in result
            assert "toxicity_detection" in result

    @pytest.mark.asyncio
    async def test_comprehensive_evaluation_performance(self, suite):
        """Test performance of comprehensive evaluation."""
        import time

        # Measure evaluation time
        start_time = time.time()

        result = await suite.evaluate_comprehensive_plus(
            "What are the benefits of renewable energy?",
            "Renewable energy provides environmental and economic benefits",
            expected_output="Benefits include sustainability and cost savings",
            context="Renewable energy sources offer multiple advantages",
            include_safety_metrics=True,
            include_conversational_metrics=True,
            conversation_history=[
                {"role": "user", "content": "Tell me about energy"},
                {"role": "assistant", "content": "Energy comes from various sources"},
            ],
        )

        end_time = time.time()
        evaluation_time = end_time - start_time

        # Should complete in reasonable time (adjust threshold as needed)
        assert evaluation_time < 30.0  # 30 seconds max for comprehensive evaluation

        # Should have comprehensive results
        assert isinstance(result, dict)
        assert len(result) >= 6  # Multiple metric categories


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
