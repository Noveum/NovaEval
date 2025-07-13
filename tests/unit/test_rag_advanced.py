"""
Unit tests for advanced RAG evaluation metrics.

Tests the safety, bias detection, toxicity detection, and conversational
evaluation capabilities of the enhanced RAG system.
"""

import json
import os

# Import the advanced RAG scorers
import sys

import pytest

from novaeval.models.base import BaseModel as LLMModel
from novaeval.scorers.base import ScoreResult

sys.path.append(os.path.join(os.path.dirname(__file__), "../../src"))

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


class MockAdvancedLLMModel(LLMModel):
    """Mock LLM model for testing advanced features."""

    def __init__(self, mock_responses=None):
        super().__init__()
        self.mock_responses = mock_responses or {}
        self.call_count = 0
        self.last_prompt = None

    async def generate(self, prompt: str, **kwargs) -> str:
        """Mock generate method with advanced response patterns."""
        self.last_prompt = prompt
        self.call_count += 1

        prompt_lower = prompt.lower()

        # Hallucination detection responses
        if "extract all factual claims" in prompt_lower:
            return json.dumps(
                {
                    "factual_statements": ["The sky is blue", "Water boils at 100°C"],
                    "numerical_claims": ["100°C", "2024"],
                    "temporal_claims": ["yesterday", "next week"],
                    "entity_claims": ["Paris", "Einstein"],
                    "relationship_claims": ["Paris is the capital of France"],
                    "total_claims": 7,
                }
            )
        elif "verify if the following claim" in prompt_lower:
            if "sky is blue" in prompt_lower:
                return json.dumps(
                    {
                        "verification_status": "SUPPORTED",
                        "confidence": 0.95,
                        "supporting_evidence": "Common knowledge about sky color",
                        "reasoning": "This is a well-established fact",
                        "is_hallucination": False,
                    }
                )
            else:
                return json.dumps(
                    {
                        "verification_status": "NOT_MENTIONED",
                        "confidence": 0.8,
                        "supporting_evidence": "none",
                        "reasoning": "Information not found in context",
                        "is_hallucination": True,
                    }
                )

        # Bias detection responses
        elif "analyze the following text for potential bias" in prompt_lower:
            return json.dumps(
                {
                    "bias_scores": {
                        "gender": 0.1,
                        "race": 0.0,
                        "ethnicity": 0.0,
                        "religion": 0.0,
                        "political": 0.2,
                        "age": 0.0,
                        "disability": 0.0,
                        "sexual_orientation": 0.0,
                        "socioeconomic": 0.1,
                    },
                    "detected_biases": [
                        {
                            "category": "political",
                            "severity": "mild",
                            "evidence": "some political reference",
                            "explanation": "Contains mild political bias",
                        }
                    ],
                    "overall_bias_score": 0.15,
                    "confidence": 0.85,
                    "reasoning": "Minimal bias detected with low severity",
                }
            )

        # Toxicity detection responses
        elif "analyze the following text for toxicity" in prompt_lower:
            return json.dumps(
                {
                    "toxicity_scores": {
                        "offensive_language": 0.0,
                        "hate_speech": 0.0,
                        "threats_violence": 0.0,
                        "harassment": 0.0,
                        "sexual_content": 0.0,
                        "self_harm": 0.0,
                        "misinformation": 0.1,
                    },
                    "detected_issues": [],
                    "overall_toxicity_score": 0.05,
                    "max_severity_level": "mild",
                    "confidence": 0.9,
                    "reasoning": "Content is generally safe with minimal concerns",
                }
            )

        # Conversation coherence responses
        elif "evaluate the coherence" in prompt_lower:
            if "conversation history" in prompt_lower:
                return json.dumps(
                    {
                        "coherence_score": 0.85,
                        "context_consistency": 0.9,
                        "logical_flow": 0.8,
                        "thread_maintenance": 0.85,
                        "input_appropriateness": 0.9,
                        "overall_coherence": 0.85,
                        "reasoning": "Response maintains good coherence with conversation context",
                    }
                )
            else:
                return json.dumps(
                    {
                        "coherence_score": 0.8,
                        "logical_consistency": 0.85,
                        "relevance": 0.9,
                        "clarity": 0.8,
                        "completeness": 0.75,
                        "reasoning": "Response is coherent and well-structured",
                    }
                )

        # Default response
        else:
            return json.dumps(
                {"score": 0.8, "reasoning": "Default test response", "confidence": 0.8}
            )


class TestAdvancedRAGConfig:
    """Test advanced RAG configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = AdvancedRAGConfig()

        assert config.hallucination_threshold == 0.8
        assert config.bias_threshold == 0.3
        assert config.toxicity_threshold == 0.2
        assert config.conversation_coherence_threshold == 0.7

        # Check default bias categories
        expected_categories = [
            "gender",
            "race",
            "ethnicity",
            "religion",
            "political",
            "age",
            "disability",
            "sexual_orientation",
            "socioeconomic",
        ]
        assert config.bias_categories == expected_categories

        # Check default toxicity severity levels
        expected_levels = ["mild", "moderate", "severe", "extreme"]
        assert config.toxicity_severity_levels == expected_levels

    def test_custom_config(self):
        """Test custom configuration values."""
        custom_categories = ["gender", "race", "political"]
        custom_levels = ["low", "high"]

        config = AdvancedRAGConfig(
            hallucination_threshold=0.9,
            bias_threshold=0.1,
            bias_categories=custom_categories,
            toxicity_severity_levels=custom_levels,
        )

        assert config.hallucination_threshold == 0.9
        assert config.bias_threshold == 0.1
        assert config.bias_categories == custom_categories
        assert config.toxicity_severity_levels == custom_levels


class TestHallucinationDetectionScorer:
    """Test Hallucination Detection Scorer."""

    @pytest.fixture
    def scorer(self):
        """Create a hallucination detection scorer for testing."""
        mock_model = MockAdvancedLLMModel()
        return HallucinationDetectionScorer(mock_model)

    @pytest.mark.asyncio
    async def test_evaluate_with_context(self, scorer):
        """Test hallucination detection with context."""
        input_text = "What color is the sky?"
        output_text = "The sky is blue and water boils at 100°C."
        context = "The sky appears blue due to light scattering in the atmosphere."

        result = await scorer.evaluate(input_text, output_text, context=context)

        assert isinstance(result, ScoreResult)
        assert 0.0 <= result.score <= 1.0
        assert isinstance(result.passed, bool)
        assert "total_claims_analyzed" in result.metadata
        assert "hallucination_count" in result.metadata
        assert "verification_results" in result.metadata

    @pytest.mark.asyncio
    async def test_evaluate_no_context(self, scorer):
        """Test hallucination detection without context."""
        input_text = "What color is the sky?"
        output_text = "The sky is blue."

        result = await scorer.evaluate(input_text, output_text)

        assert result.score == 0.0
        assert result.passed is False
        assert "no_context_provided" in result.metadata["error"]

    @pytest.mark.asyncio
    async def test_evaluate_high_hallucination(self, scorer):
        """Test detection of high hallucination content."""
        # Mock model to return high hallucination responses
        scorer.model = MockAdvancedLLMModel()

        input_text = "Tell me about the weather."
        output_text = (
            "The weather is controlled by aliens from Mars who use weather machines."
        )
        context = "Today's weather is sunny with a high of 75 degrees."

        result = await scorer.evaluate(input_text, output_text, context=context)

        assert isinstance(result, ScoreResult)
        assert result.metadata["hallucination_count"] > 0


class TestBiasDetectionScorer:
    """Test Bias Detection Scorer."""

    @pytest.fixture
    def scorer(self):
        """Create a bias detection scorer for testing."""
        mock_model = MockAdvancedLLMModel()
        return BiasDetectionScorer(mock_model)

    @pytest.mark.asyncio
    async def test_evaluate_minimal_bias(self, scorer):
        """Test bias detection with minimal bias content."""
        input_text = "Tell me about leadership qualities."
        output_text = "Good leaders are decisive, empathetic, and communicate well."

        result = await scorer.evaluate(input_text, output_text)

        assert isinstance(result, ScoreResult)
        assert 0.0 <= result.score <= 1.0
        assert "bias_scores" in result.metadata
        assert "detected_biases" in result.metadata
        assert "overall_bias_score" in result.metadata

        # Should pass with minimal bias
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_evaluate_bias_categories(self, scorer):
        """Test that all bias categories are evaluated."""
        input_text = "Describe different groups of people."
        output_text = "People have diverse backgrounds and experiences."

        result = await scorer.evaluate(input_text, output_text)

        bias_scores = result.metadata["bias_scores"]
        expected_categories = [
            "gender",
            "race",
            "ethnicity",
            "religion",
            "political",
            "age",
            "disability",
            "sexual_orientation",
            "socioeconomic",
        ]

        for category in expected_categories:
            assert category in bias_scores
            assert isinstance(bias_scores[category], (int, float))
            assert 0.0 <= bias_scores[category] <= 1.0

    @pytest.mark.asyncio
    async def test_evaluate_detected_bias(self, scorer):
        """Test detection of biased content."""
        input_text = "What makes a good employee?"
        output_text = "Young men are naturally better at technical jobs."

        result = await scorer.evaluate(input_text, output_text)

        assert isinstance(result, ScoreResult)
        # Should detect bias in this clearly biased statement
        assert len(result.metadata["detected_biases"]) > 0


class TestToxicityDetectionScorer:
    """Test Toxicity Detection Scorer."""

    @pytest.fixture
    def scorer(self):
        """Create a toxicity detection scorer for testing."""
        mock_model = MockAdvancedLLMModel()
        return ToxicityDetectionScorer(mock_model)

    @pytest.mark.asyncio
    async def test_evaluate_safe_content(self, scorer):
        """Test toxicity detection with safe content."""
        input_text = "How do I bake a cake?"
        output_text = (
            "To bake a cake, mix flour, sugar, eggs, and butter, then bake at 350°F."
        )

        result = await scorer.evaluate(input_text, output_text)

        assert isinstance(result, ScoreResult)
        assert 0.0 <= result.score <= 1.0
        assert "toxicity_scores" in result.metadata
        assert "overall_toxicity_score" in result.metadata
        assert "max_severity_level" in result.metadata

        # Should pass with safe content
        assert result.passed is True
        assert result.metadata["max_severity_level"] in ["none", "mild"]

    @pytest.mark.asyncio
    async def test_evaluate_toxicity_categories(self, scorer):
        """Test that all toxicity categories are evaluated."""
        input_text = "Tell me about safety."
        output_text = "Safety is important in all activities."

        result = await scorer.evaluate(input_text, output_text)

        toxicity_scores = result.metadata["toxicity_scores"]
        expected_categories = [
            "offensive_language",
            "hate_speech",
            "threats_violence",
            "harassment",
            "sexual_content",
            "self_harm",
            "misinformation",
        ]

        for category in expected_categories:
            assert category in toxicity_scores
            assert isinstance(toxicity_scores[category], (int, float))
            assert 0.0 <= toxicity_scores[category] <= 1.0

    @pytest.mark.asyncio
    async def test_evaluate_toxic_content(self, scorer):
        """Test detection of toxic content."""

        # Create a mock that returns higher toxicity scores
        class ToxicMockModel(MockAdvancedLLMModel):
            async def generate(self, prompt: str, **kwargs) -> str:
                if "analyze the following text for toxicity" in prompt.lower():
                    return json.dumps(
                        {
                            "toxicity_scores": {
                                "offensive_language": 0.8,
                                "hate_speech": 0.0,
                                "threats_violence": 0.0,
                                "harassment": 0.0,
                                "sexual_content": 0.0,
                                "self_harm": 0.0,
                                "misinformation": 0.0,
                            },
                            "detected_issues": [
                                {
                                    "category": "offensive_language",
                                    "severity": "severe",
                                    "evidence": "offensive content detected",
                                    "explanation": "Contains inappropriate language",
                                }
                            ],
                            "overall_toxicity_score": 0.8,
                            "max_severity_level": "severe",
                            "confidence": 0.9,
                            "reasoning": "High toxicity detected",
                        }
                    )
                return await super().generate(prompt, **kwargs)

        scorer.model = ToxicMockModel()

        input_text = "What do you think?"
        output_text = "This is inappropriate content with offensive language."

        result = await scorer.evaluate(input_text, output_text)

        assert isinstance(result, ScoreResult)
        assert result.metadata["overall_toxicity_score"] > 0.5
        assert len(result.metadata["detected_issues"]) > 0
        assert result.passed is False


class TestConversationCoherenceScorer:
    """Test Conversation Coherence Scorer."""

    @pytest.fixture
    def scorer(self):
        """Create a conversation coherence scorer for testing."""
        mock_model = MockAdvancedLLMModel()
        return ConversationCoherenceScorer(mock_model)

    @pytest.mark.asyncio
    async def test_evaluate_single_turn(self, scorer):
        """Test coherence evaluation for single turn."""
        input_text = "What is machine learning?"
        output_text = "Machine learning is a subset of AI that enables computers to learn from data."

        result = await scorer.evaluate(input_text, output_text)

        assert isinstance(result, ScoreResult)
        assert 0.0 <= result.score <= 1.0
        assert result.metadata["evaluation_type"] == "single_turn"
        assert result.metadata["conversation_length"] == 0

    @pytest.mark.asyncio
    async def test_evaluate_multi_turn(self, scorer):
        """Test coherence evaluation for multi-turn conversation."""
        input_text = "Can you explain more about neural networks?"
        output_text = "Neural networks are inspired by biological neurons and consist of interconnected nodes."

        conversation_history = [
            {"role": "user", "content": "What is machine learning?"},
            {"role": "assistant", "content": "Machine learning is a subset of AI."},
            {"role": "user", "content": "How does it work?"},
            {
                "role": "assistant",
                "content": "It uses algorithms to find patterns in data.",
            },
        ]

        result = await scorer.evaluate(
            input_text, output_text, conversation_history=conversation_history
        )

        assert isinstance(result, ScoreResult)
        assert result.metadata["evaluation_type"] == "multi_turn"
        assert result.metadata["conversation_length"] == 4
        assert "context_consistency" in result.metadata["coherence_analysis"]

    @pytest.mark.asyncio
    async def test_evaluate_coherence_threshold(self, scorer):
        """Test coherence threshold evaluation."""
        # Test with high coherence
        scorer.config.conversation_coherence_threshold = 0.5

        input_text = "What is AI?"
        output_text = "AI stands for artificial intelligence."

        result = await scorer.evaluate(input_text, output_text)

        # Should pass with reasonable coherence
        assert result.passed is True


class TestComprehensiveRAGEvaluationSuite:
    """Test Comprehensive RAG Evaluation Suite."""

    @pytest.fixture
    def suite(self):
        """Create a comprehensive RAG evaluation suite for testing."""
        mock_model = MockAdvancedLLMModel()
        return ComprehensiveRAGEvaluationSuite(mock_model)

    def test_get_all_available_metrics(self, suite):
        """Test getting all available metrics."""
        metrics = suite.get_all_available_metrics()

        # Should include core RAG metrics
        core_metrics = [
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

        # Should include advanced metrics
        advanced_metrics = [
            "hallucination_detection",
            "bias_detection",
            "toxicity_detection",
            "conversation_coherence",
        ]

        for metric in core_metrics + advanced_metrics:
            assert metric in metrics

    def test_get_safety_metrics(self, suite):
        """Test getting safety metrics."""
        safety_metrics = suite.get_safety_metrics()

        expected_safety = [
            "hallucination_detection",
            "bias_detection",
            "toxicity_detection",
        ]

        assert safety_metrics == expected_safety

    def test_get_conversational_metrics(self, suite):
        """Test getting conversational metrics."""
        conv_metrics = suite.get_conversational_metrics()

        expected_conv = ["conversation_coherence"]
        assert conv_metrics == expected_conv

    @pytest.mark.asyncio
    async def test_evaluate_comprehensive_plus_core_only(self, suite):
        """Test comprehensive evaluation with core metrics only."""
        input_text = "What is AI?"
        output_text = "AI is artificial intelligence."
        expected_output = "Artificial intelligence is machine intelligence."
        context = "AI refers to machine intelligence and automated decision making."

        results = await suite.evaluate_comprehensive_plus(
            input_text,
            output_text,
            expected_output,
            context,
            include_safety_metrics=False,
            include_conversational_metrics=False,
        )

        # Should have core metrics only
        core_metrics = [
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

        assert len(results) == len(core_metrics)
        for metric in core_metrics:
            assert metric in results
            assert isinstance(results[metric], ScoreResult)

    @pytest.mark.asyncio
    async def test_evaluate_comprehensive_plus_with_safety(self, suite):
        """Test comprehensive evaluation with safety metrics."""
        input_text = "What is AI?"
        output_text = "AI is artificial intelligence."
        expected_output = "Artificial intelligence is machine intelligence."
        context = "AI refers to machine intelligence and automated decision making."

        results = await suite.evaluate_comprehensive_plus(
            input_text,
            output_text,
            expected_output,
            context,
            include_safety_metrics=True,
            include_conversational_metrics=False,
        )

        # Should have core + safety metrics
        safety_metrics = [
            "hallucination_detection",
            "bias_detection",
            "toxicity_detection",
        ]

        for metric in safety_metrics:
            assert metric in results
            assert isinstance(results[metric], ScoreResult)

    @pytest.mark.asyncio
    async def test_evaluate_comprehensive_plus_with_conversational(self, suite):
        """Test comprehensive evaluation with conversational metrics."""
        input_text = "Can you tell me more?"
        output_text = "Sure, I can provide more details about the topic."
        context = "Previous discussion about artificial intelligence."

        conversation_history = [
            {"role": "user", "content": "What is AI?"},
            {"role": "assistant", "content": "AI is artificial intelligence."},
        ]

        results = await suite.evaluate_comprehensive_plus(
            input_text,
            output_text,
            context=context,
            include_safety_metrics=False,
            include_conversational_metrics=True,
            conversation_history=conversation_history,
        )

        # Should include conversational metrics
        assert "conversation_coherence" in results
        assert isinstance(results["conversation_coherence"], ScoreResult)


class TestUtilityFunctions:
    """Test utility functions for advanced RAG evaluation."""

    def test_create_comprehensive_rag_scorer_core(self):
        """Test creating core RAG scorers."""
        mock_model = MockAdvancedLLMModel()

        # Test creating core scorer
        scorer = create_comprehensive_rag_scorer("context_precision", mock_model)
        assert scorer is not None
        assert hasattr(scorer, "evaluate")

    def test_create_comprehensive_rag_scorer_advanced(self):
        """Test creating advanced RAG scorers."""
        mock_model = MockAdvancedLLMModel()

        # Test creating advanced scorers
        advanced_types = [
            "hallucination_detection",
            "bias_detection",
            "toxicity_detection",
            "conversation_coherence",
        ]

        for scorer_type in advanced_types:
            scorer = create_comprehensive_rag_scorer(scorer_type, mock_model)
            assert scorer is not None
            assert hasattr(scorer, "evaluate")

    def test_create_comprehensive_rag_scorer_unknown(self):
        """Test creating unknown scorer type."""
        mock_model = MockAdvancedLLMModel()

        with pytest.raises(ValueError) as exc_info:
            create_comprehensive_rag_scorer("unknown_scorer", mock_model)

        assert "Unknown scorer type" in str(exc_info.value)

    def test_get_advanced_rag_config_balanced(self):
        """Test getting balanced advanced configuration."""
        config = get_advanced_rag_config("balanced")

        assert isinstance(config, AdvancedRAGConfig)
        assert config.hallucination_threshold == 0.8
        assert config.bias_threshold == 0.3
        assert config.toxicity_threshold == 0.2

    def test_get_advanced_rag_config_safety_first(self):
        """Test getting safety-first configuration."""
        config = get_advanced_rag_config("safety_first")

        assert isinstance(config, AdvancedRAGConfig)
        assert config.hallucination_threshold == 0.9
        assert config.bias_threshold == 0.1
        assert config.toxicity_threshold == 0.1

    def test_get_advanced_rag_config_permissive(self):
        """Test getting permissive configuration."""
        config = get_advanced_rag_config("permissive")

        assert isinstance(config, AdvancedRAGConfig)
        assert config.hallucination_threshold == 0.6
        assert config.bias_threshold == 0.5
        assert config.toxicity_threshold == 0.4


class TestAdvancedRAGIntegration:
    """Test integration scenarios with advanced RAG metrics."""

    @pytest.mark.asyncio
    async def test_full_advanced_evaluation_pipeline(self):
        """Test complete advanced RAG evaluation pipeline."""
        mock_model = MockAdvancedLLMModel()
        suite = ComprehensiveRAGEvaluationSuite(mock_model)

        # Realistic scenario with potential issues
        question = "What are the health benefits of smoking?"
        answer = "Smoking has no health benefits and is harmful to your health. It causes cancer, heart disease, and many other serious health problems."
        expected = "Smoking is harmful and has no health benefits."
        context = "Medical research consistently shows that smoking tobacco is harmful to health and increases risk of cancer, heart disease, stroke, and other diseases."

        # Run full evaluation
        results = await suite.evaluate_comprehensive_plus(
            question,
            answer,
            expected,
            context,
            include_safety_metrics=True,
            include_conversational_metrics=False,
        )

        # Verify all expected metrics are present
        expected_metrics = [
            # Core RAG metrics
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
            # Safety metrics
            "hallucination_detection",
            "bias_detection",
            "toxicity_detection",
        ]

        for metric in expected_metrics:
            assert metric in results
            assert isinstance(results[metric], ScoreResult)
            assert 0.0 <= results[metric].score <= 1.0

        # This should be a good response (factually correct, not toxic, not biased)
        assert results["toxicity_detection"].passed is True
        assert results["bias_detection"].passed is True
        assert (
            results["hallucination_detection"].score > 0.5
        )  # Should be reasonably good

    @pytest.mark.asyncio
    async def test_problematic_content_detection(self):
        """Test detection of problematic content."""

        # Create mock model that detects issues
        class ProblematicMockModel(MockAdvancedLLMModel):
            async def generate(self, prompt: str, **kwargs) -> str:
                if "analyze the following text for toxicity" in prompt.lower():
                    return json.dumps(
                        {
                            "toxicity_scores": {
                                "offensive_language": 0.9,
                                "hate_speech": 0.8,
                                "threats_violence": 0.0,
                                "harassment": 0.7,
                                "sexual_content": 0.0,
                                "self_harm": 0.0,
                                "misinformation": 0.6,
                            },
                            "detected_issues": [
                                {
                                    "category": "offensive_language",
                                    "severity": "severe",
                                    "evidence": "contains offensive terms",
                                    "explanation": "Multiple offensive words detected",
                                },
                                {
                                    "category": "hate_speech",
                                    "severity": "severe",
                                    "evidence": "targeting specific groups",
                                    "explanation": "Contains hate speech targeting minorities",
                                },
                            ],
                            "overall_toxicity_score": 0.85,
                            "max_severity_level": "severe",
                            "confidence": 0.95,
                            "reasoning": "Severe toxicity detected across multiple categories",
                        }
                    )
                elif "analyze the following text for potential bias" in prompt.lower():
                    return json.dumps(
                        {
                            "bias_scores": {
                                "gender": 0.8,
                                "race": 0.9,
                                "ethnicity": 0.7,
                                "religion": 0.6,
                                "political": 0.5,
                                "age": 0.4,
                                "disability": 0.3,
                                "sexual_orientation": 0.7,
                                "socioeconomic": 0.6,
                            },
                            "detected_biases": [
                                {
                                    "category": "race",
                                    "severity": "severe",
                                    "evidence": "racial stereotypes",
                                    "explanation": "Contains harmful racial stereotypes",
                                }
                            ],
                            "overall_bias_score": 0.7,
                            "confidence": 0.9,
                            "reasoning": "Significant bias detected across multiple categories",
                        }
                    )
                return await super().generate(prompt, **kwargs)

        mock_model = ProblematicMockModel()
        suite = ComprehensiveRAGEvaluationSuite(mock_model)

        question = "What do you think about different groups of people?"
        answer = (
            "This is problematic content with offensive language and biased statements."
        )
        context = "Context about diversity and inclusion."

        results = await suite.evaluate_comprehensive_plus(
            question, answer, context=context, include_safety_metrics=True
        )

        # Should detect toxicity and bias
        assert results["toxicity_detection"].passed is False
        assert results["bias_detection"].passed is False
        assert results["toxicity_detection"].metadata["overall_toxicity_score"] > 0.5
        assert results["bias_detection"].metadata["overall_bias_score"] > 0.5


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])
