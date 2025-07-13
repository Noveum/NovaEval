"""
Comprehensive unit tests for advanced RAG evaluation system.

This test suite covers all advanced safety metrics, conversational evaluation,
and comprehensive evaluation functionality with thorough edge case testing.
"""

import asyncio
import os
import sys
from typing import Optional

import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), "../../src"))

from novaeval.scorers.base import ScoreResult
from novaeval.scorers.rag_advanced import (
    BiasDetectionScorer,
    ConversationCoherenceScorer,
    HallucinationDetectionScorer,
    ToxicityDetectionScorer,
)


class MockAdvancedLLMModel:
    """Mock LLM model for testing advanced features."""

    def __init__(
        self, response: Optional[str] = None, responses: Optional[list[str]] = None
    ):
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


class TestHallucinationDetectionScorer:
    """Test hallucination detection scorer."""

    @pytest.fixture
    def scorer(self):
        """Create a hallucination detection scorer for testing."""
        model = MockAdvancedLLMModel()
        return HallucinationDetectionScorer(model)

    @pytest.mark.asyncio
    async def test_basic_hallucination_detection(self, scorer):
        """Test basic hallucination detection."""
        result = await scorer.score(
            query="What is the capital of France?",
            answer="The capital of France is Paris.",
            context="France is a country in Europe. Paris is its capital city.",
        )

        assert isinstance(result, ScoreResult)
        assert 0.0 <= result.score <= 1.0
        assert "reasoning" in result.metadata
        assert "confidence" in result.metadata

    @pytest.mark.asyncio
    async def test_hallucination_with_contradictory_context(self, scorer):
        """Test hallucination detection with contradictory context."""
        result = await scorer.score(
            query="What is the capital of France?",
            answer="The capital of France is London.",
            context="France is a country in Europe. Paris is its capital city.",
        )

        assert isinstance(result, ScoreResult)
        assert 0.0 <= result.score <= 1.0

    @pytest.mark.asyncio
    async def test_hallucination_with_empty_context(self, scorer):
        """Test hallucination detection with empty context."""
        result = await scorer.score(
            query="What is quantum computing?",
            answer="Quantum computing uses quantum mechanics principles.",
            context="",
        )

        assert isinstance(result, ScoreResult)
        assert 0.0 <= result.score <= 1.0

    @pytest.mark.asyncio
    async def test_hallucination_error_handling(self, scorer):
        """Test error handling in hallucination detection."""
        # Test with invalid JSON response
        scorer.model = MockAdvancedLLMModel(response="invalid json")

        result = await scorer.score(
            query="Test query",
            answer="Test answer",
            context="Test context",
        )

        assert isinstance(result, ScoreResult)
        assert result.score == 0.0


class TestBiasDetectionScorer:
    """Test bias detection scorer."""

    @pytest.fixture
    def scorer(self):
        """Create a bias detection scorer for testing."""
        model = MockAdvancedLLMModel()
        return BiasDetectionScorer(model)

    @pytest.mark.asyncio
    async def test_basic_bias_detection(self, scorer):
        """Test basic bias detection."""
        result = await scorer.score(
            query="Tell me about software engineers",
            answer="Software engineers are typically male and work long hours.",
            context="Software engineering is a diverse field with professionals of all backgrounds.",
        )

        assert isinstance(result, ScoreResult)
        assert 0.0 <= result.score <= 1.0
        assert "reasoning" in result.metadata
        assert "bias_categories" in result.metadata

    @pytest.mark.asyncio
    async def test_bias_detection_categories(self, scorer):
        """Test bias detection with specific categories."""
        result = await scorer.score(
            query="Describe a good leader",
            answer="Good leaders are usually men who are assertive and dominant.",
            context="Leadership qualities include empathy, communication, and decision-making skills.",
        )

        assert isinstance(result, ScoreResult)
        assert 0.0 <= result.score <= 1.0

    @pytest.mark.asyncio
    async def test_bias_detection_neutral_content(self, scorer):
        """Test bias detection with neutral content."""
        result = await scorer.score(
            query="What is photosynthesis?",
            answer="Photosynthesis is the process by which plants convert sunlight into energy.",
            context="Plants use chlorophyll to capture light energy and convert CO2 into glucose.",
        )

        assert isinstance(result, ScoreResult)
        assert 0.0 <= result.score <= 1.0


class TestToxicityDetectionScorer:
    """Test toxicity detection scorer."""

    @pytest.fixture
    def scorer(self):
        """Create a toxicity detection scorer for testing."""
        model = MockAdvancedLLMModel()
        return ToxicityDetectionScorer(model)

    @pytest.mark.asyncio
    async def test_basic_toxicity_detection(self, scorer):
        """Test basic toxicity detection."""
        result = await scorer.score(
            query="What do you think about different cultures?",
            answer="All cultures have value and contribute to human diversity.",
            context="Cultural diversity enriches societies and promotes understanding.",
        )

        assert isinstance(result, ScoreResult)
        assert 0.0 <= result.score <= 1.0
        assert "reasoning" in result.metadata
        assert "toxicity_level" in result.metadata

    @pytest.mark.asyncio
    async def test_toxicity_detection_harmful_content(self, scorer):
        """Test toxicity detection with potentially harmful content."""
        # Mock response indicating toxicity
        scorer.model = MockAdvancedLLMModel(
            response='{"score": 0.2, "reasoning": "Contains harmful stereotypes", "toxicity_level": "moderate"}'
        )

        result = await scorer.score(
            query="Tell me about different groups",
            answer="Some groups are inherently better than others.",
            context="All human groups have equal worth and dignity.",
        )

        assert isinstance(result, ScoreResult)
        assert 0.0 <= result.score <= 1.0

    @pytest.mark.asyncio
    async def test_toxicity_detection_severity_levels(self, scorer):
        """Test toxicity detection with different severity levels."""
        # Test mild toxicity
        scorer.model = MockAdvancedLLMModel(
            response='{"score": 0.7, "reasoning": "Mild inappropriate language", "toxicity_level": "mild"}'
        )

        result = await scorer.score(
            query="How do you feel?",
            answer="I'm feeling a bit annoyed today.",
            context="It's normal to have different emotions.",
        )

        assert isinstance(result, ScoreResult)
        assert 0.0 <= result.score <= 1.0


class TestConversationCoherenceScorer:
    """Test conversation coherence scorer."""

    @pytest.fixture
    def scorer(self):
        """Create a conversation coherence scorer for testing."""
        model = MockAdvancedLLMModel()
        return ConversationCoherenceScorer(model)

    @pytest.mark.asyncio
    async def test_basic_conversation_coherence(self, scorer):
        """Test basic conversation coherence."""
        conversation_history = [
            {"role": "user", "content": "What is machine learning?"},
            {
                "role": "assistant",
                "content": "Machine learning is a subset of AI that enables computers to learn from data.",
            },
            {"role": "user", "content": "Can you give me an example?"},
        ]

        result = await scorer.score(
            query="Can you give me an example?",
            answer="Sure! Image recognition is a common example where ML models learn to identify objects in photos.",
            context="Machine learning applications include image recognition, natural language processing, and recommendation systems.",
            conversation_history=conversation_history,
        )

        assert isinstance(result, ScoreResult)
        assert 0.0 <= result.score <= 1.0
        assert "reasoning" in result.metadata
        assert "coherence_factors" in result.metadata

    @pytest.mark.asyncio
    async def test_conversation_topic_switch(self, scorer):
        """Test conversation coherence with topic switch."""
        # Mock response indicating topic switch
        scorer.model = MockAdvancedLLMModel(
            response='{"score": 0.3, "reasoning": "Abrupt topic change detected", "context_switch_detected": true, "topic_consistency": 0.2, "transition_quality": 0.1}'
        )

        conversation_history = [
            {"role": "user", "content": "What is machine learning?"},
            {"role": "assistant", "content": "Machine learning is a subset of AI."},
            {"role": "user", "content": "What's your favorite pizza topping?"},
        ]

        result = await scorer.score(
            query="What's your favorite pizza topping?",
            answer="I like pepperoni pizza the most.",
            context="Pizza toppings vary widely based on personal preference.",
            conversation_history=conversation_history,
        )

        assert isinstance(result, ScoreResult)
        assert 0.0 <= result.score <= 1.0
        assert result.metadata["context_switch_detected"]
        assert result.metadata["topic_consistency"] < 0.5
        assert "transition_quality" in result.metadata

    @pytest.mark.asyncio
    async def test_conversation_coherence_empty_history(self, scorer):
        """Test conversation coherence with empty history."""
        result = await scorer.score(
            query="What is AI?",
            answer="AI stands for Artificial Intelligence.",
            context="Artificial Intelligence is the simulation of human intelligence in machines.",
            conversation_history=[],
        )

        assert isinstance(result, ScoreResult)
        assert 0.0 <= result.score <= 1.0

    @pytest.mark.asyncio
    async def test_conversation_coherence_long_history(self, scorer):
        """Test conversation coherence with long conversation history."""
        conversation_history = []
        for i in range(20):
            conversation_history.extend(
                [
                    {"role": "user", "content": f"Question {i} about AI"},
                    {
                        "role": "assistant",
                        "content": f"Answer {i} about artificial intelligence",
                    },
                ]
            )

        result = await scorer.score(
            query="Can you summarize what we discussed?",
            answer="We've been discussing various aspects of artificial intelligence and its applications.",
            context="AI has many applications across different domains.",
            conversation_history=conversation_history,
        )

        assert isinstance(result, ScoreResult)
        assert 0.0 <= result.score <= 1.0


class TestAdvancedRAGIntegration:
    """Test integration of advanced RAG components."""

    @pytest.fixture
    def model(self):
        """Create a mock model for testing."""
        return MockAdvancedLLMModel()

    @pytest.mark.asyncio
    async def test_multiple_advanced_scorers(self, model):
        """Test using multiple advanced scorers together."""
        hallucination_scorer = HallucinationDetectionScorer(model)
        bias_scorer = BiasDetectionScorer(model)
        toxicity_scorer = ToxicityDetectionScorer(model)

        query = "Tell me about leadership qualities"
        answer = "Good leaders are empathetic, decisive, and communicate well with their teams."
        context = "Leadership research shows that effective leaders possess various skills and traits."

        # Run all scorers
        hallucination_result = await hallucination_scorer.score(query, answer, context)
        bias_result = await bias_scorer.score(query, answer, context)
        toxicity_result = await toxicity_scorer.score(query, answer, context)

        # Verify all results
        for result in [hallucination_result, bias_result, toxicity_result]:
            assert isinstance(result, ScoreResult)
            assert 0.0 <= result.score <= 1.0
            assert "reasoning" in result.metadata

    @pytest.mark.asyncio
    async def test_advanced_scorers_error_resilience(self, model):
        """Test advanced scorers' resilience to errors."""
        # Test with model that returns invalid responses
        error_model = MockAdvancedLLMModel(response="invalid json response")

        scorers = [
            HallucinationDetectionScorer(error_model),
            BiasDetectionScorer(error_model),
            ToxicityDetectionScorer(error_model),
        ]

        for scorer in scorers:
            result = await scorer.score(
                query="Test query",
                answer="Test answer",
                context="Test context",
            )

            assert isinstance(result, ScoreResult)
            # Should handle errors gracefully
            assert 0.0 <= result.score <= 1.0

    @pytest.mark.asyncio
    async def test_advanced_scorers_performance(self, model):
        """Test performance of advanced scorers with concurrent execution."""
        scorers = [
            HallucinationDetectionScorer(model),
            BiasDetectionScorer(model),
            ToxicityDetectionScorer(model),
        ]

        query = "What are the benefits of renewable energy?"
        answer = (
            "Renewable energy provides clean power and reduces environmental impact."
        )
        context = (
            "Renewable energy sources include solar, wind, and hydroelectric power."
        )

        # Run scorers concurrently
        tasks = [scorer.score(query, answer, context) for scorer in scorers]

        results = await asyncio.gather(*tasks)

        # Verify all results
        assert len(results) == 3
        for result in results:
            assert isinstance(result, ScoreResult)
            assert 0.0 <= result.score <= 1.0

    @pytest.mark.asyncio
    async def test_conversation_coherence_with_context_switches(self, model):
        """Test conversation coherence with multiple context switches."""
        scorer = ConversationCoherenceScorer(model)

        # Simulate conversation with topic switches
        conversation_history = [
            {"role": "user", "content": "What is machine learning?"},
            {
                "role": "assistant",
                "content": "ML is a subset of AI that learns from data.",
            },
            {"role": "user", "content": "What's the weather like?"},
            {
                "role": "assistant",
                "content": "I don't have access to current weather data.",
            },
            {"role": "user", "content": "Back to ML, what are neural networks?"},
        ]

        result = await scorer.score(
            query="Back to ML, what are neural networks?",
            answer="Neural networks are computing systems inspired by biological neural networks.",
            context="Neural networks consist of interconnected nodes that process information.",
            conversation_history=conversation_history,
        )

        assert isinstance(result, ScoreResult)
        assert 0.0 <= result.score <= 1.0
        assert "coherence_factors" in result.metadata
