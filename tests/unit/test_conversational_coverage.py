"""
Additional tests for conversational scorers to increase coverage.
"""

from unittest.mock import AsyncMock, patch

import pytest

from novaeval.scorers.base import ScoreResult
from novaeval.scorers.conversational import (
    ConversationalMetricsScorer,
    ConversationCompletenessScorer,
    ConversationRelevancyScorer,
    KnowledgeRetentionScorer,
    RoleAdherenceScorer,
)


class MockLLMModel:
    """Mock LLM model for testing."""

    def __init__(self, name="MockModel"):
        self.name = name

    async def generate(self, prompt: str, **kwargs) -> str:
        return "Mock response"

    def generate_batch(self, prompts: list[str], **kwargs) -> list[str]:
        return ["Mock response"] * len(prompts)


class TestConversationalCoverage:
    """Test cases to increase coverage of conversational scorers."""

    @pytest.mark.asyncio
    async def test_knowledge_retention_scorer_edge_cases(self):
        """Test KnowledgeRetentionScorer with edge cases."""
        mock_model = MockLLMModel()
        scorer = KnowledgeRetentionScorer(model=mock_model)

        # Test with empty conversation
        result = await scorer.evaluate(
            input_text="test", output_text="test", context={"conversation_history": []}
        )
        assert isinstance(result, ScoreResult)

        # Test with malformed conversation history
        result = await scorer.evaluate(
            input_text="test",
            output_text="test",
            context={"conversation_history": "invalid"},
        )
        assert isinstance(result, ScoreResult)

    @pytest.mark.asyncio
    async def test_conversation_relevancy_scorer_edge_cases(self):
        """Test ConversationRelevancyScorer with edge cases."""
        mock_model = MockLLMModel()
        scorer = ConversationRelevancyScorer(model=mock_model)

        # Test with empty conversation
        result = await scorer.evaluate(
            input_text="test", output_text="test", context={"conversation_history": []}
        )
        assert isinstance(result, ScoreResult)

        # Test with single turn conversation
        result = await scorer.evaluate(
            input_text="test",
            output_text="test",
            context={"conversation_history": [{"user": "hello", "assistant": "hi"}]},
        )
        assert isinstance(result, ScoreResult)

    @pytest.mark.asyncio
    async def test_conversation_completeness_scorer_edge_cases(self):
        """Test ConversationCompletenessScorer with edge cases."""
        mock_model = MockLLMModel()
        scorer = ConversationCompletenessScorer(model=mock_model)

        # Test with empty conversation
        result = await scorer.evaluate(
            input_text="test", output_text="test", context={"conversation_history": []}
        )
        assert isinstance(result, ScoreResult)

        # Test with malformed context
        result = await scorer.evaluate(
            input_text="test",
            output_text="test",
            context={"conversation_history": None},
        )
        assert isinstance(result, ScoreResult)

    @pytest.mark.asyncio
    async def test_role_adherence_scorer_edge_cases(self):
        """Test RoleAdherenceScorer with edge cases."""
        mock_model = MockLLMModel()
        scorer = RoleAdherenceScorer(model=mock_model)

        # Test without role definition
        result = await scorer.evaluate(
            input_text="test", output_text="test", context={}
        )
        assert isinstance(result, ScoreResult)

        # Test with empty role definition
        result = await scorer.evaluate(
            input_text="test", output_text="test", context={"role_definition": ""}
        )
        assert isinstance(result, ScoreResult)

    @pytest.mark.asyncio
    async def test_conversational_metrics_scorer_edge_cases(self):
        """Test ConversationalMetricsScorer with edge cases."""
        mock_model = MockLLMModel()
        scorer = ConversationalMetricsScorer(model=mock_model)

        # Test with empty conversation
        result = await scorer.evaluate(
            input_text="test", output_text="test", context={"conversation_history": []}
        )
        assert isinstance(result, ScoreResult)

        # Test with partial metrics enabled
        scorer = ConversationalMetricsScorer(
            model=mock_model,
            include_knowledge_retention=True,
            include_relevancy=False,
            include_completeness=False,
            include_role_adherence=False,
        )
        result = await scorer.evaluate(
            input_text="test", output_text="test", context={"conversation_history": []}
        )
        assert isinstance(result, ScoreResult)

    @pytest.mark.asyncio
    async def test_conversational_scorers_with_exceptions(self):
        """Test conversational scorers with model exceptions."""
        mock_model = MockLLMModel()

        scorers = [
            KnowledgeRetentionScorer(model=mock_model),
            ConversationRelevancyScorer(model=mock_model),
            ConversationCompletenessScorer(model=mock_model),
            RoleAdherenceScorer(model=mock_model),
        ]

        for scorer in scorers:
            # Mock the model's generate method to raise exception
            with patch.object(
                scorer.model,
                "generate",
                new=AsyncMock(side_effect=Exception("Model error")),
            ):
                result = await scorer.evaluate(
                    input_text="test",
                    output_text="test",
                    context={"conversation_history": []},
                )
            assert isinstance(result, ScoreResult)
            # The scorers have fallback behavior, so they don't always return 0.0 on exceptions
            assert 0.0 <= result.score <= 1.0

    def test_conversational_scorers_sync_wrapper(self):
        """Test sync wrapper methods for conversational scorers."""
        mock_model = MockLLMModel()

        scorers = [
            KnowledgeRetentionScorer(model=mock_model),
            ConversationRelevancyScorer(model=mock_model),
            ConversationCompletenessScorer(model=mock_model),
            RoleAdherenceScorer(model=mock_model),
        ]

        for scorer in scorers:
            # Mock the async evaluate method with AsyncMock
            with patch.object(
                scorer,
                "evaluate",
                new=AsyncMock(
                    return_value=ScoreResult(score=0.8, passed=True, reasoning="test")
                ),
            ):
                result = scorer.score(
                    "test input", "test output", context={"conversation_history": []}
                )

            assert result == 0.8

    def test_conversational_scorers_with_invalid_context(self):
        """Test conversational scorers with invalid context types."""
        mock_model = MockLLMModel()

        scorers = [
            KnowledgeRetentionScorer(model=mock_model),
            ConversationRelevancyScorer(model=mock_model),
            ConversationCompletenessScorer(model=mock_model),
            RoleAdherenceScorer(model=mock_model),
        ]

        for scorer in scorers:
            # Test with string context instead of dict
            with patch.object(
                scorer,
                "evaluate",
                new=AsyncMock(
                    return_value=ScoreResult(score=0.0, passed=False, reasoning="test")
                ),
            ):
                result = scorer.score("test input", "test output", context="invalid")

            assert result == 0.0

    @pytest.mark.asyncio
    async def test_conversational_metrics_scorer_all_metrics(self):
        """Test ConversationalMetricsScorer with all metrics enabled."""
        mock_model = MockLLMModel()
        scorer = ConversationalMetricsScorer(
            model=mock_model,
            include_knowledge_retention=True,
            include_relevancy=True,
            include_completeness=True,
            include_role_adherence=True,
        )

        result = await scorer.evaluate(
            input_text="test",
            output_text="test",
            context={
                "conversation_history": [
                    {"user": "hello", "assistant": "hi"},
                    {"user": "how are you?", "assistant": "I'm good"},
                ],
                "role_definition": "You are a helpful assistant",
            },
        )
        assert isinstance(result, ScoreResult)

    @pytest.mark.asyncio
    async def test_conversational_scorers_with_malformed_responses(self):
        """Test conversational scorers with malformed LLM responses."""
        mock_model = MockLLMModel()

        scorers = [
            KnowledgeRetentionScorer(model=mock_model),
            ConversationRelevancyScorer(model=mock_model),
            ConversationCompletenessScorer(model=mock_model),
            RoleAdherenceScorer(model=mock_model),
        ]

        for scorer in scorers:
            # Mock the model's generate method to return malformed response
            with patch.object(
                scorer.model,
                "generate",
                new=AsyncMock(return_value="invalid json response"),
            ):
                result = await scorer.evaluate(
                    input_text="test",
                    output_text="test",
                    context={"conversation_history": []},
                )
            assert isinstance(result, ScoreResult)
            # Should fallback to default scores
            assert 0.0 <= result.score <= 1.0
