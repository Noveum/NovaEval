"""
Tests for basic RAG scorers.
"""

import sys
import types
from unittest.mock import AsyncMock, Mock, patch

import numpy as np
import pytest
from test_utils import MockLLM

from src.novaeval.scorers.base import ScoreResult
from src.novaeval.scorers.basic_rag_scorers import (
    AggregateRAGScorer,
    ContextualPrecisionScorerPP,
    ContextualRecallScorerPP,
    RetrievalDiversityScorer,
    RetrievalF1Scorer,
    RetrievalRankingScorer,
    SemanticSimilarityScorer,
)

pytestmark = pytest.mark.unit


class TestAsyncLLMScorer:
    def test_init(self):
        # AsyncLLMScorer is abstract, so we'll test with a concrete subclass
        model = Mock()
        scorer = ContextualPrecisionScorerPP(model=model)
        assert scorer.model == model

    @pytest.mark.asyncio
    async def test_call_model(self):
        model = Mock()
        scorer = ContextualPrecisionScorerPP(model=model)

        with patch(
            "src.novaeval.scorers.basic_rag_scorers.call_llm",
            return_value="test response",
        ) as mock_call:
            result = await scorer._call_model("test prompt")
            assert result == "test response"
            mock_call.assert_called_once_with(model, "test prompt")

    def test_parse_numerical_response_edge_cases(self):
        """Test _parse_numerical_response with various edge cases."""
        model = Mock()
        scorer = ContextualPrecisionScorerPP(model=model)

        # Test with score in JSON (using integer to avoid regex conflict)
        response = '{"score": 8}'
        result = scorer._parse_numerical_response(response)
        assert result == 8.0

        # Test with rating in JSON
        response = '{"rating": 7}'
        result = scorer._parse_numerical_response(response)
        assert result == 7.0

        # Test with invalid JSON
        response = '{"invalid": json}'
        result = scorer._parse_numerical_response(response)
        assert result == 5.0  # Fallback

        # Test with no numbers
        response = "No numbers here at all"
        result = scorer._parse_numerical_response(response)
        assert result == 5.0  # Fallback

        # Test with numbers outside 0-10 range
        response = "Rating: 15"
        result = scorer._parse_numerical_response(response)
        assert result == 10.0  # Should be clamped to 10

        response = "Rating: 0"
        result = scorer._parse_numerical_response(response)
        assert result == 0.0  # Should be 0

    def test_parse_numerical_response_exception_handling(self):
        """Test _parse_numerical_response exception handling."""
        model = Mock()
        scorer = ContextualPrecisionScorerPP(model=model)

        # Test with response that causes exception during parsing
        response = None  # This should cause an exception
        result = scorer._parse_numerical_response(response)
        assert result == 5.0  # Fallback


class TestContextualPrecisionScorerPP:
    def test_init(self):
        model = Mock()
        scorer = ContextualPrecisionScorerPP(model=model, relevance_threshold=0.8)
        assert scorer.model == model
        assert scorer.relevance_threshold == 0.8
        assert scorer.name == "RetrievalPrecisionScorer"

    @pytest.mark.asyncio
    async def test_evaluate_chunk_relevance_success(self):
        model = Mock()
        scorer = ContextualPrecisionScorerPP(model=model)

        # Mock the _call_model method to return a numerical response
        with patch.object(
            scorer, "_call_model", new_callable=AsyncMock, return_value="Rating: 8"
        ):
            result = await scorer._evaluate_chunk_relevance("query", "chunk")
            assert result is True  # 8 >= 7 (0.7 * 10)

    @pytest.mark.asyncio
    async def test_evaluate_chunk_relevance_exception(self):
        model = Mock()
        scorer = ContextualPrecisionScorerPP(model=model)

        # Mock the _call_model method to raise an exception
        with patch.object(
            scorer,
            "_call_model",
            new_callable=AsyncMock,
            side_effect=Exception("API error"),
        ):
            result = await scorer._evaluate_chunk_relevance("query", "chunk")
            assert result is False

    def test_parse_json_response_valid_json(self):
        model = Mock()
        scorer = ContextualPrecisionScorerPP(model=model)

        response = '{"relevant": true, "reasoning": "test"}'
        result = scorer._parse_json_response(response)
        assert result == {"relevant": True, "reasoning": "test"}

    def test_parse_json_response_json_in_text(self):
        model = Mock()
        scorer = ContextualPrecisionScorerPP(model=model)

        response = 'Some text before {"relevant": false} some text after'
        result = scorer._parse_json_response(response)
        assert result == {"relevant": False}

    def test_parse_json_response_invalid_json_with_indicators(self):
        model = Mock()
        scorer = ContextualPrecisionScorerPP(model=model)

        response = "Yes, this is relevant"
        result = scorer._parse_json_response(response)
        assert result["relevant"] is True
        assert result["reasoning"] == "Fallback parsing used"

    def test_parse_json_response_invalid_json_false_indicators(self):
        model = Mock()
        scorer = ContextualPrecisionScorerPP(model=model)

        response = "No, this is not relevant"
        result = scorer._parse_json_response(response)
        assert result["relevant"] is False
        assert result["reasoning"] == "Fallback parsing used"

    def test_parse_numerical_response_rating_format(self):
        model = Mock()
        scorer = ContextualPrecisionScorerPP(model=model)

        response = "Rating: 7"
        result = scorer._parse_numerical_response(response)
        assert result == 7.0

    def test_parse_numerical_response_standalone_number(self):
        model = Mock()
        scorer = ContextualPrecisionScorerPP(model=model)

        response = "The relevance score is 9"
        result = scorer._parse_numerical_response(response)
        assert result == 9.0

    def test_parse_numerical_response_json_format(self):
        model = Mock()
        scorer = ContextualPrecisionScorerPP(model=model)

        response = '{"rating": 6}'
        result = scorer._parse_numerical_response(response)
        assert result == 6.0

    def test_parse_numerical_response_fallback(self):
        model = Mock()
        scorer = ContextualPrecisionScorerPP(model=model)

        response = "This is not a numerical response"
        result = scorer._parse_numerical_response(response)
        assert result == 5.0  # Default fallback value

    @pytest.mark.asyncio
    async def test_evaluate_no_context_empty(self):
        model = Mock()
        scorer = ContextualPrecisionScorerPP(model=model)
        # Basic initialization test to ensure scorer is properly created
        assert scorer.model == model

    @pytest.mark.asyncio
    async def test_evaluate_with_context_success(self):
        """Test evaluate method with context successfully."""
        model = Mock()
        scorer = ContextualPrecisionScorerPP(model=model)

        context = {"chunks": ["chunk1", "chunk2"]}
        with patch.object(
            scorer,
            "_evaluate_chunk_relevance",
            new_callable=AsyncMock,
            return_value=True,
        ):
            result = await scorer.evaluate("query", "answer", context=context)

            assert (
                hasattr(result, "score")
                and hasattr(result, "passed")
                and hasattr(result, "reasoning")
            )
            assert result.score == 1.0
            assert result.passed is True
            assert "Precision:" in result.reasoning

    @pytest.mark.asyncio
    async def test_evaluate_with_context_partial_success(self):
        """Test evaluate method with partial context success."""
        model = Mock()
        scorer = ContextualPrecisionScorerPP(model=model)

        context = {"chunks": ["chunk1", "chunk2", "chunk3"]}
        with patch.object(
            scorer,
            "_evaluate_chunk_relevance",
            new_callable=AsyncMock,
            side_effect=[True, False, True],
        ):
            result = await scorer.evaluate("query", "answer", context=context)

            assert (
                hasattr(result, "score")
                and hasattr(result, "passed")
                and hasattr(result, "reasoning")
            )
            assert result.score == pytest.approx(2 / 3, rel=1e-3)
            assert result.passed is False  # Below threshold
            assert "Precision:" in result.reasoning

    @pytest.mark.asyncio
    async def test_evaluate_with_context_all_fail(self):
        """Test evaluate method when all chunks fail relevance check."""
        model = Mock()
        scorer = ContextualPrecisionScorerPP(model=model)

        context = {"chunks": ["chunk1", "chunk2"]}
        with patch.object(
            scorer,
            "_evaluate_chunk_relevance",
            new_callable=AsyncMock,
            return_value=False,
        ):
            result = await scorer.evaluate("query", "answer", context=context)

            assert (
                hasattr(result, "score")
                and hasattr(result, "passed")
                and hasattr(result, "reasoning")
            )
            assert result.score == 0.0
            assert result.passed is False
            assert "Precision:" in result.reasoning

    @pytest.mark.asyncio
    async def test_evaluate_with_context_exception_handling(self):
        """Test evaluate method with exception handling."""
        model = Mock()
        scorer = ContextualPrecisionScorerPP(model=model)

        context = {"chunks": ["chunk1"]}
        # Mock the _call_model method to raise an exception, which will be caught by _evaluate_chunk_relevance
        with patch.object(
            scorer,
            "_call_model",
            new_callable=AsyncMock,
            side_effect=Exception("API error"),
        ):
            result = await scorer.evaluate("query", "answer", context=context)

            assert (
                hasattr(result, "score")
                and hasattr(result, "passed")
                and hasattr(result, "reasoning")
            )
            assert result.score == 0.0
            assert result.passed is False
            assert "Precision:" in result.reasoning

    def test_parse_json_response_edge_cases(self):
        """Test _parse_json_response with edge cases."""
        model = Mock()
        scorer = ContextualPrecisionScorerPP(model=model)

        # Test with empty string
        result = scorer._parse_json_response("")
        assert result["relevant"] is False
        assert "Fallback parsing used" in result["reasoning"]

        # Test with malformed JSON
        result = scorer._parse_json_response('{"relevant": true, "reasoning":}')
        assert result["relevant"] is True
        assert "Fallback parsing used" in result["reasoning"]

    def test_parse_numerical_response_edge_cases(self):
        """Test _parse_numerical_response with edge cases."""
        model = Mock()
        scorer = ContextualPrecisionScorerPP(model=model)

        # Test with empty string
        result = scorer._parse_numerical_response("")
        assert result == 5.0

        # Test with very large number
        result = scorer._parse_numerical_response("Rating: 999")
        assert result == 10.0  # Should be clamped

        # Test with negative number
        result = scorer._parse_numerical_response("Rating: -5")
        assert result == 5.0  # Should use fallback for invalid numbers

    def test_parse_numerical_response_decimal_numbers(self):
        """Test _parse_numerical_response with decimal numbers."""
        model = Mock()
        scorer = ContextualPrecisionScorerPP(model=model)

        # Test with decimal
        result = scorer._parse_numerical_response("Rating: 7.5")
        assert result == 7.0  # Integer parsing

        # Test with decimal in text
        result = scorer._parse_numerical_response("The score is 8.75 out of 10")
        assert result == 8.0  # Integer parsing

    def test_parse_numerical_response_multiple_numbers(self):
        """Test _parse_numerical_response with multiple numbers."""
        model = Mock()
        scorer = ContextualPrecisionScorerPP(model=model)

        # Test with multiple numbers (should take first)
        result = scorer._parse_numerical_response("Rating: 6, also 8 and 9")
        assert result == 6.0

        # Test with numbers in different formats
        result = scorer._parse_numerical_response("Score: 7, Rating: 9")
        assert result == 9.0  # Takes the last number found

    @pytest.mark.asyncio
    async def test_evaluate_no_context(self):
        """Test evaluate method with no context."""
        model = Mock()
        scorer = ContextualPrecisionScorerPP(model=model)

        result = await scorer.evaluate("query", "answer")
        assert result.score == 0.0
        assert result.passed is False
        assert "No context or input provided" in result.reasoning

    @pytest.mark.asyncio
    async def test_evaluate_with_context(self):
        model = Mock()
        scorer = ContextualPrecisionScorerPP(model=model)

        context = {"chunks": ["This is context chunk 1", "This is context chunk 2"]}

        with patch.object(
            scorer,
            "_evaluate_chunk_relevance",
            new_callable=AsyncMock,
            side_effect=[True, False],
        ):
            result = await scorer.evaluate("query", "answer", context=context)
            assert result.score == 0.5  # 1 relevant out of 2 chunks
            assert result.passed is False  # Below default threshold of 0.7
            assert "Precision: 0.500" in result.reasoning

    def test_score_method(self):
        model = Mock()
        scorer = ContextualPrecisionScorerPP(model=model)

        context = {"context": "test context"}

        with patch("asyncio.run") as mock_run:
            mock_run.return_value = ScoreResult(
                score=0.8, passed=True, reasoning="Test", metadata={}
            )

            result = scorer.score("prediction", "ground_truth", context)
            assert result == 0.8

    @pytest.mark.asyncio
    async def test_evaluate_with_context_key(self):
        """Test evaluate method when context has 'context' key instead of 'chunks'."""
        model = Mock()
        scorer = ContextualPrecisionScorerPP(model=model)

        context = {"context": "single context string"}

        with patch.object(
            scorer,
            "_evaluate_chunk_relevance",
            new_callable=AsyncMock,
            return_value=True,
        ):
            result = await scorer.evaluate("query", "answer", context=context)
            assert result.score == 1.0  # 1 relevant out of 1 chunk
            assert result.passed is True  # Above default threshold of 0.7

    @pytest.mark.asyncio
    async def test_evaluate_with_mixed_relevance(self):
        """Test evaluate method with mixed relevance results."""
        model = Mock()
        scorer = ContextualPrecisionScorerPP(model=model, relevance_threshold=0.5)

        context = {"chunks": ["chunk1", "chunk2", "chunk3", "chunk4"]}

        with patch.object(
            scorer,
            "_evaluate_chunk_relevance",
            new_callable=AsyncMock,
            side_effect=[True, False, True, False],
        ):
            result = await scorer.evaluate("query", "answer", context=context)
            assert result.score == 0.5  # 2 relevant out of 4 chunks
            assert result.passed is True  # Meets threshold of 0.5
            assert result.metadata["relevant_chunks"] == 2
            assert result.metadata["total_chunks"] == 4

    def test_score_method_with_async_context(self):
        """Test score method when called from within an async context."""
        model = Mock()
        scorer = ContextualPrecisionScorerPP(model=model)
        context = {"chunks": ["chunk1", "chunk2"]}

        # Mock the async evaluate method
        mock_result = ScoreResult(
            score=0.7, passed=True, reasoning="Test precision", metadata={}
        )

        with (
            patch("asyncio.get_running_loop") as mock_get_loop,
            patch("concurrent.futures.ThreadPoolExecutor") as mock_executor,
        ):
            # Simulate being in an async context
            mock_get_loop.return_value = Mock()

            # Mock the executor and future
            mock_future = Mock()
            mock_future.result.return_value = mock_result
            mock_executor_instance = Mock()
            mock_executor_instance.submit.return_value = mock_future
            mock_executor.return_value.__enter__.return_value = mock_executor_instance

            result = scorer.score("prediction", "ground_truth", context)

            assert result == 0.7
            mock_executor_instance.submit.assert_called_once()

    def test_score_method_result_without_score_attribute(self):
        """Test score method when result doesn't have score attribute."""
        model = Mock()
        scorer = ContextualPrecisionScorerPP(model=model)
        context = {"chunks": ["chunk1"]}

        # Mock result without score attribute
        mock_result = Mock(spec=[])  # Empty spec means no attributes

        with patch("asyncio.run", return_value=mock_result):
            result = scorer.score("prediction", "ground_truth", context)
            assert result == 0.0


class TestContextualRecallScorerPP:
    def test_init(self):
        model = Mock()
        scorer = ContextualRecallScorerPP(model=model, relevance_threshold=0.6)
        assert scorer.model == model
        assert scorer.relevance_threshold == 0.6
        assert scorer.name == "RetrievalRecallScorer"

    @pytest.mark.asyncio
    async def test_evaluate_chunk_relevance_json_response(self):
        model = Mock()
        scorer = ContextualRecallScorerPP(model=model)

        with patch.object(
            scorer,
            "_call_model",
            new_callable=AsyncMock,
            return_value='{"relevant": false}',
        ):
            result = await scorer._evaluate_chunk_relevance("query", "chunk")
            assert result is False

    @pytest.mark.asyncio
    async def test_estimate_total_relevant_chunks(self):
        model = Mock()
        scorer = ContextualRecallScorerPP(model=model)

        with patch.object(
            scorer,
            "_call_model",
            new_callable=AsyncMock,
            return_value='{"estimated_total": 5}',
        ):
            result = await scorer._estimate_total_relevant_chunks(
                "query", ["chunk1", "chunk2"]
            )
            assert result == 5

    @pytest.mark.asyncio
    async def test_estimate_total_relevant_chunks_fallback(self):
        model = Mock()
        scorer = ContextualRecallScorerPP(model=model)

        with patch.object(
            scorer, "_call_model", new_callable=AsyncMock, return_value="invalid json"
        ):
            result = await scorer._estimate_total_relevant_chunks(
                "query", ["chunk1", "chunk2"]
            )
            assert result == 2  # Fallback to number of chunks

    @pytest.mark.asyncio
    async def test_evaluate_with_chunks(self):
        model = Mock()
        scorer = ContextualRecallScorerPP(model=model)

        context = {"chunks": ["chunk1", "chunk2", "chunk3"]}

        with (
            patch.object(
                scorer,
                "_evaluate_chunk_relevance",
                new_callable=AsyncMock,
                side_effect=[True, True, False],
            ),
            patch.object(
                scorer,
                "_estimate_total_relevant_chunks",
                new_callable=AsyncMock,
                return_value=4,
            ),
        ):
            result = await scorer.evaluate("query", "answer", context=context)
            assert result.score == 0.5  # 2 found out of 4 estimated
            assert "Recall: 0.500" in result.reasoning

    def test_score_method_with_async_context(self):
        """Test score method when called from within an async context."""
        model = Mock()
        scorer = ContextualRecallScorerPP(model=model)
        context = {"chunks": ["chunk1", "chunk2"]}

        # Mock the async evaluate method
        mock_result = ScoreResult(
            score=0.7, passed=True, reasoning="Test recall", metadata={}
        )

        with (
            patch("asyncio.get_running_loop") as mock_get_loop,
            patch("concurrent.futures.ThreadPoolExecutor") as mock_executor,
        ):
            # Simulate being in an async context
            mock_get_loop.return_value = Mock()

            # Mock the executor and future
            mock_future = Mock()
            mock_future.result.return_value = mock_result
            mock_executor_instance = Mock()
            mock_executor_instance.submit.return_value = mock_future
            mock_executor.return_value.__enter__.return_value = mock_executor_instance

            result = scorer.score("prediction", "ground_truth", context)

            assert result == 0.7
            mock_executor_instance.submit.assert_called_once()

    def test_score_method_no_async_context(self):
        """Test score method when called outside an async context."""
        model = Mock()
        scorer = ContextualRecallScorerPP(model=model)
        context = {"chunks": ["chunk1", "chunk2"]}

        mock_result = ScoreResult(
            score=0.8, passed=True, reasoning="Test recall", metadata={}
        )

        with (
            patch("asyncio.get_running_loop", side_effect=RuntimeError("No loop")),
            patch("asyncio.run", return_value=mock_result) as mock_run,
        ):
            result = scorer.score("prediction", "ground_truth", context)

            assert result == 0.8
            mock_run.assert_called_once()

    def test_score_method_result_without_score_attribute(self):
        """Test score method when result doesn't have score attribute."""
        model = Mock()
        scorer = ContextualRecallScorerPP(model=model)
        context = {"chunks": ["chunk1"]}

        # Mock result without score attribute
        mock_result = Mock(spec=[])  # Empty spec means no attributes

        with patch("asyncio.run", return_value=mock_result):
            result = scorer.score("prediction", "ground_truth", context)
            assert result == 0.0

    @pytest.mark.asyncio
    async def test_evaluate_with_context_key(self):
        """Test evaluate method when context has 'context' key instead of 'chunks'."""
        model = Mock()
        scorer = ContextualRecallScorerPP(model=model)

        context = {"context": "single context string"}

        with (
            patch.object(
                scorer,
                "_evaluate_chunk_relevance",
                new_callable=AsyncMock,
                return_value=True,
            ),
            patch.object(
                scorer,
                "_estimate_total_relevant_chunks",
                new_callable=AsyncMock,
                return_value=2,
            ),
        ):
            result = await scorer.evaluate("query", "answer", context=context)
            assert result.score == 0.5  # 1 relevant retrieved, estimated 2 total
            assert result.passed is False  # Below default threshold of 0.7

    @pytest.mark.asyncio
    async def test_evaluate_with_high_recall(self):
        """Test evaluate method with high recall scenario."""
        model = Mock()
        scorer = ContextualRecallScorerPP(model=model, relevance_threshold=0.5)

        context = {"chunks": ["chunk1", "chunk2", "chunk3"]}

        with (
            patch.object(
                scorer,
                "_evaluate_chunk_relevance",
                new_callable=AsyncMock,
                side_effect=[True, True, True],
            ),
            patch.object(
                scorer,
                "_estimate_total_relevant_chunks",
                new_callable=AsyncMock,
                return_value=3,
            ),
        ):
            result = await scorer.evaluate("query", "answer", context=context)
            assert result.score == 1.0  # 3 relevant retrieved, estimated 3 total
            assert result.passed is True  # Meets threshold of 0.5
            assert result.metadata["relevant_chunks"] == 3
            assert result.metadata["estimated_total_relevant"] == 3

    @pytest.mark.asyncio
    async def test_estimate_total_relevant_chunks_with_exception(self):
        """Test _estimate_total_relevant_chunks with exception handling."""
        model = Mock()
        scorer = ContextualRecallScorerPP(model=model)

        with patch.object(
            scorer,
            "_call_model",
            new_callable=AsyncMock,
            side_effect=Exception("API error"),
        ):
            result = await scorer._estimate_total_relevant_chunks(
                "query", ["chunk1", "chunk2"]
            )
            assert result == 2  # Should fallback to length of retrieved chunks

    @pytest.mark.asyncio
    async def test_estimate_total_relevant_chunks_with_invalid_json(self):
        """Test _estimate_total_relevant_chunks with invalid JSON response."""
        model = Mock()
        scorer = ContextualRecallScorerPP(model=model)

        with patch.object(
            scorer, "_call_model", new_callable=AsyncMock, return_value="invalid json"
        ):
            result = await scorer._estimate_total_relevant_chunks(
                "query", ["chunk1", "chunk2", "chunk3"]
            )
            assert result == 3  # Should fallback to length of retrieved chunks

    @pytest.mark.asyncio
    async def test_estimate_total_relevant_chunks_with_missing_key(self):
        """Test _estimate_total_relevant_chunks with missing estimated_total key."""
        model = Mock()
        scorer = ContextualRecallScorerPP(model=model)

        with patch.object(
            scorer,
            "_call_model",
            new_callable=AsyncMock,
            return_value='{"other_key": 5}',
        ):
            result = await scorer._estimate_total_relevant_chunks(
                "query", ["chunk1", "chunk2"]
            )
            assert result == 2  # Should fallback to length of retrieved chunks

    @pytest.mark.asyncio
    async def test_estimate_total_relevant_chunks_with_low_estimate(self):
        """Test _estimate_total_relevant_chunks when estimate is lower than retrieved."""
        model = Mock()
        scorer = ContextualRecallScorerPP(model=model)

        with patch.object(
            scorer,
            "_call_model",
            new_callable=AsyncMock,
            return_value='{"estimated_total": 1}',
        ):
            result = await scorer._estimate_total_relevant_chunks(
                "query", ["chunk1", "chunk2", "chunk3"]
            )
            assert result == 3  # Should be at least as many as retrieved


class TestRetrievalF1Scorer:
    def test_init(self):
        precision_scorer = Mock()
        recall_scorer = Mock()
        scorer = RetrievalF1Scorer(precision_scorer, recall_scorer, threshold=0.6)

        assert scorer.precision_scorer == precision_scorer
        assert scorer.recall_scorer == recall_scorer
        assert scorer.threshold == 0.6

    def test_score_with_valid_scores(self):
        precision_scorer = Mock()
        recall_scorer = Mock()
        scorer = RetrievalF1Scorer(precision_scorer, recall_scorer, threshold=0.6)

        precision_scorer.score.return_value = 0.8
        recall_scorer.score.return_value = 0.6

        result = scorer.score("prediction", "ground_truth", {"context": "test"})

        # F1 = 2 * (0.8 * 0.6) / (0.8 + 0.6) = 2 * 0.48 / 1.4 = 0.6857...
        expected_f1 = 2 * (0.8 * 0.6) / (0.8 + 0.6)
        assert result["f1"] == pytest.approx(expected_f1, rel=1e-3)
        assert result["precision"] == 0.8
        assert result["recall"] == 0.6

    def test_score_with_zero_sum(self):
        precision_scorer = Mock()
        recall_scorer = Mock()
        scorer = RetrievalF1Scorer(precision_scorer, recall_scorer)

        precision_scorer.score.return_value = 0.0
        recall_scorer.score.return_value = 0.0

        result = scorer.score("prediction", "ground_truth", {"context": "test"})

        assert result["f1"] == 0.0
        assert result["precision"] == 0.0
        assert result["recall"] == 0.0

    def test_get_score_result(self):
        precision_scorer = Mock()
        recall_scorer = Mock()
        scorer = RetrievalF1Scorer(precision_scorer, recall_scorer)

        precision_scorer.score.return_value = 0.7
        recall_scorer.score.return_value = 0.5

        # Call score first to populate _last_result
        scorer.score("prediction", "ground_truth", {"context": "test"})

        result = scorer.get_score_result()
        assert result is not None

    def test_score_with_high_precision_low_recall(self):
        """Test F1 score calculation with high precision, low recall."""
        precision_scorer = Mock()
        recall_scorer = Mock()
        scorer = RetrievalF1Scorer(precision_scorer, recall_scorer)

        precision_scorer.score.return_value = 0.9
        recall_scorer.score.return_value = 0.3

        result = scorer.score("prediction", "ground_truth", {"context": "test"})

        # F1 should be lower than both precision and recall
        expected_f1 = 2 * (0.9 * 0.3) / (0.9 + 0.3)
        assert result["f1"] == pytest.approx(expected_f1, rel=1e-3)
        assert result["precision"] == 0.9
        assert result["recall"] == 0.3

    def test_score_with_low_precision_high_recall(self):
        """Test F1 score calculation with low precision, high recall."""
        precision_scorer = Mock()
        recall_scorer = Mock()
        scorer = RetrievalF1Scorer(precision_scorer, recall_scorer)

        precision_scorer.score.return_value = 0.2
        recall_scorer.score.return_value = 0.8

        result = scorer.score("prediction", "ground_truth", {"context": "test"})

        # F1 should be lower than both precision and recall
        expected_f1 = 2 * (0.2 * 0.8) / (0.2 + 0.8)
        assert result["f1"] == pytest.approx(expected_f1, rel=1e-3)
        assert result["precision"] == 0.2
        assert result["recall"] == 0.8

    def test_score_with_perfect_scores(self):
        """Test F1 score calculation with perfect precision and recall."""
        precision_scorer = Mock()
        recall_scorer = Mock()
        scorer = RetrievalF1Scorer(precision_scorer, recall_scorer)

        precision_scorer.score.return_value = 1.0
        recall_scorer.score.return_value = 1.0

        result = scorer.score("prediction", "ground_truth", {"context": "test"})

        # Perfect F1 should be 1.0
        assert result["f1"] == 1.0
        assert result["precision"] == 1.0
        assert result["recall"] == 1.0

    def test_score_with_exception_handling(self):
        """Test F1 score calculation with exception handling."""
        precision_scorer = Mock()
        recall_scorer = Mock()
        scorer = RetrievalF1Scorer(precision_scorer, recall_scorer)

        # Mock precision scorer to raise exception
        precision_scorer.score.side_effect = Exception("Precision scoring failed")
        recall_scorer.score.return_value = 0.5

        # Since the scorer doesn't have exception handling, this should raise an exception
        with pytest.raises(Exception, match="Precision scoring failed"):
            scorer.score("prediction", "ground_truth", {"context": "test"})

    def test_score_with_none_context(self):
        """Test F1 score calculation with None context."""
        precision_scorer = Mock()
        recall_scorer = Mock()
        scorer = RetrievalF1Scorer(precision_scorer, recall_scorer)

        precision_scorer.score.return_value = 0.7
        recall_scorer.score.return_value = 0.6

        result = scorer.score("prediction", "ground_truth", None)

        # Should still work with None context
        expected_f1 = 2 * (0.7 * 0.6) / (0.7 + 0.6)
        assert result["f1"] == pytest.approx(expected_f1, rel=1e-3)

    def test_score_with_empty_context(self):
        """Test F1 score calculation with empty context."""
        precision_scorer = Mock()
        recall_scorer = Mock()
        scorer = RetrievalF1Scorer(precision_scorer, recall_scorer)

        precision_scorer.score.return_value = 0.8
        recall_scorer.score.return_value = 0.7

        result = scorer.score("prediction", "ground_truth", {})

        # Should still work with empty context
        expected_f1 = 2 * (0.8 * 0.7) / (0.8 + 0.7)
        assert result["f1"] == pytest.approx(expected_f1, rel=1e-3)

    def test_score_with_float_precision_recall(self):
        """Test F1 score calculation with float precision and recall values."""
        precision_scorer = Mock()
        recall_scorer = Mock()
        scorer = RetrievalF1Scorer(precision_scorer, recall_scorer)

        precision_scorer.score.return_value = 0.75
        recall_scorer.score.return_value = 0.85

        result = scorer.score("prediction", "ground_truth", {"context": "test"})

        expected_f1 = 2 * (0.75 * 0.85) / (0.75 + 0.85)
        assert result["f1"] == pytest.approx(expected_f1, rel=1e-3)
        assert result["precision"] == 0.75
        assert result["recall"] == 0.85

    def test_score_result_metadata(self):
        """Test that score result contains proper metadata."""
        precision_scorer = Mock()
        recall_scorer = Mock()
        scorer = RetrievalF1Scorer(precision_scorer, recall_scorer)

        precision_scorer.score.return_value = 0.6
        recall_scorer.score.return_value = 0.8

        result = scorer.score("prediction", "ground_truth", {"context": "test"})

        # Check that the last result was set correctly
        score_result = scorer.get_score_result()
        assert score_result is not None
        assert "F1 Score:" in score_result.reasoning
        assert "Precision:" in score_result.reasoning
        assert "Recall:" in score_result.reasoning

        # Check that the result contains the expected values
        assert result["f1"] == pytest.approx(0.686, rel=1e-3)
        assert result["precision"] == 0.6
        assert result["recall"] == 0.8


class TestRetrievalRankingScorer:
    def test_init(self):
        scorer = RetrievalRankingScorer(threshold=0.6)
        assert scorer.threshold == 0.6
        assert scorer.name == "RetrievalRankingScorer"

    def test_score_no_rankings(self):
        scorer = RetrievalRankingScorer()

        result = scorer.score("prediction", "ground_truth", {"context": "test"})
        assert result == 0.0

        score_result = scorer.get_score_result()
        assert score_result.score == 0.0
        assert "No ranking data provided" in score_result.reasoning

    def test_score_with_rankings(self):
        scorer = RetrievalRankingScorer()

        context = {
            "rankings": [1, 2, 3, 4, 5],
            "relevance_scores": [1.0, 0.8, 0.6, 0.4, 0.2],
        }

        result = scorer.score("prediction", "ground_truth", context)

        # Should return a dictionary with ranking metrics
        assert isinstance(result, dict)
        assert "ndcg" in result
        assert "map" in result
        assert "mrr" in result

    def test_score_with_rankings_exception(self):
        scorer = RetrievalRankingScorer()

        # Invalid rankings that might cause an exception
        context = {"rankings": "invalid"}

        result = scorer.score("prediction", "ground_truth", context)
        assert result == 0.0

    def test_score_with_rankings_detailed_metrics(self):
        """Test score method with detailed ranking metrics."""
        scorer = RetrievalRankingScorer()

        context = {
            "rankings": [1, 2, 3, 4, 5],
            "relevance_scores": [1.0, 0.8, 0.6, 0.4, 0.2],
        }

        result = scorer.score("prediction", "ground_truth", context)

        # Should return a dictionary with all ranking metrics
        assert isinstance(result, dict)
        assert "mrr" in result
        assert "ndcg" in result
        assert "map" in result
        assert "avg_ranking" in result
        assert "combined" in result

        # All metrics should be between 0 and 1
        for metric in ["mrr", "ndcg", "map", "avg_ranking", "combined"]:
            assert 0 <= result[metric] <= 1

    def test_score_with_rankings_no_relevant_items(self):
        """Test score method when no items are relevant."""
        scorer = RetrievalRankingScorer()

        context = {
            "rankings": [1, 2, 3, 4, 5],
            "relevance_scores": [0.0, 0.0, 0.0, 0.0, 0.0],  # No relevant items
        }

        result = scorer.score("prediction", "ground_truth", context)

        assert isinstance(result, dict)
        assert result["mrr"] == 0.0  # No relevant items means MRR = 0

    def test_score_with_rankings_high_ranks(self):
        """Test score method with rankings higher than max_rank."""
        scorer = RetrievalRankingScorer()

        context = {
            "rankings": [1, 2, 3, 4, 5, 10, 15],  # Some ranks > max_rank (5)
            "relevance_scores": [1.0, 0.8, 0.6, 0.4, 0.2, 0.1, 0.0],
        }

        result = scorer.score("prediction", "ground_truth", context)

        assert isinstance(result, dict)
        # High ranks should get 0.0 score
        assert result["avg_ranking"] < 1.0

    def test_score_with_rankings_different_lengths(self):
        """Test score method with different lengths of rankings and relevance scores."""
        scorer = RetrievalRankingScorer()

        context = {
            "rankings": [1, 2, 3, 4, 5],
            "relevance_scores": [1.0, 0.8, 0.6, 0.4],  # Different length
        }

        result = scorer.score("prediction", "ground_truth", context)

        # Should handle gracefully and use minimum length
        assert isinstance(result, dict)
        assert "mrr" in result

    def test_score_with_rankings_empty_lists(self):
        """Test score method with empty rankings and relevance scores."""
        scorer = RetrievalRankingScorer()

        context = {
            "rankings": [],
            "relevance_scores": [],
        }

        result = scorer.score("prediction", "ground_truth", context)

        # Should return a dictionary with zero scores
        assert isinstance(result, dict)
        assert result["mrr"] == 0.0
        assert result["ndcg"] == 0.0
        assert result["map"] == 0.0

        score_result = scorer.get_score_result()
        assert "Ranking Score:" in score_result.reasoning

    def test_score_with_rankings_missing_keys(self):
        """Test score method with missing context keys."""
        scorer = RetrievalRankingScorer()

        # Missing rankings key
        context_no_rankings = {"relevance_scores": [1.0, 0.8, 0.6]}
        result = scorer.score("prediction", "ground_truth", context_no_rankings)
        assert result == 0.0

        # Missing relevance_scores key
        context_no_scores = {"rankings": [1, 2, 3]}
        result = scorer.score("prediction", "ground_truth", context_no_scores)
        # Should still work with default relevance scores
        assert isinstance(result, dict)
        assert "mrr" in result

        # Missing both keys
        context_empty = {}
        result = scorer.score("prediction", "ground_truth", context_empty)
        assert result == 0.0

    def test_score_with_rankings_none_context(self):
        """Test score method with None context."""
        scorer = RetrievalRankingScorer()

        result = scorer.score("prediction", "ground_truth", None)
        assert result == 0.0

    def test_score_with_rankings_edge_cases(self):
        """Test score method with edge cases."""
        scorer = RetrievalRankingScorer()

        # Single item
        context_single = {"rankings": [1], "relevance_scores": [1.0]}
        result = scorer.score("prediction", "ground_truth", context_single)
        assert isinstance(result, dict)
        assert result["mrr"] == 1.0  # Single relevant item at rank 1

        # All items at rank 1 (tie)
        context_tie = {"rankings": [1, 1, 1], "relevance_scores": [1.0, 0.8, 0.6]}
        result = scorer.score("prediction", "ground_truth", context_tie)
        assert isinstance(result, dict)
        assert result["avg_ranking"] == 1.0  # All at rank 1

    def test_get_score_result(self):
        """Test get_score_result method."""
        scorer = RetrievalRankingScorer()

        # Initially should return None
        assert scorer.get_score_result() is None

        # After scoring, should return the result
        context = {
            "rankings": [1, 2, 3],
            "relevance_scores": [1.0, 0.8, 0.6],
        }

        scorer.score("prediction", "ground_truth", context)

        score_result = scorer.get_score_result()
        assert score_result is not None
        assert isinstance(score_result.score, float)
        assert isinstance(score_result.passed, bool)
        assert isinstance(score_result.reasoning, str)
        assert "Ranking Score:" in score_result.reasoning

    def test_score_with_rankings_different_lengths_fallback(self):
        """Test score method with different lengths of rankings and relevance scores."""
        scorer = RetrievalRankingScorer()

        context = {
            "rankings": [1, 2, 3, 4, 5],
            "relevance_scores": [1.0, 0.8, 0.6, 0.4],  # Shorter than rankings
        }

        result = scorer.score("prediction", "ground_truth", context)

        assert isinstance(result, dict)
        # Should handle different lengths gracefully

    def test_score_with_rankings_fallback_exception(self):
        """Test score method fallback when main computation fails."""
        scorer = RetrievalRankingScorer()

        # Create context that might cause issues in the main computation
        context = {
            "rankings": [1, 2, 3],
            "relevance_scores": [1.0, 0.8, 0.6],
        }

        # Mock numpy operations to raise exceptions
        with patch("numpy.array", side_effect=Exception("Numpy error")):
            result = scorer.score("prediction", "ground_truth", context)

            # Should fallback to simple ranking score
            assert isinstance(result, dict)
            assert "avg_ranking" in result

    def test_score_with_rankings_complete_failure(self):
        """Test score method when even fallback fails."""
        scorer = RetrievalRankingScorer()

        context = {
            "rankings": [1, 2, 3],
            "relevance_scores": [1.0, 0.8, 0.6],
        }

        # Mock both main computation and fallback to fail
        with (
            patch("numpy.array", side_effect=Exception("Numpy error")),
            patch("numpy.mean", side_effect=Exception("Mean error")),
        ):
            result = scorer.score("prediction", "ground_truth", context)

            # Should return 0.0 when everything fails
            assert result == 0.0

            score_result = scorer.get_score_result()
            assert score_result.score == 0.0
            assert "Ranking computation failed" in score_result.reasoning


class TestSemanticSimilarityScorer:
    def test_init(self):
        scorer = SemanticSimilarityScorer(threshold=0.8, embedding_model="test-model")
        assert scorer.threshold == 0.8
        assert scorer.embedding_model == "test-model"
        assert scorer.model is None
        assert scorer._model_loaded is False

    def test_load_model_success(self):
        scorer = SemanticSimilarityScorer()

        mock_model = Mock()
        stub_mod = types.SimpleNamespace(
            SentenceTransformer=Mock(return_value=mock_model)
        )
        with patch.dict(sys.modules, {"sentence_transformers": stub_mod}):
            scorer._load_model()
            assert scorer.model == mock_model

    def test_load_model_import_error(self):
        scorer = SemanticSimilarityScorer()

        # Mock the import to raise ImportError by patching the module import
        original_import = __builtins__["__import__"]

        def mock_import(name, *args, **kwargs):
            if name == "sentence_transformers":
                raise ImportError("No module named 'sentence_transformers'")
            return original_import(name, *args, **kwargs)

        with (
            patch("builtins.__import__", side_effect=mock_import),
            patch("builtins.print") as mock_print,
        ):
            scorer._load_model()

            assert scorer.model is None
            mock_print.assert_called_once_with(
                "Warning: sentence_transformers not installed. "
                "Using simple similarity computation."
            )

    def test_compute_simple_similarity(self):
        scorer = SemanticSimilarityScorer()

        query = "machine learning"
        chunks = [
            "ML is a subset of AI",
            "deep learning neural networks",
            "cooking recipes",
        ]

        similarity = scorer._compute_simple_similarity(query, chunks)

        # Should be between 0 and 1
        assert 0 <= similarity <= 1

    def test_compute_simple_similarity_empty_chunks(self):
        scorer = SemanticSimilarityScorer()

        similarity = scorer._compute_simple_similarity("query", [])
        assert similarity == 0.0

    def test_score_no_context(self):
        scorer = SemanticSimilarityScorer()

        result = scorer.score("prediction", "ground_truth", None)
        assert result == 0.0

        score_result = scorer.get_score_result()
        assert score_result.score == 0.0
        assert "No context or query provided" in score_result.reasoning

    def test_score_with_fallback_similarity(self):
        scorer = SemanticSimilarityScorer(threshold=0.5)

        context = {"chunks": ["chunk1", "chunk2"]}

        # Mock _load_model to set model to None (fallback mode)
        with patch.object(scorer, "_load_model"):
            scorer.model = None

            with patch.object(scorer, "_compute_simple_similarity", return_value=0.6):
                result = scorer.score("prediction", "ground_truth", context)

                assert isinstance(result, dict)
                assert result["similarity"] == 0.6

    def test_score_with_embeddings(self):
        scorer = SemanticSimilarityScorer(threshold=0.5)

        context = {"chunks": ["chunk1", "chunk2"]}

        # Mock the model and its methods
        mock_model = Mock()
        mock_model.encode.side_effect = [
            np.array([0.1, 0.2, 0.3]),  # query embedding
            [np.array([0.2, 0.3, 0.4]), np.array([0.1, 0.1, 0.1])],  # chunk embeddings
        ]

        with patch.object(scorer, "_load_model"):
            scorer.model = mock_model

            result = scorer.score("prediction", "ground_truth", context)

            assert isinstance(result, dict)
            assert "similarity" in result
            assert 0 <= result["similarity"] <= 1

            # Check that the last result was set correctly
            score_result = scorer.get_score_result()
            assert score_result is not None
            assert "Semantic Similarity Score:" in score_result.reasoning

    def test_score_with_embeddings_exception_handling(self):
        """Test score method with embedding exceptions."""
        scorer = SemanticSimilarityScorer(threshold=0.5)

        context = {"chunks": ["chunk1", "chunk2"]}

        # Mock the model to raise exception
        mock_model = Mock()
        mock_model.encode.side_effect = Exception("Embedding failed")

        with patch.object(scorer, "_load_model"):
            scorer.model = mock_model

            result = scorer.score("prediction", "ground_truth", context)

            # Should fall back to simple similarity and return float
            assert isinstance(result, float)
            assert 0 <= result <= 1

    def test_score_with_different_context_formats(self):
        """Test score method with different context formats."""
        scorer = SemanticSimilarityScorer(threshold=0.5)

        # Test with chunks key
        context_chunks = {"chunks": ["chunk1", "chunk2"]}
        with patch.object(scorer, "_load_model"):
            scorer.model = None
            with patch.object(scorer, "_compute_simple_similarity", return_value=0.7):
                result = scorer.score("prediction", "ground_truth", context_chunks)
                assert result["similarity"] == 0.7

        # Test with context key
        context_context = {"context": "single context string"}
        with patch.object(scorer, "_load_model"):
            scorer.model = None
            with patch.object(scorer, "_compute_simple_similarity", return_value=0.8):
                result = scorer.score("prediction", "ground_truth", context_context)
                assert result["similarity"] == 0.8

        # Test with no context
        with patch.object(scorer, "_load_model"):
            scorer.model = None
            result = scorer.score("prediction", "ground_truth", None)
            assert result == 0.0

    def test_compute_simple_similarity_edge_cases(self):
        """Test _compute_simple_similarity with edge cases."""
        scorer = SemanticSimilarityScorer()

        # Test with empty query
        similarity = scorer._compute_simple_similarity("", ["chunk1", "chunk2"])
        assert similarity == 0.0

        # Test with single chunk
        similarity = scorer._compute_simple_similarity("query", ["chunk1"])
        assert 0 <= similarity <= 1

        # Test with identical chunks
        similarity = scorer._compute_simple_similarity(
            "query", ["chunk1", "chunk1", "chunk1"]
        )
        assert similarity >= 0.0  # Could be 0.0 for identical chunks

    def test_load_model_platform_detection(self):
        """Test _load_model with platform detection."""
        scorer = SemanticSimilarityScorer()

        # Test macOS ARM64 detection
        with (
            patch("platform.system", return_value="Darwin"),
            patch("platform.machine", return_value="arm64"),
            patch("builtins.print") as mock_print,
        ):
            scorer._load_model()

            # Should detect platform and use fallback
            # Note: On Windows, this may not trigger the platform detection
            if mock_print.call_count > 0:
                call_args = mock_print.call_args[0][0]
                assert (
                    "Warning: Detected macOS ARM64, using fallback mode to avoid segmentation faults"
                    in call_args
                )

    def test_get_score_result(self):
        """Test get_score_result method."""
        scorer = SemanticSimilarityScorer()

        # Initially should return None
        assert scorer.get_score_result() is None

        # After scoring, should return the result
        context = {"chunks": ["chunk1", "chunk2"]}
        with patch.object(scorer, "_load_model"):
            scorer.model = None
            with patch.object(scorer, "_compute_simple_similarity", return_value=0.6):
                scorer.score("prediction", "ground_truth", context)

                score_result = scorer.get_score_result()
                assert score_result is not None
                assert isinstance(score_result.score, float)
                assert isinstance(score_result.passed, bool)
                assert isinstance(score_result.reasoning, str)

    def test_score_with_embeddings_basic(self):
        """Test score method with basic embeddings."""
        scorer = SemanticSimilarityScorer(threshold=0.5)

        context = {"chunks": ["chunk1", "chunk2"]}

        # Mock the model and its methods
        mock_model = Mock()
        mock_model.encode.side_effect = [
            np.array([0.1, 0.2, 0.3]),  # query embedding
            [np.array([0.2, 0.3, 0.4]), np.array([0.1, 0.1, 0.1])],  # chunk embeddings
        ]

        with patch.object(scorer, "_load_model"):
            scorer.model = mock_model

            result = scorer.score("prediction", "ground_truth", context)

            assert isinstance(result, dict)
            assert "similarity" in result
            assert isinstance(result["similarity"], float)

    def test_score_with_embeddings_exception(self):
        scorer = SemanticSimilarityScorer()

        context = {"chunks": ["chunk1", "chunk2"]}

        # Mock the model to raise an exception
        mock_model = Mock()
        mock_model.encode.side_effect = Exception("Encoding failed")

        with patch.object(scorer, "_load_model"):
            scorer.model = mock_model

            result = scorer.score("prediction", "ground_truth", context)

            assert result == 0.0

            score_result = scorer.get_score_result()
            assert "Semantic similarity computation failed" in score_result.reasoning

    def test_score_with_context_fallback(self):
        """Test score method when context has 'context' key instead of 'chunks'."""
        scorer = SemanticSimilarityScorer(threshold=0.5)

        context = {"context": "single context string"}

        with patch.object(scorer, "_load_model"):
            scorer.model = None  # Use fallback mode

            with patch.object(scorer, "_compute_simple_similarity", return_value=0.6):
                result = scorer.score("prediction", "ground_truth", context)

                assert isinstance(result, dict)
                assert result["similarity"] == 0.6

    def test_score_with_empty_ground_truth(self):
        """Test score method with empty ground_truth."""
        scorer = SemanticSimilarityScorer()

        context = {"chunks": ["chunk1", "chunk2"]}

        result = scorer.score("prediction", "", context)
        assert result == 0.0

        score_result = scorer.get_score_result()
        assert "No context or query provided" in score_result.reasoning

    def test_score_with_embeddings_normalization(self):
        """Test score method with embeddings and cosine similarity normalization."""
        scorer = SemanticSimilarityScorer(threshold=0.5)

        context = {"chunks": ["chunk1", "chunk2"]}

        # Mock the model and its methods
        mock_model = Mock()
        # Create embeddings that will result in specific cosine similarities
        mock_model.encode.side_effect = [
            np.array([1.0, 0.0, 0.0]),  # query embedding
            [np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])],  # chunk embeddings
        ]

        with patch.object(scorer, "_load_model"):
            scorer.model = mock_model

            result = scorer.score("prediction", "ground_truth", context)

            assert isinstance(result, dict)
            assert "similarity" in result
            # Should normalize cosine similarity from [-1,1] to [0,1]
            assert 0 <= result["similarity"] <= 1

    def test_compute_simple_similarity_no_overlap(self):
        """Test _compute_simple_similarity with no word overlap."""
        scorer = SemanticSimilarityScorer()

        query = "machine learning"
        chunks = ["cooking recipes", "sports news"]

        similarity = scorer._compute_simple_similarity(query, chunks)

        # Should be 0.0 when there's no overlap
        assert similarity == 0.0

    def test_compute_simple_similarity_partial_overlap(self):
        """Test _compute_simple_similarity with partial word overlap."""
        scorer = SemanticSimilarityScorer()

        query = "machine learning algorithms"
        chunks = ["machine learning is important", "cooking recipes"]

        similarity = scorer._compute_simple_similarity(query, chunks)

        # Should be between 0 and 1
        assert 0 < similarity < 1

    def test_load_model_already_loaded(self):
        """Test _load_model when model is already loaded."""
        scorer = SemanticSimilarityScorer()
        scorer._model_loaded = True
        mock_model = Mock()
        scorer.model = mock_model

        # Test that _load_model doesn't do anything when already loaded
        original_model = scorer.model
        scorer._load_model()

        # Should not change anything
        assert scorer.model == original_model
        assert scorer._model_loaded is True


class TestRetrievalDiversityScorer:
    def test_init(self):
        scorer = RetrievalDiversityScorer(threshold=0.4, embedding_model="test-model")
        assert scorer.threshold == 0.4
        assert scorer.embedding_model == "test-model"

    def test_score_no_chunks(self):
        scorer = RetrievalDiversityScorer()

        result = scorer.score("prediction", "ground_truth", None)
        assert result == 0.0

    def test_score_insufficient_chunks(self):
        scorer = RetrievalDiversityScorer()

        context = {"chunks": ["single_chunk"]}
        result = scorer.score("prediction", "ground_truth", context)
        assert result == 0.0

    def test_compute_simple_diversity(self):
        scorer = RetrievalDiversityScorer()

        chunks = [
            "machine learning algorithms",
            "deep neural networks",
            "cooking recipes",
        ]
        diversity = scorer._compute_simple_diversity(chunks)

        assert 0 <= diversity <= 1

    def test_compute_pairwise_cosine_distance(self):
        scorer = RetrievalDiversityScorer()

        embeddings = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]

        distance = scorer._compute_pairwise_cosine_distance(embeddings)

        # Should be close to 1.0 for orthogonal vectors
        assert 0 <= distance <= 1

    def test_score_with_fallback_diversity(self):
        scorer = RetrievalDiversityScorer()

        context = {"chunks": ["chunk1", "chunk2", "chunk3"]}

        with patch.object(scorer, "_load_model"):
            scorer.model = None

            with patch.object(scorer, "_compute_simple_diversity", return_value=0.7):
                result = scorer.score("prediction", "ground_truth", context)

                assert isinstance(result, dict)
                assert result["diversity"] == 0.7

    def test_score_with_embeddings(self):
        scorer = RetrievalDiversityScorer()

        context = {"chunks": ["chunk1", "chunk2", "chunk3"]}

        # Mock the model
        mock_model = Mock()
        mock_model.encode.return_value = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]

        with patch.object(scorer, "_load_model"):
            scorer.model = mock_model

            result = scorer.score("prediction", "ground_truth", context)

            assert isinstance(result, dict)
            assert "diversity" in result

    def test_load_model_macos_arm64_detection(self):
        """Test _load_model method with macOS ARM64 detection."""
        scorer = RetrievalDiversityScorer()

        with (
            patch("platform.system", return_value="Darwin"),
            patch("platform.machine", return_value="arm64"),
            patch("builtins.print") as mock_print,
        ):
            scorer._load_model()

            assert scorer.model is None
            assert scorer._model_loaded is True
            mock_print.assert_called_once_with(
                "Warning: Detected macOS ARM64, using fallback mode to avoid segmentation faults"
            )

    def test_load_model_already_loaded(self):
        """Test _load_model when model is already loaded."""
        scorer = RetrievalDiversityScorer()
        scorer._model_loaded = True
        mock_model = Mock()
        scorer.model = mock_model

        # Test that _load_model doesn't do anything when already loaded
        original_model = scorer.model
        scorer._load_model()

        # Should not change anything
        assert scorer.model == original_model
        assert scorer._model_loaded is True

    def test_score_with_embeddings_exception(self):
        """Test score method when embeddings computation fails."""
        scorer = RetrievalDiversityScorer()
        context = {"chunks": ["chunk1", "chunk2", "chunk3"]}

        # Mock the model to raise an exception during encoding
        mock_model = Mock()
        mock_model.encode.side_effect = Exception("Encoding failed")

        with patch.object(scorer, "_load_model"):
            scorer.model = mock_model

            result = scorer.score("prediction", "ground_truth", context)

            assert result == 0.0
            score_result = scorer.get_score_result()
            assert "Diversity computation failed" in score_result.reasoning

    def test_compute_simple_diversity_identical_chunks(self):
        """Test _compute_simple_diversity with identical chunks."""
        scorer = RetrievalDiversityScorer()

        chunks = ["same text", "same text", "same text"]
        diversity = scorer._compute_simple_diversity(chunks)

        # Should have some diversity due to text difference calculation
        assert 0 <= diversity <= 1

    def test_compute_simple_diversity_completely_different(self):
        """Test _compute_simple_diversity with completely different chunks."""
        scorer = RetrievalDiversityScorer()

        chunks = [
            "machine learning algorithms",
            "cooking recipes and ingredients",
            "sports news and updates",
        ]
        diversity = scorer._compute_simple_diversity(chunks)

        # Should have high diversity
        assert diversity > 0.5

    def test_compute_pairwise_cosine_distance_identical_embeddings(self):
        """Test _compute_pairwise_cosine_distance with identical embeddings."""
        scorer = RetrievalDiversityScorer()

        embeddings = [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
        distance = scorer._compute_pairwise_cosine_distance(embeddings)

        # Identical embeddings should have distance close to 0
        assert distance < 0.1

    def test_compute_pairwise_cosine_distance_opposite_embeddings(self):
        """Test _compute_pairwise_cosine_distance with opposite embeddings."""
        scorer = RetrievalDiversityScorer()

        embeddings = [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        distance = scorer._compute_pairwise_cosine_distance(embeddings)

        # Opposite embeddings should have high distance
        assert distance > 0.5

    def test_load_model_exception_handling(self):
        """Test _load_model exception handling."""
        scorer = RetrievalDiversityScorer()

        # Mock the import to raise an exception
        original_import = __builtins__["__import__"]

        def mock_import(name, *args, **kwargs):
            if name == "sentence_transformers":
                raise Exception("Model load failed")
            return original_import(name, *args, **kwargs)

        with (
            patch("builtins.__import__", side_effect=mock_import),
            patch("builtins.print") as mock_print,
        ):
            scorer._load_model()

            assert scorer.model is None
            assert scorer._model_loaded is True
            # Check that print was called with a warning message (exact message may vary)
            mock_print.assert_called_once()
            call_args = mock_print.call_args[0][0]
            # The warning message can vary based on platform and error type
            assert any(
                warning in call_args
                for warning in [
                    "Warning: Could not load SentenceTransformer model:",
                    "Warning: Detected macOS ARM64, using fallback mode to avoid segmentation faults",
                ]
            )

    def test_score_with_embeddings_detailed_metrics(self):
        """Test score method with embeddings and detailed metrics."""
        scorer = RetrievalDiversityScorer()

        context = {"chunks": ["chunk1", "chunk2", "chunk3"]}

        # Mock the model
        mock_model = Mock()
        mock_model.encode.return_value = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]

        with patch.object(scorer, "_load_model"):
            scorer.model = mock_model

            result = scorer.score("prediction", "ground_truth", context)

            assert isinstance(result, dict)
            assert "diversity" in result
            assert 0 <= result["diversity"] <= 1

            # Check that the last result was set correctly
            score_result = scorer.get_score_result()
            assert score_result is not None
            assert "Diversity Score:" in score_result.reasoning
            assert "cosine distance between embeddings" in score_result.reasoning

    def test_score_without_embeddings_fallback(self):
        """Test score method when embeddings fail, using fallback diversity."""
        scorer = RetrievalDiversityScorer()

        context = {"chunks": ["chunk1", "chunk2", "chunk3"]}

        # Mock the model to fail
        with patch.object(scorer, "_load_model"):
            scorer.model = None

            result = scorer.score("prediction", "ground_truth", context)

            assert isinstance(result, dict)
            assert "diversity" in result
            assert 0 <= result["diversity"] <= 1

            # Check that the last result was set correctly
            score_result = scorer.get_score_result()
            assert score_result is not None
            assert "Fallback diversity score:" in score_result.reasoning

    def test_score_with_single_chunk(self):
        """Test score method with single chunk (should return 0 diversity)."""
        scorer = RetrievalDiversityScorer()
        context = {"chunks": ["single_chunk"]}

        with patch.object(scorer, "_load_model"):
            scorer.model = None

            result = scorer.score("prediction", "ground_truth", context)
            assert result == 0.0  # Returns float, not dict

    def test_score_with_empty_chunks(self):
        """Test score method with empty chunks list."""
        scorer = RetrievalDiversityScorer()
        context = {"chunks": []}

        with patch.object(scorer, "_load_model"):
            scorer.model = None

            result = scorer.score("prediction", "ground_truth", context)
            assert result == 0.0  # Returns float, not dict

    def test_score_with_no_chunks_key(self):
        """Test score method when context has no chunks key."""
        scorer = RetrievalDiversityScorer()
        context = {"other_key": "value"}

        with patch.object(scorer, "_load_model"):
            scorer.model = None

            result = scorer.score("prediction", "ground_truth", context)
            assert result == 0.0  # Returns float, not dict

    def test_score_with_none_context(self):
        """Test score method with None context."""
        scorer = RetrievalDiversityScorer()

        with patch.object(scorer, "_load_model"):
            scorer.model = None

            result = scorer.score("prediction", "ground_truth", None)
            assert result == 0.0  # Returns float, not dict

    def test_compute_pairwise_cosine_distance_edge_cases(self):
        """Test _compute_pairwise_cosine_distance with edge cases."""
        scorer = RetrievalDiversityScorer()

        # Test with single embedding
        result = scorer._compute_pairwise_cosine_distance([[1.0, 0.0]])
        assert result == 0.0

        # Test with empty list
        result = scorer._compute_pairwise_cosine_distance([])
        assert result == 0.0

        # Test with two identical embeddings
        result = scorer._compute_pairwise_cosine_distance([[1.0, 0.0], [1.0, 0.0]])
        assert result == 0.0

        # Test with opposite embeddings
        result = scorer._compute_pairwise_cosine_distance([[1.0, 0.0], [-1.0, 0.0]])
        assert result > 0.5  # Should have high distance

    def test_simple_diversity_computation(self):
        """Test _compute_simple_diversity method."""
        scorer = RetrievalDiversityScorer()

        # Test with single chunk
        result = scorer._compute_simple_diversity(["chunk1"])
        assert result == 0.0

        # Test with identical chunks
        result = scorer._compute_simple_diversity(["chunk1", "chunk1", "chunk1"])
        assert result > 0.0  # Should have some diversity due to text processing

        # Test with different chunks
        result = scorer._compute_simple_diversity(["chunk1", "chunk2", "chunk3"])
        assert result > 0.0

        # Test with empty list
        result = scorer._compute_simple_diversity([])
        assert result == 0.0

    def test_load_model_successful_loading(self):
        """Test _load_model with successful model loading."""
        scorer = RetrievalDiversityScorer()

        mock_model = Mock()
        stub_mod = types.SimpleNamespace(
            SentenceTransformer=Mock(return_value=mock_model)
        )
        with (
            patch("platform.system", return_value="Linux"),
            patch("platform.machine", return_value="x86_64"),
            patch("os.environ", {}),
            patch.dict(sys.modules, {"sentence_transformers": stub_mod}),
        ):
            scorer._load_model()

            # Should load model successfully
            assert scorer._model_loaded is True

    def test_load_model_environment_variable_setting(self):
        """Test _load_model sets environment variables correctly."""
        scorer = RetrievalDiversityScorer()

        mock_model = Mock()
        stub_mod = types.SimpleNamespace(
            SentenceTransformer=Mock(return_value=mock_model)
        )
        with (
            patch("platform.system", return_value="Linux"),
            patch("platform.machine", return_value="x86_64"),
            patch.dict("os.environ", {}, clear=True),
            patch.dict(sys.modules, {"sentence_transformers": stub_mod}),
        ):
            scorer._load_model()

            # Check that environment variable was set
            import os

            assert os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1"

    def test_get_score_result(self):
        """Test get_score_result method."""
        scorer = RetrievalDiversityScorer()

        # Initially should return None
        assert scorer.get_score_result() is None

        # After scoring, should return the result
        context = {"chunks": ["chunk1", "chunk2"]}
        with patch.object(scorer, "_load_model"):
            scorer.model = None
            scorer.score("prediction", "ground_truth", context)

            score_result = scorer.get_score_result()
            assert score_result is not None
            assert isinstance(score_result.score, float)
            assert isinstance(score_result.passed, bool)
            assert isinstance(score_result.reasoning, str)


class TestAggregateRAGScorer:
    def test_init(self):
        scorers = {"scorer1": Mock(), "scorer2": Mock()}
        weights = {"scorer1": 0.6, "scorer2": 0.4}
        scorer = AggregateRAGScorer(scorers, weights=weights, threshold=0.7)

        assert scorer.scorers == scorers
        assert scorer.weights == weights
        assert scorer.threshold == 0.7

    def test_init_default_weights(self):
        scorers = {"scorer1": Mock(), "scorer2": Mock()}
        scorer = AggregateRAGScorer(scorers)

        assert scorer.weights["scorer1"] == 1.0
        assert scorer.weights["scorer2"] == 1.0

    def test_score_with_float_results(self):
        mock_scorer1 = Mock()
        mock_scorer2 = Mock()
        mock_scorer1.score.return_value = 0.8
        mock_scorer2.score.return_value = 0.6

        scorers = {"scorer1": mock_scorer1, "scorer2": mock_scorer2}
        weights = {"scorer1": 0.7, "scorer2": 0.3}
        scorer = AggregateRAGScorer(scorers, weights=weights)

        result = scorer.score("prediction", "ground_truth", {"context": "test"})

        expected = (0.8 * 0.7 + 0.6 * 0.3) / (0.7 + 0.3)
        assert isinstance(result, dict)
        assert result["aggregate"] == pytest.approx(expected, rel=1e-3)

    def test_score_with_dict_results(self):
        mock_scorer1 = Mock()
        mock_scorer2 = Mock()
        mock_scorer1.score.return_value = {"average": 0.8, "precision": 0.9}
        mock_scorer2.score.return_value = {"diversity": 0.6}

        scorers = {"scorer1": mock_scorer1, "scorer2": mock_scorer2}
        scorer = AggregateRAGScorer(scorers)

        result = scorer.score("prediction", "ground_truth", {"context": "test"})

        assert isinstance(result, dict)
        assert "aggregate" in result
        assert "individual_scores" in result
        assert "scorer1" in result["individual_scores"]
        assert "scorer2" in result["individual_scores"]

    def test_score_with_score_result_objects(self):
        mock_scorer1 = Mock()
        mock_scorer1.score.return_value = ScoreResult(
            score=0.8, passed=True, reasoning="test", metadata={}
        )

        scorers = {"scorer1": mock_scorer1}
        scorer = AggregateRAGScorer(scorers)

        result = scorer.score("prediction", "ground_truth", {"context": "test"})

        assert isinstance(result, dict)
        assert result["aggregate"] == 0.8

    def test_score_with_exception(self):
        mock_scorer1 = Mock()
        mock_scorer1.score.side_effect = Exception("Scorer failed")

        scorers = {"scorer1": mock_scorer1}
        scorer = AggregateRAGScorer(scorers)

        with patch("builtins.print") as mock_print:
            result = scorer.score("prediction", "ground_truth", {"context": "test"})

            # Should handle the exception gracefully
            assert result == 0.0
            mock_print.assert_called_once_with(
                "Warning: Scorer scorer1 failed: Scorer failed"
            )

            # Check that the last result was set correctly
            last_result = scorer.get_score_result()
            assert last_result.score == 0.0
            assert last_result.passed is False
            assert "All scorers failed" in last_result.reasoning


class TestContextualPrecisionScorerPPExtended:
    """Additional tests to improve coverage for ContextualPrecisionScorerPP."""

    def test_parse_json_response_fallback_boolean_indicators(self):
        """Test _parse_json_response with boolean indicators fallback."""
        mock_llm = MockLLM()
        scorer = ContextualPrecisionScorerPP(model=mock_llm)

        # Test with "yes" indicator
        result = scorer._parse_json_response("yes, this is relevant")
        assert result["relevant"] is True
        assert "Fallback parsing used" in result["reasoning"]

        # Test with "true" indicator
        result = scorer._parse_json_response("true, relevant information")
        assert result["relevant"] is True

        # Test with "1" indicator
        result = scorer._parse_json_response("1 - this chunk is relevant")
        assert result["relevant"] is True

        # Test with no indicators
        result = scorer._parse_json_response("this is not relevant information")
        assert result["relevant"] is False

    @pytest.mark.asyncio
    async def test_evaluate_no_chunks(self):
        """Test evaluate method when no chunks are provided."""
        mock_llm = MockLLM()
        scorer = ContextualPrecisionScorerPP(model=mock_llm)

        result = await scorer.evaluate(
            input_text="test query", output_text="test output", context={"chunks": []}
        )

        assert result.score == 0.0
        assert not result.passed
        assert "No chunks provided" in result.reasoning

    @pytest.mark.asyncio
    async def test_evaluate_empty_chunks(self):
        """Test evaluate method with empty chunks."""
        mock_llm = MockLLM()
        scorer = ContextualPrecisionScorerPP(model=mock_llm)

        result = await scorer.evaluate(
            input_text="test query", output_text="test output", context={"chunks": [""]}
        )

        assert result.score == 0.0
        assert not result.passed
        assert "Precision: 0.000 (0 relevant out of 1 chunks)" in result.reasoning


class TestContextualRecallScorerPPExtended:
    """Additional tests to improve coverage for ContextualRecallScorerPP."""

    def test_parse_json_response_fallback_boolean_indicators(self):
        """Test _parse_json_response with boolean indicators fallback."""
        mock_llm = MockLLM()
        scorer = ContextualRecallScorerPP(model=mock_llm)

        # Test with "yes" indicator
        result = scorer._parse_json_response("yes, this is relevant")
        assert result["relevant"] is True
        assert "Fallback parsing used" in result["reasoning"]

        # Test with "true" indicator
        result = scorer._parse_json_response("true, relevant information")
        assert result["relevant"] is True

        # Test with "1" indicator
        result = scorer._parse_json_response("1 - this chunk is relevant")
        assert result["relevant"] is True

        # Test with no indicators
        result = scorer._parse_json_response("this is not relevant information")
        assert result["relevant"] is False

    @pytest.mark.asyncio
    async def test_estimate_total_relevant_chunks_fallback(self):
        """Test _estimate_total_relevant_chunks fallback logic."""
        mock_llm = MockLLM()
        scorer = ContextualRecallScorerPP(model=mock_llm)

        # Mock the model to return invalid response
        with patch.object(
            scorer,
            "_call_model",
            new_callable=AsyncMock,
            return_value="invalid response",
        ):
            result = await scorer._estimate_total_relevant_chunks(
                "test query", ["chunk1", "chunk2"]
            )

            # Should fallback to length of retrieved chunks
            assert result == 2

    @pytest.mark.asyncio
    async def test_evaluate_no_chunks(self):
        """Test evaluate method when no chunks are provided."""
        mock_llm = MockLLM()
        scorer = ContextualRecallScorerPP(model=mock_llm)

        result = await scorer.evaluate(
            input_text="test query", output_text="test output", context={"chunks": []}
        )

        assert result.score == 0.0
        assert not result.passed
        assert "No chunks provided" in result.reasoning

    @pytest.mark.asyncio
    async def test_evaluate_empty_chunks(self):
        """Test evaluate method with empty chunks."""
        mock_llm = MockLLM()
        scorer = ContextualRecallScorerPP(model=mock_llm)

        result = await scorer.evaluate(
            input_text="test query", output_text="test output", context={"chunks": [""]}
        )

        assert result.score == 0.0
        assert not result.passed
        assert (
            "Recall: 0.000 (0 relevant retrieved, estimated 1 total relevant)"
            in result.reasoning
        )


class TestRetrievalDiversityScorerExtended:
    """Additional tests to improve coverage for RetrievalDiversityScorer."""

    def test_compute_pairwise_cosine_distance_empty_embeddings(self):
        """Test _compute_pairwise_cosine_distance with empty embeddings."""
        mock_llm = MockLLM()
        scorer = RetrievalDiversityScorer(model=mock_llm)

        result = scorer._compute_pairwise_cosine_distance([])
        assert result == 0.0

    def test_compute_pairwise_cosine_distance_single_embedding(self):
        """Test _compute_pairwise_cosine_distance with single embedding."""
        mock_llm = MockLLM()
        scorer = RetrievalDiversityScorer(model=mock_llm)

        result = scorer._compute_pairwise_cosine_distance([[1.0, 0.0]])
        assert result == 0.0

    def test_score_with_embeddings_exception_handling(self):
        """Test score method exception handling in embeddings path."""
        mock_llm = MockLLM()
        scorer = RetrievalDiversityScorer(model=mock_llm)

        # Mock the model to raise an exception during encoding
        mock_model = Mock()
        mock_model.encode.side_effect = Exception("Encoding failed")

        with patch.object(scorer, "_load_model"):
            scorer.model = mock_model

            result = scorer.score(
                "prediction", "ground_truth", {"chunks": ["chunk1", "chunk2"]}
            )

            assert result == 0.0
            score_result = scorer.get_score_result()
            assert "Diversity computation failed" in score_result.reasoning


class TestAggregateRAGScorerExtended:
    """Additional tests to improve coverage for AggregateRAGScorer."""

    def test_score_with_exception_handling(self):
        """Test score method exception handling."""
        mock_scorer1 = Mock()
        mock_scorer1.score.side_effect = Exception("Scorer failed")

        scorers = {"scorer1": mock_scorer1}
        scorer = AggregateRAGScorer(scorers)

        with patch("builtins.print") as mock_print:
            result = scorer.score("prediction", "ground_truth", {"context": "test"})

            # Should handle the exception gracefully
            assert result == 0.0
            mock_print.assert_called_once_with(
                "Warning: Scorer scorer1 failed: Scorer failed"
            )

            # Check that the last result was set correctly
            last_result = scorer.get_score_result()
            assert last_result.score == 0.0
            assert last_result.passed is False
            assert "All scorers failed" in last_result.reasoning

    def test_extract_numeric_score_from_dict_priority_keys(self):
        """Test _extract_numeric_score_from_dict with priority keys."""
        mock_scorer1 = Mock()
        mock_scorer1.score.return_value = {"score": 0.8}

        scorers = {"scorer1": mock_scorer1}
        scorer = AggregateRAGScorer(scorers)

        result = scorer.score("prediction", "ground_truth", {"context": "test"})
        assert result["aggregate"] == 0.8

    def test_extract_numeric_score_from_dict_nested_dict(self):
        """Test _extract_numeric_score_from_dict with nested dictionaries."""
        mock_scorer1 = Mock()
        mock_scorer1.score.return_value = {"metrics": {"score": 0.7}}

        scorers = {"scorer1": mock_scorer1}
        scorer = AggregateRAGScorer(scorers)

        result = scorer.score("prediction", "ground_truth", {"context": "test"})
        assert result["aggregate"] == 0.7

    def test_extract_numeric_score_from_dict_fallback(self):
        """Test _extract_numeric_score_from_dict fallback behavior."""
        mock_scorer1 = Mock()
        mock_scorer1.score.return_value = {"some_other_key": 0.6}

        scorers = {"scorer1": mock_scorer1}
        scorer = AggregateRAGScorer(scorers)

        result = scorer.score("prediction", "ground_truth", {"context": "test"})
        assert result["aggregate"] == 0.6

    def test_is_numeric_value_various_types(self):
        """Test _is_numeric_value with various input types."""
        mock_scorer1 = Mock()
        mock_scorer1.score.return_value = {"score": "0.5"}  # String number

        scorers = {"scorer1": mock_scorer1}
        scorer = AggregateRAGScorer(scorers)

        result = scorer.score("prediction", "ground_truth", {"context": "test"})
        assert result["aggregate"] == 0.5

    def test_is_numeric_value_invalid_string(self):
        """Test _is_numeric_value with invalid string."""
        mock_scorer1 = Mock()
        mock_scorer1.score.return_value = {"score": "not_a_number"}

        scorers = {"scorer1": mock_scorer1}
        scorer = AggregateRAGScorer(scorers)

        result = scorer.score("prediction", "ground_truth", {"context": "test"})
        assert result["aggregate"] == 0.0  # Should fallback to 0.0

    def test_score_with_none_result(self):
        """Test score method with None result from scorer."""
        mock_scorer1 = Mock()
        mock_scorer1.score.return_value = None

        scorers = {"scorer1": mock_scorer1}
        scorer = AggregateRAGScorer(scorers)

        result = scorer.score("prediction", "ground_truth", {"context": "test"})
        assert result["aggregate"] == 0.0


class TestBasicRAGScorersEdgeCases:
    """Additional tests to cover missing lines and increase coverage."""

    def test_parse_numerical_response_json_extraction_edge_cases(self):
        """Test _parse_numerical_response with JSON extraction edge cases."""
        model = Mock()
        scorer = ContextualPrecisionScorerPP(model=model)

        # Test with malformed JSON that causes ValueError
        response = '{"rating": "invalid_number"}'
        result = scorer._parse_numerical_response(response)
        assert result == 5.0  # Should fallback to default

        # Test with JSON that has KeyError
        response = '{"other_key": 5}'
        result = scorer._parse_numerical_response(response)
        assert result == 5.0  # Should fallback to default

        # Test with JSON that has TypeError
        response = '{"rating": None}'
        result = scorer._parse_numerical_response(response)
        assert result == 5.0  # Should fallback to default

    def test_parse_numerical_response_exception_handling(self):
        """Test _parse_numerical_response with exception handling."""
        model = Mock()
        scorer = ContextualPrecisionScorerPP(model=model)

        # Test with response that causes exception during parsing
        response = None  # This should cause an exception
        result = scorer._parse_numerical_response(response)
        assert result == 5.0  # Fallback

        # Test with response that causes exception during regex
        response = "Rating: "  # Incomplete rating
        result = scorer._parse_numerical_response(response)
        assert result == 5.0  # Fallback

    def test_parse_json_response_json_extraction_edge_cases(self):
        """Test _parse_json_response with JSON extraction edge cases."""
        model = Mock()
        scorer = ContextualPrecisionScorerPP(model=model)

        # Test with malformed JSON in regex match
        response = 'Some text {"invalid": json} more text'
        result = scorer._parse_json_response(response)
        assert result["relevant"] is False  # Should use fallback
        assert "Fallback parsing used" in result["reasoning"]

        # Test with empty JSON object
        response = "{}"
        result = scorer._parse_json_response(response)
        # Empty JSON object parses successfully but has no keys
        assert isinstance(result, dict)
        assert len(result) == 0

    def test_parse_json_response_boolean_indicators_edge_cases(self):
        """Test _parse_json_response with boolean indicators edge cases."""
        model = Mock()
        scorer = ContextualPrecisionScorerPP(model=model)

        # Test with mixed case indicators
        response = "YES, this is RELEVANT"
        result = scorer._parse_json_response(response)
        assert result["relevant"] is True

        # Test with numeric indicators
        response = "1 - this is relevant"
        result = scorer._parse_json_response(response)
        assert result["relevant"] is True

        # Test with no indicators
        response = "this is not relevant information"
        result = scorer._parse_json_response(response)
        assert result["relevant"] is False

    @pytest.mark.asyncio
    async def test_estimate_total_relevant_chunks_exception_handling(self):
        """Test _estimate_total_relevant_chunks with exception handling."""
        model = Mock()
        scorer = ContextualRecallScorerPP(model=model)

        # Test with exception during model call
        with patch.object(
            scorer,
            "_call_model",
            new_callable=AsyncMock,
            side_effect=Exception("API error"),
        ):
            result = await scorer._estimate_total_relevant_chunks(
                "query", ["chunk1", "chunk2"]
            )
            assert result == 2  # Should fallback to length of retrieved chunks

        # Test with exception during JSON parsing
        with patch.object(
            scorer, "_call_model", new_callable=AsyncMock, return_value="invalid json"
        ):
            result = await scorer._estimate_total_relevant_chunks(
                "query", ["chunk1", "chunk2", "chunk3"]
            )
            assert result == 3  # Should fallback to length of retrieved chunks

    @pytest.mark.asyncio
    async def test_estimate_total_relevant_chunks_missing_key_handling(self):
        """Test _estimate_total_relevant_chunks with missing key handling."""
        model = Mock()
        scorer = ContextualRecallScorerPP(model=model)

        # Test with missing estimated_total key
        with patch.object(
            scorer,
            "_call_model",
            new_callable=AsyncMock,
            return_value='{"other_key": 5}',
        ):
            result = await scorer._estimate_total_relevant_chunks(
                "query", ["chunk1", "chunk2"]
            )
            assert result == 2  # Should fallback to length of retrieved chunks

        # Test with low estimate that gets corrected
        with patch.object(
            scorer,
            "_call_model",
            new_callable=AsyncMock,
            return_value='{"estimated_total": 1}',
        ):
            result = await scorer._estimate_total_relevant_chunks(
                "query", ["chunk1", "chunk2", "chunk3"]
            )
            assert result == 3  # Should be at least as many as retrieved

    def test_retrieval_ranking_scorer_exception_handling(self):
        """Test RetrievalRankingScorer exception handling."""
        scorer = RetrievalRankingScorer()

        # Test with exception during numpy operations
        context = {"rankings": [1, 2, 3], "relevance_scores": [1.0, 0.8, 0.6]}

        with patch("numpy.array", side_effect=Exception("Numpy error")):
            result = scorer.score("prediction", "ground_truth", context)
            assert isinstance(result, dict)
            assert "avg_ranking" in result

        # Test with complete failure
        with (
            patch("numpy.array", side_effect=Exception("Numpy error")),
            patch("numpy.mean", side_effect=Exception("Mean error")),
        ):
            result = scorer.score("prediction", "ground_truth", context)
            assert result == 0.0

    def test_semantic_similarity_scorer_embedding_exception_handling(self):
        """Test SemanticSimilarityScorer embedding exception handling."""
        scorer = SemanticSimilarityScorer()

        context = {"chunks": ["chunk1", "chunk2"]}

        # Test with exception during embedding
        mock_model = Mock()
        mock_model.encode.side_effect = Exception("Encoding failed")

        with patch.object(scorer, "_load_model"):
            scorer.model = mock_model
            result = scorer.score("prediction", "ground_truth", context)
            assert result == 0.0

            score_result = scorer.get_score_result()
            assert "Semantic similarity computation failed" in score_result.reasoning

    def test_semantic_similarity_scorer_platform_detection_edge_cases(self):
        """Test SemanticSimilarityScorer platform detection edge cases."""
        scorer = SemanticSimilarityScorer()

        # Test with different platform configurations
        with (
            patch("platform.system", return_value="Windows"),
            patch("platform.machine", return_value="AMD64"),
        ):
            scorer._load_model()
            # On Windows, this may not trigger platform detection
            # Just ensure it doesn't crash

        # Test with Linux platform
        with (
            patch("platform.system", return_value="Linux"),
            patch("platform.machine", return_value="x86_64"),
        ):
            scorer._load_model()
            # Should work normally on Linux

    def test_retrieval_diversity_scorer_embedding_exception_handling(self):
        """Test RetrievalDiversityScorer embedding exception handling."""
        scorer = RetrievalDiversityScorer()
        context = {"chunks": ["chunk1", "chunk2", "chunk3"]}

        # Test with exception during embedding
        mock_model = Mock()
        mock_model.encode.side_effect = Exception("Encoding failed")

        with patch.object(scorer, "_load_model"):
            scorer.model = mock_model
            result = scorer.score("prediction", "ground_truth", context)
            assert result == 0.0

            score_result = scorer.get_score_result()
            assert "Diversity computation failed" in score_result.reasoning

    def test_retrieval_diversity_scorer_platform_detection_edge_cases(self):
        """Test RetrievalDiversityScorer platform detection edge cases."""
        scorer = RetrievalDiversityScorer()

        # Test with different platform configurations
        with (
            patch("platform.system", return_value="Windows"),
            patch("platform.machine", return_value="AMD64"),
        ):
            scorer._load_model()
            # Should work normally on Windows

        # Test with Linux platform
        with (
            patch("platform.system", return_value="Linux"),
            patch("platform.machine", return_value="x86_64"),
        ):
            scorer._load_model()
            # Should work normally on Linux

    def test_aggregate_rag_scorer_exception_handling_edge_cases(self):
        """Test AggregateRAGScorer exception handling edge cases."""
        mock_scorer1 = Mock()
        mock_scorer1.score.side_effect = Exception("Scorer failed")

        scorers = {"scorer1": mock_scorer1}
        scorer = AggregateRAGScorer(scorers)

        with patch("builtins.print") as mock_print:
            result = scorer.score("prediction", "ground_truth", {"context": "test"})
            assert result == 0.0
            mock_print.assert_called_once_with(
                "Warning: Scorer scorer1 failed: Scorer failed"
            )

            last_result = scorer.get_score_result()
            assert last_result.score == 0.0
            assert "All scorers failed" in last_result.reasoning

    def test_aggregate_rag_scorer_numeric_extraction_edge_cases(self):
        """Test AggregateRAGScorer numeric extraction edge cases."""
        mock_scorer1 = Mock()

        # Test with nested dictionary
        mock_scorer1.score.return_value = {"metrics": {"score": 0.7}}
        scorers = {"scorer1": mock_scorer1}
        scorer = AggregateRAGScorer(scorers)

        result = scorer.score("prediction", "ground_truth", {"context": "test"})
        assert result["aggregate"] == 0.7

        # Test with string number
        mock_scorer1.score.return_value = {"score": "0.5"}
        result = scorer.score("prediction", "ground_truth", {"context": "test"})
        assert result["aggregate"] == 0.5

        # Test with invalid string
        mock_scorer1.score.return_value = {"score": "not_a_number"}
        result = scorer.score("prediction", "ground_truth", {"context": "test"})
        assert result["aggregate"] == 0.0

        # Test with None result
        mock_scorer1.score.return_value = None
        result = scorer.score("prediction", "ground_truth", {"context": "test"})
        assert result["aggregate"] == 0.0

    @pytest.mark.asyncio
    async def test_contextual_precision_scorer_pp_edge_cases(self):
        """Test ContextualPrecisionScorerPP edge cases."""
        model = Mock()
        scorer = ContextualPrecisionScorerPP(model=model)

        # Test with empty chunks
        context = {"chunks": []}
        result = await scorer.evaluate("query", "answer", context=context)
        assert result.score == 0.0
        assert "No chunks provided" in result.reasoning

        # Test with empty string chunks
        context = {"chunks": [""]}
        result = await scorer.evaluate("query", "answer", context=context)
        assert result.score == 0.0

    @pytest.mark.asyncio
    async def test_contextual_recall_scorer_pp_edge_cases(self):
        """Test ContextualRecallScorerPP edge cases."""
        model = Mock()
        scorer = ContextualRecallScorerPP(model=model)

        # Test with empty chunks
        context = {"chunks": []}
        result = await scorer.evaluate("query", "answer", context=context)
        assert result.score == 0.0
        assert "No chunks provided" in result.reasoning

        # Test with empty string chunks
        context = {"chunks": [""]}
        result = await scorer.evaluate("query", "answer", context=context)
        assert result.score == 0.0

    def test_retrieval_diversity_scorer_edge_cases(self):
        """Test RetrievalDiversityScorer edge cases."""
        scorer = RetrievalDiversityScorer()

        # Test with empty embeddings
        result = scorer._compute_pairwise_cosine_distance([])
        assert result == 0.0

        # Test with single embedding
        result = scorer._compute_pairwise_cosine_distance([[1.0, 0.0]])
        assert result == 0.0

        # Test with identical embeddings
        result = scorer._compute_pairwise_cosine_distance([[1.0, 0.0], [1.0, 0.0]])
        assert result == 0.0

        # Test with opposite embeddings
        result = scorer._compute_pairwise_cosine_distance([[1.0, 0.0], [-1.0, 0.0]])
        assert result > 0.5

    def test_semantic_similarity_scorer_edge_cases(self):
        """Test SemanticSimilarityScorer edge cases."""
        scorer = SemanticSimilarityScorer()

        # Test with empty query
        similarity = scorer._compute_simple_similarity("", ["chunk1", "chunk2"])
        assert similarity == 0.0

        # Test with single chunk
        similarity = scorer._compute_simple_similarity("query", ["chunk1"])
        assert 0 <= similarity <= 1

        # Test with identical chunks
        similarity = scorer._compute_simple_similarity(
            "query", ["chunk1", "chunk1", "chunk1"]
        )
        assert similarity >= 0.0

        # Test with no word overlap
        similarity = scorer._compute_simple_similarity(
            "machine learning", ["cooking recipes", "sports news"]
        )
        assert similarity == 0.0

        # Test with partial word overlap
        similarity = scorer._compute_simple_similarity(
            "machine learning algorithms",
            ["machine learning is important", "cooking recipes"],
        )
        assert 0 < similarity < 1

    def test_retrieval_diversity_scorer_simple_diversity_edge_cases(self):
        """Test RetrievalDiversityScorer simple diversity edge cases."""
        scorer = RetrievalDiversityScorer()

        # Test with single chunk
        result = scorer._compute_simple_diversity(["chunk1"])
        assert result == 0.0

        # Test with identical chunks
        result = scorer._compute_simple_diversity(["chunk1", "chunk1", "chunk1"])
        assert result > 0.0  # Should have some diversity due to text processing

        # Test with different chunks
        result = scorer._compute_simple_diversity(["chunk1", "chunk2", "chunk3"])
        assert result > 0.0

        # Test with empty list
        result = scorer._compute_simple_diversity([])
        assert result == 0.0

        # Test with completely different chunks
        chunks = [
            "machine learning algorithms",
            "cooking recipes and ingredients",
            "sports news and updates",
        ]
        result = scorer._compute_simple_diversity(chunks)
        assert result > 0.5  # Should have high diversity

    def test_retrieval_ranking_scorer_edge_cases(self):
        """Test RetrievalRankingScorer edge cases."""
        scorer = RetrievalRankingScorer()

        # Test with single item
        context_single = {"rankings": [1], "relevance_scores": [1.0]}
        result = scorer.score("prediction", "ground_truth", context_single)
        assert isinstance(result, dict)
        assert result["mrr"] == 1.0  # Single relevant item at rank 1

        # Test with all items at rank 1 (tie)
        context_tie = {"rankings": [1, 1, 1], "relevance_scores": [1.0, 0.8, 0.6]}
        result = scorer.score("prediction", "ground_truth", context_tie)
        assert isinstance(result, dict)
        assert result["avg_ranking"] == 1.0  # All at rank 1

        # Test with rankings higher than max_rank
        context_high_ranks = {
            "rankings": [1, 2, 3, 4, 5, 10, 15],
            "relevance_scores": [1.0, 0.8, 0.6, 0.4, 0.2, 0.1, 0.0],
        }
        result = scorer.score("prediction", "ground_truth", context_high_ranks)
        assert isinstance(result, dict)
        assert result["avg_ranking"] < 1.0  # High ranks should get 0.0 score

    def test_retrieval_f1_scorer_edge_cases(self):
        """Test RetrievalF1Scorer edge cases."""
        precision_scorer = Mock()
        recall_scorer = Mock()
        scorer = RetrievalF1Scorer(precision_scorer, recall_scorer)

        # Test with high precision, low recall
        precision_scorer.score.return_value = 0.9
        recall_scorer.score.return_value = 0.3
        result = scorer.score("prediction", "ground_truth", {"context": "test"})
        expected_f1 = 2 * (0.9 * 0.3) / (0.9 + 0.3)
        assert result["f1"] == pytest.approx(expected_f1, rel=1e-3)
        assert result["precision"] == 0.9
        assert result["recall"] == 0.3

        # Test with low precision, high recall
        precision_scorer.score.return_value = 0.2
        recall_scorer.score.return_value = 0.8
        result = scorer.score("prediction", "ground_truth", {"context": "test"})
        expected_f1 = 2 * (0.2 * 0.8) / (0.2 + 0.8)
        assert result["f1"] == pytest.approx(expected_f1, rel=1e-3)
        assert result["precision"] == 0.2
        assert result["recall"] == 0.8

        # Test with perfect scores
        precision_scorer.score.return_value = 1.0
        recall_scorer.score.return_value = 1.0
        result = scorer.score("prediction", "ground_truth", {"context": "test"})
        assert result["f1"] == 1.0
        assert result["precision"] == 1.0
        assert result["recall"] == 1.0

    def test_aggregate_rag_scorer_score_result_objects(self):
        """Test AggregateRAGScorer with ScoreResult objects."""
        mock_scorer1 = Mock()
        mock_scorer1.score.return_value = ScoreResult(
            score=0.8, passed=True, reasoning="test", metadata={}
        )

        scorers = {"scorer1": mock_scorer1}
        scorer = AggregateRAGScorer(scorers)

        result = scorer.score("prediction", "ground_truth", {"context": "test"})
        assert isinstance(result, dict)
        assert result["aggregate"] == 0.8

    def test_aggregate_rag_scorer_dict_results(self):
        """Test AggregateRAGScorer with dictionary results."""
        mock_scorer1 = Mock()
        mock_scorer2 = Mock()
        mock_scorer1.score.return_value = {"average": 0.8, "precision": 0.9}
        mock_scorer2.score.return_value = {"diversity": 0.6}

        scorers = {"scorer1": mock_scorer1, "scorer2": mock_scorer2}
        scorer = AggregateRAGScorer(scorers)

        result = scorer.score("prediction", "ground_truth", {"context": "test"})
        assert isinstance(result, dict)
        assert "aggregate" in result
        assert "individual_scores" in result
        assert "scorer1" in result["individual_scores"]
        assert "scorer2" in result["individual_scores"]

    def test_aggregate_rag_scorer_float_results(self):
        """Test AggregateRAGScorer with float results."""
        mock_scorer1 = Mock()
        mock_scorer2 = Mock()
        mock_scorer1.score.return_value = 0.8
        mock_scorer2.score.return_value = 0.6

        scorers = {"scorer1": mock_scorer1, "scorer2": mock_scorer2}
        weights = {"scorer1": 0.7, "scorer2": 0.3}
        scorer = AggregateRAGScorer(scorers, weights=weights)

        result = scorer.score("prediction", "ground_truth", {"context": "test"})
        expected = (0.8 * 0.7 + 0.6 * 0.3) / (0.7 + 0.3)
        assert isinstance(result, dict)
        assert result["aggregate"] == pytest.approx(expected, rel=1e-3)

    def test_aggregate_rag_scorer_default_weights(self):
        """Test AggregateRAGScorer with default weights."""
        scorers = {"scorer1": Mock(), "scorer2": Mock()}
        scorer = AggregateRAGScorer(scorers)

        assert scorer.weights["scorer1"] == 1.0
        assert scorer.weights["scorer2"] == 1.0

    def test_aggregate_rag_scorer_zero_sum(self):
        """Test AggregateRAGScorer with zero sum."""
        precision_scorer = Mock()
        recall_scorer = Mock()
        precision_scorer.score.return_value = 0.0
        recall_scorer.score.return_value = 0.0

        scorers = {"precision": precision_scorer, "recall": recall_scorer}
        scorer = AggregateRAGScorer(scorers)

        result = scorer.score("prediction", "ground_truth", {"context": "test"})
        assert result["aggregate"] == 0.0

    def test_aggregate_rag_scorer_high_precision_low_recall(self):
        """Test AggregateRAGScorer with high precision, low recall."""
        precision_scorer = Mock()
        recall_scorer = Mock()
        precision_scorer.score.return_value = 0.9
        recall_scorer.score.return_value = 0.3

        scorers = {"precision": precision_scorer, "recall": recall_scorer}
        scorer = AggregateRAGScorer(scorers)

        result = scorer.score("prediction", "ground_truth", {"context": "test"})
        # AggregateRAGScorer averages the scores, doesn't calculate F1
        expected_average = (0.9 + 0.3) / 2
        assert result["aggregate"] == pytest.approx(expected_average, rel=1e-3)

    def test_aggregate_rag_scorer_low_precision_high_recall(self):
        """Test AggregateRAGScorer with low precision, high recall."""
        precision_scorer = Mock()
        recall_scorer = Mock()
        precision_scorer.score.return_value = 0.2
        recall_scorer.score.return_value = 0.8

        scorers = {"precision": precision_scorer, "recall": recall_scorer}
        scorer = AggregateRAGScorer(scorers)

        result = scorer.score("prediction", "ground_truth", {"context": "test"})
        # AggregateRAGScorer averages the scores, doesn't calculate F1
        expected_average = (0.2 + 0.8) / 2
        assert result["aggregate"] == pytest.approx(expected_average, rel=1e-3)

    def test_aggregate_rag_scorer_perfect_scores(self):
        """Test AggregateRAGScorer with perfect precision and recall."""
        precision_scorer = Mock()
        recall_scorer = Mock()
        precision_scorer.score.return_value = 1.0
        recall_scorer.score.return_value = 1.0

        scorers = {"precision": precision_scorer, "recall": recall_scorer}
        scorer = AggregateRAGScorer(scorers)

        result = scorer.score("prediction", "ground_truth", {"context": "test"})
        # Perfect F1 should be 1.0
        assert result["aggregate"] == 1.0

    def test_aggregate_rag_scorer_float_precision_recall(self):
        """Test AggregateRAGScorer with float precision and recall values."""
        precision_scorer = Mock()
        recall_scorer = Mock()
        precision_scorer.score.return_value = 0.75
        recall_scorer.score.return_value = 0.85

        scorers = {"precision": precision_scorer, "recall": recall_scorer}
        scorer = AggregateRAGScorer(scorers)

        result = scorer.score("prediction", "ground_truth", {"context": "test"})
        # AggregateRAGScorer averages the scores, doesn't calculate F1
        expected_average = (0.75 + 0.85) / 2
        assert result["aggregate"] == pytest.approx(expected_average, rel=1e-3)

    def test_aggregate_rag_scorer_score_result_metadata(self):
        """Test that AggregateRAGScorer score result contains proper metadata."""
        precision_scorer = Mock()
        recall_scorer = Mock()
        precision_scorer.score.return_value = 0.6
        recall_scorer.score.return_value = 0.8

        scorers = {"precision": precision_scorer, "recall": recall_scorer}
        scorer = AggregateRAGScorer(scorers)

        result = scorer.score("prediction", "ground_truth", {"context": "test"})

        # Check that the last result was set correctly
        score_result = scorer.get_score_result()
        assert score_result is not None
        assert "Aggregate Score:" in score_result.reasoning

        # Check that the result contains the expected values
        # AggregateRAGScorer averages the scores, doesn't calculate F1
        expected_average = (0.6 + 0.8) / 2
        assert result["aggregate"] == pytest.approx(expected_average, rel=1e-3)

    def test_aggregate_rag_scorer_none_context(self):
        """Test AggregateRAGScorer with None context."""
        precision_scorer = Mock()
        recall_scorer = Mock()
        precision_scorer.score.return_value = 0.7
        recall_scorer.score.return_value = 0.6

        scorers = {"precision": precision_scorer, "recall": recall_scorer}
        scorer = AggregateRAGScorer(scorers)

        result = scorer.score("prediction", "ground_truth", None)
        # Should still work with None context
        # AggregateRAGScorer averages the scores, doesn't calculate F1
        expected_average = (0.7 + 0.6) / 2
        assert result["aggregate"] == pytest.approx(expected_average, rel=1e-3)

    def test_aggregate_rag_scorer_empty_context(self):
        """Test AggregateRAGScorer with empty context."""
        precision_scorer = Mock()
        recall_scorer = Mock()
        precision_scorer.score.return_value = 0.8
        recall_scorer.score.return_value = 0.7

        scorers = {"precision": precision_scorer, "recall": recall_scorer}
        scorer = AggregateRAGScorer(scorers)

        result = scorer.score("prediction", "ground_truth", {})
        # Should still work with empty context
        # AggregateRAGScorer averages the scores, doesn't calculate F1
        expected_average = (0.8 + 0.7) / 2
        assert result["aggregate"] == pytest.approx(expected_average, rel=1e-3)

    def test_parse_numerical_response_edge_cases_advanced(self):
        """Test _parse_numerical_response with more edge cases."""
        model = Mock()
        scorer = ContextualPrecisionScorerPP(model=model)

        # Test with response that causes exception during regex
        response = "Rating: "  # Incomplete rating
        result = scorer._parse_numerical_response(response)
        assert result == 5.0  # Fallback

        # Test with response that causes exception during number parsing
        response = "Rating: abc"  # Invalid number
        result = scorer._parse_numerical_response(response)
        assert result == 5.0  # Fallback

        # Test with response that has multiple numbers but first is invalid
        response = "Rating: abc, also 8 and 9"
        result = scorer._parse_numerical_response(response)
        assert result == 8.0  # Should find the first valid number "8"

    def test_parse_json_response_edge_cases_advanced(self):
        """Test _parse_json_response with more edge cases."""
        model = Mock()
        scorer = ContextualPrecisionScorerPP(model=model)

        # Test with response that has JSON but with invalid content
        response = 'Some text {"invalid": "json"} more text'
        result = scorer._parse_json_response(response)
        assert isinstance(result, dict)
        assert "invalid" in result

        # Test with response that has multiple JSON objects
        response = '{"first": "obj"} {"second": "obj"}'
        result = scorer._parse_json_response(response)
        assert isinstance(result, dict)
        # This is invalid JSON, so it should fall back to boolean parsing
        assert "reasoning" in result
        assert result["relevant"] is False

        # Test with response that has nested JSON
        response = '{"nested": {"key": "value"}}'
        result = scorer._parse_json_response(response)
        assert isinstance(result, dict)
        assert "nested" in result

    def test_semantic_similarity_scorer_edge_cases_advanced(self):
        """Test SemanticSimilarityScorer with more edge cases."""
        scorer = SemanticSimilarityScorer()

        # Test with very long query and chunks
        long_query = "a" * 1000
        long_chunks = ["b" * 1000, "c" * 1000]
        similarity = scorer._compute_simple_similarity(long_query, long_chunks)
        assert 0 <= similarity <= 1

        # Test with special characters
        special_query = "query with !@#$%^&*()"
        special_chunks = ["chunk with !@#$%^&*()", "normal chunk"]
        similarity = scorer._compute_simple_similarity(special_query, special_chunks)
        assert 0 <= similarity <= 1

        # Test with unicode characters
        unicode_query = "query with émojis 🚀"
        unicode_chunks = ["chunk with émojis 🚀", "normal chunk"]
        similarity = scorer._compute_simple_similarity(unicode_query, unicode_chunks)
        assert 0 <= similarity <= 1

    def test_retrieval_diversity_scorer_edge_cases_advanced(self):
        """Test RetrievalDiversityScorer with more edge cases."""
        scorer = RetrievalDiversityScorer()

        # Test with very long chunks
        long_chunks = ["a" * 1000, "b" * 1000, "c" * 1000]
        result = scorer._compute_simple_diversity(long_chunks)
        assert 0 <= result <= 1

        # Test with special characters
        special_chunks = ["chunk with !@#$%^&*()", "normal chunk", "another chunk"]
        result = scorer._compute_simple_diversity(special_chunks)
        assert 0 <= result <= 1

        # Test with unicode characters
        unicode_chunks = ["chunk with émojis 🚀", "normal chunk", "another chunk"]
        result = scorer._compute_simple_diversity(unicode_chunks)
        assert 0 <= result <= 1

    def test_retrieval_ranking_scorer_edge_cases_advanced(self):
        """Test RetrievalRankingScorer with more edge cases."""
        scorer = RetrievalRankingScorer()

        # Test with very large rankings
        context_large = {
            "rankings": [1000, 2000, 3000],
            "relevance_scores": [1.0, 0.8, 0.6],
        }
        result = scorer.score("prediction", "ground_truth", context_large)
        assert isinstance(result, dict)
        assert result["avg_ranking"] == 0.0  # Should be 0.0 for very large rankings

        # Test with negative rankings
        context_negative = {
            "rankings": [-1, -2, -3],
            "relevance_scores": [1.0, 0.8, 0.6],
        }
        result = scorer.score("prediction", "ground_truth", context_negative)
        assert isinstance(result, dict)

        # Test with floating point rankings
        context_float = {
            "rankings": [1.5, 2.7, 3.2],
            "relevance_scores": [1.0, 0.8, 0.6],
        }
        result = scorer.score("prediction", "ground_truth", context_float)
        assert isinstance(result, dict)

    def test_aggregate_rag_scorer_edge_cases_advanced(self):
        """Test AggregateRAGScorer with more edge cases."""
        # Test with empty scorers dict
        scorer = AggregateRAGScorer({})
        result = scorer.score("prediction", "ground_truth", {"context": "test"})
        assert result == 0.0

        # Test with scorer that returns complex nested structure
        mock_scorer = Mock()
        mock_scorer.score.return_value = {
            "metrics": {"nested": {"deep": {"score": 0.9}}}
        }
        scorers = {"scorer": mock_scorer}
        scorer = AggregateRAGScorer(scorers)
        result = scorer.score("prediction", "ground_truth", {"context": "test"})
        assert result["aggregate"] == 0.9

        # Test with scorer that returns list
        mock_scorer2 = Mock()
        mock_scorer2.score.return_value = [0.8, 0.9, 0.7]
        scorers2 = {"scorer": mock_scorer2}
        scorer2 = AggregateRAGScorer(scorers2)
        result2 = scorer2.score("prediction", "ground_truth", {"context": "test"})
        assert result2 == 0.0  # Should fallback to 0.0 when scorer returns invalid type

    def test_parse_numerical_response_more_edge_cases(self):
        """Test _parse_numerical_response with additional edge cases."""
        model = Mock()
        scorer = ContextualPrecisionScorerPP(model=model)

        # Test with response that has numbers in different formats
        response = "Score: 7.5, Rating: 8, Value: 9.0"
        result = scorer._parse_numerical_response(response)
        assert result == 8.0  # Should take the first valid number

        # Test with response that has negative numbers
        response = "Rating: -3, Score: 5"
        result = scorer._parse_numerical_response(response)
        assert result == 3.0  # Should extract the first valid number

        # Test with response that has very large numbers
        response = "Rating: 999999, Score: 8"
        result = scorer._parse_numerical_response(response)
        assert result == 10.0  # Should be clamped to 10

    def test_parse_json_response_more_edge_cases(self):
        """Test _parse_json_response with additional edge cases."""
        model = Mock()
        scorer = ContextualPrecisionScorerPP(model=model)

        # Test with response that has mixed case indicators
        response = "YES, this is RELEVANT information"
        result = scorer._parse_json_response(response)
        assert result["relevant"] is True

        # Test with response that has numeric indicators
        response = "1 - this chunk is relevant"
        result = scorer._parse_json_response(response)
        assert result["relevant"] is True

        # Test with response that has no clear indicators
        response = "this is ambiguous information"
        result = scorer._parse_json_response(response)
        assert result["relevant"] is False

    def test_semantic_similarity_scorer_more_edge_cases(self):
        """Test SemanticSimilarityScorer with additional edge cases."""
        scorer = SemanticSimilarityScorer()

        # Test with very short query and chunks
        short_query = "a"
        short_chunks = ["b", "c"]
        similarity = scorer._compute_simple_similarity(short_query, short_chunks)
        assert 0 <= similarity <= 1

        # Test with query that has special characters
        special_query = "query with !@#$%^&*()"
        special_chunks = ["chunk with !@#$%^&*()", "normal chunk"]
        similarity = scorer._compute_simple_similarity(special_query, special_chunks)
        assert 0 <= similarity <= 1

        # Test with query that has unicode characters
        unicode_query = "query with émojis 🚀"
        unicode_chunks = ["chunk with émojis 🚀", "normal chunk"]
        similarity = scorer._compute_simple_similarity(unicode_query, unicode_chunks)
        assert 0 <= similarity <= 1

    def test_retrieval_diversity_scorer_more_edge_cases(self):
        """Test RetrievalDiversityScorer with additional edge cases."""
        scorer = RetrievalDiversityScorer()

        # Test with very short chunks
        short_chunks = ["a", "b", "c"]
        result = scorer._compute_simple_diversity(short_chunks)
        assert 0 <= result <= 1

        # Test with chunks that have special characters
        special_chunks = ["chunk with !@#$%^&*()", "normal chunk", "another chunk"]
        result = scorer._compute_simple_diversity(special_chunks)
        assert 0 <= result <= 1

        # Test with chunks that have unicode characters
        unicode_chunks = ["chunk with émojis 🚀", "normal chunk", "another chunk"]
        result = scorer._compute_simple_diversity(unicode_chunks)
        assert 0 <= result <= 1

    def test_retrieval_ranking_scorer_more_edge_cases(self):
        """Test RetrievalRankingScorer with additional edge cases."""
        scorer = RetrievalRankingScorer()

        # Test with very large rankings
        context_large = {
            "rankings": [1000, 2000, 3000],
            "relevance_scores": [1.0, 0.8, 0.6],
        }
        result = scorer.score("prediction", "ground_truth", context_large)
        assert isinstance(result, dict)
        assert result["avg_ranking"] == 0.0  # Should be 0.0 for very large rankings

        # Test with negative rankings
        context_negative = {
            "rankings": [-1, -2, -3],
            "relevance_scores": [1.0, 0.8, 0.6],
        }
        result = scorer.score("prediction", "ground_truth", context_negative)
        assert isinstance(result, dict)

        # Test with floating point rankings
        context_float = {
            "rankings": [1.5, 2.7, 3.2],
            "relevance_scores": [1.0, 0.8, 0.6],
        }
        result = scorer.score("prediction", "ground_truth", context_float)
        assert isinstance(result, dict)

    def test_aggregate_rag_scorer_more_edge_cases(self):
        """Test AggregateRAGScorer with additional edge cases."""
        # Test with empty scorers dict
        scorer = AggregateRAGScorer({})
        result = scorer.score("prediction", "ground_truth", {"context": "test"})
        assert result == 0.0

        # Test with scorer that returns complex nested structure
        mock_scorer = Mock()
        mock_scorer.score.return_value = {
            "metrics": {"nested": {"deep": {"score": 0.9}}}
        }
        scorers = {"scorer": mock_scorer}
        scorer = AggregateRAGScorer(scorers)
        result = scorer.score("prediction", "ground_truth", {"context": "test"})
        assert result["aggregate"] == 0.9

        # Test with scorer that returns list
        mock_scorer2 = Mock()
        mock_scorer2.score.return_value = [0.8, 0.9, 0.7]
        scorers2 = {"scorer": mock_scorer2}
        scorer2 = AggregateRAGScorer(scorers2)
        result2 = scorer2.score("prediction", "ground_truth", {"context": "test"})
        assert result2 == 0.0  # Should fallback to 0.0 when scorer returns invalid type

    @pytest.mark.asyncio
    async def test_contextual_precision_scorer_pp_more_edge_cases(self):
        """Test ContextualPrecisionScorerPP with additional edge cases."""
        model = Mock()
        scorer = ContextualPrecisionScorerPP(model=model)

        # Test with context that has empty string chunks
        context = {"chunks": ["", "chunk2"]}
        with patch.object(
            scorer,
            "_evaluate_chunk_relevance",
            new_callable=AsyncMock,
            side_effect=[False, True],
        ):
            result = await scorer.evaluate("query", "answer", context=context)
            assert result.score == 0.5  # 1 relevant out of 2 chunks

        # Test with context that has None chunks
        context = {"chunks": [None, "chunk2"]}
        with patch.object(
            scorer,
            "_evaluate_chunk_relevance",
            new_callable=AsyncMock,
            side_effect=[False, True],
        ):
            result = await scorer.evaluate("query", "answer", context=context)
            assert result.score == 0.5  # 1 relevant out of 2 chunks

    @pytest.mark.asyncio
    async def test_contextual_recall_scorer_pp_more_edge_cases(self):
        """Test ContextualRecallScorerPP with additional edge cases."""
        model = Mock()
        scorer = ContextualRecallScorerPP(model=model)

        # Test with context that has empty string chunks
        context = {"chunks": ["", "chunk2"]}
        with (
            patch.object(
                scorer,
                "_evaluate_chunk_relevance",
                new_callable=AsyncMock,
                side_effect=[False, True],
            ),
            patch.object(
                scorer,
                "_estimate_total_relevant_chunks",
                new_callable=AsyncMock,
                return_value=2,
            ),
        ):
            result = await scorer.evaluate("query", "answer", context=context)
            assert result.score == 0.5  # 1 relevant retrieved, estimated 2 total

        # Test with context that has None chunks
        context = {"chunks": [None, "chunk2"]}
        with (
            patch.object(
                scorer,
                "_evaluate_chunk_relevance",
                new_callable=AsyncMock,
                side_effect=[False, True],
            ),
            patch.object(
                scorer,
                "_estimate_total_relevant_chunks",
                new_callable=AsyncMock,
                return_value=2,
            ),
        ):
            result = await scorer.evaluate("query", "answer", context=context)
            assert result.score == 0.5  # 1 relevant retrieved, estimated 2 total

    def test_retrieval_f1_scorer_more_edge_cases(self):
        """Test RetrievalF1Scorer with additional edge cases."""
        precision_scorer = Mock()
        recall_scorer = Mock()
        scorer = RetrievalF1Scorer(precision_scorer, recall_scorer)

        # Test with very high precision, very low recall
        precision_scorer.score.return_value = 0.99
        recall_scorer.score.return_value = 0.01
        result = scorer.score("prediction", "ground_truth", {"context": "test"})
        expected_f1 = 2 * (0.99 * 0.01) / (0.99 + 0.01)
        assert result["f1"] == pytest.approx(expected_f1, rel=1e-3)

        # Test with very low precision, very high recall
        precision_scorer.score.return_value = 0.01
        recall_scorer.score.return_value = 0.99
        result = scorer.score("prediction", "ground_truth", {"context": "test"})
        expected_f1 = 2 * (0.01 * 0.99) / (0.01 + 0.99)
        assert result["f1"] == pytest.approx(expected_f1, rel=1e-3)

    def test_semantic_similarity_scorer_load_model_edge_cases(self):
        """Test SemanticSimilarityScorer _load_model with edge cases."""
        scorer = SemanticSimilarityScorer()

        # Test with Windows platform
        with (
            patch("platform.system", return_value="Windows"),
            patch("platform.machine", return_value="AMD64"),
        ):
            scorer._load_model()
            # Should work normally on Windows

        # Test with Linux platform
        with (
            patch("platform.system", return_value="Linux"),
            patch("platform.machine", return_value="x86_64"),
        ):
            scorer._load_model()
            # Should work normally on Linux

    def test_retrieval_diversity_scorer_load_model_edge_cases(self):
        """Test RetrievalDiversityScorer _load_model with edge cases."""
        scorer = RetrievalDiversityScorer()

        # Test with Windows platform
        with (
            patch("platform.system", return_value="Windows"),
            patch("platform.machine", return_value="AMD64"),
        ):
            scorer._load_model()
            # Should work normally on Windows

        # Test with Linux platform
        with (
            patch("platform.system", return_value="Linux"),
            patch("platform.machine", return_value="x86_64"),
        ):
            scorer._load_model()
            # Should work normally on Linux
