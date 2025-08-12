import asyncio
from unittest.mock import patch

import pytest

# Import shared test utilities
from test_utils import MockLLM

from novaeval.scorers.base import ScoreResult
from novaeval.scorers.rag import (
    AnswerRelevancyScorer,
    ContextualPrecisionScorer,
    ContextualRecallScorer,
    FaithfulnessScorer,
    RAGASScorer,
)


class TestAnswerRelevancyScorer:
    """Test class for AnswerRelevancyScorer to improve coverage."""

    def test_load_embedding_model_import_error(self):
        """Test _load_embedding_model with ImportError."""
        mock_llm = MockLLM()
        scorer = AnswerRelevancyScorer(model=mock_llm)

        with (
            patch(
                "builtins.__import__",
                side_effect=ImportError("No module named 'sentence_transformers'"),
            ),
            patch("builtins.print") as mock_print,
        ):
            scorer._load_embedding_model()

            assert scorer.embedding_model is None
            assert scorer._model_loaded is True
            mock_print.assert_called_once_with(
                "Warning: sentence_transformers not installed. "
                "Answer relevancy scoring will use fallback method."
            )

    def test_load_embedding_model_exception(self):
        """Test _load_embedding_model with general Exception."""
        mock_llm = MockLLM()
        scorer = AnswerRelevancyScorer(model=mock_llm)

        with (
            patch(
                "sentence_transformers.SentenceTransformer",
                side_effect=Exception("Model loading failed"),
            ),
            patch("builtins.print") as mock_print,
        ):
            scorer._load_embedding_model()

            assert scorer.embedding_model is None
            assert scorer._model_loaded is True
            mock_print.assert_called_once_with(
                "Warning: Could not load SentenceTransformer model: Model loading failed"
            )

    @pytest.mark.asyncio
    async def test_evaluate_with_fallback_similarity(self):
        """Test evaluate method using fallback text similarity when embedding model is None."""
        mock_llm = MockLLM()
        scorer = AnswerRelevancyScorer(model=mock_llm)

        # Force embedding model to be None
        scorer.embedding_model = None
        scorer._model_loaded = True

        # Mock the question generation to return predictable results
        with patch.object(
            scorer,
            "_parse_questions",
            return_value=["What is the capital?", "Which city is the capital?"],
        ):
            result = await scorer.evaluate(
                input_text="What is the capital of France?",
                output_text="Paris is the capital of France.",
                context="Paris is the capital of France.",
            )

            assert isinstance(result, ScoreResult)
            assert 0 <= result.score <= 1
            assert isinstance(result.passed, bool)
            assert "Answer Relevancy" in result.reasoning

    @pytest.mark.asyncio
    async def test_evaluate_fallback_similarity_edge_cases(self):
        """Test fallback similarity with edge cases like empty words."""
        mock_llm = MockLLM()
        scorer = AnswerRelevancyScorer(model=mock_llm)

        # Force embedding model to be None
        scorer.embedding_model = None
        scorer._model_loaded = True

        # Test with empty generated questions
        with patch.object(scorer, "_parse_questions", return_value=[""]):
            result = await scorer.evaluate(
                input_text="What is the capital?",
                output_text="Paris",
                context="Paris is the capital",
            )

            assert isinstance(result, ScoreResult)
            assert result.score >= 0

    def test_score_method_sync(self):
        """Test the synchronous score method."""
        mock_llm = MockLLM()
        scorer = AnswerRelevancyScorer(model=mock_llm)

        # Mock the embedding model to be None to trigger fallback
        scorer.embedding_model = None
        scorer._model_loaded = True

        with patch.object(
            scorer, "_parse_questions", return_value=["What is the capital?"]
        ):
            result = scorer.score(
                prediction="Paris is the capital of France.",
                ground_truth="What is the capital of France?",
                context={"context": "Paris is the capital of France."},
            )

            assert isinstance(result, (float, dict))


@pytest.mark.asyncio
async def test_answer_relevancy_scorer():
    """Test AnswerRelevancyScorer."""
    mock_llm = MockLLM()
    scorer = AnswerRelevancyScorer(model=mock_llm, threshold=0.7)

    result = await scorer.evaluate(
        input_text="What is the capital of France?",
        output_text="Paris is the capital of France.",
        context="Paris is the capital of France.",
    )

    assert isinstance(result, ScoreResult)
    assert 0 <= result.score <= 1
    assert isinstance(result.passed, bool)
    assert isinstance(result.reasoning, str)
    assert isinstance(result.metadata, dict)
    print(f"Answer Relevancy: {result.score:.3f} (Passed: {result.passed})")


@pytest.mark.asyncio
async def test_faithfulness_scorer():
    """Test FaithfulnessScorer."""
    mock_llm = MockLLM()
    scorer = FaithfulnessScorer(model=mock_llm, threshold=0.8)

    result = await scorer.evaluate(
        input_text="What is the capital of France?",
        output_text="Paris is the capital of France.",
        context="Paris is the capital of France. France is in Europe.",
    )

    assert isinstance(result, ScoreResult)
    assert 0 <= result.score <= 1
    assert isinstance(result.passed, bool)
    assert isinstance(result.reasoning, str)
    assert isinstance(result.metadata, dict)
    print(f"Faithfulness: {result.score:.3f} (Passed: {result.passed})")


@pytest.mark.asyncio
async def test_contextual_precision_scorer():
    """Test ContextualPrecisionScorer."""
    mock_llm = MockLLM()
    scorer = ContextualPrecisionScorer(model=mock_llm, threshold=0.7)

    result = await scorer.evaluate(
        input_text="What is the capital of France?",
        output_text="Paris is the capital of France.",
        context="Paris is the capital of France.\n\nFrance is in Europe.\n\nParis has the Eiffel Tower.",
    )

    assert isinstance(result, ScoreResult)
    assert 0 <= result.score <= 1
    assert isinstance(result.passed, bool)
    assert isinstance(result.reasoning, str)
    assert isinstance(result.metadata, dict)
    print(f"Contextual Precision: {result.score:.3f} (Passed: {result.passed})")


@pytest.mark.asyncio
async def test_contextual_recall_scorer():
    """Test ContextualRecallScorer."""
    mock_llm = MockLLM()
    scorer = ContextualRecallScorer(model=mock_llm, threshold=0.7)

    result = await scorer.evaluate(
        input_text="What is the capital of France?",
        output_text="Paris is the capital of France.",
        expected_output="Paris is the capital of France and is known for the Eiffel Tower.",
        context="Paris is the capital of France.\n\nFrance is in Europe.\n\nParis has the Eiffel Tower.",
    )

    assert isinstance(result, ScoreResult)
    assert 0 <= result.score <= 1
    assert isinstance(result.passed, bool)
    assert isinstance(result.reasoning, str)
    assert isinstance(result.metadata, dict)
    print(f"Contextual Recall: {result.score:.3f} (Passed: {result.passed})")


@pytest.mark.asyncio
async def test_ragas_scorer():
    """Test RAGASScorer (composite scorer)."""
    mock_llm = MockLLM()
    scorer = RAGASScorer(model=mock_llm, threshold=0.7)

    result = await scorer.evaluate(
        input_text="What is the capital of France?",
        output_text="Paris is the capital of France.",
        expected_output="Paris is the capital of France and is known for the Eiffel Tower.",
        context="Paris is the capital of France.\n\nFrance is in Europe.\n\nParis has the Eiffel Tower.",
    )

    assert isinstance(result, ScoreResult)
    assert 0 <= result.score <= 1
    assert isinstance(result.passed, bool)
    assert isinstance(result.reasoning, str)
    assert isinstance(result.metadata, dict)
    print(f"RAGAS Score: {result.score:.3f} (Passed: {result.passed})")


# @pytest.mark.asyncio
# async def test_sync_score_methods():
#     """Test synchronous score methods."""
#     mock_llm = MockLLM()
#
#     # Test AnswerRelevancyScorer sync method
#     scorer = AnswerRelevancyScorer(model=mock_llm, threshold=0.7)
#     score = scorer.score(
#         prediction="Paris is the capital of France.",
#         ground_truth="What is the capital of France?",
#         context={"context": "Paris is the capital of France."}
#     )
#     assert isinstance(score, (float, dict))
#     print(f"Sync Answer Relevancy: {score}")


@pytest.mark.asyncio
async def test_error_handling():
    """Test comprehensive error handling with various failure scenarios."""

    # Test with missing context
    mock_llm = MockLLM()
    scorer = AnswerRelevancyScorer(model=mock_llm, threshold=0.7)

    result = await scorer.evaluate(
        input_text="What is the capital of France?",
        output_text="Paris is the capital of France.",
        context=None,
    )

    # Should handle gracefully
    assert isinstance(result, ScoreResult)
    print(f"Missing context test passed: {result.score}")

    # Test LLM exception handling

    class ExceptionMockLLM:

        async def generate(self, prompt):
            raise Exception("LLM API error: Rate limit exceeded")

    exception_mock_llm = ExceptionMockLLM()
    scorer = AnswerRelevancyScorer(model=exception_mock_llm, threshold=0.7)

    result = await scorer.evaluate(
        input_text="What is the capital of France?",
        output_text="Paris is the capital of France.",
        context="Paris is the capital of France.",
    )

    # Should handle LLM exceptions gracefully
    assert isinstance(result, ScoreResult)
    assert result.score == 0.0
    assert not result.passed
    assert "LLM API error" in result.reasoning or "failed" in result.reasoning.lower()
    print(f"LLM exception test passed: {result.score}")

    # Test malformed response handling

    class MalformedMockLLM:

        async def generate(self, prompt):
            return "This is not a valid response format"

    malformed_mock_llm = MalformedMockLLM()
    scorer = AnswerRelevancyScorer(model=malformed_mock_llm, threshold=0.7)

    result = await scorer.evaluate(
        input_text="What is the capital of France?",
        output_text="Paris is the capital of France.",
        context="Paris is the capital of France.",
    )

    # Should handle malformed responses gracefully
    assert isinstance(result, ScoreResult)
    assert result.score == 0.0
    assert not result.passed
    print(f"Malformed response test passed: {result.score}")

    # Test empty response handling

    class EmptyMockLLM:

        async def generate(self, prompt):
            return ""

    empty_mock_llm = EmptyMockLLM()
    scorer = AnswerRelevancyScorer(model=empty_mock_llm, threshold=0.7)

    result = await scorer.evaluate(
        input_text="What is the capital of France?",
        output_text="Paris is the capital of France.",
        context="Paris is the capital of France.",
    )

    # Should handle empty responses gracefully
    assert isinstance(result, ScoreResult)
    assert result.score == 0.0
    assert not result.passed
    print(f"Empty response test passed: {result.score}")

    # Test unexpected response format

    class UnexpectedMockLLM:

        async def generate(self, prompt):
            return "Rating: invalid\nExplanation: This is not a number"

    unexpected_mock_llm = UnexpectedMockLLM()
    scorer = AnswerRelevancyScorer(model=unexpected_mock_llm, threshold=0.7)

    result = await scorer.evaluate(
        input_text="What is the capital of France?",
        output_text="Paris is the capital of France.",
        context="Paris is the capital of France.",
    )

    # Should handle unexpected response formats gracefully
    assert isinstance(result, ScoreResult)
    assert result.score == 0.0
    assert not result.passed
    print(f"Unexpected format test passed: {result.score}")

    # Test network timeout simulation

    class TimeoutMockLLM:

        async def generate(self, prompt):

            await asyncio.sleep(0.1)  # Simulate delay
            raise TimeoutError("LLM request timed out")

    timeout_mock_llm = TimeoutMockLLM()
    scorer = AnswerRelevancyScorer(model=timeout_mock_llm, threshold=0.7)

    result = await scorer.evaluate(
        input_text="What is the capital of France?",
        output_text="Paris is the capital of France.",
        context="Paris is the capital of France.",
    )

    # Should handle timeout errors gracefully
    assert isinstance(result, ScoreResult)
    assert result.score == 0.0
    assert not result.passed
    assert "timeout" in result.reasoning.lower() or "failed" in result.reasoning.lower()
    print(f"Timeout error test passed: {result.score}")

    # Test with None inputs
    result = await scorer.evaluate(input_text=None, output_text=None, context=None)

    # Should handle None inputs gracefully
    assert isinstance(result, ScoreResult)
    assert result.score == 0.0
    assert not result.passed
    print(f"None inputs test passed: {result.score}")

    # Test with empty string inputs
    result = await scorer.evaluate(input_text="", output_text="", context="")

    # Should handle empty string inputs gracefully
    assert isinstance(result, ScoreResult)
    assert result.score == 0.0
    assert not result.passed
    print(f"Empty string inputs test passed: {result.score}")

    print("All error handling tests passed successfully!")


if __name__ == "__main__":
    # Run all tests
    pytest.main([__file__, "-v", "-s"])
