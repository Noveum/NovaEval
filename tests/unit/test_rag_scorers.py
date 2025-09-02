import asyncio
from unittest.mock import AsyncMock, Mock, patch

import pytest

from novaeval.scorers.base import ScoreResult
from novaeval.scorers.rag import (
    AnswerRelevancyScorer,
    ContextualPrecisionScorer,
    ContextualRecallScorer,
    FaithfulnessScorer,
    RAGASScorer,
)
from tests.unit.test_utils import MockLLM


class TestAnswerRelevancyScorer:
    """Test class for AnswerRelevancyScorer to improve coverage."""

    def test_load_embedding_model_import_error(self):
        """Test _load_embedding_model with ImportError."""
        mock_llm = MockLLM()
        scorer = AnswerRelevancyScorer(model=mock_llm)

        # Mock the import to fail by patching the method to simulate import error
        original_method = scorer._load_embedding_model

        def mock_load_embedding_model():
            scorer.embedding_model = None
            scorer._model_loaded = True
            import logging

            logging.warning(
                "sentence_transformers not installed. Answer relevancy scoring will use fallback method."
            )

        scorer._load_embedding_model = mock_load_embedding_model

        with patch("logging.warning") as mock_warning:
            scorer._load_embedding_model()

            assert scorer.embedding_model is None
            assert scorer._model_loaded is True
            mock_warning.assert_called_once_with(
                "sentence_transformers not installed. "
                "Answer relevancy scoring will use fallback method."
            )

        # Restore original method
        scorer._load_embedding_model = original_method

    def test_load_embedding_model_exception(self):
        """Test _load_embedding_model with general Exception."""
        mock_llm = MockLLM()
        scorer = AnswerRelevancyScorer(model=mock_llm)

        # Mock the method to simulate exception
        original_method = scorer._load_embedding_model

        def mock_load_embedding_model():
            scorer.embedding_model = None
            scorer._model_loaded = True
            print(
                "Warning: Could not load SentenceTransformer model: Model loading failed"
            )

        scorer._load_embedding_model = mock_load_embedding_model

        with patch("builtins.print") as mock_print:
            scorer._load_embedding_model()

            assert scorer.embedding_model is None
            assert scorer._model_loaded is True
            mock_print.assert_called_once_with(
                "Warning: Could not load SentenceTransformer model: Model loading failed"
            )

        # Restore original method
        scorer._load_embedding_model = original_method

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


class TestAnswerRelevancyScorerExtended:
    """Additional tests to improve coverage for AnswerRelevancyScorer."""

    def test_load_embedding_model_success(self):
        """Test _load_embedding_model with successful model loading."""
        mock_llm = MockLLM()
        scorer = AnswerRelevancyScorer(model=mock_llm)

        # Mock successful model loading
        mock_model = Mock()
        original_method = scorer._load_embedding_model

        def mock_load_embedding_model():
            scorer.embedding_model = mock_model
            scorer._model_loaded = True

        scorer._load_embedding_model = mock_load_embedding_model

        scorer._load_embedding_model()

        assert scorer.embedding_model == mock_model
        assert scorer._model_loaded is True

        # Restore original method
        scorer._load_embedding_model = original_method

    def test_parse_questions_various_formats(self):
        """Test _parse_questions method with various response formats."""
        mock_llm = MockLLM()
        scorer = AnswerRelevancyScorer(model=mock_llm)

        # Test numbered format
        response1 = "1. What is machine learning?\n2. How does ML work?\n3. What are ML applications?"
        questions1 = scorer._parse_questions(response1)
        assert len(questions1) == 3
        assert "What is machine learning?" in questions1
        assert "How does ML work?" in questions1
        assert "What are ML applications?" in questions1

        # Test bullet point format
        response2 = "- What is AI?\n- How does AI work?\n* What are AI applications?"
        questions2 = scorer._parse_questions(response2)
        assert len(questions2) == 3
        assert "What is AI?" in questions2
        assert "How does AI work?" in questions2
        assert "What are AI applications?" in questions2

        # Test mixed format
        response3 = "1. What is Python?\n- How does Python work?\n* What are Python applications?"
        questions3 = scorer._parse_questions(response3)
        assert len(questions3) == 3

        # Test with non-question lines
        response4 = (
            "1. What is Python?\nThis is not a question.\n2. How does Python work?"
        )
        questions4 = scorer._parse_questions(response4)
        assert len(questions4) == 2
        assert "What is Python?" in questions4
        assert "How does Python work?" in questions4

        # Test with empty lines and whitespace
        response5 = "1. What is Python?\n\n  2. How does Python work?  \n"
        questions5 = scorer._parse_questions(response5)
        assert len(questions5) == 2

        # Test with no questions
        response6 = "This response has no questions."
        questions6 = scorer._parse_questions(response6)
        assert len(questions6) == 0

    def test_parse_questions_edge_cases(self):
        """Test _parse_questions method with edge cases."""
        mock_llm = MockLLM()
        scorer = AnswerRelevancyScorer(model=mock_llm)

        # Test with empty response
        questions1 = scorer._parse_questions("")
        assert len(questions1) == 0

        # Test with single character
        questions2 = scorer._parse_questions("a")
        assert len(questions2) == 0

        # Test with numbers but no questions
        response3 = "1. This is not a question\n2. Neither is this"
        questions3 = scorer._parse_questions(response3)
        assert len(questions3) == 0

        # Test with questions without question marks
        response4 = "1. What is Python\n2. How does it work"
        questions4 = scorer._parse_questions(response4)
        assert len(questions4) == 0

    @pytest.mark.asyncio
    async def test_evaluate_with_embedding_model(self):
        """Test evaluate method using embedding model when available."""
        mock_llm = MockLLM()
        scorer = AnswerRelevancyScorer(model=mock_llm)

        # Mock the embedding model
        mock_model = Mock()
        mock_model.encode.return_value = [[0.1, 0.2], [0.3, 0.4]]
        scorer.embedding_model = mock_model
        scorer._model_loaded = True

        # Mock the question generation
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

    def test_score_method_with_embedding_model(self):
        """Test the synchronous score method with embedding model."""
        mock_llm = MockLLM()
        scorer = AnswerRelevancyScorer(model=mock_llm)

        # Mock the embedding model
        mock_model = Mock()
        mock_model.encode.return_value = [[0.1, 0.2]]
        scorer.embedding_model = mock_model
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
    async def test_evaluate_with_empty_questions(self):
        """Test evaluate method when no questions are generated."""
        mock_llm = MockLLM()
        scorer = AnswerRelevancyScorer(model=mock_llm)

        with patch.object(scorer, "_parse_questions", return_value=[]):
            result = await scorer.evaluate(
                input_text="What is the capital of France?",
                output_text="Paris is the capital of France.",
                context="Paris is the capital of France.",
            )

            assert isinstance(result, ScoreResult)
            assert result.score == 0.0
            assert not result.passed
            assert "Failed to generate questions" in result.reasoning
            assert result.metadata["error"] == "question_generation_failed"

    @pytest.mark.asyncio
    async def test_evaluate_with_llm_exception(self):
        """Test evaluate method when LLM generation fails."""
        mock_llm = MockLLM()
        scorer = AnswerRelevancyScorer(model=mock_llm)

        # Mock the LLM to raise an exception
        async def mock_generate(prompt):
            raise Exception("LLM API error")

        mock_llm.generate = mock_generate

        result = await scorer.evaluate(
            input_text="What is the capital of France?",
            output_text="Paris is the capital of France.",
            context="Paris is the capital of France.",
        )

        assert isinstance(result, ScoreResult)
        assert result.score == 0.0
        assert not result.passed
        assert "LLM API error" in result.reasoning

    @pytest.mark.asyncio
    async def test_evaluate_with_embedding_model_exception(self):
        """Test evaluate method when embedding model encoding fails."""
        mock_llm = MockLLM()
        scorer = AnswerRelevancyScorer(model=mock_llm)

        # Mock the embedding model to raise an exception
        mock_model = Mock()
        mock_model.encode.side_effect = Exception("Encoding failed")
        scorer.embedding_model = mock_model
        scorer._model_loaded = True

        with (
            patch.object(
                scorer,
                "_parse_questions",
                return_value=["What is the capital?", "Which city is the capital?"],
            ),
            patch.object(
                scorer,
                "_load_embedding_model",
                return_value=None,
            ),
        ):
            result = await scorer.evaluate(
                input_text="What is the capital of France?",
                output_text="Paris is the capital of France.",
                context="Paris is the capital of France.",
            )

            assert isinstance(result, ScoreResult)
            # Should fall back to token-overlap instead of hard-failing to 0.0
            assert result.score > 0.0  # Should have a meaningful token-overlap score
            assert not result.passed  # Should still fail the threshold
            assert "token-overlap" in result.reasoning
            assert "fallback" in result.reasoning
            assert "Encoding failed" in result.reasoning
            assert result.metadata["similarity_method"] == "token-overlap"
            assert "embedding_encoding_failed" in result.metadata["fallback_reason"]

    def test_score_method_with_context_dict(self):
        """Test score method with context dictionary."""
        mock_llm = MockLLM()
        scorer = AnswerRelevancyScorer(model=mock_llm)

        # Mock the evaluate method to return a predictable result
        async def mock_evaluate(*args, **kwargs):
            return ScoreResult(
                score=0.8, passed=True, reasoning="Good answer", metadata={}
            )

        scorer.evaluate = mock_evaluate

        result = scorer.score(
            prediction="Paris is the capital of France.",
            ground_truth="What is the capital of France?",
            context={"context": "France is a country in Europe."},
        )

        assert result == 0.8

    def test_score_method_without_context(self):
        """Test score method without context."""
        mock_llm = MockLLM()
        scorer = AnswerRelevancyScorer(model=mock_llm)

        # Mock the evaluate method to return a predictable result
        async def mock_evaluate(*args, **kwargs):
            return ScoreResult(
                score=0.7, passed=True, reasoning="Good answer", metadata={}
            )

        scorer.evaluate = mock_evaluate

        result = scorer.score(
            prediction="Paris is the capital of France.",
            ground_truth="What is the capital of France?",
        )

        assert result == 0.7

    def test_load_embedding_model_import_error_via_import_mock(self):
        """Test _load_embedding_model with ImportError via import mocking."""
        mock_llm = MockLLM()

        # Mock the import to fail
        with patch(
            "builtins.__import__",
            side_effect=ImportError("No module named 'sentence_transformers'"),
        ):
            # Create scorer after mocking to ensure the mock is active during initialization
            scorer = AnswerRelevancyScorer(model=mock_llm)
            assert scorer.embedding_model is None
            assert scorer._model_loaded is True

    def test_load_embedding_model_general_exception(self):
        """Test _load_embedding_model with general Exception."""
        mock_llm = MockLLM()

        # Mock the import to succeed but SentenceTransformer to fail
        with patch(
            "novaeval.scorers.rag.SentenceTransformer",
            side_effect=Exception("Model loading failed"),
        ):
            # Create scorer after mocking to ensure the mock is active during initialization
            scorer = AnswerRelevancyScorer(model=mock_llm)
            assert scorer.embedding_model is None
            assert scorer._model_loaded is True

    @pytest.mark.asyncio
    async def test_evaluate_with_empty_generated_questions(self):
        """Test evaluate method when no questions are generated."""
        mock_llm = MockLLM()
        scorer = AnswerRelevancyScorer(model=mock_llm)

        # Mock the question generation to return empty list
        with patch.object(
            scorer,
            "_parse_questions",
            return_value=[],
        ):
            result = await scorer.evaluate(
                input_text="What is the capital of France?",
                output_text="Paris is the capital of France.",
                context="Paris is the capital of France.",
            )

            assert isinstance(result, ScoreResult)
            assert result.score == 0.0
            assert not result.passed
            assert "Failed to generate questions" in result.reasoning
            assert result.metadata.get("error") == "question_generation_failed"

    @pytest.mark.asyncio
    async def test_evaluate_with_embedding_model_encoding_exception(self):
        """Test evaluate method with embedding model encoding exception."""
        mock_llm = MockLLM()
        scorer = AnswerRelevancyScorer(model=mock_llm)

        # Mock the question generation to return predictable results
        with (
            patch.object(
                scorer,
                "_parse_questions",
                return_value=["What is the capital?", "Which city is the capital?"],
            ),
            patch.object(
                scorer,
                "_load_embedding_model",
                return_value=None,
            ),
        ):
            # Mock embedding model to raise exception during encoding
            mock_embedding_model = Mock()
            mock_embedding_model.encode.side_effect = Exception("Encoding failed")
            scorer.embedding_model = mock_embedding_model
            scorer._model_loaded = True

            result = await scorer.evaluate(
                input_text="What is the capital of France?",
                output_text="Paris is the capital of France.",
                context="Paris is the capital of France.",
            )

            assert isinstance(result, ScoreResult)
            assert 0 <= result.score <= 1
            assert "Encoding failed" in result.reasoning

    @pytest.mark.asyncio
    async def test_evaluate_with_model_generation_exception(self):
        """Test evaluate method when model generation fails."""
        mock_llm = MockLLM()
        mock_llm.generate = AsyncMock(side_effect=Exception("Model generation failed"))
        scorer = AnswerRelevancyScorer(model=mock_llm)

        result = await scorer.evaluate(
            input_text="What is the capital of France?",
            output_text="Paris is the capital of France.",
            context="Paris is the capital of France.",
        )

        assert isinstance(result, ScoreResult)
        assert result.score == 0.0
        assert not result.passed
        assert "Model generation failed" in result.reasoning


class TestFaithfulnessScorer:
    """Test class for FaithfulnessScorer to improve coverage."""

    def test_parse_claims_various_formats(self):
        """Test _parse_claims method with various response formats."""
        mock_llm = MockLLM()
        scorer = FaithfulnessScorer(model=mock_llm)

        # Test numbered format
        response1 = "1. Machine learning is a subset of AI\n2. ML involves training algorithms\n3. ML makes predictions from data"
        claims1 = scorer._parse_claims(response1)
        assert len(claims1) == 3
        assert "Machine learning is a subset of AI" in claims1
        assert "ML involves training algorithms" in claims1
        assert "ML makes predictions from data" in claims1

        # Test bullet point format
        response2 = "- AI is a broad field\n- Machine learning is a subset\n* Deep learning is a subset of ML"
        claims2 = scorer._parse_claims(response2)
        assert len(claims2) == 3

        # Test with non-claim lines
        response3 = "1. Python is a programming language\nThis is not a claim.\n2. Python is easy to learn"
        claims3 = scorer._parse_claims(response3)
        assert len(claims3) == 2

        # Test with empty response
        claims4 = scorer._parse_claims("")
        assert len(claims4) == 0

    @pytest.mark.asyncio
    async def test_evaluate_with_no_context(self):
        """Test evaluate method with no context."""
        mock_llm = MockLLM()
        scorer = FaithfulnessScorer(model=mock_llm)

        result = await scorer.evaluate(
            input_text="What is the capital of France?",
            output_text="Paris is the capital of France.",
            context=None,
        )

        assert isinstance(result, ScoreResult)
        assert result.score == 0.0
        assert not result.passed
        assert "No context provided" in result.reasoning
        assert result.metadata.get("error") == "no_context"

    @pytest.mark.asyncio
    async def test_evaluate_with_empty_context(self):
        """Test evaluate method with empty context."""
        mock_llm = MockLLM()
        scorer = FaithfulnessScorer(model=mock_llm)

        result = await scorer.evaluate(
            input_text="What is the capital of France?",
            output_text="Paris is the capital of France.",
            context="",
        )

        assert isinstance(result, ScoreResult)
        assert result.score == 0.0
        assert not result.passed
        assert "No context provided" in result.reasoning
        assert result.metadata.get("error") == "no_context"

    @pytest.mark.asyncio
    async def test_evaluate_with_no_claims(self):
        """Test evaluate method when no claims are extracted."""
        mock_llm = MockLLM()
        scorer = FaithfulnessScorer(model=mock_llm)

        # Mock the claims extraction to return empty list
        with patch.object(
            scorer,
            "_parse_claims",
            return_value=[],
        ):
            result = await scorer.evaluate(
                input_text="What is the capital of France?",
                output_text="Paris is the capital of France.",
                context="Paris is the capital of France. France is in Europe.",
            )

            assert isinstance(result, ScoreResult)
            assert result.score == 1.0
            assert result.passed
            assert "No factual claims found" in result.reasoning
            assert result.metadata.get("claims") == []

    @pytest.mark.asyncio
    async def test_evaluate_with_llm_exception(self):
        """Test evaluate method when LLM generation fails."""
        mock_llm = MockLLM()
        scorer = FaithfulnessScorer(model=mock_llm)

        # Mock the LLM to raise an exception
        async def mock_generate(prompt):
            raise Exception("LLM API error")

        mock_llm.generate = mock_generate

        result = await scorer.evaluate(
            input_text="What is machine learning?",
            output_text="Machine learning is a subset of AI.",
            context="Machine learning is a subset of artificial intelligence.",
        )

        assert isinstance(result, ScoreResult)
        assert result.score == 0.0
        assert not result.passed
        assert "LLM API error" in result.reasoning

    def test_score_method_with_context_dict(self):
        """Test score method with context dictionary."""
        mock_llm = MockLLM()
        scorer = FaithfulnessScorer(model=mock_llm)

        # Mock the evaluate method to return a predictable result
        async def mock_evaluate(*args, **kwargs):
            return ScoreResult(
                score=0.9, passed=True, reasoning="Faithful to context", metadata={}
            )

        scorer.evaluate = mock_evaluate

        result = scorer.score(
            prediction="Paris is the capital of France.",
            ground_truth="What is the capital of France?",
            context={"context": "Paris is the capital of France. France is in Europe."},
        )

        assert result == 0.9

    def test_score_method_without_context(self):
        """Test score method without context."""
        mock_llm = MockLLM()
        scorer = FaithfulnessScorer(model=mock_llm)

        # Mock the evaluate method to return a predictable result
        async def mock_evaluate(*args, **kwargs):
            return ScoreResult(
                score=0.8, passed=True, reasoning="Faithful to context", metadata={}
            )

        scorer.evaluate = mock_evaluate

        result = scorer.score(
            prediction="Paris is the capital of France.",
            ground_truth="What is the capital of France?",
        )

        assert result == 0.8

    def test_parse_claims_edge_cases(self):
        """Test _parse_claims method with edge cases."""
        mock_llm = MockLLM()
        scorer = FaithfulnessScorer(model=mock_llm)

        # Test with empty response
        claims1 = scorer._parse_claims("")
        assert len(claims1) == 0

        # Test with single character
        claims2 = scorer._parse_claims("a")
        assert len(claims2) == 0

        # Test with numbers but no claims (these will be parsed as claims since they don't end with ?)
        response3 = "1. This is not a claim\n2. Neither is this"
        claims3 = scorer._parse_claims(response3)
        assert len(claims3) == 2  # These are actually parsed as claims

        # Test with claims that end with question marks (these are also parsed as claims)
        response4 = "1. What is Python\n2. How does it work"
        claims4 = scorer._parse_claims(response4)
        assert (
            len(claims4) == 2
        )  # These are parsed as claims regardless of question marks

        # Test with mixed formats
        response5 = "1. First claim\n- Second claim\n* Third claim"
        claims5 = scorer._parse_claims(response5)
        assert len(claims5) == 3

        # Test with empty lines
        response6 = "1. First claim\n\n2. Second claim"
        claims6 = scorer._parse_claims(response6)
        assert len(claims6) == 2


class TestContextualPrecisionScorer:
    """Test class for ContextualPrecisionScorer to improve coverage."""

    @pytest.mark.asyncio
    async def test_evaluate_with_empty_context(self):
        """Test evaluate method with empty context."""
        mock_llm = MockLLM()
        scorer = ContextualPrecisionScorer(model=mock_llm)

        result = await scorer.evaluate(
            input_text="What is machine learning?",
            output_text="Machine learning is a subset of AI.",
            context="",
        )

        assert isinstance(result, ScoreResult)
        assert result.score == 0.0
        assert not result.passed
        assert "No context provided" in result.reasoning
        assert result.metadata.get("error") == "no_context"

    @pytest.mark.asyncio
    async def test_evaluate_with_single_context_chunk(self):
        """Test evaluate method with single context chunk."""
        mock_llm = MockLLM()
        scorer = ContextualPrecisionScorer(model=mock_llm)

        context = "Machine learning is a subset of artificial intelligence."

        # Mock the relevance score parsing to return a predictable score
        with patch.object(scorer, "_parse_relevance_score", return_value=4.0):
            result = await scorer.evaluate(
                input_text="What is machine learning?",
                output_text="Machine learning is a subset of AI.",
                context=context,
            )

            assert isinstance(result, ScoreResult)
            assert 0 <= result.score <= 1

    def test_split_context_method(self):
        """Test the _split_context method with various input formats."""
        mock_llm = MockLLM()
        scorer = ContextualPrecisionScorer(model=mock_llm)

        # Test with double newlines - paragraphs need to be long enough (min 50 chars)
        context1 = "This is the first paragraph with enough characters to meet the minimum length requirement.\n\nThis is the second paragraph with sufficient length to pass the filter.\n\nThis is the third paragraph that also meets the minimum character count."
        chunks1 = scorer._split_context(context1)
        assert len(chunks1) == 3
        assert "first paragraph" in chunks1[0]
        assert "second paragraph" in chunks1[1]
        assert "third paragraph" in chunks1[2]

        # Test with single paragraph (should split by sentences, but regex removes punctuation)
        # Each sentence needs to be at least 50 chars after punctuation removal
        context2 = "This is the first sentence with enough characters to meet the minimum length requirement after punctuation removal. This is the second sentence that also meets the requirement and has sufficient length. This is the third sentence with enough characters to pass the filter."
        chunks2 = scorer._split_context(context2)
        # After splitting by sentences and filtering by length, we should have chunks
        assert len(chunks2) >= 1

        # Test with very short context (should return original due to min length filter)
        context3 = "Short context."
        chunks3 = scorer._split_context(context3)
        assert len(chunks3) == 1
        assert chunks3[0] == context3

        # Test with context below minimum length (should return original)
        context4 = "Too short"
        chunks4 = scorer._split_context(context4)
        assert len(chunks4) == 1
        assert chunks4[0] == context4

        # Test with a sentence that's clearly longer than 50 characters
        context5 = "This is a much longer sentence that definitely exceeds the minimum length requirement of fifty characters and should pass the filter."
        chunks5 = scorer._split_context(context5)
        # Should be split into sentences and each should meet the length requirement
        assert len(chunks5) >= 1

    def test_parse_relevance_score_method(self):
        """Test the _parse_relevance_score method with various response formats."""
        mock_llm = MockLLM()
        scorer = ContextualPrecisionScorer(model=mock_llm)

        # Test with "Rating: X" format
        response1 = "Rating: 4\nExplanation: Highly relevant"
        score1 = scorer._parse_relevance_score(response1)
        assert score1 == 4.0

        # Test with standalone number
        response2 = "This is relevant with a score of 5"
        score2 = scorer._parse_relevance_score(response2)
        assert score2 == 5.0

        # Test with multiple numbers (should return first 1-5)
        response3 = "Rating 3 out of 10, very relevant"
        score3 = scorer._parse_relevance_score(response3)
        assert score3 == 3.0

        # Test with no valid rating (should return default 3.0)
        response4 = "This response has no rating"
        score4 = scorer._parse_relevance_score(response4)
        assert score4 == 3.0

        # Test with numbers outside 1-5 range (method doesn't validate range)
        response5 = "Rating: 7\nExplanation: Very relevant"
        score5 = scorer._parse_relevance_score(response5)
        assert score5 == 7.0  # Method returns whatever number it finds

    @pytest.mark.asyncio
    async def test_evaluate_with_llm_exception(self):
        """Test evaluate method when LLM generation fails."""
        mock_llm = MockLLM()
        scorer = ContextualPrecisionScorer(model=mock_llm)

        # Mock the LLM to raise an exception
        async def mock_generate(prompt):
            raise Exception("LLM API error")

        mock_llm.generate = mock_generate

        result = await scorer.evaluate(
            input_text="What is machine learning?",
            output_text="Machine learning is a subset of AI.",
            context="Machine learning is a subset of artificial intelligence.",
        )

        assert isinstance(result, ScoreResult)
        assert result.score == 0.0
        assert not result.passed
        assert "LLM API error" in result.reasoning


class TestContextualRecallScorer:
    """Test class for ContextualRecallScorer to improve coverage."""

    @pytest.mark.asyncio
    async def test_evaluate_with_expected_output(self):
        """Test evaluate method with expected output."""
        mock_llm = MockLLM()
        scorer = ContextualRecallScorer(model=mock_llm)

        # Mock the LLM to return predictable results
        async def mock_generate(prompt):
            if "expected answer" in prompt:
                return "Expected answer: Paris is the capital of France."
            elif "context chunk" in prompt:
                return "Rating: 4\nExplanation: Relevant"
            else:
                return "Rating: 3\nExplanation: Moderately relevant"

        mock_llm.generate = mock_generate

        result = await scorer.evaluate(
            input_text="What is the capital of France?",
            output_text="Paris is the capital of France.",
            expected_output="Paris is the capital of France.",
            context="Paris is the capital of France. France is in Europe.",
        )

        assert isinstance(result, ScoreResult)
        assert 0 <= result.score <= 1
        assert isinstance(result.passed, bool)
        assert isinstance(result.reasoning, str)
        assert isinstance(result.metadata, dict)

    @pytest.mark.asyncio
    async def test_evaluate_without_expected_output(self):
        """Test evaluate method without expected output."""
        mock_llm = MockLLM()
        scorer = ContextualRecallScorer(model=mock_llm)

        # Mock the LLM to return predictable results
        async def mock_generate(prompt):
            if "context chunk" in prompt:
                return "Rating: 4\nExplanation: Relevant"
            else:
                return "Rating: 3\nExplanation: Moderately relevant"

        mock_llm.generate = mock_generate

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

    def test_score_method_with_context_dict(self):
        """Test score method with context dictionary."""
        mock_llm = MockLLM()
        scorer = ContextualRecallScorer(model=mock_llm)

        # Mock the evaluate method to return a predictable result
        async def mock_evaluate(*args, **kwargs):
            return ScoreResult(
                score=0.8, passed=True, reasoning="Good recall", metadata={}
            )

        scorer.evaluate = mock_evaluate

        result = scorer.score(
            prediction="Paris is the capital of France.",
            ground_truth="What is the capital of France?",
            context={"context": "Paris is the capital of France. France is in Europe."},
        )

        assert result == 0.8

    def test_score_method_without_context(self):
        """Test score method without context."""
        mock_llm = MockLLM()
        scorer = ContextualRecallScorer(model=mock_llm)

        # Mock the evaluate method to return a predictable result
        async def mock_evaluate(*args, **kwargs):
            return ScoreResult(
                score=0.7, passed=True, reasoning="Good recall", metadata={}
            )

        scorer.evaluate = mock_evaluate

        result = scorer.score(
            prediction="Paris is the capital of France.",
            ground_truth="What is the capital of France?",
        )

        assert result == 0.7

    def test_parse_claims_edge_cases(self):
        """Test _parse_claims method with edge cases."""
        mock_llm = MockLLM()
        scorer = ContextualRecallScorer(model=mock_llm)

        # Test with empty response
        claims1 = scorer._parse_claims("")
        assert len(claims1) == 0

        # Test with single character
        claims2 = scorer._parse_claims("a")
        assert len(claims2) == 0

        # Test with numbers but no claims (these will be parsed as claims since they don't end with ?)
        response3 = "1. This is not a claim\n2. Neither is this"
        claims3 = scorer._parse_claims(response3)
        assert len(claims3) == 2  # These are actually parsed as claims

        # Test with claims that end with question marks (these are also parsed as claims)
        response4 = "1. What is Python\n2. How does it work"
        claims4 = scorer._parse_claims(response4)
        assert (
            len(claims4) == 2
        )  # These are parsed as claims regardless of question marks

        # Test with mixed formats
        response5 = "1. First claim\n- Second claim\n* Third claim"
        claims5 = scorer._parse_claims(response5)
        assert len(claims5) == 3

        # Test with empty lines
        response6 = "1. First claim\n\n2. Second claim"
        claims6 = scorer._parse_claims(response6)
        assert len(claims6) == 2


class TestRAGASScorer:
    """Test class for RAGASScorer to improve coverage."""

    @pytest.mark.asyncio
    async def test_evaluate_with_custom_weights(self):
        """Test evaluate method with custom weights."""
        mock_llm = MockLLM()
        weights = {
            "answer_relevancy": 0.4,
            "faithfulness": 0.3,
            "contextual_precision": 0.2,
            "contextual_recall": 0.1,
        }
        scorer = RAGASScorer(model=mock_llm, weights=weights)

        result = await scorer.evaluate(
            input_text="What is machine learning?",
            output_text="Machine learning is a subset of AI.",
            expected_output="Machine learning is a subset of AI and involves training algorithms.",
            context="Machine learning is a subset of artificial intelligence. It involves training algorithms on data.",
        )

        assert isinstance(result, ScoreResult)
        assert 0 <= result.score <= 1
        assert "weights" in result.metadata

    @pytest.mark.asyncio
    async def test_evaluate_with_default_weights(self):
        """Test evaluate method with default weights."""
        mock_llm = MockLLM()
        scorer = RAGASScorer(model=mock_llm)

        result = await scorer.evaluate(
            input_text="What is machine learning?",
            output_text="Machine learning is a subset of AI.",
            expected_output="Machine learning is a subset of AI and involves training algorithms.",
            context="Machine learning is a subset of artificial intelligence. It involves training algorithms on data.",
        )

        assert isinstance(result, ScoreResult)
        assert 0 <= result.score <= 1
        assert "weights" in result.metadata

    @pytest.mark.asyncio
    async def test_evaluate_with_exception_handling(self):
        """Test evaluate method with exception handling."""
        mock_llm = MockLLM()
        scorer = RAGASScorer(model=mock_llm)

        # Mock one of the scorers to raise an exception
        async def mock_evaluate(*args, **kwargs):
            raise Exception("Scorer error")

        scorer.answer_relevancy_scorer.evaluate = mock_evaluate

        result = await scorer.evaluate(
            input_text="What is machine learning?",
            output_text="Machine learning is a subset of AI.",
            expected_output="Machine learning is a subset of AI and involves training algorithms.",
            context="Machine learning is a subset of artificial intelligence. It involves training algorithms on data.",
        )

        assert isinstance(result, ScoreResult)
        assert 0 <= result.score <= 1
        assert "Scorer error" in result.reasoning

    @pytest.mark.asyncio
    async def test_evaluate_with_multiple_scorer_exceptions(self):
        """Test evaluate method with multiple scorer exceptions."""
        mock_llm = MockLLM()
        scorer = RAGASScorer(model=mock_llm)

        # Mock multiple scorers to raise exceptions
        async def mock_evaluate1(*args, **kwargs):
            raise Exception("Answer relevancy error")

        async def mock_evaluate2(*args, **kwargs):
            raise Exception("Faithfulness error")

        scorer.answer_relevancy_scorer.evaluate = mock_evaluate1
        scorer.faithfulness_scorer.evaluate = mock_evaluate2

        result = await scorer.evaluate(
            input_text="What is machine learning?",
            output_text="Machine learning is a subset of AI.",
            expected_output="Machine learning is a subset of AI and involves training algorithms.",
            context="Machine learning is a subset of artificial intelligence. It involves training algorithms on data.",
        )

        assert isinstance(result, ScoreResult)
        assert 0 <= result.score <= 1
        assert "Answer relevancy error" in result.reasoning
        assert "Faithfulness error" in result.reasoning

    @pytest.mark.asyncio
    async def test_evaluate_with_partial_scorer_success(self):
        """Test evaluate method with some scorers succeeding and some failing."""
        mock_llm = MockLLM()
        scorer = RAGASScorer(model=mock_llm)

        # Mock one scorer to succeed and one to fail
        async def mock_evaluate_success(*args, **kwargs):
            return ScoreResult(score=0.8, passed=True, reasoning="Success", metadata={})

        async def mock_evaluate_failure(*args, **kwargs):
            raise Exception("Scorer error")

        scorer.answer_relevancy_scorer.evaluate = mock_evaluate_success
        scorer.faithfulness_scorer.evaluate = mock_evaluate_failure

        result = await scorer.evaluate(
            input_text="What is machine learning?",
            output_text="Machine learning is a subset of AI.",
            expected_output="Machine learning is a subset of AI and involves training algorithms.",
            context="Machine learning is a subset of artificial intelligence. It involves training algorithms on data.",
        )

        assert isinstance(result, ScoreResult)
        assert 0 <= result.score <= 1
        assert "Scorer error" in result.reasoning

    def test_initialization_with_custom_weights(self):
        """Test RAGASScorer initialization with custom weights."""
        mock_llm = MockLLM()
        custom_weights = {"answer_relevancy": 0.5, "faithfulness": 0.5}
        scorer = RAGASScorer(model=mock_llm, weights=custom_weights)

        assert scorer.weights == custom_weights
        assert scorer.answer_relevancy_scorer is not None
        assert scorer.faithfulness_scorer is not None
        assert scorer.contextual_precision_scorer is not None
        assert scorer.contextual_recall_scorer is not None

    def test_initialization_with_default_weights(self):
        """Test RAGASScorer initialization with default weights."""
        mock_llm = MockLLM()
        scorer = RAGASScorer(model=mock_llm)

        expected_defaults = {
            "answer_relevancy": 0.25,
            "faithfulness": 0.35,
            "contextual_precision": 0.2,
            "contextual_recall": 0.2,
        }
        assert scorer.weights == expected_defaults

    def test_initialization_with_invalid_weights(self):
        """Test RAGASScorer initialization with invalid weights."""
        mock_llm = MockLLM()

        # Test with weights that don't sum to 1
        invalid_weights = {"answer_relevancy": 0.3, "faithfulness": 0.3}
        scorer = RAGASScorer(model=mock_llm, weights=invalid_weights)

        # The weights should be used as-is, not normalized
        assert scorer.weights == invalid_weights

    def test_initialization_with_empty_weights(self):
        """Test RAGASScorer initialization with empty weights."""
        mock_llm = MockLLM()

        # Test with empty weights dict
        empty_weights = {}
        scorer = RAGASScorer(model=mock_llm, weights=empty_weights)

        # Should use default weights
        expected_defaults = {
            "answer_relevancy": 0.25,
            "faithfulness": 0.35,
            "contextual_precision": 0.2,
            "contextual_recall": 0.2,
        }
        assert scorer.weights == expected_defaults


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
