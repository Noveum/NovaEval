"""
Unit tests for RAG (Retrieval-Augmented Generation) scorers.
"""

from unittest.mock import patch

import pytest

from novaeval.scorers.base import ScoreResult
from novaeval.scorers.rag import (
    AnswerRelevancyScorer,
    ContextualPrecisionScorer,
    ContextualRecallScorer,
    FaithfulnessScorer,
    RAGASScorer,
)

pytestmark = pytest.mark.unit


class MockLLMModel:
    """Mock LLM model for testing."""

    async def generate(self, prompt: str) -> str:
        """Mock generate method."""
        if "question" in prompt.lower():
            return "1. What is the capital of France?\n2. What is the largest city in France?\n3. What country is Paris in?"
        elif "claim" in prompt.lower():
            return "1. Paris is the capital of France\n2. France is a European country\n3. The Eiffel Tower is in Paris"
        elif "verification" in prompt.lower():
            return "Verification: SUPPORTED\nExplanation: This information is present in the context"
        elif "relevance" in prompt.lower():
            return "Rating: 4\nExplanation: This context is highly relevant"
        elif "status" in prompt.lower():
            return (
                "Status: PRESENT\nExplanation: This information is found in the context"
            )
        elif "key information" in prompt.lower() or "key fact" in prompt.lower():
            return "1. Paris is the capital of France\n2. France is a European country\n3. The Eiffel Tower is in Paris"
        else:
            return "Mock response"


class MockSentenceTransformer:
    """Mock SentenceTransformer for testing."""

    def __init__(self, model_name: str):
        self.model_name = model_name

    def encode(self, texts, **kwargs):
        """Mock encode method that returns predictable embeddings."""
        import numpy as np

        # Return deterministic embeddings based on text content
        # This ensures consistent test results without actual model inference
        embeddings = []
        for text in texts:
            # Create a simple hash-based embedding for consistency
            hash_val = hash(text) % 1000
            np.random.seed(hash_val)
            embedding = np.random.rand(384)  # Standard embedding size
            # Normalize the embedding
            embedding = embedding / np.linalg.norm(embedding)
            embeddings.append(embedding)

        if len(texts) == 1:
            return np.array(embeddings)
        return np.array(embeddings)


@pytest.fixture
def mock_sentence_transformer():
    """Fixture to mock SentenceTransformer."""
    with patch("novaeval.scorers.rag.SentenceTransformer", MockSentenceTransformer):
        yield MockSentenceTransformer("all-MiniLM-L6-v2")


class TestAnswerRelevancyScorer:
    """Test cases for AnswerRelevancyScorer class."""

    def test_init_default(self, mock_sentence_transformer):
        """Test initialization with default parameters."""
        mock_model = MockLLMModel()
        scorer = AnswerRelevancyScorer(mock_model, name="answer_relevancy")

        assert scorer.threshold == 0.7
        assert scorer.model == mock_model
        assert scorer.embedding_model is not None

    def test_init_with_custom_params(self, mock_sentence_transformer):
        """Test initialization with custom parameters."""
        mock_model = MockLLMModel()
        scorer = AnswerRelevancyScorer(
            mock_model,
            name="custom_answer_relevancy",
            threshold=0.8,
            embedding_model="all-MiniLM-L6-v2",
        )

        assert scorer.threshold == 0.8
        assert scorer.embedding_model is not None

    @pytest.mark.asyncio
    async def test_evaluate_success(self, mock_sentence_transformer):
        """Test successful evaluation."""
        mock_model = MockLLMModel()
        scorer = AnswerRelevancyScorer(mock_model, name="answer_relevancy")

        result = await scorer.evaluate(
            input_text="What is the capital of France?",
            output_text="Paris is the capital of France.",
        )

        assert isinstance(result, ScoreResult)
        assert 0.0 <= result.score <= 1.0
        assert "Answer Relevancy Analysis" in result.reasoning
        assert "generated_questions" in result.metadata

    @pytest.mark.asyncio
    async def test_evaluate_with_context(self, mock_sentence_transformer):
        """Test evaluation with context."""
        mock_model = MockLLMModel()
        scorer = AnswerRelevancyScorer(mock_model, name="answer_relevancy")

        result = await scorer.evaluate(
            input_text="What is the capital of France?",
            output_text="Paris is the capital of France.",
            context="France is a country in Europe. Paris is its capital city.",
        )

        assert isinstance(result, ScoreResult)
        assert 0.0 <= result.score <= 1.0

    @pytest.mark.asyncio
    async def test_evaluate_multiple_queries(self, mock_sentence_transformer):
        """Test evaluation of multiple queries."""
        mock_model = MockLLMModel()
        scorer = AnswerRelevancyScorer(mock_model, name="answer_relevancy")

        queries = ["What is the capital?", "What is the population?"]
        contexts = [["Context 1"], ["Context 2"]]

        scores = await scorer.evaluate_multiple_queries(
            queries, contexts, "Paris is the capital of France."
        )

        assert len(scores) == 2
        assert all(isinstance(score, float) for score in scores)

    @pytest.mark.asyncio
    async def test_evaluate_multiple_queries_length_mismatch(
        self, mock_sentence_transformer
    ):
        """Test evaluation with mismatched query and context lengths."""
        mock_model = MockLLMModel()
        scorer = AnswerRelevancyScorer(mock_model, name="answer_relevancy")

        queries = ["What is the capital?"]
        contexts = [["Context 1"], ["Context 2"]]  # Mismatch

        with pytest.raises(ValueError, match="Length mismatch"):
            await scorer.evaluate_multiple_queries(
                queries, contexts, "Paris is the capital of France."
            )

    @pytest.mark.asyncio
    async def test_evaluate_single_query_with_contexts(self, mock_sentence_transformer):
        """Test evaluation of single query with multiple contexts."""
        mock_model = MockLLMModel()
        scorer = AnswerRelevancyScorer(mock_model, name="answer_relevancy")

        score = await scorer._evaluate_single_query_with_contexts(
            "What is the capital?",
            ["Context 1", "Context 2"],
            "Paris is the capital of France.",
        )

        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    @pytest.mark.asyncio
    async def test_evaluate_single_query_with_contexts_empty(
        self, mock_sentence_transformer
    ):
        """Test evaluation with empty contexts."""
        mock_model = MockLLMModel()
        scorer = AnswerRelevancyScorer(mock_model, name="answer_relevancy")

        score = await scorer._evaluate_single_query_with_contexts(
            "What is the capital?", [], "Paris is the capital of France."
        )

        assert score == 0.0

    def test_score_sync_wrapper(self, mock_sentence_transformer):
        """Test synchronous score wrapper."""
        mock_model = MockLLMModel()
        scorer = AnswerRelevancyScorer(mock_model, name="answer_relevancy")

        score = scorer.score(
            prediction="Paris is the capital of France.",
            ground_truth="What is the capital of France?",
            context={"context": "France is a country in Europe."},
        )

        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_parse_questions(self, mock_sentence_transformer):
        """Test parsing of generated questions."""
        mock_model = MockLLMModel()
        scorer = AnswerRelevancyScorer(mock_model, name="answer_relevancy")

        response = "1. What is the capital?\n2. What is the population?\n3. Where is it located?"
        questions = scorer._parse_questions(response)

        assert len(questions) == 3
        assert all(q.endswith("?") for q in questions)

    def test_parse_questions_with_bullets(self, mock_sentence_transformer):
        """Test parsing questions with bullet points."""
        mock_model = MockLLMModel()
        scorer = AnswerRelevancyScorer(mock_model, name="answer_relevancy")

        response = (
            "- What is the capital?\n* What is the population?\n1. Where is it located?"
        )
        questions = scorer._parse_questions(response)

        assert len(questions) == 3
        assert all(q.endswith("?") for q in questions)


class TestFaithfulnessScorer:
    """Test cases for FaithfulnessScorer class."""

    def test_init_default(self):
        """Test initialization with default parameters."""
        mock_model = MockLLMModel()
        scorer = FaithfulnessScorer(mock_model, name="faithfulness")

        assert scorer.threshold == 0.8
        assert scorer.model == mock_model

    def test_init_with_custom_params(self):
        """Test initialization with custom parameters."""
        mock_model = MockLLMModel()
        scorer = FaithfulnessScorer(mock_model, name="faithfulness", threshold=0.9)

        assert scorer.threshold == 0.9

    @pytest.mark.asyncio
    async def test_evaluate_no_context(self):
        """Test evaluation without context."""
        mock_model = MockLLMModel()
        scorer = FaithfulnessScorer(mock_model, name="faithfulness")

        result = await scorer.evaluate(
            input_text="What is the capital?",
            output_text="Paris is the capital of France.",
        )

        assert result.score == 0.0
        assert not result.passed
        assert "No context provided" in result.reasoning

    @pytest.mark.asyncio
    async def test_evaluate_success(self):
        """Test successful evaluation."""
        mock_model = MockLLMModel()
        scorer = FaithfulnessScorer(mock_model, name="faithfulness")

        result = await scorer.evaluate(
            input_text="What is the capital?",
            output_text="Paris is the capital of France.",
            context="France is a country in Europe. Paris is its capital city.",
        )

        assert isinstance(result, ScoreResult)
        assert 0.0 <= result.score <= 1.0
        assert "Faithfulness Analysis" in result.reasoning

    @pytest.mark.asyncio
    async def test_evaluate_multiple_queries(self):
        """Test evaluation of multiple queries."""
        mock_model = MockLLMModel()
        scorer = FaithfulnessScorer(mock_model, name="faithfulness")

        queries = ["What is the capital?", "What is the population?"]
        contexts = [["Context 1"], ["Context 2"]]

        scores = await scorer.evaluate_multiple_queries(
            queries, contexts, "Paris is the capital of France."
        )

        assert len(scores) == 2
        assert all(isinstance(score, float) for score in scores)

    @pytest.mark.asyncio
    async def test_evaluate_single_query_with_contexts(self):
        """Test evaluation of single query with multiple contexts."""
        mock_model = MockLLMModel()
        scorer = FaithfulnessScorer(mock_model, name="faithfulness")

        score = await scorer._evaluate_single_query_with_contexts(
            "What is the capital?",
            ["Context 1", "Context 2"],
            "Paris is the capital of France.",
        )

        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_score_sync_wrapper(self):
        """Test synchronous score wrapper."""
        mock_model = MockLLMModel()
        scorer = FaithfulnessScorer(mock_model, name="faithfulness")

        score = scorer.score(
            prediction="Paris is the capital of France.",
            ground_truth="What is the capital of France?",
            context={"context": "France is a country in Europe."},
        )

        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_parse_claims(self):
        """Test parsing of claims."""
        mock_model = MockLLMModel()
        scorer = FaithfulnessScorer(mock_model, name="faithfulness")

        response = "1. Paris is the capital\n2. France is in Europe\n3. Eiffel Tower is in Paris"
        claims = scorer._parse_claims(response)

        assert len(claims) == 3
        assert "Paris is the capital" in claims


class TestContextualPrecisionScorer:
    """Test cases for ContextualPrecisionScorer class."""

    def test_init_default(self):
        """Test initialization with default parameters."""
        mock_model = MockLLMModel()
        scorer = ContextualPrecisionScorer(mock_model, name="contextual_precision")

        assert scorer.threshold == 0.7
        assert scorer.model == mock_model

    @pytest.mark.asyncio
    async def test_evaluate_no_context(self):
        """Test evaluation without context."""
        mock_model = MockLLMModel()
        scorer = ContextualPrecisionScorer(mock_model, name="contextual_precision")

        result = await scorer.evaluate(
            input_text="What is the capital?",
            output_text="Paris is the capital of France.",
        )

        assert result.score == 0.0
        assert not result.passed
        assert "No context provided" in result.reasoning

    @pytest.mark.asyncio
    async def test_evaluate_success(self):
        """Test successful evaluation."""
        mock_model = MockLLMModel()
        scorer = ContextualPrecisionScorer(mock_model, name="contextual_precision")

        result = await scorer.evaluate(
            input_text="What is the capital?",
            output_text="Paris is the capital of France.",
            context="France is a country in Europe. Paris is its capital city.",
        )

        assert isinstance(result, ScoreResult)
        assert 0.0 <= result.score <= 1.0
        assert "Contextual Precision Analysis" in result.reasoning

    @pytest.mark.asyncio
    async def test_evaluate_multiple_queries(self):
        """Test evaluation of multiple queries."""
        mock_model = MockLLMModel()
        scorer = ContextualPrecisionScorer(mock_model, name="contextual_precision")

        queries = ["What is the capital?", "What is the population?"]
        contexts = [["Context 1"], ["Context 2"]]

        scores = await scorer.evaluate_multiple_queries(
            queries, contexts, "Paris is the capital of France."
        )

        assert len(scores) == 2
        assert all(isinstance(score, float) for score in scores)

    @pytest.mark.asyncio
    async def test_evaluate_single_query_with_contexts(self):
        """Test evaluation of single query with multiple contexts."""
        mock_model = MockLLMModel()
        scorer = ContextualPrecisionScorer(mock_model, name="contextual_precision")

        score = await scorer._evaluate_single_query_with_contexts(
            "What is the capital?",
            ["Context 1", "Context 2"],
            "Paris is the capital of France.",
        )

        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_score_sync_wrapper(self):
        """Test synchronous score wrapper."""
        mock_model = MockLLMModel()
        scorer = ContextualPrecisionScorer(mock_model, name="contextual_precision")

        score = scorer.score(
            prediction="Paris is the capital of France.",
            ground_truth="What is the capital of France?",
            context={"context": "France is a country in Europe."},
        )

        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_split_context(self):
        """Test context splitting functionality."""
        mock_model = MockLLMModel()
        scorer = ContextualPrecisionScorer(mock_model, name="contextual_precision")

        context = "This is the first paragraph with enough characters to meet the minimum length requirement.\n\nThis is the second paragraph that also has sufficient length to pass the filter.\n\nThis is the third paragraph that meets the minimum character count as well."
        chunks = scorer._split_context(context)

        assert len(chunks) == 3
        assert all(len(chunk) >= 50 for chunk in chunks)

    def test_split_context_sentences(self):
        """Test context splitting by sentences."""
        mock_model = MockLLMModel()
        scorer = ContextualPrecisionScorer(mock_model, name="contextual_precision")

        context = "First sentence. Second sentence. Third sentence."
        chunks = scorer._split_context(context)

        assert len(chunks) >= 1

    def test_parse_relevance_score(self):
        """Test parsing of relevance scores."""
        mock_model = MockLLMModel()
        scorer = ContextualPrecisionScorer(mock_model, name="contextual_precision")

        response = "Rating: 4\nExplanation: Highly relevant"
        score = scorer._parse_relevance_score(response)

        assert score == 4.0

    def test_parse_relevance_score_fallback(self):
        """Test parsing fallback for relevance scores."""
        mock_model = MockLLMModel()
        scorer = ContextualPrecisionScorer(mock_model, name="contextual_precision")

        response = "This context is moderately relevant"
        score = scorer._parse_relevance_score(response)

        assert score == 3.0  # Default fallback


class TestContextualRecallScorer:
    """Test cases for ContextualRecallScorer class."""

    def test_init_default(self):
        """Test initialization with default parameters."""
        mock_model = MockLLMModel()
        scorer = ContextualRecallScorer(mock_model, name="contextual_recall")

        assert scorer.threshold == 0.7
        assert scorer.model == mock_model

    @pytest.mark.asyncio
    async def test_evaluate_missing_inputs(self):
        """Test evaluation with missing inputs."""
        mock_model = MockLLMModel()
        scorer = ContextualRecallScorer(mock_model, name="contextual_recall")

        result = await scorer.evaluate(
            input_text="What is the capital?",
            output_text="Paris is the capital of France.",
        )

        assert result.score == 0.0
        assert not result.passed
        assert "Both context and expected output are required" in result.reasoning

    @pytest.mark.asyncio
    async def test_evaluate_success(self):
        """Test successful evaluation."""
        mock_model = MockLLMModel()
        scorer = ContextualRecallScorer(mock_model, name="contextual_recall")

        result = await scorer.evaluate(
            input_text="What is the capital?",
            output_text="Paris is the capital of France.",
            context="France is a country in Europe. Paris is its capital city.",
            expected_output="Paris is the capital of France.",
        )

        assert isinstance(result, ScoreResult)
        assert 0.0 <= result.score <= 1.0
        assert "Contextual Recall Analysis" in result.reasoning

    @pytest.mark.asyncio
    async def test_evaluate_multiple_queries(self):
        """Test evaluation of multiple queries."""
        mock_model = MockLLMModel()
        scorer = ContextualRecallScorer(mock_model, name="contextual_recall")

        queries = ["What is the capital?", "What is the population?"]
        contexts = [["Context 1"], ["Context 2"]]

        scores = await scorer.evaluate_multiple_queries(
            queries, contexts, "Paris is the capital of France."
        )

        assert len(scores) == 2
        assert all(isinstance(score, float) for score in scores)

    @pytest.mark.asyncio
    async def test_evaluate_single_query_with_contexts(self):
        """Test evaluation of single query with multiple contexts."""
        mock_model = MockLLMModel()
        scorer = ContextualRecallScorer(mock_model, name="contextual_recall")

        score = await scorer._evaluate_single_query_with_contexts(
            "What is the capital?",
            ["Context 1", "Context 2"],
            "Paris is the capital of France.",
            expected_output="Paris is the capital of France.",
        )

        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_score_sync_wrapper(self):
        """Test synchronous score wrapper."""
        mock_model = MockLLMModel()
        scorer = ContextualRecallScorer(mock_model, name="contextual_recall")

        score = scorer.score(
            prediction="Paris is the capital of France.",
            ground_truth="What is the capital of France?",
            context={
                "context": "France is a country in Europe.",
                "expected_output": "Paris is the capital of France.",
            },
        )

        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0


class TestRAGASScorer:
    """Test cases for RAGASScorer class."""

    def test_init_default(self):
        """Test initialization with default parameters."""
        mock_model = MockLLMModel()
        scorer = RAGASScorer(mock_model)

        assert scorer.threshold == 0.7
        assert scorer.model == mock_model
        assert scorer.name == "ragas_scorer"
        assert "answer_relevancy" in scorer.weights
        assert "faithfulness" in scorer.weights

    def test_init_with_custom_params(self):
        """Test initialization with custom parameters."""
        mock_model = MockLLMModel()
        custom_weights = {"answer_relevancy": 0.5, "faithfulness": 0.5}
        scorer = RAGASScorer(mock_model, threshold=0.8, weights=custom_weights)

        assert scorer.threshold == 0.8
        assert scorer.weights == custom_weights

    def test_init_with_name_override(self):
        """Test initialization with name override."""
        mock_model = MockLLMModel()
        scorer = RAGASScorer(mock_model, name="custom_ragas")

        assert scorer.name == "custom_ragas"

    @pytest.mark.asyncio
    async def test_evaluate_success(self):
        """Test successful evaluation."""
        mock_model = MockLLMModel()
        scorer = RAGASScorer(mock_model)

        result = await scorer.evaluate(
            input_text="What is the capital?",
            output_text="Paris is the capital of France.",
            context="France is a country in Europe. Paris is its capital city.",
        )

        assert isinstance(result, ScoreResult)
        assert 0.0 <= result.score <= 1.0
        assert "RAGAS Evaluation Results" in result.reasoning
        assert "individual_scores" in result.metadata

    @pytest.mark.asyncio
    async def test_evaluate_multiple_queries(self):
        """Test evaluation of multiple queries."""
        mock_model = MockLLMModel()
        scorer = RAGASScorer(mock_model)

        queries = ["What is the capital?", "What is the population?"]
        contexts = [["Context 1"], ["Context 2"]]

        scores = await scorer.evaluate_multiple_queries(
            queries, contexts, "Paris is the capital of France."
        )

        assert len(scores) == 2
        assert all(isinstance(score, float) for score in scores)

    @pytest.mark.asyncio
    async def test_evaluate_single_query_with_contexts(self):
        """Test evaluation of single query with multiple contexts."""
        mock_model = MockLLMModel()
        scorer = RAGASScorer(mock_model)

        score = await scorer._evaluate_single_query_with_contexts(
            "What is the capital?",
            ["Context 1", "Context 2"],
            "Paris is the capital of France.",
        )

        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_score_sync_wrapper(self):
        """Test synchronous score wrapper."""
        mock_model = MockLLMModel()
        scorer = RAGASScorer(mock_model)

        score = scorer.score(
            prediction="Paris is the capital of France.",
            ground_truth="What is the capital of France?",
            context={"context": "France is a country in Europe."},
        )

        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
