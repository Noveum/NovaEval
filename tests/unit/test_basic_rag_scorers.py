"""
Tests for basic RAG scorers.
"""

from unittest.mock import Mock, patch

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
        with patch.object(scorer, "_call_model", return_value="Rating: 8"):
            result = await scorer._evaluate_chunk_relevance("query", "chunk")
            assert result is True  # 8 >= 7 (0.7 * 10)

    @pytest.mark.asyncio
    async def test_evaluate_chunk_relevance_exception(self):
        model = Mock()
        scorer = ContextualPrecisionScorerPP(model=model)

        # Mock the _call_model method to raise an exception
        with patch.object(scorer, "_call_model", side_effect=Exception("API error")):
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
    async def test_evaluate_no_context(self):
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
            scorer, "_evaluate_chunk_relevance", side_effect=[True, False]
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

        with patch.object(scorer, "_call_model", return_value='{"relevant": false}'):
            result = await scorer._evaluate_chunk_relevance("query", "chunk")
            assert result is False

    @pytest.mark.asyncio
    async def test_estimate_total_relevant_chunks(self):
        model = Mock()
        scorer = ContextualRecallScorerPP(model=model)

        with patch.object(scorer, "_call_model", return_value='{"estimated_total": 5}'):
            result = await scorer._estimate_total_relevant_chunks(
                "query", ["chunk1", "chunk2"]
            )
            assert result == 5

    @pytest.mark.asyncio
    async def test_estimate_total_relevant_chunks_fallback(self):
        model = Mock()
        scorer = ContextualRecallScorerPP(model=model)

        with patch.object(scorer, "_call_model", return_value="invalid json"):
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
                scorer, "_evaluate_chunk_relevance", side_effect=[True, True, False]
            ),
            patch.object(scorer, "_estimate_total_relevant_chunks", return_value=4),
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


class TestSemanticSimilarityScorer:
    def test_init(self):
        scorer = SemanticSimilarityScorer(threshold=0.8, embedding_model="test-model")
        assert scorer.threshold == 0.8
        assert scorer.embedding_model == "test-model"
        assert scorer.model is None
        assert scorer._model_loaded is False

    def test_load_model_success(self):
        scorer = SemanticSimilarityScorer()

        with patch("sentence_transformers.SentenceTransformer") as mock_st:
            mock_model = Mock()
            mock_st.return_value = mock_model

            scorer._load_model()

            assert scorer.model == mock_model
            mock_st.assert_called_once_with("all-MiniLM-L6-v2")

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
        with patch.object(scorer, "_call_model", return_value="invalid response"):
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
