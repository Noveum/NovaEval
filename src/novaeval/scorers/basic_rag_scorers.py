"""
Basic RAG Scorers for NovaEval.

This module contains fundamental scorers for RAG evaluation including:
- Basic RAG scorers (from rag.py)
- Advanced retrieval scorers (precision, recall, ranking, diversity)
- Semantic similarity scorers
- Aggregate scoring
"""

from __future__ import annotations

import asyncio
import re
from typing import TYPE_CHECKING, Any

import numpy as np
from sklearn.metrics import average_precision_score, ndcg_score

from novaeval.scorers.base import BaseScorer, ScoreResult
from novaeval.scorers.rag_prompts import RAGPrompts
from novaeval.utils.llm import call_llm

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer  # noqa: TC004


class AsyncLLMScorer(BaseScorer):
    """
    Base class for scorers that use async LLM calls.
    Provides shared functionality for async model interaction.
    """

    def __init__(self, model: Any, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.model = model

    async def _call_model(self, prompt: str) -> str:
        """Async wrapper for call_llm."""
        return await asyncio.to_thread(call_llm, self.model, prompt)

    def _parse_numerical_response(self, response: str) -> float:
        """Parse numerical response from LLM and extract 0-10 score."""
        try:
            # Try to extract numerical score from response
            import re

            # Look for "Rating: X" pattern
            rating_match = re.search(r"Rating:\s*(\d+)", response, re.IGNORECASE)
            if rating_match:
                score_int = int(rating_match.group(1))
                return max(0, min(10, score_int))  # Ensure score is between 0-10

            # Look for standalone numbers
            numbers = re.findall(r"\b(\d+)\b", response)
            if numbers:
                score_int = int(numbers[0])
                return max(0, min(10, score_int))  # Ensure score is between 0-10

            # Fallback: try to extract from JSON if present
            try:
                import json

                result = json.loads(response)
                if "rating" in result:
                    score_float = float(result["rating"])
                    return max(0, min(10, score_float))
                elif "score" in result:
                    score_float = float(result["score"])
                    return max(0, min(10, score_float))
            except (ValueError, KeyError, TypeError):
                pass

            # Final fallback: default to 5 (moderate relevance)
            return 5.0

        except Exception:
            return 5.0  # Default to moderate relevance


class ContextualPrecisionScorerPP(AsyncLLMScorer):
    """
    Computes precision for retrieved chunks.
    Precision = (Number of relevant chunks retrieved) ÷ (Total number of chunks retrieved)
    """

    def __init__(
        self, model: Any, relevance_threshold: float = 0.7, **kwargs: Any
    ) -> None:
        super().__init__(name="RetrievalPrecisionScorer", model=model, **kwargs)
        self.relevance_threshold = relevance_threshold

    async def _evaluate_chunk_relevance(self, query: str, chunk: str) -> bool:
        """Evaluate if a single chunk is relevant to the query using 0-10 numerical scoring."""
        prompt = RAGPrompts.get_numerical_chunk_relevance_0_10(query, chunk)

        try:
            response = await self._call_model(prompt)
            numerical_score = self._parse_numerical_response(response)

            # Convert numerical score to boolean using threshold
            # Convert 0-1 threshold to 0-10 scale (e.g., 0.7 threshold = 7.0 score)
            threshold_score = self.relevance_threshold * 10
            return numerical_score >= threshold_score

        except Exception:
            return False

    def _parse_json_response(self, response: str) -> dict:
        """Parse JSON response from model."""
        import json

        # Try to extract JSON from response
        json_match = re.search(r"\{.*\}", response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

        # Fallback: try to parse the entire response as JSON
        try:
            return json.loads(response.strip())
        except json.JSONDecodeError:
            # Final fallback: look for boolean indicators
            response_lower = response.strip().lower()
            return {
                "relevant": "yes" in response_lower
                or "true" in response_lower
                or "1" in response_lower,
                "reasoning": "Fallback parsing used",
            }

    async def evaluate(
        self,
        input_text: str,
        output_text: str,
        expected_output: str | None = None,
        context: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> ScoreResult:
        if not context or not input_text:
            return ScoreResult(
                score=0.0,
                passed=False,
                reasoning="No context or input provided",
                metadata={},
            )

        # Extract chunks from context
        chunks = context.get("chunks")
        if not chunks and context.get("context"):
            chunks = [context["context"]]
        if not chunks:
            return ScoreResult(
                score=0.0,
                passed=False,
                reasoning="No chunks provided",
                metadata={},
            )

        # Evaluate each chunk for relevance
        relevant_chunks = 0
        total_chunks = len(chunks)
        chunk_results = []

        for _i, chunk in enumerate(chunks):
            is_relevant = await self._evaluate_chunk_relevance(input_text, chunk)
            chunk_results.append({"chunk": chunk, "relevant": is_relevant})
            if is_relevant:
                relevant_chunks += 1

        # Calculate precision
        precision = relevant_chunks / total_chunks if total_chunks > 0 else 0.0
        passed = precision >= self.relevance_threshold

        return ScoreResult(
            score=precision,
            passed=passed,
            reasoning=f"Precision: {precision:.3f} ({relevant_chunks} relevant out of {total_chunks} chunks)",
            metadata={
                "relevant_chunks": relevant_chunks,
                "total_chunks": total_chunks,
                "chunk_results": chunk_results,
            },
        )

    def score(
        self,
        prediction: str,
        ground_truth: str,
        context: dict[str, Any] | None = None,
    ) -> float | dict[str, float]:
        import asyncio

        # Check if we're already in an async context
        try:
            asyncio.get_running_loop()
            # We're in an async context, run in a separate thread
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    asyncio.run,
                    self.evaluate(
                        input_text=ground_truth,
                        output_text=prediction,
                        context=context,
                    ),
                )
                result = future.result()
        except RuntimeError:
            # No running loop, use asyncio.run directly
            result = asyncio.run(
                self.evaluate(
                    input_text=ground_truth,
                    output_text=prediction,
                    context=context,
                )
            )

        # Extract score from ScoreResult and return as float
        if hasattr(result, "score"):
            return result.score
        else:
            return 0.0


class ContextualRecallScorerPP(AsyncLLMScorer):
    """
    Computes recall for retrieved chunks.
    Note: Since we don't have access to all available chunks, this is an approximation.
    Recall = (Number of relevant chunks retrieved) ÷ (Estimated total relevant chunks)
    """

    def __init__(
        self, model: Any, relevance_threshold: float = 0.7, **kwargs: Any
    ) -> None:
        super().__init__(name="RetrievalRecallScorer", model=model, **kwargs)
        self.relevance_threshold = relevance_threshold

    async def _evaluate_chunk_relevance(self, query: str, chunk: str) -> bool:
        """Evaluate if a single chunk is relevant to the query using 0-10 numerical scoring."""
        prompt = RAGPrompts.get_numerical_chunk_relevance_0_10(query, chunk)

        try:
            response = await self._call_model(prompt)
            numerical_score = self._parse_numerical_response(response)

            # Convert numerical score to boolean using threshold
            # Convert 0-1 threshold to 0-10 scale (e.g., 0.7 threshold = 7.0 score)
            threshold_score = self.relevance_threshold * 10
            return numerical_score >= threshold_score

        except Exception:
            return False

    def _parse_json_response(self, response: str) -> dict:
        """Parse JSON response from model."""
        import json

        # Try to extract JSON from response
        json_match = re.search(r"\{.*\}", response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

        # Fallback: try to parse the entire response as JSON
        try:
            return json.loads(response.strip())
        except json.JSONDecodeError:
            # Final fallback: look for boolean indicators
            response_lower = response.strip().lower()
            return {
                "relevant": "yes" in response_lower
                or "true" in response_lower
                or "1" in response_lower,
                "reasoning": "Fallback parsing used",
            }

    async def _estimate_total_relevant_chunks(
        self, query: str, retrieved_chunks: list[str]
    ) -> int:
        """Estimate the total number of relevant chunks available."""
        prompt = RAGPrompts.get_estimate_total_relevant(query, len(retrieved_chunks))

        try:
            response = await self._call_model(prompt)
            result = self._parse_json_response(response)
            estimated_total = result.get("estimated_total", len(retrieved_chunks))
            return max(
                int(estimated_total), len(retrieved_chunks)
            )  # At least as many as retrieved
        except Exception:
            return len(retrieved_chunks)  # Default fallback

    async def evaluate(
        self,
        input_text: str,
        output_text: str,
        expected_output: str | None = None,
        context: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> ScoreResult:
        if not context or not input_text:
            return ScoreResult(
                score=0.0,
                passed=False,
                reasoning="No context or input provided",
                metadata={},
            )

        # Extract chunks from context
        chunks = context.get("chunks")
        if not chunks and context.get("context"):
            chunks = [context["context"]]
        if not chunks:
            return ScoreResult(
                score=0.0,
                passed=False,
                reasoning="No chunks provided",
                metadata={},
            )

        # Evaluate each chunk for relevance
        relevant_chunks = 0
        chunk_results = []

        for _i, chunk in enumerate(chunks):
            is_relevant = await self._evaluate_chunk_relevance(input_text, chunk)
            chunk_results.append({"chunk": chunk, "relevant": is_relevant})
            if is_relevant:
                relevant_chunks += 1

        # Estimate total relevant chunks available
        estimated_total_relevant = await self._estimate_total_relevant_chunks(
            input_text, chunks
        )

        # Calculate recall
        recall = (
            relevant_chunks / estimated_total_relevant
            if estimated_total_relevant > 0
            else 0.0
        )
        passed = recall >= self.relevance_threshold

        return ScoreResult(
            score=recall,
            passed=passed,
            reasoning=f"Recall: {recall:.3f} ({relevant_chunks} relevant retrieved, estimated {estimated_total_relevant} total relevant)",
            metadata={
                "relevant_chunks": relevant_chunks,
                "estimated_total_relevant": estimated_total_relevant,
                "chunk_results": chunk_results,
            },
        )

    def score(
        self,
        prediction: str,
        ground_truth: str,
        context: dict[str, Any] | None = None,
    ) -> float | dict[str, float]:
        import asyncio

        # Check if we're already in an async context
        try:
            asyncio.get_running_loop()
            # We're in an async context, run in a separate thread
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    asyncio.run,
                    self.evaluate(
                        input_text=ground_truth,
                        output_text=prediction,
                        context=context,
                    ),
                )
                result = future.result()
        except RuntimeError:
            # No running loop, use asyncio.run directly
            result = asyncio.run(
                self.evaluate(
                    input_text=ground_truth,
                    output_text=prediction,
                    context=context,
                )
            )

        # Extract score from ScoreResult and return as float
        if hasattr(result, "score"):
            return result.score
        else:
            return 0.0


class RetrievalF1Scorer(BaseScorer):
    """
    F1 score combining precision and recall for contextual evaluation.
    """

    def __init__(
        self,
        precision_scorer: Any,
        recall_scorer: Any,
        threshold: float = 0.5,
        **kwargs: Any,
    ) -> None:
        super().__init__(name="ContextualF1Scorer", **kwargs)
        self.precision_scorer = precision_scorer
        self.recall_scorer = recall_scorer
        self.threshold = threshold
        self._last_result: ScoreResult | None = None

    def score(
        self,
        prediction: str,
        ground_truth: str,
        context: dict[str, Any] | None = None,
    ) -> float | dict[str, float]:
        # Calls the precision and recall scorers, then computes F1
        precision = self.precision_scorer.score(prediction, ground_truth, context)
        recall = self.recall_scorer.score(prediction, ground_truth, context)

        if precision + recall == 0:
            f1_score = 0.0
        else:
            f1_score = 2 * (precision * recall) / (precision + recall)

        passed = f1_score >= self.threshold

        # Store the full result internally
        self._last_result = ScoreResult(
            score=f1_score,
            passed=passed,
            reasoning=f"F1 Score: {f1_score:.3f} (Precision: {precision:.3f}, Recall: {recall:.3f})",
            metadata={"precision": precision, "recall": recall},
        )

        # Return dictionary with precision and recall as required by BaseScorer
        # interface
        return {"f1": f1_score, "precision": precision, "recall": recall}

    def get_score_result(self) -> ScoreResult | None:
        """Get the full ScoreResult from the last evaluation."""
        return self._last_result


class RetrievalRankingScorer(BaseScorer):
    """
    Computes ranking metrics for retrieved context.
    """

    def __init__(self, threshold: float = 0.5, **kwargs: Any) -> None:
        super().__init__(name="RetrievalRankingScorer", **kwargs)
        self.threshold = threshold
        self._last_result: ScoreResult | None = None

    def score(
        self,
        prediction: str,
        ground_truth: str,
        context: dict[str, Any] | None = None,
    ) -> float | dict[str, float]:
        # Computes ranking scores based on rankings (1,2,3,4,5) and relevance scores
        if not context or "rankings" not in context:
            self._last_result = ScoreResult(
                score=0.0,
                passed=False,
                reasoning="No ranking data provided",
                metadata={},
            )
            return 0.0

        rankings = context["rankings"]
        relevance_labels = context.get("relevance_scores", [1.0] * len(rankings))

        # Define max_rank outside try block to avoid UnboundLocalError in except paths
        max_rank = 5  # Configurable maximum rank for scoring

        try:
            # Convert rankings to scores (1,2,3,4,5 -> 1.0, 0.8, 0.6, 0.4, 0.2)
            # Higher ranking (lower number) should get higher score
            ranking_scores = []
            for rank in rankings:
                score = (
                    (max_rank + 1 - rank) / max_rank if rank <= max_rank else 0.0
                )  # 1->1.0, 2->0.8, 3->0.6, 4->0.4, 5->0.2
                ranking_scores.append(score)

            # Derive predicted scores from rankings (higher ranking -> higher predicted score)
            # Convert rankings to predicted scores where lower rank number = higher score
            predicted_scores = []
            for rank in rankings:
                # Higher ranking (lower number) gets higher predicted score
                predicted_score = (
                    (max_rank + 1 - rank) / max_rank if rank <= max_rank else 0.0
                )
                predicted_scores.append(predicted_score)

            # Ensure relevance_labels and predicted_scores have the same length
            min_length = min(len(relevance_labels), len(predicted_scores))
            relevance_labels = relevance_labels[:min_length]
            predicted_scores = predicted_scores[:min_length]

            # Compute MRR (Mean Reciprocal Rank) - find best relevant rank
            mrr = 0.0
            try:
                # Find the minimum rank among items whose relevance label meets the threshold
                relevant_ranks = []
                for i, relevance in enumerate(relevance_labels):
                    if relevance > 0:  # Item is relevant
                        relevant_ranks.append(
                            rankings[i]
                        )  # Use actual rank, not position

                if relevant_ranks:
                    min_rank = min(relevant_ranks)
                    mrr = 1.0 / min_rank
            except Exception:
                mrr = 0.0

            # Compute NDCG using relevance labels and predicted scores
            try:
                # Convert to numpy arrays and ensure proper shape for sklearn
                relevance_array = np.array(relevance_labels).reshape(1, -1)
                predicted_array = np.array(predicted_scores).reshape(1, -1)
                ndcg = ndcg_score(
                    relevance_array, predicted_array, k=len(predicted_scores)
                )
            except Exception:
                ndcg = 0.0

            # Compute MAP using relevance labels and predicted scores
            try:
                # Convert to numpy arrays for sklearn
                relevance_array = np.array(relevance_labels)
                predicted_array = np.array(predicted_scores)
                map_score = average_precision_score(relevance_array, predicted_array)
            except Exception:
                map_score = 0.0

            # Average ranking score
            avg_ranking_score = np.mean(ranking_scores) if ranking_scores else 0.0

            # Combined score
            combined_score = (mrr + ndcg + map_score + avg_ranking_score) / 4.0
            passed = combined_score >= self.threshold

            self._last_result = ScoreResult(
                score=combined_score,
                passed=passed,
                reasoning=f"Ranking Score: {combined_score:.3f} (MRR: {mrr:.3f}, NDCG: {ndcg:.3f}, MAP: {map_score:.3f}, Avg: {avg_ranking_score:.3f})",
                metadata={
                    "mrr": mrr,
                    "ndcg": ndcg,
                    "map": map_score,
                    "avg_ranking": avg_ranking_score,
                    "ranking_scores": ranking_scores,
                    "original_rankings": rankings,
                },
            )
            return {
                "mrr": float(mrr),
                "ndcg": float(ndcg),
                "map": float(map_score),
                "avg_ranking": float(avg_ranking_score),
                "combined": float(combined_score),
            }
        except Exception as e:
            # Fallback to simple ranking score if computation fails
            try:
                # Simple fallback: just use average ranking score
                ranking_scores = []
                for rank in rankings:
                    score = (
                        (max_rank + 1 - rank) / max_rank if rank <= max_rank else 0.0
                    )
                    ranking_scores.append(score)

                avg_score = np.mean(ranking_scores) if ranking_scores else 0.0
                passed = avg_score >= self.threshold

                self._last_result = ScoreResult(
                    score=float(avg_score),
                    passed=bool(passed),
                    reasoning=f"Fallback Ranking Score: {avg_score:.3f} (average ranking score only)",
                    metadata={"avg_ranking": float(avg_score), "method": "fallback"},
                )
                return {"avg_ranking": float(avg_score)}
            except Exception:
                self._last_result = ScoreResult(
                    score=0.0,
                    passed=False,
                    reasoning=f"Ranking computation failed: {e!s}",
                    metadata={},
                )
                return 0.0

    def get_score_result(self) -> ScoreResult | None:
        """Get the full ScoreResult from the last evaluation."""
        return self._last_result


class SemanticSimilarityScorer(BaseScorer):
    """
    Computes semantic similarity between query and retrieved context.
    """

    def __init__(
        self,
        threshold: float = 0.7,
        embedding_model: str = "all-MiniLM-L6-v2",
        **kwargs: Any,
    ) -> None:
        super().__init__(name="SemanticSimilarityScorer", **kwargs)
        self.threshold = threshold
        self.embedding_model = embedding_model
        self.model: SentenceTransformer | None = None
        self._model_loaded = False
        self._last_result: ScoreResult | None = None

    def _load_model(self) -> None:
        """Load the sentence transformer model."""
        if self.model is None and not self._model_loaded:
            try:
                from sentence_transformers import SentenceTransformer

                self.model = SentenceTransformer(self.embedding_model)
                self._model_loaded = True
            except ImportError:
                self.model = None
                self._model_loaded = True  # Mark as loaded to avoid retrying
                print(
                    "Warning: sentence_transformers not installed. "
                    "Using simple similarity computation."
                )

    def _compute_simple_similarity(self, query: str, chunks: list[str]) -> float:
        """Fallback similarity computation without embeddings."""
        # Simple text-based similarity as fallback
        query_lower = query.lower()
        total_similarity = 0.0

        for chunk in chunks:
            chunk_lower = chunk.lower()
            # Simple word overlap similarity
            query_words = set(query_lower.split())
            chunk_words = set(chunk_lower.split())

            if query_words and chunk_words:
                overlap = len(query_words.intersection(chunk_words))
                union = len(query_words.union(chunk_words))
                similarity = overlap / union if union > 0 else 0.0
                total_similarity += similarity

        return total_similarity / len(chunks) if chunks else 0.0

    def score(
        self,
        prediction: str,
        ground_truth: str,
        context: dict[str, Any] | None = None,
    ) -> float | dict[str, float]:
        # Embeds the query and all chunks, computes mean semantic similarity
        if not context or not ground_truth:
            self._last_result = ScoreResult(
                score=0.0,
                passed=False,
                reasoning="No context or query provided",
                metadata={},
            )
            return 0.0

        try:
            if not self._model_loaded:
                self._load_model()

            query = ground_truth
            chunks = context.get("chunks")
            if not chunks and context.get("context"):
                chunks = [context["context"]]

            if self.model is None:
                # Use fallback similarity computation
                similarity = self._compute_simple_similarity(query, chunks or [])
                passed = similarity >= self.threshold

                self._last_result = ScoreResult(
                    score=similarity,
                    passed=passed,
                    reasoning=f"Fallback similarity score: {similarity:.3f} (using text-based similarity)",
                    metadata={"similarity": similarity, "method": "fallback"},
                )
                return {"similarity": similarity}

            # Compute embeddings
            query_embedding = self.model.encode([query])[0]
            chunk_embeddings = self.model.encode(chunks or [])

            # Compute similarities
            similarities = []
            for chunk_emb in chunk_embeddings:
                sim = np.dot(query_embedding, chunk_emb) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(chunk_emb)
                )
                # Normalize cosine similarity from [-1,1] to [0,1] to match fallback scale
                sim_norm = (sim + 1.0) / 2.0
                similarities.append(sim_norm)

            mean_similarity = np.mean(similarities)
            passed = bool(mean_similarity >= self.threshold)

            self._last_result = ScoreResult(
                score=float(mean_similarity),
                passed=passed,
                reasoning=f"Semantic Similarity Score: {mean_similarity:.3f}",
                metadata={
                    "mean_similarity": float(mean_similarity),
                    "similarities": similarities,
                },
            )
            return {
                "similarity": float(mean_similarity),
            }
        except Exception as e:
            self._last_result = ScoreResult(
                score=0.0,
                passed=False,
                reasoning=f"Semantic similarity computation failed: {e!s}",
                metadata={},
            )
            return 0.0

    def get_score_result(self) -> ScoreResult | None:
        """Get the full ScoreResult from the last evaluation."""
        return self._last_result


class RetrievalDiversityScorer(BaseScorer):
    """
    Evaluates diversity of retrieved context chunks using cosine distance between embeddings.
    """

    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        threshold: float = 0.3,
        **kwargs: Any,
    ) -> None:
        super().__init__(name="RetrievalDiversityScorer", **kwargs)
        self.embedding_model = embedding_model
        self.threshold = threshold
        self.model: SentenceTransformer | None = None
        self._model_loaded = False
        self._last_result: ScoreResult | None = None

    def _load_model(self) -> None:
        if self.model is None and not self._model_loaded:
            try:
                # Add safety checks for macOS/ARM64 issues
                import os
                import platform

                # Check if we're on macOS ARM64 (M1/M2) which has known issues
                if platform.system() == "Darwin" and platform.machine() == "arm64":
                    print(
                        "Warning: Detected macOS ARM64, using fallback mode to avoid segmentation faults"
                    )
                    self.model = None
                    self._model_loaded = True
                    return

                # Set environment variables to help with macOS issues
                os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

                # Try to load the model with error handling
                self.model = SentenceTransformer(self.embedding_model)
                self._model_loaded = True
            except Exception as e:
                # If model loading fails, we'll use a fallback approach
                print(f"Warning: Could not load SentenceTransformer model: {e}")
                self.model = None
                self._model_loaded = True  # Prevent retry

    def _compute_pairwise_cosine_distance(self, embeddings: list[list[float]]) -> float:
        """Compute pairwise cosine distances between embeddings."""
        if len(embeddings) < 2:
            return 0.0

        distances = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                # Compute cosine distance: 1 - cosine_similarity
                cos_sim = np.dot(embeddings[i], embeddings[j]) / (
                    np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
                )
                # Normalize cosine similarity from [-1,1] to [0,1], then compute distance
                cos_sim_norm = (cos_sim + 1.0) / 2.0
                distance = 1.0 - cos_sim_norm
                distances.append(distance)

        return float(np.mean(distances)) if distances else 0.0

    def _compute_simple_diversity(self, chunks: list[str]) -> float:
        """Fallback diversity computation without embeddings."""
        if len(chunks) <= 1:
            return 0.0

        # Simple diversity based on unique content and text differences
        unique_chunks = set(chunks)
        base_diversity = len(unique_chunks) / len(chunks)

        # Additional diversity based on text differences
        total_diff = 0.0
        comparisons = 0

        for i in range(len(chunks)):
            for j in range(i + 1, len(chunks)):
                # Simple text difference metric
                words_i = set(chunks[i].lower().split())
                words_j = set(chunks[j].lower().split())

                if words_i or words_j:
                    intersection = len(words_i.intersection(words_j))
                    union = len(words_i.union(words_j))
                    diff_ratio = 1.0 - (intersection / union if union > 0 else 0.0)
                    total_diff += diff_ratio
                    comparisons += 1

        avg_diff = total_diff / comparisons if comparisons > 0 else 0.0
        diversity_score = (base_diversity + avg_diff) / 2.0

        return diversity_score

    def score(
        self,
        prediction: str,
        ground_truth: str,
        context: dict[str, Any] | None = None,
    ) -> float | dict[str, float]:
        if not context or "chunks" not in context:
            self._last_result = ScoreResult(
                score=0.0, passed=False, reasoning="No chunks provided", metadata={}
            )
            return 0.0

        chunks = context["chunks"]

        if len(chunks) <= 1:
            self._last_result = ScoreResult(
                score=0.0,
                passed=False,
                reasoning="Insufficient chunks for diversity calculation",
                metadata={},
            )
            return 0.0

        try:
            if not self._model_loaded:
                self._load_model()

            if self.model is None:
                # Use fallback diversity computation
                diversity_score = self._compute_simple_diversity(chunks)

                self._last_result = ScoreResult(
                    score=diversity_score,
                    passed=diversity_score > self.threshold,  # Configurable threshold
                    reasoning=f"Fallback diversity score: {diversity_score:.3f} (using text-based diversity)",
                    metadata={"diversity": diversity_score, "method": "fallback"},
                )
                return {"diversity": diversity_score}

            # Compute embeddings for all chunks
            chunk_embeddings = self.model.encode(chunks)

            # Compute pairwise cosine distances
            diversity_score = self._compute_pairwise_cosine_distance(chunk_embeddings)

            self._last_result = ScoreResult(
                score=diversity_score,
                passed=diversity_score > self.threshold,  # Configurable threshold
                reasoning=f"Diversity Score: {diversity_score:.3f} (using cosine distance between embeddings)",
                metadata={
                    "diversity": diversity_score,
                    "num_chunks": len(chunks),
                    "method": "cosine_distance",
                },
            )
            return {"diversity": diversity_score}
        except Exception as e:
            self._last_result = ScoreResult(
                score=0.0,
                passed=False,
                reasoning=f"Diversity computation failed: {e!s}",
                metadata={},
            )
            return 0.0

    def get_score_result(self) -> ScoreResult | None:
        """Get the full ScoreResult from the last evaluation."""
        return self._last_result


class AggregateRAGScorer(BaseScorer):
    """
    Combines multiple retrieval scorers with weighted averaging.
    """

    def __init__(
        self,
        scorers: dict[str, Any],
        weights: dict[str, float] | None = None,
        threshold: float = 0.5,
        **kwargs: Any,
    ) -> None:
        super().__init__(name="AggregateRetrievalScorer", **kwargs)
        self.scorers = scorers
        self.weights = weights or dict.fromkeys(scorers.keys(), 1.0)
        self.threshold = threshold
        self._last_result: ScoreResult | None = None

    def score(
        self,
        prediction: str,
        ground_truth: str,
        context: dict[str, Any] | None = None,
    ) -> float | dict[str, Any]:
        # Calls each scorer, extracts main score, and computes weighted average
        scores: dict[str, float] = {}
        total_weight = 0.0
        weighted_sum = 0.0

        for name, scorer in self.scorers.items():
            try:
                result = scorer.score(prediction, ground_truth, context)
                # Handle both ScoreResult objects and direct score values
                if hasattr(result, "score"):
                    score = result.score
                elif isinstance(result, dict):
                    # Extract the main score from dictionary results with robust key prioritization
                    score = self._extract_numeric_score_from_dict(result)
                else:
                    score = float(result) if result is not None else 0.0

                weight = self.weights.get(name, 1.0)

                scores[name] = score
                weighted_sum += score * weight
                total_weight += weight
            except Exception as e:
                scores[name] = 0.0
                print(f"Warning: Scorer {name} failed: {e}")

        if total_weight == 0:
            self._last_result = ScoreResult(
                score=0.0, passed=False, reasoning="All scorers failed", metadata=scores
            )
            return 0.0

        final_score = weighted_sum / total_weight
        passed = final_score >= self.threshold

        self._last_result = ScoreResult(
            score=final_score,
            passed=passed,
            reasoning=f"Aggregate Score: {final_score:.3f}",
            metadata={"individual_scores": scores, "weights": self.weights},
        )
        return {"aggregate": final_score, "individual_scores": scores}

    def _extract_numeric_score_from_dict(self, result: dict[str, Any]) -> float:
        """
        Extract a numeric score from a dictionary with robust key prioritization.

        Args:
            result: Dictionary that may contain score information

        Returns:
            First numeric value found, or 0.0 if none found
        """
        # Prioritized list of keys to look for
        priority_keys = [
            "score",
            "aggregate",
            "average",
            "similarity",
            "f1",
            "combined",
            "diversity",
            "precision",
            "recall",
            "accuracy",
        ]

        # First, try the prioritized keys
        for key in priority_keys:
            if key in result:
                value = result[key]
                if self._is_numeric_value(value):
                    return float(value)
                elif isinstance(value, dict):
                    # Recursively try to extract from nested dict
                    nested_score = self._extract_numeric_score_from_dict(value)
                    if nested_score != 0.0:
                        return nested_score

        # If no prioritized keys found, try all keys in the dict
        for _key, value in result.items():
            if self._is_numeric_value(value):
                return float(value)
            elif isinstance(value, dict):
                # Recursively try to extract from nested dict
                nested_score = self._extract_numeric_score_from_dict(value)
                if nested_score != 0.0:
                    return nested_score

        # Final fallback
        return 0.0

    def _is_numeric_value(self, value: Any) -> bool:
        """
        Check if a value is numeric or can be cast to float.

        Args:
            value: Value to check

        Returns:
            True if value is numeric or castable to float, False otherwise
        """
        if isinstance(value, (int, float)):
            return True
        if isinstance(value, str):
            try:
                float(value)
                return True
            except (ValueError, TypeError):
                return False
        return False

    def get_score_result(self) -> ScoreResult | None:
        """Get the full ScoreResult from the last evaluation."""
        return self._last_result
