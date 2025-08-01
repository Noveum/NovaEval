"""
Basic RAG Scorers for NovaEval.

This module contains fundamental scorers for RAG evaluation including:
- Basic RAG scorers (from rag.py)
- Advanced retrieval scorers (precision, recall, ranking, diversity)
- Semantic similarity scorers
- Aggregate scoring
"""
import re

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics import average_precision_score, ndcg_score

from typing import Any, Optional, Union

from novaeval.scorers.base import BaseScorer, ScoreResult
from novaeval.utils.llm import call_llm
from novaeval.utils.parsing import parse_simple_claims


class ContextualPrecisionScorerPP(BaseScorer):
    """
    Enhanced contextual precision scorer with ranking awareness.
    Evaluates how relevant the retrieved context is to the query.
    """

    def __init__(self, model, threshold=0.7, **kwargs):
        super().__init__(name="ContextualPrecisionScorerPP", **kwargs)
        self.threshold = threshold
        self.model = model

    async def evaluate(self, input_text: str, output_text: str, expected_output: Optional[str] = None, context: Optional[str] = None, **kwargs: Any) -> ScoreResult:
        if not context or not input_text:
            return ScoreResult(score=0.0, passed=False, reasoning="No context or input provided", metadata={})

        prompt = f"""
        Evaluate the relevance of the retrieved context to the query.

        Query: {input_text}
        Retrieved Context: {context}

        Rate the relevance from 0-10 where:
        0: Completely irrelevant
        5: Somewhat relevant
        10: Highly relevant and directly addresses the query

        Provide only the numerical score (0-10):
        """

        try:
            response = await self._call_model(prompt)
            score = self._parse_relevance_score(response)
            passed = score >= self.threshold * 10

            return ScoreResult(
                score=score / 10.0,  # Normalize to 0-1
                passed=passed,
                reasoning=f"Context relevance score: {score}/10",
                metadata={"raw_score": score, "threshold": self.threshold * 10}
            )
        except Exception as e:
            return ScoreResult(score=0.0, passed=False, reasoning=f"Evaluation failed: {e!s}", metadata={})

    async def _call_model(self, prompt: str):
        # Async wrapper for call_llm
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, call_llm, self.model, prompt)

    def _parse_relevance_score(self, resp):
        # Extract numerical score from response
        numbers = re.findall(r"\b(?:10|[0-9])\b", resp)
        if numbers:
            return float(numbers[-1])
        return 5.0  # Default to neutral

    def score(self, prediction: str, ground_truth: str, context: Optional[dict[str, Any]] = None) -> Union[float, dict[str, float]]:
        import asyncio
        
        # Extract context from dict if available
        context_text = context.get("context") if context else None
        
        # Check if we're already in an async context
        try:
            loop = asyncio.get_running_loop()
            # We're in an async context, run in a separate thread
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, self.evaluate(
                    input_text=ground_truth,
                    output_text=prediction,
                    context=context_text
                ))
                result = future.result()
        except RuntimeError:
            # No running loop, use asyncio.run directly
            result = asyncio.run(
                self.evaluate(
                    input_text=ground_truth,
                    output_text=prediction,
                    context=context_text
                )
            )
        
        return result.score if hasattr(result, "score") else result


class ContextualRecallScorerPP(BaseScorer):
    """
    Enhanced contextual recall scorer with comprehensive coverage analysis.
    Evaluates how much of the relevant information is captured in the retrieved context.
    """

    def __init__(self, model, threshold=0.7, top_k=5, **kwargs):
        super().__init__(name="ContextualRecallScorerPP", **kwargs)
        self.threshold = threshold
        self.model = model
        self.top_k = top_k

    async def evaluate(self, input_text, output_text, expected_output=None, context=None, **kwargs: Any) -> ScoreResult:
        if not context or not input_text:
            return ScoreResult(score=0.0, passed=False, reasoning="No context or input provided", metadata={})

        # Extract key information points from the query
        claims = self._parse_claims(input_text)

        if not claims:
            return ScoreResult(score=0.0, passed=False, reasoning="No claims found in input", metadata={})

        prompt = f"""
        Evaluate how well the retrieved context covers the key information points from the query.

        Query: {input_text}
        Key Information Points: {', '.join(claims)}
        Retrieved Context: {context}

        For each information point, rate coverage from 0-10:
        0: Not covered at all
        5: Partially covered
        10: Fully covered

        Provide the average coverage score (0-10):
        """

        try:
            response = await self._call_model(prompt)
            score = self._parse_relevance_score(response)
            passed = score >= self.threshold * 10

            return ScoreResult(
                score=score / 10.0,  # Normalize to 0-1
                passed=passed,
                reasoning=f"Context recall score: {score}/10",
                metadata={"raw_score": score, "claims": claims, "threshold": self.threshold * 10}
            )
        except Exception as e:
            return ScoreResult(score=0.0, passed=False, reasoning=f"Evaluation failed: {e!s}", metadata={})

    async def _call_model(self, prompt: str):
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, call_llm, self.model, prompt)

    def _parse_claims(self, text):
        # Simple claim extraction - can be enhanced
        return parse_simple_claims(text, min_length=10, max_claims=self.top_k)

    def score(self, prediction: str, ground_truth: str, context: Optional[dict[str, Any]] = None) -> Union[float, dict[str, float]]:
        import asyncio
        
        # Extract context from dict if available
        context_text = context.get("context") if context else None
        
        # Check if we're already in an async context
        try:
            loop = asyncio.get_running_loop()
            # We're in an async context, run in a separate thread
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, self.evaluate(
                    input_text=ground_truth,
                    output_text=prediction,
                    context=context_text
                ))
                result = future.result()
        except RuntimeError:
            # No running loop, use asyncio.run directly
            result = asyncio.run(
                self.evaluate(
                    input_text=ground_truth,
                    output_text=prediction,
                    context=context_text
                )
            )
        
        return result.score if hasattr(result, "score") else result


class ContextualF1Scorer(BaseScorer):
    """
    F1 score combining precision and recall for contextual evaluation.
    """

    def __init__(self, precision_scorer, recall_scorer, threshold=0.5, **kwargs):
        super().__init__(name="ContextualF1Scorer", **kwargs)
        self.precision_scorer = precision_scorer
        self.recall_scorer = recall_scorer
        self.threshold = threshold

    def score(self, prediction: str, ground_truth: str, context: Optional[dict[str, Any]] = None) -> Union[float, dict[str, float]]:
        # Calls the precision and recall scorers, then computes F1
        precision = self.precision_scorer.score(prediction, ground_truth, context)
        recall = self.recall_scorer.score(prediction, ground_truth, context)

        if precision + recall == 0:
            f1_score = 0.0
        else:
            f1_score = 2 * (precision * recall) / (precision + recall)

        passed = f1_score >= self.threshold

        return ScoreResult(
            score=f1_score,
            passed=passed,
            reasoning=f"F1 Score: {f1_score:.3f} (Precision: {precision:.3f}, Recall: {recall:.3f})",
            metadata={"precision": precision, "recall": recall}
        )


class RetrievalRankingScorer(BaseScorer):
    """
    Computes ranking metrics for retrieved context.
    """

    def __init__(self, threshold=0.5, **kwargs):
        super().__init__(name="RetrievalRankingScorer", **kwargs)
        self.threshold = threshold

    def score(self, prediction: str, ground_truth: str, context: Optional[dict[str, Any]] = None) -> Union[float, dict[str, float]]:
        # Computes NDCG, MAP, and MRR for the retrieved context
        if not context or "rankings" not in context:
            return ScoreResult(score=0.0, passed=False, reasoning="No ranking data provided", metadata={})

        rankings = context["rankings"]
        relevance_scores = context.get("relevance_scores", [1.0] * len(rankings))

        try:
            # Add safety checks for macOS/ARM64 issues
            import os
            import platform

            # Check if we're on macOS ARM64 (M1/M2) which has known issues
            if platform.system() == "Darwin" and platform.machine() == "arm64":
                print("Warning: Detected macOS ARM64, using fallback mode for ranking scorer")
                # Use fallback mode immediately
                mrr = 0.0
                for i, relevance in enumerate(relevance_scores):
                    if relevance > 0:
                        mrr = 1.0 / (i + 1)  # i is the 0-based position
                        break

                passed = mrr >= self.threshold
                return ScoreResult(
                    score=mrr,
                    passed=passed,
                    reasoning=f"Fallback Ranking Score: {mrr:.3f} (MRR only, macOS ARM64 detected)",
                    metadata={"mrr": mrr, "method": "fallback_macos"}
                )

            os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

            # NDCG
            ndcg = ndcg_score([relevance_scores], [rankings], k=len(rankings))

            # MAP
            map_score = average_precision_score(relevance_scores, rankings)

            # MRR
            mrr = 0.0
            for i, relevance in enumerate(relevance_scores):
                if relevance > 0:
                    mrr = 1.0 / (i + 1)  # i is the 0-based position
                    break

            avg_score = (ndcg + map_score + mrr) / 3.0
            passed = avg_score >= self.threshold

            return ScoreResult(
                score=avg_score,
                passed=passed,
                reasoning=f"Ranking Score: {avg_score:.3f} (NDCG: {ndcg:.3f}, MAP: {map_score:.3f}, MRR: {mrr:.3f})",
                metadata={"ndcg": ndcg, "map": map_score, "mrr": mrr}
            )
        except Exception as e:
            # Fallback to simple ranking score if sklearn fails
            try:
                # Simple fallback: just use MRR
                mrr = 0.0
                for i, relevance in enumerate(relevance_scores):
                    if relevance > 0:
                        mrr = 1.0 / (i + 1)  # i is the 0-based position
                        break

                passed = mrr >= self.threshold
                return ScoreResult(
                    score=mrr,
                    passed=passed,
                    reasoning=f"Fallback Ranking Score: {mrr:.3f} (MRR only, sklearn failed)",
                    metadata={"mrr": mrr, "method": "fallback"}
                )
            except Exception:
                return ScoreResult(score=0.0, passed=False, reasoning=f"Ranking computation failed: {e!s}", metadata={})


class SemanticSimilarityScorer(BaseScorer):
    """
    Computes semantic similarity between query and retrieved context.
    """

    def __init__(self, threshold=0.7, embedding_model="all-MiniLM-L6-v2", **kwargs):
        super().__init__(name="SemanticSimilarityScorer", **kwargs)
        self.threshold = threshold
        self.embedding_model = embedding_model
        self.model = None
        self._model_loaded = False

    def _load_model(self):
        if self.model is None and not self._model_loaded:
            try:
                # Add safety checks for macOS/ARM64 issues
                import os
                import platform

                # Check if we're on macOS ARM64 (M1/M2) which has known issues
                if platform.system() == "Darwin" and platform.machine() == "arm64":
                    print("Warning: Detected macOS ARM64, using fallback mode to avoid segmentation faults")
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

    def _compute_simple_similarity(self, query: str, chunks: list) -> float:
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

    def score(self, prediction: str, ground_truth: str, context: Optional[dict[str, Any]] = None) -> Union[float, dict[str, float]]:
        # Embeds the query and all chunks, computes mean similarity and diversity
        if not context or not ground_truth:
            return ScoreResult(score=0.0, passed=False, reasoning="No context or query provided", metadata={})

        try:
            self._load_model()

            query = ground_truth
            chunks = context.get("chunks", [context.get("context", "")])

            if self.model is None:
                # Use fallback similarity computation
                similarity = self._compute_simple_similarity(query, chunks)
                passed = similarity >= self.threshold

                return ScoreResult(
                    score=similarity,
                    passed=passed,
                    reasoning=f"Fallback similarity score: {similarity:.3f} (using text-based similarity)",
                    metadata={"similarity": similarity, "method": "fallback"}
                )

            # Compute embeddings
            query_embedding = self.model.encode([query])[0]
            chunk_embeddings = self.model.encode(chunks)

            # Compute similarities
            similarities = []
            for chunk_emb in chunk_embeddings:
                sim = np.dot(query_embedding, chunk_emb) / (np.linalg.norm(query_embedding) * np.linalg.norm(chunk_emb))
                similarities.append(sim)

            mean_similarity = np.mean(similarities)
            diversity = 1.0 - np.std(similarities)  # Higher std = lower diversity

            avg_score = (mean_similarity + diversity) / 2.0
            passed = avg_score >= self.threshold

            return ScoreResult(
                score=avg_score,
                passed=passed,
                reasoning=f"Semantic Score: {avg_score:.3f} (Mean Similarity: {mean_similarity:.3f}, Diversity: {diversity:.3f})",
                metadata={"mean_similarity": mean_similarity, "diversity": diversity, "similarities": similarities}
            )
        except Exception as e:
            return ScoreResult(score=0.0, passed=False, reasoning=f"Semantic similarity computation failed: {e!s}", metadata={})


class RetrievalDiversityScorer(BaseScorer):
    """
    Evaluates diversity of retrieved context chunks.
    """

    def __init__(self, **kwargs):
        super().__init__(name="RetrievalDiversityScorer", **kwargs)

    def score(self, prediction: str, ground_truth: str, context: Optional[dict[str, Any]] = None) -> Union[float, dict[str, float]]:
        if not context or "chunks" not in context:
            return ScoreResult(score=0.0, passed=False, reasoning="No chunks provided", metadata={})

        chunks = context["chunks"]

        if len(chunks) <= 1:
            return ScoreResult(score=0.0, passed=False, reasoning="Insufficient chunks for diversity calculation", metadata={})

        try:
            # Simple diversity based on unique content
            unique_chunks = set(chunks)
            diversity_score = len(unique_chunks) / len(chunks)

            return ScoreResult(
                score=diversity_score,
                passed=diversity_score > 0.5,
                reasoning=f"Diversity Score: {diversity_score:.3f} ({len(unique_chunks)} unique out of {len(chunks)} chunks)",
                metadata={"unique_chunks": len(unique_chunks), "total_chunks": len(chunks)}
            )
        except Exception as e:
            return ScoreResult(score=0.0, passed=False, reasoning=f"Diversity computation failed: {e!s}", metadata={})


class AggregateRetrievalScorer(BaseScorer):
    """
    Combines multiple retrieval scorers with weighted averaging.
    """

    def __init__(self, scorers: dict, weights: Optional[dict] = None, **kwargs):
        super().__init__(name="AggregateRetrievalScorer", **kwargs)
        self.scorers = scorers
        self.weights = weights or dict.fromkeys(scorers.keys(), 1.0)

    def score(self, prediction: str, ground_truth: str, context: Optional[dict[str, Any]] = None) -> Union[float, dict[str, float]]:
        # Calls each scorer, extracts main score, and computes weighted average
        scores = {}
        total_weight = 0.0
        weighted_sum = 0.0

        for name, scorer in self.scorers.items():
            try:
                result = scorer.score(prediction, ground_truth, context)
                score = result.score if hasattr(result, "score") else result
                weight = self.weights.get(name, 1.0)

                scores[name] = score
                weighted_sum += score * weight
                total_weight += weight
            except Exception as e:
                scores[name] = 0.0
                print(f"Warning: Scorer {name} failed: {e}")

        if total_weight == 0:
            return ScoreResult(score=0.0, passed=False, reasoning="All scorers failed", metadata=scores)

        final_score = weighted_sum / total_weight
        passed = final_score >= 0.5  # Default threshold

        return ScoreResult(
            score=final_score,
            passed=passed,
            reasoning=f"Aggregate Score: {final_score:.3f}",
            metadata={"individual_scores": scores, "weights": self.weights}
        )
