"""
RAG (Retrieval-Augmented Generation) specific scorers for NovaEval.

This module implements various metrics for evaluating RAG systems including:
- Answer Relevancy
- Faithfulness
- Contextual Precision
- Contextual Recall
- Contextual Relevancy
- RAGAS metrics
"""

import asyncio
import logging
from typing import Any, Optional, Union

import numpy as np

from novaeval.models.base import BaseModel as LLMModel
from novaeval.scorers.base import BaseScorer, ScoreResult
from novaeval.scorers.conversational import _run_async_in_sync_context
from novaeval.scorers.rag_prompts import RAGPrompts
from novaeval.utils.parsing import parse_claims

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    # Fallback for when sentence_transformers is not installed
    SentenceTransformer = None  # type: ignore


class RAGScorerMixin:
    """
    Mixin class providing common methods for RAG scorers.

    This mixin provides the evaluate_multiple_queries and _evaluate_single_query_with_contexts
    methods that are expected by the test suite for all RAG scorers.
    """

    async def _call_generate(self, *args: Any, **kwargs: Any) -> str:
        """
        Coroutine-safe adapter for model.generate that handles both sync and async implementations.

        This method detects whether self.model.generate is a coroutine function or returns an
        awaitable, and if it's synchronous, runs it safely in a thread via asyncio.to_thread
        to avoid blocking the event loop.
        """
        import asyncio
        import inspect

        gen = self.model.generate  # type: ignore[attr-defined]
        if inspect.iscoroutinefunction(gen):
            return await gen(*args, **kwargs)  # type: ignore[misc]
        # Offload sync call
        result = await asyncio.to_thread(gen, *args, **kwargs)
        # Some sync wrappers may return an awaitable; handle it
        if inspect.isawaitable(result):
            return await result  # type: ignore[misc]
        return result  # type: ignore[return-value]

    async def evaluate_multiple_queries(
        self,
        queries: list[str],
        contexts: list[list[str]],
        answer: str,
    ) -> list[float]:
        """Evaluate multiple queries and return scores."""
        if len(queries) != len(contexts):
            raise ValueError("Length mismatch between queries and contexts")

        scores = []
        for query, context_list in zip(queries, contexts):
            context_text = " ".join(context_list) if context_list else ""
            # Type ignore: evaluate method is provided by the concrete scorer class
            result = await self.evaluate(  # type: ignore[attr-defined]
                input_text=query,
                output_text=answer,
                context=context_text,
            )
            scores.append(result.score)
        return scores

    async def _evaluate_single_query_with_contexts(
        self,
        query: str,
        contexts: list[str],
        answer: str,
        expected_output: Optional[str] = None,
    ) -> float:
        """Evaluate a single query with multiple contexts."""
        if not contexts:
            return 0.0

        context_text = " ".join(contexts) if contexts else ""
        # Type ignore: evaluate method is provided by the concrete scorer class
        result = await self.evaluate(  # type: ignore[attr-defined]
            input_text=query,
            output_text=answer,
            expected_output=expected_output,
            context=context_text,
        )
        return result.score


class AnswerRelevancyScorer(RAGScorerMixin, BaseScorer):
    """
    Evaluates how relevant the answer is to the given question.

    This metric measures whether the response directly addresses the question
    and provides relevant information.
    """

    def __init__(
        self,
        model: LLMModel,
        threshold: float = 0.7,
        embedding_model: str = "all-MiniLM-L6-v2",
        **kwargs: Any,
    ) -> None:
        super().__init__(name=kwargs.pop("name", "AnswerRelevancyScorer"), **kwargs)
        self.threshold = threshold
        self.model = model
        self.embedding_model_name = embedding_model
        self.embedding_model: Optional[SentenceTransformer] = None
        self._model_loaded = False

        # Load embedding model during initialization
        self._load_embedding_model()

    def _load_embedding_model(self) -> None:
        """Load the sentence transformer model with error handling."""
        if self._model_loaded:
            return  # Already loaded, return early to make function idempotent

        try:
            if SentenceTransformer is None:
                raise ImportError("sentence_transformers not available")

            self.embedding_model = SentenceTransformer(self.embedding_model_name)
            self._model_loaded = True
        except ImportError:
            self.embedding_model = None
            logging.warning(
                "sentence_transformers not installed. "
                "Answer relevancy scoring will use fallback method."
            )
            self._model_loaded = True  # Set to True to prevent re-attempting
        except Exception as e:
            self.embedding_model = None
            logging.exception(f"Could not load SentenceTransformer model: {e}")
            self._model_loaded = True  # Set to True to prevent re-attempting

    def score(
        self,
        prediction: str,
        ground_truth: str,
        context: Optional[dict[str, Any]] = None,
    ) -> Union[float, dict[str, float]]:
        """Synchronous wrapper for the async evaluate method."""

        # Extract context from dict if available
        context_text = context.get("context") if context else None
        question = context.get("question") if context else ground_truth

        # Run async evaluation (safe in both sync/async callers)
        result = _run_async_in_sync_context(
            self.evaluate(
                input_text=(
                    question if question is not None else ""
                ),  # Use actual question as input
                output_text=prediction,
                context=context_text,
            )
        )

        return result.score

    async def evaluate(
        self,
        input_text: str,
        output_text: str,
        expected_output: Optional[str] = None,
        context: Optional[str] = None,
        **kwargs: Any,
    ) -> ScoreResult:
        """Evaluate answer relevancy."""

        # Generate questions that the answer could be answering
        question_generation_prompt = f"""
        Given the following answer, generate 3-5 questions that this answer could be responding to.
        The questions should be specific and directly related to the content of the answer.

        Answer: {output_text}

        Generate questions in the following format:
        1. [Question 1]
        2. [Question 2]
        3. [Question 3]
        ...
        """

        try:
            generated_questions_response = await self._call_generate(
                question_generation_prompt
            )
            generated_questions = self._parse_questions(generated_questions_response)

            if not generated_questions:
                return ScoreResult(
                    score=0.0,
                    passed=False,
                    reasoning="Failed to generate questions from the answer",
                    metadata={"error": "question_generation_failed"},
                )

            # Calculate semantic similarity between original question and generated
            # questions
            self._load_embedding_model()

            # Track whether we used a fallback method
            used_fallback = False
            fallback_reason = None

            if self.embedding_model is None:
                # Fallback to simple text similarity if embedding model is not available
                used_fallback = True
                fallback_reason = "embedding_model_not_available"
                similarities = []
                input_words = set(input_text.lower().split())
                for gen_question in generated_questions:
                    gen_words = set(gen_question.lower().split())
                    if input_words and gen_words:
                        overlap = len(input_words.intersection(gen_words))
                        union = len(input_words.union(gen_words))
                        similarity = overlap / union if union > 0 else 0.0
                    else:
                        similarity = 0.0
                    similarities.append(similarity)
            else:
                # Use embedding model for semantic similarity
                try:
                    original_embedding = await asyncio.to_thread(
                        self.embedding_model.encode, [input_text]
                    )
                    generated_embeddings = await asyncio.to_thread(
                        self.embedding_model.encode, generated_questions
                    )

                    # Calculate cosine similarities
                    similarities = []
                    for gen_embedding in generated_embeddings:
                        similarity = np.dot(original_embedding[0], gen_embedding) / (
                            np.linalg.norm(original_embedding[0])
                            * np.linalg.norm(gen_embedding)
                        )
                        # Normalize cosine similarity from [-1,1] to [0,1] to match token-overlap scale
                        similarity_norm = (similarity + 1.0) / 2.0
                        similarities.append(similarity_norm)
                except Exception as e:
                    # Log the embedding error and fall back to token-overlap heuristic
                    logging.warning(
                        f"Embedding encoding failed, falling back to token-overlap: {e}"
                    )
                    used_fallback = True
                    fallback_reason = f"embedding_encoding_failed: {e!s}"

                    # Fallback to token-overlap similarity
                    similarities = []
                    input_words = set(input_text.lower().split())
                    for gen_question in generated_questions:
                        gen_words = set(gen_question.lower().split())
                        if input_words and gen_words:
                            overlap = len(input_words.intersection(gen_words))
                            union = len(input_words.union(gen_words))
                            similarity = overlap / union if union > 0 else 0.0
                        else:
                            similarity = 0.0
                        similarities.append(similarity)

            # Use mean similarity as the relevancy score
            relevancy_score = float(np.mean(similarities))

            # Build reasoning with fallback information
            similarity_method = (
                "token-overlap" if used_fallback else "semantic embedding"
            )
            fallback_info = f" (fallback: {fallback_reason})" if used_fallback else ""

            reasoning = f"""
            Answer Relevancy Analysis:
            - Generated {len(generated_questions)} questions from the answer
            - Calculated {similarity_method} similarity with original question{fallback_info}
            - Individual similarities: {[f'{s:.3f}' for s in similarities]}
            - Mean relevancy score: {relevancy_score:.3f}

            Generated questions:
            {chr(10).join(f'{i+1}. {q}' for i, q in enumerate(generated_questions))}
            """

            metadata = {
                "generated_questions": generated_questions,
                "similarities": similarities,
                "mean_similarity": relevancy_score,
                "similarity_method": similarity_method,
            }

            if used_fallback:
                metadata["fallback_reason"] = fallback_reason

            return ScoreResult(
                score=relevancy_score,
                passed=relevancy_score >= self.threshold,
                reasoning=reasoning.strip(),
                metadata=metadata,
            )

        except Exception as e:
            return ScoreResult(
                score=0.0,
                passed=False,
                reasoning=f"Answer relevancy evaluation failed: {e!s}",
                metadata={"error": str(e)},
            )

    def _parse_questions(self, response: str) -> list[str]:
        """Parse generated questions from LLM response."""
        questions = []
        lines = response.strip().split("\n")

        for line in lines:
            line = line.strip()
            if line and (
                line[0].isdigit() or line.startswith("-") or line.startswith("*")
            ):
                # Remove numbering and bullet points
                question = line
                for prefix in ["1.", "2.", "3.", "4.", "5.", "-", "*"]:
                    if question.startswith(prefix):
                        question = question[len(prefix) :].strip()
                        break

                if question and question.endswith("?"):
                    questions.append(question)

        return questions


class FaithfulnessScorer(RAGScorerMixin, BaseScorer):
    """
    Evaluates whether the answer is faithful to the provided context.

    This metric measures if the response contains information that can be
    verified from the given context without hallucinations.
    """

    def __init__(self, model: LLMModel, threshold: float = 0.8, **kwargs: Any) -> None:
        super().__init__(name=kwargs.pop("name", "FaithfulnessScorer"), **kwargs)
        self.threshold = threshold
        self.model = model

    def score(
        self,
        prediction: str,
        ground_truth: str,
        context: Optional[dict[str, Any]] = None,
    ) -> Union[float, dict[str, float]]:
        """Synchronous wrapper for the async evaluate method."""

        # Extract context from dict if available
        context_text = context.get("context") if context else None
        question = context.get("question") if context else ground_truth

        # Run async evaluation (safe in both sync/async callers)
        result = _run_async_in_sync_context(
            self.evaluate(
                input_text=(
                    question if question is not None else ""
                ),  # Use actual question as input
                output_text=prediction,
                context=context_text,
            )
        )

        return result.score

    async def evaluate(
        self,
        input_text: str,
        output_text: str,
        expected_output: Optional[str] = None,
        context: Optional[str] = None,
        **kwargs: Any,
    ) -> ScoreResult:
        """Evaluate faithfulness to context."""

        if not context:
            return ScoreResult(
                score=0.0,
                passed=False,
                reasoning="No context provided for faithfulness evaluation",
                metadata={"error": "no_context"},
            )

        # Extract claims from the answer
        claims_extraction_prompt = f"""
        Extract all factual claims from the following answer.
        List each claim as a separate statement that can be verified.

        Answer: {output_text}

        Format your response as:
        1. [Claim 1]
        2. [Claim 2]
        3. [Claim 3]
        ...
        """

        try:
            claims_response = await self._call_generate(claims_extraction_prompt)
            claims = self._parse_claims(claims_response)

            if not claims:
                return ScoreResult(
                    score=1.0,  # No claims means no unfaithful content
                    passed=True,
                    reasoning="No factual claims found in the answer",
                    metadata={"claims": []},
                )

            # Verify each claim against the context
            verification_results = []
            for claim in claims:
                verification_prompt = f"""
                Context: {context}

                Claim to verify: {claim}

                Can this claim be verified from the given context?
                Respond with either "SUPPORTED", "NOT_SUPPORTED", or "PARTIALLY_SUPPORTED".
                Then provide a brief explanation.

                Format:
                Verification: [SUPPORTED/NOT_SUPPORTED/PARTIALLY_SUPPORTED]
                Explanation: [Brief explanation]
                """

                verification_response = await self._call_generate(verification_prompt)
                verification_results.append((claim, verification_response))

            # Calculate faithfulness score
            supported_count = 0
            partial_count = 0

            for _claim, verification in verification_results:
                if "SUPPORTED" in verification and "NOT_SUPPORTED" not in verification:
                    supported_count += 1
                elif "PARTIALLY_SUPPORTED" in verification:
                    partial_count += 1

            # Score calculation: full points for supported, half points for partial
            total_claims = len(claims)
            partial_weight = 0.5  # Configurable weight for partial support
            faithfulness_score = (
                supported_count + partial_weight * partial_count
            ) / total_claims

            reasoning = f"""
            Faithfulness Analysis:
            - Extracted {total_claims} claims from the answer
            - Fully supported claims: {supported_count}
            - Partially supported claims: {partial_count}
            - Unsupported claims: {total_claims - supported_count - partial_count}
            - Faithfulness score: {faithfulness_score:.3f}

            Claim verification details:
            {chr(10).join(f'• {claim}: {verification.split("Explanation:")[0].replace("Verification:", "").strip()}' for claim, verification in verification_results)}
            """

            return ScoreResult(
                score=faithfulness_score,
                passed=faithfulness_score >= self.threshold,
                reasoning=reasoning.strip(),
                metadata={
                    "claims": claims,
                    "verification_results": verification_results,
                    "supported_count": supported_count,
                    "partial_count": partial_count,
                    "total_claims": total_claims,
                },
            )

        except Exception as e:
            return ScoreResult(
                score=0.0,
                passed=False,
                reasoning=f"Faithfulness evaluation failed: {e!s}",
                metadata={"error": str(e)},
            )

    def _parse_claims(self, response: str) -> list[str]:
        """Parse claims from LLM response."""
        return parse_claims(response)


class ContextualPrecisionScorer(RAGScorerMixin, BaseScorer):
    """
    Evaluates the precision of the retrieved context.

    This metric measures whether the retrieved context contains relevant
    information for answering the question.
    """

    def __init__(self, model: LLMModel, threshold: float = 0.7, **kwargs: Any) -> None:
        super().__init__(name=kwargs.pop("name", "ContextualPrecisionScorer"), **kwargs)
        self.threshold = threshold
        self.model = model

    def score(
        self,
        prediction: str,
        ground_truth: str,
        context: Optional[dict[str, Any]] = None,
    ) -> Union[float, dict[str, float]]:
        """Synchronous wrapper for the async evaluate method."""

        # Extract context from dict if available
        context_text = context.get("context") if context else None
        question = context.get("question") if context else ground_truth

        # Run async evaluation (safe in both sync/async callers)
        result = _run_async_in_sync_context(
            self.evaluate(
                input_text=(
                    question if question is not None else ""
                ),  # Use actual question as input
                output_text=prediction,
                context=context_text,
            )
        )

        return result.score

    async def evaluate(
        self,
        input_text: str,
        output_text: str,
        expected_output: Optional[str] = None,
        context: Optional[str] = None,
        **kwargs: Any,
    ) -> ScoreResult:
        """Evaluate contextual precision."""

        if not context:
            return ScoreResult(
                score=0.0,
                passed=False,
                reasoning="No context provided for contextual precision evaluation",
                metadata={"error": "no_context"},
            )

        # Split context into chunks (assuming it's a concatenated retrieval result)
        context_chunks = self._split_context(context)

        if not context_chunks:
            return ScoreResult(
                score=0.0,
                passed=False,
                reasoning="No context chunks found",
                metadata={"error": "no_chunks"},
            )

        try:
            # Evaluate relevance of each context chunk
            relevance_scores = []

            for _i, chunk in enumerate(context_chunks):
                relevance_prompt = RAGPrompts.get_numerical_chunk_relevance_1_5(
                    input_text, chunk
                )

                relevance_response = await self._call_generate(relevance_prompt)
                score = self._parse_relevance_score(relevance_response)
                relevance_scores.append(score)

            # Calculate precision as the average relevance score normalized to 0-1
            precision_score = sum(relevance_scores) / len(relevance_scores) / 5.0

            reasoning = f"""
            Contextual Precision Analysis:
            - Evaluated {len(context_chunks)} context chunks
            - Individual relevance scores (1-5): {relevance_scores}
            - Average relevance: {sum(relevance_scores) / len(relevance_scores):.2f}
            - Precision score (normalized): {precision_score:.3f}
            """

            return ScoreResult(
                score=precision_score,
                passed=precision_score >= self.threshold,
                reasoning=reasoning.strip(),
                metadata={
                    "context_chunks": len(context_chunks),
                    "relevance_scores": relevance_scores,
                    "average_relevance": sum(relevance_scores) / len(relevance_scores),
                },
            )

        except Exception as e:
            return ScoreResult(
                score=0.0,
                passed=False,
                reasoning=f"Contextual precision evaluation failed: {e!s}",
                metadata={"error": str(e)},
            )

    def _split_context(self, context: str) -> list[str]:
        """Split context into chunks for evaluation."""
        # Simple splitting by double newlines or sentences
        chunks = []

        # Try splitting by double newlines first
        parts = context.split("\n\n")
        if len(parts) > 1:
            chunks = [part.strip() for part in parts if part.strip()]
        else:
            # Split by sentences if no paragraph breaks
            import re

            sentences = re.split(r"[.!?]+", context)
            chunks = [sent.strip() for sent in sentences if sent.strip()]

        # Ensure chunks are not too small
        min_length = 50
        filtered_chunks = [chunk for chunk in chunks if len(chunk) >= min_length]

        return filtered_chunks if filtered_chunks else [context]

    def _parse_relevance_score(self, response: str) -> float:
        """Parse relevance score from LLM response."""
        import re

        # Look for "Rating: X" pattern
        rating_match = re.search(r"Rating:\s*(\d+)", response)
        if rating_match:
            return float(rating_match.group(1))

        # Look for standalone numbers 1-5
        numbers = re.findall(r"\b([1-5])\b", response)
        if numbers:
            return float(numbers[0])

        # Default to middle score if parsing fails
        return 3.0


class ContextualRecallScorer(RAGScorerMixin, BaseScorer):
    """
    Evaluates the recall of the retrieved context.

    This metric measures whether all necessary information for answering
    the question is present in the retrieved context.
    """

    def __init__(self, model: LLMModel, threshold: float = 0.7, **kwargs: Any) -> None:
        super().__init__(name=kwargs.pop("name", "ContextualRecallScorer"), **kwargs)
        self.threshold = threshold
        self.model = model

    def score(
        self,
        prediction: str,
        ground_truth: str,
        context: Optional[dict[str, Any]] = None,
    ) -> Union[float, dict[str, float]]:
        """Synchronous wrapper for the async evaluate method."""

        # Extract context and expected_output from dict if available
        context_text = context.get("context") if context else None
        expected_output = context.get("expected_output") if context else None
        question = context.get("question") if context else ground_truth

        # Run async evaluation (safe in both sync/async callers)
        result = _run_async_in_sync_context(
            self.evaluate(
                input_text=(
                    question if question is not None else ""
                ),  # Use actual question as input
                output_text=prediction,
                expected_output=expected_output,
                context=context_text,
            )
        )

        return result.score

    async def evaluate(
        self,
        input_text: str,
        output_text: str,
        expected_output: Optional[str] = None,
        context: Optional[str] = None,
        **kwargs: Any,
    ) -> ScoreResult:
        """Evaluate contextual recall."""

        if not context or not expected_output:
            return ScoreResult(
                score=0.0,
                passed=False,
                reasoning="Both context and expected output are required for contextual recall evaluation",
                metadata={"error": "missing_inputs"},
            )

        try:
            # Extract key information from expected output
            key_info_prompt = f"""
            Extract the key pieces of information from the following expected answer.
            List each key fact or piece of information separately.

            Expected Answer: {expected_output}

            Format your response as:
            1. [Key information 1]
            2. [Key information 2]
            3. [Key information 3]
            ...
            """

            key_info_response = await self._call_generate(key_info_prompt)
            key_information = self._parse_claims(key_info_response)

            if not key_information:
                return ScoreResult(
                    score=1.0,  # No key info means perfect recall
                    passed=True,
                    reasoning="No key information extracted from expected output",
                    metadata={"key_information": []},
                )

            # Check if each key information is present in the context
            recall_results = []
            for info in key_information:
                recall_prompt = f"""
                Context: {context}

                Key information to find: {info}

                Is this information present in the context?
                Respond with either "PRESENT", "NOT_PRESENT", or "PARTIALLY_PRESENT".
                Then provide a brief explanation.

                Format:
                Status: [PRESENT/NOT_PRESENT/PARTIALLY_PRESENT]
                Explanation: [Brief explanation]
                """

                recall_response = await self._call_generate(recall_prompt)
                recall_results.append((info, recall_response))

            # Calculate recall score
            present_count = 0
            partial_count = 0

            for _info, result in recall_results:
                if "PRESENT" in result and "NOT_PRESENT" not in result:
                    present_count += 1
                elif "PARTIALLY_PRESENT" in result:
                    partial_count += 1

            total_info = len(key_information)
            partial_weight = 0.5  # Configurable weight for partial presence
            recall_score = (present_count + partial_weight * partial_count) / total_info

            reasoning = f"""
            Contextual Recall Analysis:
            - Extracted {total_info} key pieces of information from expected output
            - Fully present in context: {present_count}
            - Partially present in context: {partial_count}
            - Missing from context: {total_info - present_count - partial_count}
            - Recall score: {recall_score:.3f}

            Information presence details:
            {chr(10).join(f'• {info}: {result.split("Explanation:")[0].replace("Status:", "").strip()}' for info, result in recall_results)}
            """

            return ScoreResult(
                score=recall_score,
                passed=recall_score >= self.threshold,
                reasoning=reasoning.strip(),
                metadata={
                    "key_information": key_information,
                    "recall_results": recall_results,
                    "present_count": present_count,
                    "partial_count": partial_count,
                    "total_info": total_info,
                },
            )

        except Exception as e:
            return ScoreResult(
                score=0.0,
                passed=False,
                reasoning=f"Contextual recall evaluation failed: {e!s}",
                metadata={"error": str(e)},
            )

    def _parse_claims(self, response: str) -> list[str]:
        """Parse claims/information from LLM response."""
        return parse_claims(response)


class RAGASScorer(RAGScorerMixin, BaseScorer):
    """
    Composite RAGAS (Retrieval-Augmented Generation Assessment) scorer.

    Combines multiple RAG metrics into a single comprehensive score.
    """

    def __init__(
        self,
        model: LLMModel,
        threshold: float = 0.7,
        weights: Optional[dict[str, float]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(name=kwargs.pop("name", "ragas_scorer"), **kwargs)
        self.threshold = threshold
        self.model = model

        # Default weights for different metrics
        self.weights = weights or {
            "answer_relevancy": 0.25,
            "faithfulness": 0.35,
            "contextual_precision": 0.2,
            "contextual_recall": 0.2,
        }

        # Initialize individual scorers
        self.answer_relevancy_scorer = AnswerRelevancyScorer(model, threshold=0.7)
        self.faithfulness_scorer = FaithfulnessScorer(model, threshold=0.8)
        self.contextual_precision_scorer = ContextualPrecisionScorer(
            model, threshold=0.7
        )
        self.contextual_recall_scorer = ContextualRecallScorer(model, threshold=0.7)

    def score(
        self,
        prediction: str,
        ground_truth: str,
        context: Optional[dict[str, Any]] = None,
    ) -> Union[float, dict[str, float]]:
        """Synchronous wrapper for the async evaluate method."""

        # Extract context from dict if available
        context_text = context.get("context") if context else None
        expected_output = context.get("expected_output") if context else None
        question = context.get("question") if context else ground_truth

        # Run async evaluation (safe in both sync/async callers)
        result = _run_async_in_sync_context(
            self.evaluate(
                input_text=(
                    question if question is not None else ""
                ),  # Use actual question as input
                output_text=prediction,
                context=context_text,
                expected_output=expected_output,
            )
        )

        return result.score

    async def evaluate(
        self,
        input_text: str,
        output_text: str,
        expected_output: Optional[str] = None,
        context: Optional[str] = None,
        **kwargs: Any,
    ) -> ScoreResult:
        """Evaluate using RAGAS methodology."""

        try:
            # Run all individual evaluations in parallel
            results = await asyncio.gather(
                self.answer_relevancy_scorer.evaluate(
                    input_text, output_text, expected_output, context
                ),
                self.faithfulness_scorer.evaluate(
                    input_text, output_text, expected_output, context
                ),
                self.contextual_precision_scorer.evaluate(
                    input_text, output_text, expected_output, context
                ),
                self.contextual_recall_scorer.evaluate(
                    input_text, output_text, expected_output, context
                ),
                return_exceptions=True,
            )

            # Extract scores and handle exceptions
            scores = {}
            reasonings = {}

            metric_names = [
                "answer_relevancy",
                "faithfulness",
                "contextual_precision",
                "contextual_recall",
            ]

            for _i, (metric_name, result) in enumerate(zip(metric_names, results)):
                if isinstance(result, Exception):
                    scores[metric_name] = 0.0
                    reasonings[metric_name] = f"Error: {result!s}"
                elif hasattr(result, "score") and hasattr(result, "reasoning"):
                    scores[metric_name] = result.score  # type: ignore
                    reasonings[metric_name] = result.reasoning  # type: ignore
                else:
                    scores[metric_name] = 0.0
                    reasonings[metric_name] = "Unknown result type"

            # Calculate weighted average
            total_weight = sum(self.weights.values())
            ragas_score = (
                sum(scores[metric] * self.weights[metric] for metric in scores)
                / total_weight
            )

            # Compile comprehensive reasoning
            reasoning = f"""
            RAGAS Evaluation Results:

            Individual Metric Scores:
            • Answer Relevancy: {scores['answer_relevancy']:.3f} (weight: {self.weights['answer_relevancy']})
            • Faithfulness: {scores['faithfulness']:.3f} (weight: {self.weights['faithfulness']})
            • Contextual Precision: {scores['contextual_precision']:.3f} (weight: {self.weights['contextual_precision']})
            • Contextual Recall: {scores['contextual_recall']:.3f} (weight: {self.weights['contextual_recall']})

            Weighted RAGAS Score: {ragas_score:.3f}

            Detailed Analysis:
            {chr(10).join(f'{metric.replace("_", " ").title()}:{chr(10)}{reasoning}{chr(10)}' for metric, reasoning in reasonings.items())}
            """

            return ScoreResult(
                score=ragas_score,
                passed=ragas_score >= self.threshold,
                reasoning=reasoning.strip(),
                metadata={
                    "individual_scores": scores,
                    "weights": self.weights,
                    "ragas_score": ragas_score,
                },
            )

        except Exception as e:
            return ScoreResult(
                score=0.0,
                passed=False,
                reasoning=f"RAGAS evaluation failed: {e!s}",
                metadata={"error": str(e)},
            )


# Make SentenceTransformer available at module level for tests/mocking/etc.
SentenceTransformer = SentenceTransformer
