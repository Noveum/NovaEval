"""
RAG Pipeline Evaluator - Comprehensive evaluation of RAG pipeline stages and workflow.

This module provides a complete evaluation framework for RAG pipelines, including:
- Stage-specific evaluators (Query, Retrieval, Reranking, Generation, Post-processing)
- Workflow orchestration scorers
- Pipeline coordination analysis
- Performance and resource monitoring
- Integration with all existing RAG scorers
"""

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Union

import numpy as np
from pydantic import BaseModel

from novaeval.models.base import BaseModel as ModelBase

# Import all scorers from advanced_generation_scorers
from .advanced_generation_scorers import (
    AnswerCompletenessScorer,
    BiasDetectionScorer,
    CitationQualityScorer,
    ClaimVerificationScorer,
    ClarityAndCoherenceScorer,
    ConflictResolutionScorer,
    ContextCompletenessScorer,
    ContextConsistencyScorer,
    ContextFaithfulnessScorerPP,
    ContextGroundednessScorer,
    ContextPrioritizationScorer,
    CrossContextSynthesisScorer,
    FactualAccuracyScorer,
    HallucinationDetectionScorer,
    InformationDensityScorer,
    QuestionAnswerAlignmentScorer,
    RAGAnswerQualityScorer,
    SourceAttributionScorer,
    TechnicalAccuracyScorer,
    TerminologyConsistencyScorer,
    ToneConsistencyScorer,
)
from .base import BaseScorer, ScoreResult

# Import the required scorers from other modules
# Import all scorers from basic_rag_scorers
from .basic_rag_scorers import (
    ContextualPrecisionScorerPP,
    ContextualRecallScorerPP,
    RetrievalDiversityScorer,
    RetrievalRankingScorer,
    SemanticSimilarityScorer,
)
from .rag import AnswerRelevancyScorer, FaithfulnessScorer, RAGASScorer

# Define AgentData for compatibility


class AgentData(BaseModel):
    """AgentData structure for RAG pipeline evaluation."""

    user_id: Optional[str] = None
    task_id: Optional[str] = None
    ground_truth: Optional[str] = None
    agent_response: Optional[str] = None
    retrieval_query: Optional[str] = None
    retrieved_context: Optional[str] = None
    metadata: Optional[str] = None


@dataclass
class RAGContext:
    """Represents a retrieved context chunk with metadata."""

    content: str
    source: str
    relevance_score: float = 0.0
    rank: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RAGSample:
    """Represents a complete RAG evaluation sample."""

    query: str
    ground_truth: str
    generated_answer: str
    retrieved_contexts: list[RAGContext]
    pipeline_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class StageMetrics:
    """Metrics for a specific pipeline stage."""

    stage_name: str
    latency_ms: float
    success: bool
    error_message: Optional[str] = None
    metrics: dict[str, float] = field(default_factory=dict)
    resource_usage: dict[str, float] = field(default_factory=dict)


@dataclass
class RAGEvaluationResult:
    """Complete evaluation result for a RAG pipeline."""

    overall_score: float
    stage_metrics: dict[str, StageMetrics]
    retrieval_score: float
    generation_score: float
    pipeline_coordination_score: float
    latency_analysis: dict[str, Union[float, dict[str, float]]]
    resource_utilization: dict[str, float]
    error_propagation_score: float
    detailed_scores: dict[str, Union[float, dict[str, Any], ScoreResult]]
    recommendations: list[str]
    # Enhanced scoring results
    basic_rag_scores: dict[str, Union[float, dict[str, Any]]] = field(
        default_factory=dict
    )
    advanced_generation_scores: dict[str, Union[float, dict[str, Any]]] = field(
        default_factory=dict
    )
    comprehensive_scores: dict[str, Union[float, dict[str, Any]]] = field(
        default_factory=dict
    )


class QueryProcessingEvaluator(BaseScorer):
    """Evaluates query understanding and processing quality."""

    def __init__(self, llm: Union[str, Any], name: str = "query_processing") -> None:
        super().__init__(name=name)
        self.llm = llm

    def score(
        self,
        prediction: str,
        ground_truth: str,
        context: Optional[dict[str, Any]] = None,
        weights: Optional[dict[str, float]] = None,
    ) -> Union[float, dict[str, Any]]:
        """
        Evaluate query processing quality with configurable weights.

        Args:
            prediction: The query to evaluate
            ground_truth: Not used for query processing evaluation
            context: Optional context dictionary
            weights: Optional dictionary of weights for different metrics.
                    Default weights are:
                    - clarity: 0.25
                    - intent: 0.25
                    - preprocessing: 0.2
                    - specificity: 0.15
                    - complexity: 0.1
                    - ambiguity: 0.05
        """
        try:
            # Use default weights if not provided
            default_weights = {
                "clarity": 0.25,
                "intent": 0.25,
                "preprocessing": 0.2,
                "specificity": 0.15,
                "complexity": 0.1,
                "ambiguity": 0.05,
            }
            weights = weights or default_weights

            query = prediction

            # Basic query analysis
            clarity_score = self._evaluate_query_clarity(query)
            intent_score = self._evaluate_intent_detection(query)
            preprocessing_score = self._evaluate_preprocessing(query)

            # Enhanced analysis
            specificity_score = self._evaluate_specificity(query)
            complexity_score = self._evaluate_complexity(query)
            ambiguity_score = self._evaluate_ambiguity(query)

            # Weighted overall score using provided weights
            overall_score = (
                clarity_score * weights.get("clarity", 0.25)
                + intent_score * weights.get("intent", 0.25)
                + preprocessing_score * weights.get("preprocessing", 0.2)
                + specificity_score * weights.get("specificity", 0.15)
                + complexity_score * weights.get("complexity", 0.1)
                + ambiguity_score * weights.get("ambiguity", 0.05)
            )

            return {
                "score": overall_score,
                "reasoning": f"Clarity: {clarity_score:.2f}, Intent: {intent_score:.2f}, Preprocessing: {preprocessing_score:.2f}, Specificity: {specificity_score:.2f}, Complexity: {complexity_score:.2f}, Ambiguity: {ambiguity_score:.2f}",
                "details": {
                    "clarity_score": clarity_score,
                    "intent_score": intent_score,
                    "preprocessing_score": preprocessing_score,
                    "specificity_score": specificity_score,
                    "complexity_score": complexity_score,
                    "ambiguity_score": ambiguity_score,
                    "weights_used": weights,
                },
            }
        except Exception as e:
            return {
                "score": 0.0,
                "reasoning": f"Error in enhanced query processing evaluation: {e!s}",
                "details": {"error": str(e)},
            }

    def _evaluate_query_clarity(self, query: str) -> float:
        """Evaluate how clear and specific the query is."""
        words = query.split()
        if len(words) < 3:
            return 0.3  # Too short
        elif len(words) > 20:
            return 0.8  # Detailed query
        else:
            return 0.6  # Moderate clarity

    def _evaluate_intent_detection(self, query: str) -> float:
        """Evaluate if the query intent is clear."""
        question_words = ["what", "how", "why", "when", "where", "who", "which"]
        has_question = any(word.lower() in query.lower() for word in question_words)
        has_specific_terms = len([w for w in query.split() if len(w) > 4]) > 1

        if has_question and has_specific_terms:
            return 0.9
        elif has_question or has_specific_terms:
            return 0.6
        else:
            return 0.3

    def _evaluate_preprocessing(self, query: str) -> float:
        """Evaluate query preprocessing effectiveness."""
        clean_query = " ".join(query.split())
        if clean_query == query and len(query.strip()) > 0:
            return 0.8
        else:
            return 0.5

    def _evaluate_specificity(self, query: str) -> float:
        """Evaluate query specificity."""
        # Check for specific terms, numbers, proper nouns
        specific_indicators = ["specific", "exact", "precise", "detailed", "particular"]
        has_specific_terms = any(
            indicator in query.lower() for indicator in specific_indicators
        )
        has_numbers = any(char.isdigit() for char in query)
        has_proper_nouns = any(
            word[0].isupper() for word in query.split() if len(word) > 2
        )

        # Use configurable weights for specificity scoring
        base_score = 0.5  # Base score
        specific_terms_weight = 0.2
        numbers_weight = 0.15
        proper_nouns_weight = 0.15

        score = base_score
        if has_specific_terms:
            score += specific_terms_weight
        if has_numbers:
            score += numbers_weight
        if has_proper_nouns:
            score += proper_nouns_weight
        return min(score, 1.0)

    def _evaluate_complexity(self, query: str) -> float:
        """Evaluate query complexity."""
        words = query.split()
        avg_word_length = np.mean([len(word) for word in words]) if words else 0
        unique_words = len(set(words))
        total_words = len(words)

        # Complexity based on vocabulary diversity and word length
        if total_words == 0:
            return 0.0
        elif unique_words / total_words > 0.8 and avg_word_length > 6:
            return 0.9  # High complexity
        elif unique_words / total_words > 0.6 and avg_word_length > 5:
            return 0.7  # Medium complexity
        else:
            return 0.5  # Low complexity

    def _evaluate_ambiguity(self, query: str) -> float:
        """Evaluate query ambiguity (lower is better)."""
        ambiguous_words = ["it", "this", "that", "these", "those", "thing", "stuff"]
        ambiguous_count = sum(
            1 for word in query.lower().split() if word in ambiguous_words
        )

        # Lower ambiguity score is better, so invert
        ambiguity_score = min(ambiguous_count / 3.0, 1.0)
        return 1.0 - ambiguity_score


class RetrievalStageEvaluator(BaseScorer):
    """Retrieval stage performance evaluation."""

    def __init__(
        self, llm: Union[str, Any], name: str = "enhanced_retrieval_stage"
    ) -> None:
        super().__init__(name=name)
        self.llm = llm

        # Initialize all retrieval scorers
        self.precision_scorer = ContextualPrecisionScorerPP(llm)
        self.recall_scorer = ContextualRecallScorerPP(llm)
        # Note: ContextualF1Scorer is not available, so we'll calculate F1 manually
        self.ranking_scorer = RetrievalRankingScorer()
        self.similarity_scorer = SemanticSimilarityScorer()
        self.diversity_scorer = RetrievalDiversityScorer()

    def score(
        self,
        prediction: str,
        ground_truth: str,
        context: Optional[dict[str, Any]] = None,
        weights: Optional[dict[str, float]] = None,
    ) -> Union[float, dict[str, Any]]:
        """
        Evaluate retrieval stage quality with configurable weights.

        This scorer adapts the BaseScorer interface to work with retrieval evaluation.
        The retrieved_contexts are expected to be passed via the context parameter.

        Args:
            prediction: Not used for retrieval evaluation
            ground_truth: The ground truth for evaluation
            context: Dictionary containing 'retrieved_contexts' key
            weights: Optional dictionary of weights for different metrics.
                    Default weights are:
                    - precision: 0.25
                    - recall: 0.25
                    - f1: 0.2
                    - ranking: 0.15
                    - similarity: 0.1
                    - diversity: 0.05
        """
        try:
            # Use default weights if not provided
            default_weights = {
                "precision": 0.25,
                "recall": 0.25,
                "f1": 0.2,
                "ranking": 0.15,
                "similarity": 0.1,
                "diversity": 0.05,
            }
            weights = weights or default_weights

            # Extract retrieved_contexts from context to maintain compatibility
            retrieved_contexts = context.get("retrieved_contexts") if context else None

            if not retrieved_contexts:
                return {
                    "score": 0.0,
                    "reasoning": "No retrieved contexts provided",
                    "details": {"error": "No retrieved contexts provided"},
                }

            # Run all retrieval evaluations
            precision_result = self.precision_scorer.score(
                "", ground_truth, {"retrieved_contexts": retrieved_contexts}
            )
            recall_result = self.recall_scorer.score(
                "", ground_truth, {"retrieved_contexts": retrieved_contexts}
            )
            # Calculate F1 manually since ContextualF1Scorer is not available
            f1_result = {
                "score": 0.0
            }  # Placeholder - F1 would be calculated from precision and recall
            ranking_result = self.ranking_scorer.score(
                "", ground_truth, {"retrieved_contexts": retrieved_contexts}
            )
            similarity_result = self.similarity_scorer.score(
                "", ground_truth, {"retrieved_contexts": retrieved_contexts}
            )

            # Calculate manual diversity
            manual_diversity = self._calculate_diversity(retrieved_contexts)

            def extract_score(
                result: Union[float, dict[str, Any], ScoreResult],
            ) -> float:
                if isinstance(result, ScoreResult):
                    return result.score
                elif isinstance(result, dict):
                    return result.get("score", 0.0)
                return float(result) if result is not None else 0.0

            precision_score = extract_score(precision_result)
            recall_score = extract_score(recall_result)
            f1_score = extract_score(f1_result)
            ranking_score = extract_score(ranking_result)
            similarity_score = extract_score(similarity_result)
            diversity_score = manual_diversity  # Use the manually calculated diversity

            # Weighted overall retrieval score using provided weights
            overall_score = (
                precision_score * weights.get("precision", 0.25)
                + recall_score * weights.get("recall", 0.25)
                + f1_score * weights.get("f1", 0.2)
                + ranking_score * weights.get("ranking", 0.15)
                + similarity_score * weights.get("similarity", 0.1)
                + diversity_score * weights.get("diversity", 0.05)
            )

            return {
                "score": overall_score,
                "reasoning": f"Precision: {precision_score:.2f}, Recall: {recall_score:.2f}, F1: {f1_score:.2f}, Ranking: {ranking_score:.2f}, Similarity: {similarity_score:.2f}, Diversity: {diversity_score:.2f}",
                "details": {
                    "precision": precision_score,
                    "recall": recall_score,
                    "f1": f1_score,
                    "ranking": ranking_score,
                    "similarity": similarity_score,
                    "diversity": diversity_score,
                    "manual_diversity": manual_diversity,
                    "num_contexts": len(retrieved_contexts),
                    "weights_used": weights,
                },
            }
        except Exception as e:
            return {
                "score": 0.0,
                "reasoning": f"Error in retrieval evaluation: {e!s}",
                "details": {"error": str(e)},
            }

    def _calculate_diversity(self, contexts: list[RAGContext]) -> float:
        """Calculate diversity of retrieved contexts."""
        if len(contexts) <= 1:
            return 0.0

        # Simple diversity based on source variety
        sources = {ctx.source for ctx in contexts}
        return min(len(sources) / len(contexts), 1.0)


class RerankingEvaluator(BaseScorer):
    """Evaluates reranking stage effectiveness."""

    def __init__(
        self, llm: Union[str, Any], name: str = "enhanced_generation_stage"
    ) -> None:
        super().__init__(name=name)
        self.llm = llm

        # Initialize all generation scorers
        self.bias_scorer = BiasDetectionScorer(llm)
        self.factual_scorer = FactualAccuracyScorer(llm)
        self.claim_scorer = ClaimVerificationScorer(llm)
        self.density_scorer = InformationDensityScorer(llm)
        self.clarity_scorer = ClarityAndCoherenceScorer(llm)
        self.conflict_scorer = ConflictResolutionScorer(llm)
        self.prioritization_scorer = ContextPrioritizationScorer(llm)
        self.citation_scorer = CitationQualityScorer(llm)
        self.tone_scorer = ToneConsistencyScorer(llm)
        self.terminology_scorer = TerminologyConsistencyScorer(llm)
        self.faithfulness_scorer = ContextFaithfulnessScorerPP(llm)
        self.groundedness_scorer = ContextGroundednessScorer(llm)
        self.completeness_scorer = ContextCompletenessScorer(llm)
        self.consistency_scorer = ContextConsistencyScorer(llm)
        self.quality_scorer = RAGAnswerQualityScorer(llm)
        self.hallucination_scorer = HallucinationDetectionScorer(llm)
        self.attribution_scorer = SourceAttributionScorer(llm)
        self.answer_completeness_scorer = AnswerCompletenessScorer(llm)
        self.alignment_scorer = QuestionAnswerAlignmentScorer(llm)
        self.synthesis_scorer = CrossContextSynthesisScorer(llm)
        self.technical_scorer = TechnicalAccuracyScorer(llm)

    def score(
        self,
        prediction: str,
        ground_truth: str,
        context: Optional[dict[str, Any]] = None,
        weights: Optional[dict[str, float]] = None,
    ) -> Union[float, dict[str, Any]]:
        """
        Evaluate generation stage quality with comprehensive metrics and configurable weights.

        This scorer adapts the BaseScorer interface to work with generation evaluation.
        The generated_answer and retrieved_contexts are expected to be passed via the context parameter.

        Args:
            prediction: Not used for generation evaluation
            ground_truth: The ground truth for evaluation
            context: Dictionary containing 'generated_answer' and 'retrieved_contexts' keys
            weights: Optional dictionary of weights for different metrics.
                    Default weights are:
                    - quality: 0.15
                    - faithfulness: 0.12
                    - groundedness: 0.12
                    - factual: 0.10
                    - clarity: 0.10
                    - completeness: 0.08
                    - consistency: 0.08
                    - bias: 0.06
                    - hallucination: 0.06
                    - alignment: 0.05
                    - synthesis: 0.04
                    - technical: 0.04
        """
        try:
            # Use default weights if not provided
            default_weights = {
                "quality": 0.15,
                "faithfulness": 0.12,
                "groundedness": 0.12,
                "factual": 0.10,
                "clarity": 0.10,
                "completeness": 0.08,
                "consistency": 0.08,
                "bias": 0.06,
                "hallucination": 0.06,
                "alignment": 0.05,
                "synthesis": 0.04,
                "technical": 0.04,
            }
            weights = weights or default_weights

            # Extract context information
            generated_answer = context.get("generated_answer") if context else None
            retrieved_contexts = context.get("retrieved_contexts") if context else None

            if not generated_answer:
                return {
                    "score": 0.0,
                    "reasoning": "No generated answer provided",
                    "details": {"error": "No generated answer provided"},
                }

            # Prepare context for scorers
            context_text = ""
            if retrieved_contexts:
                context_text = "\n\n".join([ctx.content for ctx in retrieved_contexts])

            context_dict = {
                "context": context_text,
                "retrieved_context": (
                    [ctx.content for ctx in retrieved_contexts]
                    if retrieved_contexts
                    else []
                ),
                "relevant_indices": [],
            }

            # Run all generation evaluations
            quality_result = self.quality_scorer.score(
                generated_answer, ground_truth, context_dict
            )
            faithfulness_result = self.faithfulness_scorer.score(
                generated_answer, ground_truth, context_dict
            )
            groundedness_result = self.groundedness_scorer.score(
                generated_answer, ground_truth, context_dict
            )
            factual_result = self.factual_scorer.score(
                generated_answer, ground_truth, context_dict
            )
            clarity_result = self.clarity_scorer.score(
                generated_answer, ground_truth, context_dict
            )
            completeness_result = self.completeness_scorer.score(
                generated_answer, ground_truth, context_dict
            )
            consistency_result = self.consistency_scorer.score(
                generated_answer, ground_truth, context_dict
            )
            bias_result = self.bias_scorer.score(
                generated_answer, ground_truth, context_dict
            )
            hallucination_result = self.hallucination_scorer.score(
                generated_answer, ground_truth, context_dict
            )
            alignment_result = self.alignment_scorer.score(
                generated_answer, ground_truth, context_dict
            )
            synthesis_result = self.synthesis_scorer.score(
                generated_answer, ground_truth, context_dict
            )
            technical_result = self.technical_scorer.score(
                generated_answer, ground_truth, context_dict
            )

            def extract_score(
                result: Union[float, dict[str, Any], ScoreResult],
            ) -> float:
                if isinstance(result, ScoreResult):
                    return result.score
                elif isinstance(result, dict):
                    return result.get("score", 0.0)
                return float(result) if result is not None else 0.0

            quality_score = extract_score(quality_result)
            faithfulness_score = extract_score(faithfulness_result)
            groundedness_score = extract_score(groundedness_result)
            factual_score = extract_score(factual_result)
            clarity_score = extract_score(clarity_result)
            completeness_score = extract_score(completeness_result)
            consistency_score = extract_score(consistency_result)
            bias_score = extract_score(bias_result)
            hallucination_score = extract_score(hallucination_result)
            alignment_score = extract_score(alignment_result)
            synthesis_score = extract_score(synthesis_result)
            technical_score = extract_score(technical_result)

            # Weighted overall generation score using provided weights
            overall_score = (
                quality_score * weights.get("quality", 0.15)
                + faithfulness_score * weights.get("faithfulness", 0.12)
                + groundedness_score * weights.get("groundedness", 0.12)
                + factual_score * weights.get("factual", 0.10)
                + clarity_score * weights.get("clarity", 0.10)
                + completeness_score * weights.get("completeness", 0.08)
                + consistency_score * weights.get("consistency", 0.08)
                + bias_score * weights.get("bias", 0.06)
                + hallucination_score * weights.get("hallucination", 0.06)
                + alignment_score * weights.get("alignment", 0.05)
                + synthesis_score * weights.get("synthesis", 0.04)
                + technical_score * weights.get("technical", 0.04)
            )

            return {
                "score": overall_score,
                "reasoning": f"Quality: {quality_score:.2f}, Faithfulness: {faithfulness_score:.2f}, Groundedness: {groundedness_score:.2f}, Factual: {factual_score:.2f}, Clarity: {clarity_score:.2f}",
                "details": {
                    "quality": quality_score,
                    "faithfulness": faithfulness_score,
                    "groundedness": groundedness_score,
                    "factual": factual_score,
                    "clarity": clarity_score,
                    "completeness": completeness_score,
                    "consistency": consistency_score,
                    "bias": bias_score,
                    "hallucination": hallucination_score,
                    "alignment": alignment_score,
                    "synthesis": synthesis_score,
                    "technical": technical_score,
                    "weights_used": weights,
                },
            }
        except Exception as e:
            return {
                "score": 0.0,
                "reasoning": f"Error in enhanced generation evaluation: {e!s}",
                "details": {"error": str(e)},
            }


class PipelineCoordinationScorer(BaseScorer):
    """Evaluates how well pipeline stages work together."""

    def __init__(self, name: str = "pipeline_coordination") -> None:
        super().__init__(name=name)

    def score(
        self,
        prediction: str,
        ground_truth: str,
        context: Optional[dict[str, Any]] = None,
    ) -> Union[float, dict[str, Any]]:
        """
        Evaluate pipeline coordination.

        This scorer adapts the BaseScorer interface to work with pipeline stage metrics.
        The stage_metrics are expected to be passed via the context parameter.

        Args:
            prediction: Not used for pipeline coordination evaluation
            ground_truth: Not used for pipeline coordination evaluation
            context: Dictionary containing 'stage_metrics' key with pipeline stage metrics
        """
        try:
            # Extract stage_metrics from context to maintain compatibility
            stage_metrics = context.get("stage_metrics") if context else None

            if not stage_metrics:
                return 0.0

            # Calculate coordination scores
            success_rate = self._calculate_success_rate(stage_metrics)
            latency_consistency = self._calculate_latency_consistency(stage_metrics)
            data_flow_quality = self._calculate_data_flow_quality(stage_metrics)

            overall_score = (success_rate + latency_consistency + data_flow_quality) / 3

            return {
                "score": overall_score,
                "reasoning": f"Success rate: {success_rate:.2f}, Latency consistency: {latency_consistency:.2f}, Data flow: {data_flow_quality:.2f}",
                "details": {
                    "success_rate": success_rate,
                    "latency_consistency": latency_consistency,
                    "data_flow_quality": data_flow_quality,
                },
            }
        except Exception as e:
            return {
                "score": 0.0,
                "reasoning": f"Error in coordination evaluation: {e!s}",
                "details": {"error": str(e)},
            }

    def _calculate_success_rate(self, metrics: dict[str, StageMetrics]) -> float:
        """Calculate overall success rate across stages."""
        successful_stages = sum(1 for stage in metrics.values() if stage.success)
        return successful_stages / len(metrics) if metrics else 0.0

    def _calculate_latency_consistency(self, metrics: dict[str, StageMetrics]) -> float:
        """Calculate latency consistency across stages."""
        latencies = [stage.latency_ms for stage in metrics.values()]
        if not latencies:
            return 0.0

        mean_latency = float(np.mean(latencies))
        std_latency = float(np.std(latencies))

        # Lower coefficient of variation is better
        cv = std_latency / mean_latency if mean_latency > 0 else 1.0
        return max(0.0, 1.0 - cv)

    def _calculate_data_flow_quality(self, metrics: dict[str, StageMetrics]) -> float:
        """Calculate data flow quality between stages."""
        # Simple heuristic: check if stages have reasonable latencies
        reasonable_latencies = sum(
            1 for stage in metrics.values() if 10 <= stage.latency_ms <= 10000
        )  # 10ms to 10s
        return reasonable_latencies / len(metrics) if metrics else 0.0


class LatencyAnalysisScorer(BaseScorer):
    """Analyzes latency across pipeline stages."""

    def __init__(self, name: str = "latency_analysis") -> None:
        super().__init__(name=name)

    def score(
        self,
        prediction: str,
        ground_truth: str,
        context: Optional[dict[str, Any]] = None,
    ) -> Union[float, dict[str, Any]]:
        """
        Analyze latency performance.

        This scorer adapts the BaseScorer interface to work with pipeline stage metrics.
        The stage_metrics are expected to be passed via the context parameter.

        Args:
            prediction: Not used for latency analysis evaluation
            ground_truth: Not used for latency analysis evaluation
            context: Dictionary containing 'stage_metrics' key with pipeline stage metrics
        """
        try:
            # Extract stage_metrics from context to maintain compatibility
            stage_metrics = context.get("stage_metrics") if context else None

            if not stage_metrics:
                return {
                    "score": 0.0,
                    "reasoning": "No stage metrics provided in context",
                    "details": {"error": "No metrics"},
                }

            # Calculate latency metrics
            total_latency = sum(stage.latency_ms for stage in stage_metrics.values())
            avg_latency = total_latency / len(stage_metrics)
            max_latency = max(stage.latency_ms for stage in stage_metrics.values())
            latency_distribution = self._calculate_latency_distribution(stage_metrics)

            # Score based on configurable latency thresholds
            latency_thresholds = {
                "excellent": 1000,  # Under 1 second
                "good": 5000,  # Under 5 seconds
                "acceptable": 10000,  # Under 10 seconds
            }
            latency_scores = {
                "excellent": 0.9,
                "good": 0.7,
                "acceptable": 0.5,
                "poor": 0.2,
            }

            if total_latency < latency_thresholds["excellent"]:
                score = latency_scores["excellent"]
            elif total_latency < latency_thresholds["good"]:
                score = latency_scores["good"]
            elif total_latency < latency_thresholds["acceptable"]:
                score = latency_scores["acceptable"]
            else:
                score = latency_scores["poor"]

            return {
                "score": score,
                "reasoning": f"Total latency: {total_latency:.1f}ms, Avg: {avg_latency:.1f}ms, Max: {max_latency:.1f}ms",
                "details": {
                    "total_latency": total_latency,
                    "avg_latency": avg_latency,
                    "max_latency": max_latency,
                    "latency_distribution": latency_distribution,
                },
            }
        except Exception as e:
            return {
                "score": 0.0,
                "reasoning": f"Error in latency analysis: {e!s}",
                "details": {"error": str(e)},
            }

    def _calculate_latency_distribution(
        self, metrics: dict[str, StageMetrics]
    ) -> dict[str, float]:
        """Calculate latency distribution across stages."""
        return {stage_name: stage.latency_ms for stage_name, stage in metrics.items()}


class ResourceUtilizationScorer(BaseScorer):
    """Analyzes resource utilization across pipeline stages."""

    def __init__(self, name: str = "resource_utilization") -> None:
        super().__init__(name=name)

    def score(
        self,
        prediction: str,
        ground_truth: str,
        context: Optional[dict[str, Any]] = None,
    ) -> Union[float, dict[str, Any]]:
        """
        Analyze resource utilization.

        This scorer adapts the BaseScorer interface to work with pipeline stage metrics.
        The stage_metrics are expected to be passed via the context parameter.

        Args:
            prediction: Not used for resource utilization evaluation
            ground_truth: Not used for resource utilization evaluation
            context: Dictionary containing 'stage_metrics' key with pipeline stage metrics
        """
        try:
            # Extract stage_metrics from context to maintain compatibility
            stage_metrics = context.get("stage_metrics") if context else None

            if not stage_metrics:
                return 0.0

            # Calculate resource metrics
            cpu_usage = self._calculate_cpu_usage(stage_metrics)
            memory_usage = self._calculate_memory_usage(stage_metrics)
            resource_efficiency = self._calculate_resource_efficiency(stage_metrics)

            overall_score = (cpu_usage + memory_usage + resource_efficiency) / 3

            return {
                "score": overall_score,
                "reasoning": f"CPU efficiency: {cpu_usage:.2f}, Memory efficiency: {memory_usage:.2f}, Overall efficiency: {resource_efficiency:.2f}",
                "details": {
                    "cpu_usage": cpu_usage,
                    "memory_usage": memory_usage,
                    "resource_efficiency": resource_efficiency,
                },
            }
        except Exception as e:
            return {
                "score": 0.0,
                "reasoning": f"Error in resource analysis: {e!s}",
                "details": {"error": str(e)},
            }

    def _calculate_cpu_usage(self, metrics: dict[str, StageMetrics]) -> float:
        """Calculate CPU usage efficiency."""
        # Simple heuristic based on latency and success
        efficient_stages = sum(
            1 for stage in metrics.values() if stage.success and stage.latency_ms < 1000
        )
        return efficient_stages / len(metrics) if metrics else 0.0

    def _calculate_memory_usage(self, metrics: dict[str, StageMetrics]) -> float:
        """Calculate memory usage efficiency."""
        # Simple heuristic based on stage success
        successful_stages = sum(1 for stage in metrics.values() if stage.success)
        return successful_stages / len(metrics) if metrics else 0.0

    def _calculate_resource_efficiency(self, metrics: dict[str, StageMetrics]) -> float:
        """Calculate overall resource efficiency."""
        # Combine CPU and memory efficiency
        cpu_efficiency = self._calculate_cpu_usage(metrics)
        memory_efficiency = self._calculate_memory_usage(metrics)
        return (cpu_efficiency + memory_efficiency) / 2


class ErrorPropagationScorer(BaseScorer):
    """Analyzes error propagation across pipeline stages."""

    def __init__(self, name: str = "error_propagation") -> None:
        super().__init__(name=name)

    def score(
        self,
        prediction: str,
        ground_truth: str,
        context: Optional[dict[str, Any]] = None,
    ) -> Union[float, dict[str, Any]]:
        """
        Analyze error propagation patterns.

        This scorer adapts the BaseScorer interface to work with pipeline stage metrics.
        The stage_metrics are expected to be passed via the context parameter.

        Args:
            prediction: Not used for error propagation evaluation
            ground_truth: Not used for error propagation evaluation
            context: Dictionary containing 'stage_metrics' key with pipeline stage metrics
        """
        try:
            # Extract stage_metrics from context to maintain compatibility
            stage_metrics = context.get("stage_metrics") if context else None

            if not stage_metrics:
                return {
                    "score": 0.0,
                    "reasoning": "No stage metrics provided in context",
                    "details": {"error": "No metrics"},
                }

            # Calculate error metrics
            error_rate = self._calculate_error_rate(stage_metrics)
            error_isolation = self._calculate_error_isolation(stage_metrics)
            recovery_effectiveness = self._calculate_recovery_effectiveness(
                stage_metrics
            )

            overall_score = (
                error_isolation + recovery_effectiveness
            ) / 2  # Lower error rate is better

            return {
                "score": overall_score,
                "reasoning": f"Error rate: {error_rate:.2f}, Error isolation: {error_isolation:.2f}, Recovery: {recovery_effectiveness:.2f}",
                "details": {
                    "error_rate": error_rate,
                    "error_isolation": error_isolation,
                    "recovery_effectiveness": recovery_effectiveness,
                },
            }
        except Exception as e:
            return {
                "score": 0.0,
                "reasoning": f"Error in error propagation analysis: {e!s}",
                "details": {"error": str(e)},
            }

    def _calculate_error_rate(self, metrics: dict[str, StageMetrics]) -> float:
        """Calculate overall error rate."""
        failed_stages = sum(1 for stage in metrics.values() if not stage.success)
        return failed_stages / len(metrics) if metrics else 0.0

    def _calculate_error_isolation(self, metrics: dict[str, StageMetrics]) -> float:
        """Calculate how well errors are isolated."""
        # Check if errors are contained to individual stages
        failed_stages = [stage for stage in metrics.values() if not stage.success]

        if not failed_stages:
            return 1.0  # No errors

        # Simple heuristic: fewer failed stages is better
        return max(0, 1 - len(failed_stages) / len(metrics))

    def _calculate_recovery_effectiveness(
        self, metrics: dict[str, StageMetrics]
    ) -> float:
        """Calculate error recovery effectiveness."""
        # Check if pipeline can continue despite some stage failures
        list(metrics.keys())
        critical_stages = ["query_processing", "retrieval", "generation"]

        failed_critical = sum(
            1
            for stage_name in critical_stages
            if stage_name in metrics and not metrics[stage_name].success
        )

        if failed_critical == 0:
            return 1.0  # No critical failures
        elif failed_critical < len(critical_stages):
            return 0.5  # Some critical failures but pipeline continues
        else:
            return 0.0  # All critical stages failed


class RAGPipelineEvaluator:
    """Main entry point for comprehensive RAG pipeline evaluation."""

    def __init__(self, llm: Union[str, Any]) -> None:
        self.llm = llm

        # Enhanced stage-specific evaluators
        self.query_evaluator = QueryProcessingEvaluator(llm)
        self.retrieval_evaluator = RetrievalStageEvaluator(llm)
        self.generation_evaluator = RerankingEvaluator(llm)

        # Basic scorers for overall evaluation
        # Convert llm to proper model instance if needed
        if isinstance(llm, str):
            # Create a simple model wrapper for string-based LLM
            class SimpleLLMModel(ModelBase):
                def __init__(self, model_name: str):
                    super().__init__(name="simple_llm", model_name=model_name)

                def generate(
                    self,
                    prompt: str,
                    max_tokens: Optional[int] = None,
                    temperature: Optional[float] = None,
                    stop: Optional[Union[str, list[str]]] = None,
                    **kwargs: Any,
                ) -> str:
                    # This is a placeholder - in real usage, you'd implement actual LLM calls
                    return f"Generated response for: {prompt[:50]}..."

                def generate_batch(
                    self,
                    prompts: list[str],
                    max_tokens: Optional[int] = None,
                    temperature: Optional[float] = None,
                    stop: Optional[Union[str, list[str]]] = None,
                    **kwargs: Any,
                ) -> list[str]:
                    return [self.generate(prompt) for prompt in prompts]

                def get_provider(self) -> str:
                    return "simple_llm"

            model_instance = SimpleLLMModel(llm)
        else:
            model_instance = llm

        self.answer_relevancy_scorer = AnswerRelevancyScorer(model_instance)
        self.faithfulness_scorer = FaithfulnessScorer(model_instance)
        self.ragas_scorer = RAGASScorer(model_instance)

        # Workflow orchestration scorers
        self.coordination_scorer = PipelineCoordinationScorer()
        self.latency_scorer = LatencyAnalysisScorer()
        self.resource_scorer = ResourceUtilizationScorer()
        self.error_scorer = ErrorPropagationScorer()

    def _convert_agent_data_to_rag_sample(self, agent_data: AgentData) -> RAGSample:
        """Convert AgentData to RAGSample for pipeline evaluation."""
        # Parse retrieved context into RAGContext objects
        retrieved_contexts = []
        if agent_data.retrieved_context:
            # Split context by double newlines (common separator)
            context_chunks = agent_data.retrieved_context.split("\n\n")
            for i, chunk in enumerate(context_chunks):
                if chunk.strip():
                    retrieved_contexts.append(
                        RAGContext(
                            content=chunk.strip(),
                            source=f"chunk_{i}",
                            relevance_score=0.8,  # Default score, can be enhanced
                            rank=i,
                            metadata={"original_index": i},
                        )
                    )

        return RAGSample(
            query=agent_data.retrieval_query or "",
            ground_truth=agent_data.ground_truth or "",
            generated_answer=agent_data.agent_response or "",
            retrieved_contexts=retrieved_contexts,
            pipeline_metadata={
                "user_id": agent_data.user_id,
                "task_id": agent_data.task_id,
                "metadata": agent_data.metadata,
            },
        )

    def evaluate_from_agent_data(
        self,
        agent_data: AgentData,
        weights: Optional[dict[str, Any]] = None,
    ) -> RAGEvaluationResult:
        """
        Evaluate RAG pipeline from agent data with configurable weights.

        Args:
            agent_data: The agent data to evaluate
            weights: Optional dictionary of weights for different components.
                    Can include:
                    - query_weights: Weights for query processing metrics
                    - retrieval_weights: Weights for retrieval metrics
                    - generation_weights: Weights for generation metrics
                    - pipeline_weights: Weights for overall pipeline scoring
        """
        # Convert agent data to RAG sample
        rag_sample = self._convert_agent_data_to_rag_sample(agent_data)

        # Extract retrieved contexts and generated answer
        retrieved_contexts = (
            [
                RAGContext(
                    content=agent_data.retrieved_context or "", source="agent_data"
                )
            ]
            if agent_data.retrieved_context
            else []
        )
        generated_answer = agent_data.agent_response or ""

        # Evaluate the pipeline
        return self.evaluate_rag_pipeline(
            rag_sample, retrieved_contexts, generated_answer, weights
        )

    def test_pipeline_compatibility(
        self, sample_agent_data: AgentData
    ) -> dict[str, Any]:
        """Test if the pipeline works correctly with AgentData."""
        try:
            # Test conversion
            self._convert_agent_data_to_rag_sample(sample_agent_data)

            # Test evaluation
            result = self.evaluate_from_agent_data(sample_agent_data)

            return {
                "success": True,
                "conversion_works": True,
                "evaluation_works": True,
                "overall_score": result.overall_score,
                "stage_count": len(result.stage_metrics),
                "recommendations": result.recommendations,
                "error": None,
            }
        except Exception as e:
            return {
                "success": False,
                "conversion_works": False,
                "evaluation_works": False,
                "overall_score": 0.0,
                "stage_count": 0,
                "recommendations": [],
                "error": str(e),
            }

    def evaluate_rag_pipeline(
        self,
        rag_sample: RAGSample,
        retrieved_contexts: list[RAGContext],
        generated_answer: str,
        weights: Optional[dict[str, Any]] = None,
    ) -> RAGEvaluationResult:
        """
        Evaluate the complete RAG pipeline with comprehensive scoring and configurable weights.

        Args:
            rag_sample: The RAG sample to evaluate
            retrieved_contexts: List of retrieved contexts
            generated_answer: The generated answer
            weights: Optional dictionary of weights for different components.
                    Can include:
                    - query_weights: Weights for query processing metrics
                    - retrieval_weights: Weights for retrieval metrics
                    - generation_weights: Weights for generation metrics
                    - pipeline_weights: Weights for overall pipeline scoring
        """
        time.time()
        stage_metrics = {}
        detailed_scores: dict[str, Union[float, dict[str, Any], ScoreResult]] = {}

        try:
            # Extract weights for different components
            query_weights = weights.get("query_weights") if weights else None
            retrieval_weights = weights.get("retrieval_weights") if weights else None
            generation_weights = weights.get("generation_weights") if weights else None
            pipeline_weights = weights.get("pipeline_weights") if weights else None

            # Stage 1: Query Processing Evaluation
            query_start = time.time()
            # Pass query as prediction parameter to maintain BaseScorer interface compatibility
            query_result = self.query_evaluator.score(
                rag_sample.query, "", None, query_weights
            )
            query_latency = (time.time() - query_start) * 1000
            # Extract score and success from JSON result
            success_threshold = 0.6  # Configurable success threshold
            if isinstance(query_result, dict):
                query_score = query_result.get("score", 0.0)
                query_success = query_score >= success_threshold
                query_reasoning = query_result.get("reasoning", "")
            else:
                query_score = (
                    float(query_result)
                    if isinstance(query_result, (int, float))
                    else 0.0
                )
                query_success = query_score >= success_threshold
                query_reasoning = f"Query processing score: {query_score}"

            stage_metrics["query_processing"] = StageMetrics(
                stage_name="query_processing",
                latency_ms=query_latency,
                success=query_success,
                error_message=query_reasoning if not query_success else None,
                metrics={"score": query_score},
            )
            detailed_scores["query_processing"] = query_result

            # Stage 2: Retrieval Evaluation
            retrieval_start = time.time()
            # Pass retrieved_contexts via context parameter to maintain BaseScorer interface compatibility
            retrieval_context = {"retrieved_contexts": retrieved_contexts}
            # Convert ground_truth string to list of references for the updated interface
            retrieval_result = self.retrieval_evaluator.score(
                "", rag_sample.ground_truth, retrieval_context, retrieval_weights
            )
            retrieval_latency = (time.time() - retrieval_start) * 1000

            # Extract score and success from JSON result
            if isinstance(retrieval_result, dict):
                retrieval_score = retrieval_result.get("score", 0.0)
                retrieval_success = retrieval_score >= success_threshold
                retrieval_reasoning = retrieval_result.get("reasoning", "")
            else:
                retrieval_score = (
                    float(retrieval_result)
                    if isinstance(retrieval_result, (int, float))
                    else 0.0
                )
                retrieval_success = retrieval_score >= success_threshold
                retrieval_reasoning = f"Retrieval score: {retrieval_score}"

            stage_metrics["retrieval"] = StageMetrics(
                stage_name="retrieval",
                latency_ms=retrieval_latency,
                success=retrieval_success,
                error_message=retrieval_reasoning if not retrieval_success else None,
                metrics={"score": retrieval_score},
            )
            detailed_scores["retrieval"] = retrieval_result

            # Stage 3: Generation Evaluation
            generation_start = time.time()
            # Pass generated_answer and retrieved_contexts via context parameter to maintain BaseScorer interface compatibility
            generation_context = {
                "generated_answer": generated_answer,
                "retrieved_contexts": retrieved_contexts,
            }
            generation_result = self.generation_evaluator.score(
                "", rag_sample.ground_truth, generation_context, generation_weights
            )
            generation_latency = (time.time() - generation_start) * 1000

            # Extract score and success from JSON result
            if isinstance(generation_result, dict):
                generation_score = generation_result.get("score", 0.0)
                generation_success = generation_score >= success_threshold
                generation_reasoning = generation_result.get("reasoning", "")
            else:
                generation_score = (
                    float(generation_result)
                    if isinstance(generation_result, (int, float))
                    else 0.0
                )
                generation_success = generation_score >= success_threshold
                generation_reasoning = f"Generation score: {generation_score}"

            stage_metrics["generation"] = StageMetrics(
                stage_name="generation",
                latency_ms=generation_latency,
                success=generation_success,
                error_message=generation_reasoning if not generation_success else None,
                metrics={"score": generation_score},
            )
            detailed_scores["generation"] = generation_result

            # Comprehensive scoring with all available scorers
            comprehensive_scores = self._run_comprehensive_evaluation(
                rag_sample, retrieved_contexts, generated_answer
            )
            detailed_scores.update(comprehensive_scores)

            # Workflow Orchestration Evaluation
            # Pass stage_metrics via context parameter to maintain BaseScorer interface compatibility
            context_with_metrics = {"stage_metrics": stage_metrics}
            coordination_result = self.coordination_scorer.score(
                "", "", context_with_metrics
            )
            latency_result = self.latency_scorer.score("", "", context_with_metrics)
            resource_result = self.resource_scorer.score("", "", context_with_metrics)
            error_result = self.error_scorer.score("", "", context_with_metrics)

            detailed_scores["coordination"] = coordination_result  # type: ignore
            detailed_scores["latency"] = latency_result  # type: ignore
            detailed_scores["resource"] = resource_result  # type: ignore
            detailed_scores["error_propagation"] = error_result  # type: ignore

            # Overall evaluation
            answer_relevancy_score = self.answer_relevancy_scorer.score(
                generated_answer, rag_sample.ground_truth
            )
            faithfulness_score = self.faithfulness_scorer.score(
                generated_answer, rag_sample.ground_truth
            )

            # Convert float scores to JSON format
            if isinstance(answer_relevancy_score, (int, float)):
                answer_relevancy_result: Union[float, dict[str, Any], ScoreResult] = {
                    "score": float(answer_relevancy_score),
                    "reasoning": f"Answer relevancy score: {answer_relevancy_score}",
                    "details": {"relevancy_score": float(answer_relevancy_score)},
                }
            else:
                answer_relevancy_result = answer_relevancy_score  # type: ignore[assignment]

            if isinstance(faithfulness_score, (int, float)):
                faithfulness_result: Union[float, dict[str, Any], ScoreResult] = {
                    "score": float(faithfulness_score),
                    "reasoning": f"Faithfulness score: {faithfulness_score}",
                    "details": {"faithfulness_score": float(faithfulness_score)},
                }
            else:
                faithfulness_result = faithfulness_score  # type: ignore[assignment]

            detailed_scores["answer_relevancy"] = answer_relevancy_result
            detailed_scores["faithfulness"] = faithfulness_result

            # Calculate overall scores - extract from JSON results
            def extract_score(
                result: Union[float, dict[str, Any], ScoreResult],
            ) -> float:
                if isinstance(result, ScoreResult):
                    return result.score
                elif isinstance(result, dict):
                    return result.get("score", 0.0)
                elif isinstance(result, (int, float)):
                    return float(result)
                else:
                    return 0.0

            retrieval_score = extract_score(retrieval_result)
            generation_score = extract_score(generation_result)
            pipeline_coordination_score = extract_score(coordination_result)
            error_propagation_score = extract_score(error_result)

            # Overall score (weighted average) using configurable weights
            default_pipeline_weights = {
                "retrieval": 0.3,
                "generation": 0.4,
                "pipeline_coordination": 0.2,
                "error_propagation": 0.1,
            }
            pipeline_weights = pipeline_weights or default_pipeline_weights

            overall_score = (
                retrieval_score * pipeline_weights.get("retrieval", 0.3)
                + generation_score * pipeline_weights.get("generation", 0.4)
                + pipeline_coordination_score
                * pipeline_weights.get("pipeline_coordination", 0.2)
                + error_propagation_score
                * pipeline_weights.get("error_propagation", 0.1)
            )

            # Latency analysis
            latency_values = [stage.latency_ms for stage in stage_metrics.values()]
            latency_analysis: dict[str, Union[float, dict[str, float]]] = {
                "total_latency": float(sum(latency_values)),
                "avg_latency": (
                    float(np.mean(latency_values)) if latency_values else 0.0
                ),
                "max_latency": float(max(latency_values)) if latency_values else 0.0,
                "latency_distribution": {
                    name: float(stage.latency_ms)
                    for name, stage in stage_metrics.items()
                },
            }

            # Resource utilization - extract from JSON result
            if isinstance(resource_result, dict):
                resource_details = resource_result.get("details", {})
                resource_utilization: dict[str, float] = {
                    "cpu_efficiency": resource_details.get("cpu_usage", 0.0),
                    "memory_efficiency": resource_details.get("memory_usage", 0.0),
                    "overall_efficiency": resource_details.get(
                        "resource_efficiency", 0.0
                    ),
                }
            else:
                resource_utilization = {
                    "cpu_efficiency": 0.0,
                    "memory_efficiency": 0.0,
                    "overall_efficiency": 0.0,
                }

            # Generate recommendations
            recommendations = self._generate_recommendations(
                stage_metrics, detailed_scores
            )

            return RAGEvaluationResult(
                overall_score=overall_score,
                stage_metrics=stage_metrics,
                retrieval_score=retrieval_score,
                generation_score=generation_score,
                pipeline_coordination_score=pipeline_coordination_score,
                latency_analysis=latency_analysis,
                resource_utilization=resource_utilization,
                error_propagation_score=error_propagation_score,
                detailed_scores=detailed_scores,
                recommendations=recommendations,
            )

        except Exception as e:
            # Return error result
            pipeline_error_result = RAGEvaluationResult(
                overall_score=0.0,
                stage_metrics={},
                retrieval_score=0.0,
                generation_score=0.0,
                pipeline_coordination_score=0.0,
                latency_analysis={},
                resource_utilization={},
                error_propagation_score=0.0,
                detailed_scores={},
                recommendations=[f"Pipeline evaluation failed: {e!s}"],
            )
            return pipeline_error_result

    def _get_scorer_factories(self) -> list[tuple[str, Callable]]:
        """
        Get list of (scorer_name, factory_function) tuples for comprehensive evaluation.

        Returns:
            List of tuples containing scorer names and their factory functions
        """
        return [
            # Basic RAG scorers
            ("contextual_precision_pp", lambda: ContextualPrecisionScorerPP(self.llm)),
            ("contextual_recall_pp", lambda: ContextualRecallScorerPP(self.llm)),
            (
                "contextual_f1",
                lambda: {
                    "score": 0.0,
                    "reasoning": "ContextualF1Scorer not available",
                    "details": {"error": "ContextualF1Scorer not implemented"},
                },
            ),
            ("retrieval_ranking", lambda: RetrievalRankingScorer()),
            ("semantic_similarity", lambda: SemanticSimilarityScorer()),
            ("retrieval_diversity", lambda: RetrievalDiversityScorer()),
            # Advanced generation scorers
            ("bias_detection", lambda: BiasDetectionScorer(self.llm)),
            ("factual_accuracy", lambda: FactualAccuracyScorer(self.llm)),
            ("claim_verification", lambda: ClaimVerificationScorer(self.llm)),
            ("information_density", lambda: InformationDensityScorer(self.llm)),
            ("clarity_coherence", lambda: ClarityAndCoherenceScorer(self.llm)),
            ("conflict_resolution", lambda: ConflictResolutionScorer(self.llm)),
            ("context_prioritization", lambda: ContextPrioritizationScorer(self.llm)),
            ("citation_quality", lambda: CitationQualityScorer(self.llm)),
            ("tone_consistency", lambda: ToneConsistencyScorer(self.llm)),
            ("terminology_consistency", lambda: TerminologyConsistencyScorer(self.llm)),
            ("context_faithfulness_pp", lambda: ContextFaithfulnessScorerPP(self.llm)),
            ("context_groundedness", lambda: ContextGroundednessScorer(self.llm)),
            ("context_completeness", lambda: ContextCompletenessScorer(self.llm)),
            ("context_consistency", lambda: ContextConsistencyScorer(self.llm)),
            ("rag_answer_quality", lambda: RAGAnswerQualityScorer(self.llm)),
            ("hallucination_detection", lambda: HallucinationDetectionScorer(self.llm)),
            ("source_attribution", lambda: SourceAttributionScorer(self.llm)),
            ("answer_completeness", lambda: AnswerCompletenessScorer(self.llm)),
            (
                "question_answer_alignment",
                lambda: QuestionAnswerAlignmentScorer(self.llm),
            ),
            ("cross_context_synthesis", lambda: CrossContextSynthesisScorer(self.llm)),
            ("technical_accuracy", lambda: TechnicalAccuracyScorer(self.llm)),
        ]

    def _evaluate_single_scorer(
        self,
        scorer_name: str,
        factory_func: Callable,
        generated_answer: str,
        ground_truth: str,
        context_dict: dict[str, Any],
    ) -> Union[float, dict[str, Any], ScoreResult]:
        """
        Evaluate a single scorer with unified error handling.

        Args:
            scorer_name: Name of the scorer
            factory_func: Function that creates the scorer instance
            generated_answer: The generated answer to evaluate
            ground_truth: The ground truth answer
            context_dict: Context dictionary for the scorer

        Returns:
            Score result (float, dict, or ScoreResult)
        """
        try:
            scorer = factory_func()
            result = scorer.score(generated_answer, ground_truth, context_dict)

            # Return the result directly (it's already the correct type)
            return result

        except Exception as e:
            return {
                "score": 0.0,
                "reasoning": f"Error evaluating {scorer_name}: {e!s}",
                "details": {"error": str(e)},
            }

    def _run_comprehensive_evaluation(
        self,
        rag_sample: RAGSample,
        retrieved_contexts: list[RAGContext],
        generated_answer: str,
    ) -> dict[str, Union[float, dict[str, Any], ScoreResult]]:
        """Run comprehensive evaluation using all available scorers."""
        comprehensive_scores = {}
        context_text = " ".join([ctx.content for ctx in retrieved_contexts])
        context_dict = {"context": context_text}

        # Get scorer factories
        scorer_factories = self._get_scorer_factories()

        # Evaluate all scorers with unified error handling
        for scorer_name, factory_func in scorer_factories:
            result = self._evaluate_single_scorer(
                scorer_name,
                factory_func,
                generated_answer,
                rag_sample.ground_truth,
                context_dict,
            )
            comprehensive_scores[scorer_name] = result

        return comprehensive_scores

    def _generate_recommendations(
        self,
        stage_metrics: dict[str, StageMetrics],
        detailed_scores: dict[str, Union[float, dict[str, Any], ScoreResult]],
    ) -> list[str]:
        """Generate improvement recommendations based on evaluation results."""
        recommendations = []

        # Check for failed stages
        failed_stages = [
            name for name, stage in stage_metrics.items() if not stage.success
        ]
        if failed_stages:
            recommendations.append(
                f"Critical: Fix failed stages: {', '.join(failed_stages)}"
            )

        # Check for slow stages
        slow_stages = [
            name for name, stage in stage_metrics.items() if stage.latency_ms > 2000
        ]  # Over 2 seconds
        if slow_stages:
            recommendations.append(
                f"Performance: Optimize slow stages: {', '.join(slow_stages)}"
            )

        # Check for low scores
        def extract_score(result: Union[float, dict[str, Any], ScoreResult]) -> float:
            if isinstance(result, ScoreResult):
                return result.score
            elif isinstance(result, dict):
                return result.get("score", 0.0)
            elif isinstance(result, (int, float)):
                return float(result)
            else:
                return 0.0

        low_score_stages = [
            name
            for name, score_result in detailed_scores.items()
            if extract_score(score_result) < 0.6
        ]
        if low_score_stages:
            recommendations.append(
                f"Quality: Improve low-scoring stages: {', '.join(low_score_stages)}"
            )

        # Check coordination
        coordination_score = extract_score(detailed_scores.get("coordination", 0.0))
        if coordination_score < 0.7:
            recommendations.append(
                "Architecture: Improve pipeline coordination and data flow"
            )

        # Check resource utilization
        resource_score = extract_score(detailed_scores.get("resource", 0.0))
        if resource_score < 0.6:
            recommendations.append("Resources: Optimize CPU and memory usage")

        if not recommendations:
            recommendations.append("Pipeline is performing well across all metrics")

        return recommendations
