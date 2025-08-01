"""
RAG Pipeline Evaluator - Comprehensive evaluation of RAG pipeline stages and workflow.

This module provides a complete evaluation framework for RAG pipelines, including:
- Stage-specific evaluators (Query, Retrieval, Reranking, Generation, Post-processing)
- Workflow orchestration scorers
- Pipeline coordination analysis
- Performance and resource monitoring
- Integration with all existing RAG scorers
"""

import asyncio
import time
import traceback
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import psutil
import numpy as np
from pydantic import BaseModel, Field

from .base import BaseScorer, ScoreResult
from .rag import AnswerRelevancyScorer, FaithfulnessScorer, ContextualPrecisionScorer, ContextualRecallScorer, RAGASScorer
from .g_eval import GEvalScorer, GEvalCriteria

# Import the required scorers from other modules
from .rag import ContextualPrecisionScorer, ContextualRecallScorer

# Import all scorers from basic_rag_scorers
from .basic_rag_scorers import (
    ContextualPrecisionScorerPP,
    ContextualRecallScorerPP,
    ContextualF1Scorer,
    RetrievalRankingScorer,
    SemanticSimilarityScorer,
    RetrievalDiversityScorer,
    AggregateRetrievalScorer
)

# Import all scorers from advanced_generation_scorers
from .advanced_generation_scorers import (
    BiasDetectionScorer,
    FactualAccuracyScorer,
    ClaimVerificationScorer,
    InformationDensityScorer,
    ClarityAndCoherenceScorer,
    ConflictResolutionScorer,
    ContextPrioritizationScorer,
    CitationQualityScorer,
    ToneConsistencyScorer,
    TerminologyConsistencyScorer,
    ContextFaithfulnessScorerPP,
    ContextGroundednessScorer,
    ContextCompletenessScorer,
    ContextConsistencyScorer,
    RAGAnswerQualityScorer,
    HallucinationDetectionScorer,
    SourceAttributionScorer,
    AnswerCompletenessScorer,
    QuestionAnswerAlignmentScorer,
    CrossContextSynthesisScorer,
    TechnicalAccuracyScorer
)

# Import AgentData for compatibility
try:
    from .rag_assessment import AgentData
except ImportError:
    # Fallback if rag_assessment is not available
    from dataclasses import dataclass
    from typing import Optional, Dict, Any
    
    @dataclass
    class AgentData:
        """Fallback AgentData structure."""
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
    metadata: Dict[str, Any] = None


@dataclass
class RAGSample:
    """Represents a complete RAG evaluation sample."""
    query: str
    ground_truth: str
    generated_answer: str
    retrieved_contexts: List[RAGContext]
    pipeline_metadata: Dict[str, Any] = None


@dataclass
class StageMetrics:
    """Metrics for a specific pipeline stage."""
    stage_name: str
    latency_ms: float
    success: bool
    error_message: Optional[str] = None
    metrics: Dict[str, float] = None
    resource_usage: Dict[str, float] = None


@dataclass
class RAGEvaluationResult:
    """Complete evaluation result for a RAG pipeline."""
    overall_score: float
    stage_metrics: Dict[str, StageMetrics]
    retrieval_score: float
    generation_score: float
    pipeline_coordination_score: float
    latency_analysis: Dict[str, float]
    resource_utilization: Dict[str, float]
    error_propagation_score: float
    detailed_scores: Dict[str, ScoreResult]
    recommendations: List[str]
    # Enhanced scoring results
    basic_rag_scores: Dict[str, ScoreResult] = None
    advanced_generation_scores: Dict[str, ScoreResult] = None
    comprehensive_scores: Dict[str, ScoreResult] = None


class QueryProcessingEvaluator(BaseScorer):
    """Evaluates query understanding and processing quality."""
    
    def __init__(self, llm: Union[str, Any], name: str = "query_processing") -> None:
        super().__init__(name=name)
        self.llm = llm
        
    def score(self, query: str, context: Optional[Dict[str, Any]] = None) -> ScoreResult:
        """Evaluate query processing quality."""
        try:
            # Basic query analysis
            clarity_score = self._evaluate_query_clarity(query)
            intent_score = self._evaluate_intent_detection(query)
            preprocessing_score = self._evaluate_preprocessing(query)
            
            # Enhanced analysis
            specificity_score = self._evaluate_specificity(query)
            complexity_score = self._evaluate_complexity(query)
            ambiguity_score = self._evaluate_ambiguity(query)
            
            # Weighted overall score
            overall_score = (
                clarity_score * 0.25 +
                intent_score * 0.25 +
                preprocessing_score * 0.2 +
                specificity_score * 0.15 +
                complexity_score * 0.1 +
                ambiguity_score * 0.05
            )
            
            return ScoreResult(
                score=overall_score,
                passed=overall_score >= 0.7,
                reasoning=f"Clarity: {clarity_score:.2f}, Intent: {intent_score:.2f}, Preprocessing: {preprocessing_score:.2f}, Specificity: {specificity_score:.2f}, Complexity: {complexity_score:.2f}, Ambiguity: {ambiguity_score:.2f}",
                metadata={
                    "clarity_score": clarity_score,
                    "intent_score": intent_score,
                    "preprocessing_score": preprocessing_score,
                    "specificity_score": specificity_score,
                    "complexity_score": complexity_score,
                    "ambiguity_score": ambiguity_score
                }
            )
        except Exception as e:
            return ScoreResult(
                score=0.0,
                passed=False,
                reasoning=f"Error in enhanced query processing evaluation: {str(e)}",
                metadata={"error": str(e)}
            )
    
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
        question_words = ['what', 'how', 'why', 'when', 'where', 'who', 'which']
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
        clean_query = ' '.join(query.split())
        if clean_query == query and len(query.strip()) > 0:
            return 0.8
        else:
            return 0.5
    
    def _evaluate_specificity(self, query: str) -> float:
        """Evaluate query specificity."""
        # Check for specific terms, numbers, proper nouns
        specific_indicators = ['specific', 'exact', 'precise', 'detailed', 'particular']
        has_specific_terms = any(indicator in query.lower() for indicator in specific_indicators)
        has_numbers = any(char.isdigit() for char in query)
        has_proper_nouns = any(word[0].isupper() for word in query.split() if len(word) > 2)
        
        score = 0.5  # Base score
        if has_specific_terms:
            score += 0.2
        if has_numbers:
            score += 0.15
        if has_proper_nouns:
            score += 0.15
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
        ambiguous_words = ['it', 'this', 'that', 'these', 'those', 'thing', 'stuff']
        ambiguous_count = sum(1 for word in query.lower().split() if word in ambiguous_words)
        
        # Lower ambiguity score is better, so invert
        ambiguity_score = min(ambiguous_count / 3.0, 1.0)
        return 1.0 - ambiguity_score


class RetrievalStageEvaluator(BaseScorer):
    """Retrieval stage performance evaluation."""
    
    def __init__(self, llm: Union[str, Any], name: str = "enhanced_retrieval_stage") -> None:
        super().__init__(name=name)
        self.llm = llm
        
        # Initialize all retrieval scorers
        self.precision_scorer = ContextualPrecisionScorerPP(llm)
        self.recall_scorer = ContextualRecallScorerPP(llm)
        self.f1_scorer = ContextualF1Scorer(self.precision_scorer, self.recall_scorer)
        self.ranking_scorer = RetrievalRankingScorer()
        self.similarity_scorer = SemanticSimilarityScorer()
        self.diversity_scorer = RetrievalDiversityScorer()
        
    def score(self, retrieved_contexts: List[RAGContext], ground_truth: str, context: Optional[Dict[str, Any]] = None) -> ScoreResult:
        """Evaluate retrieval stage performance with comprehensive metrics."""
        try:
            if not retrieved_contexts:
                return ScoreResult(
                    score=0.0,
                    passed=False,
                    reasoning="No retrieved contexts provided",
                    metadata={"error": "No contexts"}
                )
            
            # Prepare context for scorers
            context_text = "\n\n".join([ctx.content for ctx in retrieved_contexts])
            context_dict = {
                "context": context_text,
                "retrieved_context": [ctx.content for ctx in retrieved_contexts],
                "relevant_indices": []
            }
            
            # Calculate all retrieval metrics
            precision_result = self.precision_scorer.score("", ground_truth, context_dict)
            recall_result = self.recall_scorer.score("", ground_truth, context_dict)
            f1_result = self.f1_scorer.score("", ground_truth, context_dict)
            ranking_result = self.ranking_scorer.score("", ground_truth, context_dict)
            similarity_result = self.similarity_scorer.score("", ground_truth, context_dict)
            diversity_result = self.diversity_scorer.score("", ground_truth, context_dict)
            
            # Calculate diversity manually as well
            manual_diversity = self._calculate_diversity(retrieved_contexts)
            
            # Weighted overall retrieval score
            overall_score = (
                precision_result.score * 0.25 +
                recall_result.score * 0.25 +
                f1_result.score * 0.2 +
                ranking_result.score * 0.15 +
                similarity_result.score * 0.1 +
                diversity_result.score * 0.05
            )
            
            return ScoreResult(
                score=overall_score,
                passed=overall_score >= 0.6,
                reasoning=f"Precision: {precision_result.score:.2f}, Recall: {recall_result.score:.2f}, F1: {f1_result.score:.2f}, Ranking: {ranking_result.score:.2f}, Similarity: {similarity_result.score:.2f}, Diversity: {diversity_result.score:.2f}",
                metadata={
                    "precision": precision_result.score,
                    "recall": recall_result.score,
                    "f1": f1_result.score,
                    "ranking": ranking_result.score,
                    "similarity": similarity_result.score,
                    "diversity": diversity_result.score,
                    "manual_diversity": manual_diversity,
                    "num_contexts": len(retrieved_contexts)
                }
            )
        except Exception as e:
            return ScoreResult(
                score=0.0,
                passed=False,
                reasoning=f"Error in retrieval evaluation: {str(e)}",
                metadata={"error": str(e)}
            )
    
    def _calculate_diversity(self, contexts: List[RAGContext]) -> float:
        """Calculate diversity of retrieved contexts."""
        if len(contexts) <= 1:
            return 0.0
        
        # Simple diversity based on source variety
        sources = set(ctx.source for ctx in contexts)
        return min(len(sources) / len(contexts), 1.0)


class RerankingEvaluator(BaseScorer):
    """Evaluates reranking stage effectiveness."""
    
    def __init__(self, llm: Union[str, Any], name: str = "enhanced_generation_stage") -> None:
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
        
    def score(self, generated_answer: str, retrieved_contexts: List[RAGContext], 
              ground_truth: str, context: Optional[Dict[str, Any]] = None) -> ScoreResult:
        """Evaluate generation stage quality with comprehensive metrics."""
        try:
            # Prepare context
            context_text = "\n\n".join([ctx.content for ctx in retrieved_contexts])
            
            # Calculate all generation metrics
            bias_result = self.bias_scorer.score(generated_answer, ground_truth, {"context": context_text})
            factual_result = self.factual_scorer.score(generated_answer, ground_truth, {"context": context_text})
            claim_result = self.claim_scorer.score(generated_answer, ground_truth, {"context": context_text})
            density_result = self.density_scorer.score(generated_answer, ground_truth, {"context": context_text})
            clarity_result = self.clarity_scorer.score(generated_answer, ground_truth, {"context": context_text})
            conflict_result = self.conflict_scorer.score(generated_answer, ground_truth, {"context": context_text})
            prioritization_result = self.prioritization_scorer.score(generated_answer, ground_truth, {"context": context_text})
            citation_result = self.citation_scorer.score(generated_answer, ground_truth, {"context": context_text})
            tone_result = self.tone_scorer.score(generated_answer, ground_truth, {"context": context_text})
            terminology_result = self.terminology_scorer.score(generated_answer, ground_truth, {"context": context_text})
            faithfulness_result = self.faithfulness_scorer.score(generated_answer, ground_truth, {"context": context_text})
            groundedness_result = self.groundedness_scorer.score(generated_answer, ground_truth, {"context": context_text})
            completeness_result = self.completeness_scorer.score(generated_answer, ground_truth, {"context": context_text})
            consistency_result = self.consistency_scorer.score(generated_answer, ground_truth, {"context": context_text})
            quality_result = self.quality_scorer.score(generated_answer, ground_truth, {"context": context_text})
            hallucination_result = self.hallucination_scorer.score(generated_answer, ground_truth, {"context": context_text})
            attribution_result = self.attribution_scorer.score(generated_answer, ground_truth, {"context": context_text})
            answer_completeness_result = self.answer_completeness_scorer.score(generated_answer, ground_truth, {"context": context_text})
            alignment_result = self.alignment_scorer.score(generated_answer, ground_truth, {"context": context_text})
            synthesis_result = self.synthesis_scorer.score(generated_answer, ground_truth, {"context": context_text})
            technical_result = self.technical_scorer.score(generated_answer, ground_truth, {"context": context_text})
            
            # Weighted overall generation score (focusing on key metrics)
            overall_score = (
                quality_result.score * 0.15 +
                faithfulness_result.score * 0.12 +
                groundedness_result.score * 0.12 +
                factual_result.score * 0.10 +
                clarity_result.score * 0.10 +
                completeness_result.score * 0.08 +
                consistency_result.score * 0.08 +
                bias_result.score * 0.06 +
                hallucination_result.score * 0.06 +
                alignment_result.score * 0.05 +
                synthesis_result.score * 0.04 +
                technical_result.score * 0.04
            )
            
            return ScoreResult(
                score=overall_score,
                passed=overall_score >= 0.7,
                reasoning=f"Quality: {quality_result.score:.2f}, Faithfulness: {faithfulness_result.score:.2f}, Groundedness: {groundedness_result.score:.2f}, Factual: {factual_result.score:.2f}, Clarity: {clarity_result.score:.2f}",
                metadata={
                    "quality": quality_result.score,
                    "faithfulness": faithfulness_result.score,
                    "groundedness": groundedness_result.score,
                    "factual": factual_result.score,
                    "clarity": clarity_result.score,
                    "completeness": completeness_result.score,
                    "consistency": consistency_result.score,
                    "bias": bias_result.score,
                    "hallucination": hallucination_result.score,
                    "alignment": alignment_result.score,
                    "synthesis": synthesis_result.score,
                    "technical": technical_result.score
                }
            )
        except Exception as e:
            return ScoreResult(
                score=0.0,
                passed=False,
                reasoning=f"Error in enhanced generation evaluation: {str(e)}",
                metadata={"error": str(e)}
            )


class PipelineCoordinationScorer(BaseScorer):
    """Evaluates how well pipeline stages work together."""
    
    def __init__(self, name: str = "pipeline_coordination") -> None:
        super().__init__(name=name)
        
    def score(self, stage_metrics: Dict[str, StageMetrics], context: Optional[Dict[str, Any]] = None) -> ScoreResult:
        """Evaluate pipeline coordination."""
        try:
            if not stage_metrics:
                return ScoreResult(
                    score=0.0,
                    passed=False,
                    reasoning="No stage metrics provided",
                    metadata={"error": "No metrics"}
                )
            
            # Calculate coordination scores
            success_rate = self._calculate_success_rate(stage_metrics)
            latency_consistency = self._calculate_latency_consistency(stage_metrics)
            data_flow_quality = self._calculate_data_flow_quality(stage_metrics)
            
            overall_score = (success_rate + latency_consistency + data_flow_quality) / 3
            
            return ScoreResult(
                score=overall_score,
                passed=overall_score >= 0.7,
                reasoning=f"Success rate: {success_rate:.2f}, Latency consistency: {latency_consistency:.2f}, Data flow: {data_flow_quality:.2f}",
                metadata={
                    "success_rate": success_rate,
                    "latency_consistency": latency_consistency,
                    "data_flow_quality": data_flow_quality
                }
            )
        except Exception as e:
            return ScoreResult(
                score=0.0,
                passed=False,
                reasoning=f"Error in coordination evaluation: {str(e)}",
                metadata={"error": str(e)}
            )
    
    def _calculate_success_rate(self, metrics: Dict[str, StageMetrics]) -> float:
        """Calculate overall success rate across stages."""
        successful_stages = sum(1 for stage in metrics.values() if stage.success)
        return successful_stages / len(metrics) if metrics else 0.0
    
    def _calculate_latency_consistency(self, metrics: Dict[str, StageMetrics]) -> float:
        """Calculate latency consistency across stages."""
        latencies = [stage.latency_ms for stage in metrics.values()]
        if not latencies:
            return 0.0
        
        mean_latency = np.mean(latencies)
        std_latency = np.std(latencies)
        
        # Lower coefficient of variation is better
        cv = std_latency / mean_latency if mean_latency > 0 else 1.0
        return max(0, 1 - cv)
    
    def _calculate_data_flow_quality(self, metrics: Dict[str, StageMetrics]) -> float:
        """Calculate data flow quality between stages."""
        # Simple heuristic: check if stages have reasonable latencies
        reasonable_latencies = sum(1 for stage in metrics.values() 
                                 if 10 <= stage.latency_ms <= 10000)  # 10ms to 10s
        return reasonable_latencies / len(metrics) if metrics else 0.0


class LatencyAnalysisScorer(BaseScorer):
    """Analyzes latency across pipeline stages."""
    
    def __init__(self, name: str = "latency_analysis") -> None:
        super().__init__(name=name)
        
    def score(self, stage_metrics: Dict[str, StageMetrics], context: Optional[Dict[str, Any]] = None) -> ScoreResult:
        """Analyze latency performance."""
        try:
            if not stage_metrics:
                return ScoreResult(
                    score=0.0,
                    passed=False,
                    reasoning="No stage metrics provided",
                    metadata={"error": "No metrics"}
                )
            
            # Calculate latency metrics
            total_latency = sum(stage.latency_ms for stage in stage_metrics.values())
            avg_latency = total_latency / len(stage_metrics)
            max_latency = max(stage.latency_ms for stage in stage_metrics.values())
            latency_distribution = self._calculate_latency_distribution(stage_metrics)
            
            # Score based on latency thresholds
            if total_latency < 1000:  # Under 1 second
                score = 0.9
            elif total_latency < 5000:  # Under 5 seconds
                score = 0.7
            elif total_latency < 10000:  # Under 10 seconds
                score = 0.5
            else:
                score = 0.2
            
            return ScoreResult(
                score=score,
                passed=score >= 0.7,
                reasoning=f"Total latency: {total_latency:.1f}ms, Avg: {avg_latency:.1f}ms, Max: {max_latency:.1f}ms",
                metadata={
                    "total_latency": total_latency,
                    "avg_latency": avg_latency,
                    "max_latency": max_latency,
                    "latency_distribution": latency_distribution
                }
            )
        except Exception as e:
            return ScoreResult(
                score=0.0,
                passed=False,
                reasoning=f"Error in latency analysis: {str(e)}",
                metadata={"error": str(e)}
            )
    
    def _calculate_latency_distribution(self, metrics: Dict[str, StageMetrics]) -> Dict[str, float]:
        """Calculate latency distribution across stages."""
        return {stage_name: stage.latency_ms for stage_name, stage in metrics.items()}


class ResourceUtilizationScorer(BaseScorer):
    """Analyzes resource utilization across pipeline stages."""
    
    def __init__(self, name: str = "resource_utilization") -> None:
        super().__init__(name=name)
        
    def score(self, stage_metrics: Dict[str, StageMetrics], context: Optional[Dict[str, Any]] = None) -> ScoreResult:
        """Analyze resource utilization."""
        try:
            if not stage_metrics:
                return ScoreResult(
                    score=0.0,
                    passed=False,
                    reasoning="No stage metrics provided",
                    metadata={"error": "No metrics"}
                )
            
            # Calculate resource metrics
            cpu_usage = self._calculate_cpu_usage(stage_metrics)
            memory_usage = self._calculate_memory_usage(stage_metrics)
            resource_efficiency = self._calculate_resource_efficiency(stage_metrics)
            
            overall_score = (cpu_usage + memory_usage + resource_efficiency) / 3
            
            return ScoreResult(
                score=overall_score,
                passed=overall_score >= 0.6,
                reasoning=f"CPU efficiency: {cpu_usage:.2f}, Memory efficiency: {memory_usage:.2f}, Overall efficiency: {resource_efficiency:.2f}",
                metadata={
                    "cpu_usage": cpu_usage,
                    "memory_usage": memory_usage,
                    "resource_efficiency": resource_efficiency
                }
            )
        except Exception as e:
            return ScoreResult(
                score=0.0,
                passed=False,
                reasoning=f"Error in resource analysis: {str(e)}",
                metadata={"error": str(e)}
            )
    
    def _calculate_cpu_usage(self, metrics: Dict[str, StageMetrics]) -> float:
        """Calculate CPU usage efficiency."""
        # Simple heuristic based on latency and success
        efficient_stages = sum(1 for stage in metrics.values() 
                             if stage.success and stage.latency_ms < 1000)
        return efficient_stages / len(metrics) if metrics else 0.0
    
    def _calculate_memory_usage(self, metrics: Dict[str, StageMetrics]) -> float:
        """Calculate memory usage efficiency."""
        # Simple heuristic based on stage success
        successful_stages = sum(1 for stage in metrics.values() if stage.success)
        return successful_stages / len(metrics) if metrics else 0.0
    
    def _calculate_resource_efficiency(self, metrics: Dict[str, StageMetrics]) -> float:
        """Calculate overall resource efficiency."""
        # Combine CPU and memory efficiency
        cpu_efficiency = self._calculate_cpu_usage(metrics)
        memory_efficiency = self._calculate_memory_usage(metrics)
        return (cpu_efficiency + memory_efficiency) / 2


class ErrorPropagationScorer(BaseScorer):
    """Analyzes error propagation across pipeline stages."""
    
    def __init__(self, name: str = "error_propagation") -> None:
        super().__init__(name=name)
        
    def score(self, stage_metrics: Dict[str, StageMetrics], context: Optional[Dict[str, Any]] = None) -> ScoreResult:
        """Analyze error propagation patterns."""
        try:
            if not stage_metrics:
                return ScoreResult(
                    score=0.0,
                    passed=False,
                    reasoning="No stage metrics provided",
                    metadata={"error": "No metrics"}
                )
            
            # Calculate error metrics
            error_rate = self._calculate_error_rate(stage_metrics)
            error_isolation = self._calculate_error_isolation(stage_metrics)
            recovery_effectiveness = self._calculate_recovery_effectiveness(stage_metrics)
            
            overall_score = (error_isolation + recovery_effectiveness) / 2  # Lower error rate is better
            
            return ScoreResult(
                score=overall_score,
                passed=overall_score >= 0.7,
                reasoning=f"Error rate: {error_rate:.2f}, Error isolation: {error_isolation:.2f}, Recovery: {recovery_effectiveness:.2f}",
                metadata={
                    "error_rate": error_rate,
                    "error_isolation": error_isolation,
                    "recovery_effectiveness": recovery_effectiveness
                }
            )
        except Exception as e:
            return ScoreResult(
                score=0.0,
                passed=False,
                reasoning=f"Error in error propagation analysis: {str(e)}",
                metadata={"error": str(e)}
            )
    
    def _calculate_error_rate(self, metrics: Dict[str, StageMetrics]) -> float:
        """Calculate overall error rate."""
        failed_stages = sum(1 for stage in metrics.values() if not stage.success)
        return failed_stages / len(metrics) if metrics else 0.0
    
    def _calculate_error_isolation(self, metrics: Dict[str, StageMetrics]) -> float:
        """Calculate how well errors are isolated."""
        # Check if errors are contained to individual stages
        failed_stages = [stage for stage in metrics.values() if not stage.success]
        
        if not failed_stages:
            return 1.0  # No errors
        
        # Simple heuristic: fewer failed stages is better
        return max(0, 1 - len(failed_stages) / len(metrics))
    
    def _calculate_recovery_effectiveness(self, metrics: Dict[str, StageMetrics]) -> float:
        """Calculate error recovery effectiveness."""
        # Check if pipeline can continue despite some stage failures
        stages = list(metrics.keys())
        critical_stages = ['query_processing', 'retrieval', 'generation']
        
        failed_critical = sum(1 for stage_name in critical_stages 
                            if stage_name in metrics and not metrics[stage_name].success)
        
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
        self.answer_relevancy_scorer = AnswerRelevancyScorer(llm)
        self.faithfulness_scorer = FaithfulnessScorer(llm)
        self.ragas_scorer = RAGASScorer(llm)
        
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
            context_chunks = agent_data.retrieved_context.split('\n\n')
            for i, chunk in enumerate(context_chunks):
                if chunk.strip():
                    retrieved_contexts.append(RAGContext(
                        content=chunk.strip(),
                        source=f"chunk_{i}",
                        relevance_score=0.8,  # Default score, can be enhanced
                        rank=i,
                        metadata={"original_index": i}
                    ))
        
        return RAGSample(
            query=agent_data.retrieval_query or "",
            ground_truth=agent_data.ground_truth or "",
            generated_answer=agent_data.agent_response or "",
            retrieved_contexts=retrieved_contexts,
            pipeline_metadata={
                "user_id": agent_data.user_id,
                "task_id": agent_data.task_id,
                "metadata": agent_data.metadata
            }
        )
    
    def evaluate_from_agent_data(self, agent_data: AgentData) -> RAGEvaluationResult:
        """Evaluate RAG pipeline using AgentData input."""
        # Convert AgentData to RAGSample
        rag_sample = self._convert_agent_data_to_rag_sample(agent_data)
        
        # Extract retrieved contexts
        retrieved_contexts = rag_sample.retrieved_contexts
        
        # Use the generated answer as the final output
        generated_answer = rag_sample.generated_answer
        
        # Run the pipeline evaluation
        return self.evaluate_rag_pipeline(rag_sample, retrieved_contexts, generated_answer)
    
    def test_pipeline_compatibility(self, sample_agent_data: AgentData) -> Dict[str, Any]:
        """Test if the pipeline works correctly with AgentData."""
        try:
            # Test conversion
            rag_sample = self._convert_agent_data_to_rag_sample(sample_agent_data)
            
            # Test evaluation
            result = self.evaluate_from_agent_data(sample_agent_data)
            
            return {
                "success": True,
                "conversion_works": True,
                "evaluation_works": True,
                "overall_score": result.overall_score,
                "stage_count": len(result.stage_metrics),
                "recommendations": result.recommendations,
                "error": None
            }
        except Exception as e:
            return {
                "success": False,
                "conversion_works": False,
                "evaluation_works": False,
                "overall_score": 0.0,
                "stage_count": 0,
                "recommendations": [],
                "error": str(e)
            }
    
    def evaluate_rag_pipeline(self, rag_sample: RAGSample, 
                            retrieved_contexts: List[RAGContext],
                            generated_answer: str) -> RAGEvaluationResult:
        """Evaluate the complete RAG pipeline with comprehensive scoring."""
        start_time = time.time()
        stage_metrics = {}
        detailed_scores = {}
        
        try:
            # Stage 1: Query Processing Evaluation
            query_start = time.time()
            query_result = self.query_evaluator.score(rag_sample.query)
            query_latency = (time.time() - query_start) * 1000
            stage_metrics['query_processing'] = StageMetrics(
                stage_name='query_processing',
                latency_ms=query_latency,
                success=query_result.passed,
                error_message=query_result.reasoning if not query_result.passed else None,
                metrics={'score': query_result.score}
            )
            detailed_scores['query_processing'] = query_result
            
            # Stage 2: Retrieval Evaluation
            retrieval_start = time.time()
            retrieval_result = self.retrieval_evaluator.score(
                retrieved_contexts, 
                rag_sample.ground_truth
            )
            retrieval_latency = (time.time() - retrieval_start) * 1000
            stage_metrics['retrieval'] = StageMetrics(
                stage_name='retrieval',
                latency_ms=retrieval_latency,
                success=retrieval_result.passed,
                error_message=retrieval_result.reasoning if not retrieval_result.passed else None,
                metrics={'score': retrieval_result.score}
            )
            detailed_scores['retrieval'] = retrieval_result
            
            # Stage 3: Generation Evaluation
            generation_start = time.time()
            generation_result = self.generation_evaluator.score(
                generated_answer,
                retrieved_contexts,
                rag_sample.ground_truth
            )
            generation_latency = (time.time() - generation_start) * 1000
            stage_metrics['generation'] = StageMetrics(
                stage_name='generation',
                latency_ms=generation_latency,
                success=generation_result.passed,
                error_message=generation_result.reasoning if not generation_result.passed else None,
                metrics={'score': generation_result.score}
            )
            detailed_scores['generation'] = generation_result
            
            # Comprehensive scoring with all available scorers
            comprehensive_scores = self._run_comprehensive_evaluation(
                rag_sample, retrieved_contexts, generated_answer
            )
            detailed_scores.update(comprehensive_scores)
            
            # Workflow Orchestration Evaluation
            coordination_result = self.coordination_scorer.score(stage_metrics)
            latency_result = self.latency_scorer.score(stage_metrics)
            resource_result = self.resource_scorer.score(stage_metrics)
            error_result = self.error_scorer.score(stage_metrics)
            
            detailed_scores['coordination'] = coordination_result
            detailed_scores['latency'] = latency_result
            detailed_scores['resource'] = resource_result
            detailed_scores['error_propagation'] = error_result
            
            # Overall evaluation
            answer_relevancy_score = self.answer_relevancy_scorer.score(
                generated_answer, 
                rag_sample.ground_truth
            )
            faithfulness_score = self.faithfulness_scorer.score(
                generated_answer,
                " ".join([ctx.content for ctx in retrieved_contexts])
            )
            
            # Convert float scores to ScoreResult objects
            answer_relevancy_result = ScoreResult(
                score=answer_relevancy_score if isinstance(answer_relevancy_score, float) else 0.0,
                passed=answer_relevancy_score >= 0.7 if isinstance(answer_relevancy_score, float) else False,
                reasoning=f"Answer relevancy score: {answer_relevancy_score}"
            )
            faithfulness_result = ScoreResult(
                score=faithfulness_score if isinstance(faithfulness_score, float) else 0.0,
                passed=faithfulness_score >= 0.8 if isinstance(faithfulness_score, float) else False,
                reasoning=f"Faithfulness score: {faithfulness_score}"
            )
            
            detailed_scores['answer_relevancy'] = answer_relevancy_result
            detailed_scores['faithfulness'] = faithfulness_result
            
            # Calculate overall scores
            retrieval_score = retrieval_result.score
            generation_score = generation_result.score
            pipeline_coordination_score = coordination_result.score
            error_propagation_score = error_result.score
            
            # Overall score (weighted average)
            overall_score = (
                retrieval_score * 0.3 +
                generation_score * 0.4 +
                pipeline_coordination_score * 0.2 +
                error_propagation_score * 0.1
            )
            
            # Latency analysis
            latency_analysis = {
                'total_latency': sum(stage.latency_ms for stage in stage_metrics.values()),
                'avg_latency': np.mean([stage.latency_ms for stage in stage_metrics.values()]),
                'max_latency': max(stage.latency_ms for stage in stage_metrics.values()),
                'latency_distribution': {name: stage.latency_ms for name, stage in stage_metrics.items()}
            }
            
            # Resource utilization
            resource_utilization = {
                'cpu_efficiency': resource_result.metadata.get('cpu_usage', 0.0),
                'memory_efficiency': resource_result.metadata.get('memory_usage', 0.0),
                'overall_efficiency': resource_result.metadata.get('resource_efficiency', 0.0)
            }
            
            # Generate recommendations
            recommendations = self._generate_recommendations(stage_metrics, detailed_scores)
            
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
                recommendations=recommendations
            )
            
        except Exception as e:
            # Return error result
            error_result = RAGEvaluationResult(
                overall_score=0.0,
                stage_metrics={},
                retrieval_score=0.0,
                generation_score=0.0,
                pipeline_coordination_score=0.0,
                latency_analysis={},
                resource_utilization={},
                error_propagation_score=0.0,
                detailed_scores={},
                recommendations=[f"Pipeline evaluation failed: {str(e)}"]
            )
            return error_result
    
    def _get_scorer_factories(self) -> List[Tuple[str, callable]]:
        """
        Get list of (scorer_name, factory_function) tuples for comprehensive evaluation.
        
        Returns:
            List of tuples containing scorer names and their factory functions
        """
        return [
            # Basic RAG scorers
            ("contextual_precision_pp", lambda: ContextualPrecisionScorerPP(self.llm)),
            ("contextual_recall_pp", lambda: ContextualRecallScorerPP(self.llm)),
            ("contextual_f1", lambda: ContextualF1Scorer(
                ContextualPrecisionScorerPP(self.llm), 
                ContextualRecallScorerPP(self.llm)
            )),
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
            ("question_answer_alignment", lambda: QuestionAnswerAlignmentScorer(self.llm)),
            ("cross_context_synthesis", lambda: CrossContextSynthesisScorer(self.llm)),
            ("technical_accuracy", lambda: TechnicalAccuracyScorer(self.llm)),
        ]

    def _evaluate_single_scorer(self, scorer_name: str, factory_func: callable, 
                               generated_answer: str, ground_truth: str, 
                               context_dict: Dict[str, Any]) -> ScoreResult:
        """
        Evaluate a single scorer with unified error handling.
        
        Args:
            scorer_name: Name of the scorer
            factory_func: Function that creates the scorer instance
            generated_answer: The generated answer to evaluate
            ground_truth: The ground truth answer
            context_dict: Context dictionary for the scorer
            
        Returns:
            ScoreResult with evaluation result or error
        """
        try:
            scorer = factory_func()
            result = scorer.score(generated_answer, ground_truth, context_dict)
            
            # Ensure we have a ScoreResult object
            if isinstance(result, float):
                return ScoreResult(
                    score=result, 
                    passed=result >= 0.6, 
                    reasoning=f"{scorer_name} score: {result}"
                )
            else:
                return result
                
        except Exception as e:
            return ScoreResult(0.0, False, f"Error evaluating {scorer_name}: {str(e)}")

    def _run_comprehensive_evaluation(self, rag_sample: RAGSample, 
                                    retrieved_contexts: List[RAGContext],
                                    generated_answer: str) -> Dict[str, ScoreResult]:
        """Run comprehensive evaluation using all available scorers."""
        comprehensive_scores = {}
        context_text = " ".join([ctx.content for ctx in retrieved_contexts])
        context_dict = {"context": context_text}
        
        # Get scorer factories
        scorer_factories = self._get_scorer_factories()
        
        # Evaluate all scorers with unified error handling
        for scorer_name, factory_func in scorer_factories:
            result = self._evaluate_single_scorer(
                scorer_name, factory_func, generated_answer, 
                rag_sample.ground_truth, context_dict
            )
            comprehensive_scores[scorer_name] = result
        
        return comprehensive_scores
    
    def _generate_recommendations(self, stage_metrics: Dict[str, StageMetrics], 
                                 detailed_scores: Dict[str, ScoreResult]) -> List[str]:
        """Generate improvement recommendations based on evaluation results."""
        recommendations = []
        
        # Check for failed stages
        failed_stages = [name for name, stage in stage_metrics.items() if not stage.success]
        if failed_stages:
            recommendations.append(f"Critical: Fix failed stages: {', '.join(failed_stages)}")
        
        # Check for slow stages
        slow_stages = [name for name, stage in stage_metrics.items() 
                      if stage.latency_ms > 2000]  # Over 2 seconds
        if slow_stages:
            recommendations.append(f"Performance: Optimize slow stages: {', '.join(slow_stages)}")
        
        # Check for low scores
        low_score_stages = [name for name, score_result in detailed_scores.items() 
                           if score_result.score < 0.6]
        if low_score_stages:
            recommendations.append(f"Quality: Improve low-scoring stages: {', '.join(low_score_stages)}")
        
        # Check coordination
        if detailed_scores.get('coordination', ScoreResult(score=0.0, passed=False, reasoning="")).score < 0.7:
            recommendations.append("Architecture: Improve pipeline coordination and data flow")
        
        # Check resource utilization
        if detailed_scores.get('resource', ScoreResult(score=0.0, passed=False, reasoning="")).score < 0.6:
            recommendations.append("Resources: Optimize CPU and memory usage")
        
        if not recommendations:
            recommendations.append("Pipeline is performing well across all metrics")
        
        return recommendations
