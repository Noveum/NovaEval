"""
Advanced Retrieval Assessment for NovaEval.

This module implements advanced metrics for evaluating retrieval-augmented generation (RAG) and retrieval systems, including:
- Enhanced Contextual Precision (ranking-aware)
- Enhanced Contextual Recall (comprehensive)
- Contextual F1 Score
- Retrieval Ranking Metrics (NDCG, MAP, MRR)
- Semantic Similarity for context relevance
- Retrieval Diversity (uniqueness of retrieved chunks)
- Aggregate scorer for overall retrieval quality

LLM USAGE MODES:
----------------
Simple way:
    Pass a string, like "gpt-3.5-turbo". NovaEval will use its built-in code to talk to OpenAI.
Advanced way:
    If you use LangChain in your project, you might already have a model object, like my_llm = ChatOpenAI(model="gpt-3.5-turbo") or a local model. If LangChain is installed, you can pass this object directly and NovaEval will use it for generation.

The scorers will auto-detect which mode is being used.
"""

# LangChain detection
try:
    import langchain
    from langchain_core.language_models.base import BaseLanguageModel
    _HAS_LANGCHAIN = True
except ImportError:
    _HAS_LANGCHAIN = False
    BaseLanguageModel = None

from typing import Any, Optional, Union, List, TYPE_CHECKING
from novaeval.scorers.base import BaseScorer, ScoreResult
from novaeval.scorers.rag import (
    AnswerRelevancyScorer,
    FaithfulnessScorer,
    ContextualPrecisionScorer,
    ContextualRecallScorer,
    RAGASScorer
)
from novaeval.scorers.g_eval import GEvalScorer, CommonGEvalCriteria
from novaeval.scorers.basic_rag_scorers import (
    ContextualPrecisionScorerPP,
    ContextualRecallScorerPP,
    ContextualF1Scorer,
    RetrievalRankingScorer,
    SemanticSimilarityScorer,
    RetrievalDiversityScorer,
    AggregateRetrievalScorer
)
from novaeval.scorers.advanced_generation_scorers import (
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

import re
import numpy as np
from sklearn.metrics import ndcg_score, average_precision_score
from sentence_transformers import SentenceTransformer
from pydantic import BaseModel

# Forward reference for type hints
if TYPE_CHECKING:
    from typing_extensions import TypedDict

# AgentData structure definition
class ToolSchema(BaseModel):
    """Schema for tool definitions."""
    name: str
    description: Optional[str] = None
    parameters: Optional[dict[str, Any]] = None

class ToolCall(BaseModel):
    """Represents a tool call."""
    name: str
    arguments: Optional[dict[str, Any]] = None

class ToolResult(BaseModel):
    """Represents the result of a tool call."""
    name: str
    result: Optional[str] = None
    error: Optional[str] = None

class AgentData(BaseModel):
    """Data structure for agent evaluation data."""
    user_id: Optional[str] = None
    task_id: Optional[str] = None
    turn_id: Optional[str] = None
    ground_truth: Optional[str] = None
    expected_tool_call: Optional[ToolCall] = None
    agent_name: Optional[str] = None
    agent_role: Optional[str] = None
    agent_task: Optional[str] = None
    system_prompt: Optional[str] = None
    agent_response: Optional[str] = None
    trace: Optional[list[dict[str, Any]]] = None
    tools_available: list[ToolSchema] = []
    tool_calls: list[ToolCall] = []
    parameters_passed: dict[str, Any] = {}
    tool_call_results: Optional[list[ToolResult]] = None
    retrieval_query: Optional[str] = None
    retrieved_context: Optional[str] = None
    metadata: Optional[str] = None


def _call_llm(model, prompt: str):
    """
    Helper to call the LLM, supporting both string (OpenAI) and LangChain LLM objects.
    """
    if _HAS_LANGCHAIN and BaseLanguageModel and isinstance(model, BaseLanguageModel):
        # LangChain LLM: use .invoke or .generate
        if hasattr(model, "invoke"):
            return model.invoke(prompt)
        elif hasattr(model, "generate"):
            return model.generate([prompt]).generations[0][0].text
        else:
            raise ValueError("LangChain LLM does not support invoke or generate.")
    elif isinstance(model, str):
        # Built-in: string model name, use OpenAI API
        raise NotImplementedError("String-based LLM calling not implemented. Please provide a LangChain LLM or implement OpenAI call here.")
    else:
        # Assume model has a .generate method (NovaEval style)
        return model.generate(prompt)


class RAGAssessmentEngine:
    """
    Comprehensive RAG assessment engine that works with AgentData.
    
    This class provides methods to evaluate RAG systems using AgentData instances,
    integrating both basic RAG metrics from rag.py and advanced retrieval metrics.
    """
    
    def __init__(self, model, threshold: float = 0.7, **kwargs):
        """
        Initialize the RAG assessment engine.
        
        Args:
            model: LLM model (string, LangChain LLM, or NovaEval model)
            threshold: Default threshold for pass/fail decisions
            **kwargs: Additional configuration
        """
        self.model = model
        self.threshold = threshold
        
        # Initialize scorers from rag.py
        self.answer_relevancy_scorer = AnswerRelevancyScorer(model=model, threshold=threshold)
        self.faithfulness_scorer = FaithfulnessScorer(model=model, threshold=threshold)
        self.contextual_precision_scorer = ContextualPrecisionScorer(model=model, threshold=threshold)
        self.contextual_recall_scorer = ContextualRecallScorer(model=model, threshold=threshold)
        self.ragas_scorer = RAGASScorer(model=model, threshold=threshold)
        
        # Initialize advanced scorers
        self.contextual_precision_pp = ContextualPrecisionScorerPP(model=model, threshold=threshold)
        self.contextual_recall_pp = ContextualRecallScorerPP(model=model, threshold=threshold)
        self.retrieval_ranking_scorer = RetrievalRankingScorer(threshold=threshold)
        self.semantic_similarity_scorer = SemanticSimilarityScorer(threshold=threshold)
        self.retrieval_diversity_scorer = RetrievalDiversityScorer()
        
        # Add G-Eval scorers
        self.helpfulness_scorer = GEvalScorer(model=model, criteria=CommonGEvalCriteria.helpfulness())
        self.correctness_scorer = GEvalScorer(model=model, criteria=CommonGEvalCriteria.correctness())
        
        # Initialize F1 scorer
        self.contextual_f1_scorer = ContextualF1Scorer(
            precision_scorer=self.contextual_precision_pp,
            recall_scorer=self.contextual_recall_pp,
            threshold=threshold
        )
        
        # Initialize Context-Aware Generation Scorers
        self.context_faithfulness_pp = ContextFaithfulnessScorerPP(model=model, threshold=threshold)
        self.context_groundedness_scorer = ContextGroundednessScorer(model=model, threshold=threshold)
        self.context_completeness_scorer = ContextCompletenessScorer(model=model, threshold=threshold)
        self.context_consistency_scorer = ContextConsistencyScorer(model=model, threshold=threshold)
        
        # Initialize Answer Quality Enhancement Scorers
        self.rag_answer_quality_scorer = RAGAnswerQualityScorer(model=model, threshold=threshold)
        
        # Initialize Hallucination Detection Scorers
        self.hallucination_detection_scorer = HallucinationDetectionScorer(model=model, threshold=threshold)
        self.source_attribution_scorer = SourceAttributionScorer(model=model, threshold=threshold)
        self.factual_accuracy_scorer = FactualAccuracyScorer(model=model, threshold=threshold)
        self.claim_verification_scorer = ClaimVerificationScorer(model=model, threshold=threshold)
        
        # Initialize Answer Completeness and Relevance Scorers
        self.answer_completeness_scorer = AnswerCompletenessScorer(model=model, threshold=threshold)
        self.question_answer_alignment_scorer = QuestionAnswerAlignmentScorer(model=model, threshold=threshold)
        self.information_density_scorer = InformationDensityScorer(model=model, threshold=threshold)
        self.clarity_coherence_scorer = ClarityAndCoherenceScorer(model=model, threshold=threshold)
        
        # Initialize Multi-Context Integration Scorers
        self.cross_context_synthesis_scorer = CrossContextSynthesisScorer(model=model, threshold=threshold)
        self.conflict_resolution_scorer = ConflictResolutionScorer(model=model, threshold=threshold)
        self.context_prioritization_scorer = ContextPrioritizationScorer(model=model, threshold=threshold)
        self.citation_quality_scorer = CitationQualityScorer(model=model, threshold=threshold)
        
        # Initialize Domain-Specific Evaluation Scorers
        self.technical_accuracy_scorer = TechnicalAccuracyScorer(model=model, threshold=threshold)
        self.bias_detection_scorer = BiasDetectionScorer(model=model, threshold=threshold)
        self.tone_consistency_scorer = ToneConsistencyScorer(model=model, threshold=threshold)
        self.terminology_consistency_scorer = TerminologyConsistencyScorer(model=model, threshold=threshold)
        
        # Initialize aggregate scorer
        self.aggregate_scorer = AggregateRetrievalScorer(
            scorers={
                "answer_relevancy": self.answer_relevancy_scorer,
                "faithfulness": self.faithfulness_scorer,
                "contextual_precision": self.contextual_precision_scorer,
                "contextual_recall": self.contextual_recall_scorer,
                "semantic_similarity": self.semantic_similarity_scorer,
                "retrieval_diversity": self.retrieval_diversity_scorer,
                "helpfulness": self.helpfulness_scorer,
                "correctness": self.correctness_scorer,
                "context_faithfulness_pp": self.context_faithfulness_pp,
                "rag_answer_quality": self.rag_answer_quality_scorer,
                "hallucination_detection": self.hallucination_detection_scorer,
                "factual_accuracy": self.factual_accuracy_scorer,
                "answer_completeness": self.answer_completeness_scorer,
                "information_density": self.information_density_scorer,
                "clarity_coherence": self.clarity_coherence_scorer,
                "cross_context_synthesis": self.cross_context_synthesis_scorer,
                "technical_accuracy": self.technical_accuracy_scorer,
                "bias_detection": self.bias_detection_scorer,
            },
            weights={
                "answer_relevancy": 0.06,
                "faithfulness": 0.06,
                "contextual_precision": 0.06,
                "contextual_recall": 0.06,
                "semantic_similarity": 0.05,
                "retrieval_diversity": 0.04,
                "helpfulness": 0.06,
                "correctness": 0.06,
                "context_faithfulness_pp": 0.06,
                "rag_answer_quality": 0.06,
                "hallucination_detection": 0.06,
                "factual_accuracy": 0.06,
                "answer_completeness": 0.06,
                "information_density": 0.04,
                "clarity_coherence": 0.05,
                "cross_context_synthesis": 0.06,
                "technical_accuracy": 0.04,
                "bias_detection": 0.06,
            }
        )
    
    async def evaluate(self, agent_data: AgentData) -> dict[str, Any]:
        """
        Evaluate a single AgentData instance using all available metrics.
        
        Args:
            agent_data: AgentData instance to evaluate
            
        Returns:
            Dictionary containing all evaluation results
        """
        if not agent_data.retrieved_context:
            return {"error": "No retrieved context available for evaluation"}
        
        # Prepare context for scorers
        context_dict = {
            "context": agent_data.retrieved_context,
            "retrieved_context": agent_data.retrieved_context.split("\n\n") if agent_data.retrieved_context else [],
            "relevant_indices": []  # You can populate this based on your relevance criteria
        }
        
        results = {}
        
        # Basic RAG metrics (from rag.py)
        try:
            results["answer_relevancy"] = await self.answer_relevancy_scorer.evaluate(
                input_text=agent_data.ground_truth or "",
                output_text=agent_data.agent_response or "",
                context=agent_data.retrieved_context
            )
        except Exception as e:
            results["answer_relevancy"] = ScoreResult(0.0, False, f"Error: {str(e)}")
        
        try:
            results["faithfulness"] = await self.faithfulness_scorer.evaluate(
                input_text=agent_data.ground_truth or "",
                output_text=agent_data.agent_response or "",
                context=agent_data.retrieved_context
            )
        except Exception as e:
            results["faithfulness"] = ScoreResult(0.0, False, f"Error: {str(e)}")
        
        try:
            results["contextual_precision"] = await self.contextual_precision_scorer.evaluate(
                input_text=agent_data.ground_truth or "",
                output_text=agent_data.agent_response or "",
                context=agent_data.retrieved_context
            )
        except Exception as e:
            results["contextual_precision"] = ScoreResult(0.0, False, f"Error: {str(e)}")
        
        try:
            results["contextual_recall"] = await self.contextual_recall_scorer.evaluate(
                input_text=agent_data.ground_truth or "",
                output_text=agent_data.agent_response or "",
                expected_output=agent_data.ground_truth or "",
                context=agent_data.retrieved_context
            )
        except Exception as e:
            results["contextual_recall"] = ScoreResult(0.0, False, f"Error: {str(e)}")
        
        # Advanced retrieval metrics
        try:
            results["contextual_precision_pp"] = await self.contextual_precision_pp.evaluate(
                input_text=agent_data.ground_truth or "",
                output_text=agent_data.agent_response or "",
                context=agent_data.retrieved_context
            )
        except Exception as e:
            results["contextual_precision_pp"] = ScoreResult(0.0, False, f"Error: {str(e)}")
        
        try:
            results["contextual_recall_pp"] = await self.contextual_recall_pp.evaluate(
                input_text=agent_data.ground_truth or "",
                output_text=agent_data.agent_response or "",
                expected_output=agent_data.ground_truth or "",
                context=agent_data.retrieved_context
            )
        except Exception as e:
            results["contextual_recall_pp"] = ScoreResult(0.0, False, f"Error: {str(e)}")
        
        # Non-async scorers
        try:
            results["contextual_f1"] = self.contextual_f1_scorer.score(
                prediction=agent_data.agent_response or "",
                ground_truth=agent_data.ground_truth or "",
                context=context_dict
            )
        except Exception as e:
            results["contextual_f1"] = ScoreResult(0.0, False, f"Error: {str(e)}")
        
        try:
            results["retrieval_ranking"] = self.retrieval_ranking_scorer.score(
                prediction=agent_data.agent_response or "",
                ground_truth=agent_data.ground_truth or "",
                context=context_dict
            )
        except Exception as e:
            results["retrieval_ranking"] = {"error": str(e)}
        
        try:
            results["semantic_similarity"] = self.semantic_similarity_scorer.score(
                prediction=agent_data.agent_response or "",
                ground_truth=agent_data.ground_truth or "",
                context=context_dict
            )
        except Exception as e:
            results["semantic_similarity"] = {"error": str(e)}
        
        try:
            results["retrieval_diversity"] = self.retrieval_diversity_scorer.score(
                prediction=agent_data.agent_response or "",
                ground_truth=agent_data.ground_truth or "",
                context=context_dict
            )
        except Exception as e:
            results["retrieval_diversity"] = {"error": str(e)}
        
        # Add G-Eval evaluations
        try:
            results["helpfulness"] = await self.helpfulness_scorer.evaluate(
                input_text=agent_data.ground_truth or "",
                output_text=agent_data.agent_response or "",
                context=agent_data.retrieved_context
            )
        except Exception as e:
            results["helpfulness"] = ScoreResult(score=0.0, passed=False, reasoning=f"Error: {str(e)}")
        
        try:
            results["correctness"] = await self.correctness_scorer.evaluate(
                input_text=agent_data.ground_truth or "",
                output_text=agent_data.agent_response or "",
                context=agent_data.retrieved_context
            )
        except Exception as e:
            results["correctness"] = ScoreResult(score=0.0, passed=False, reasoning=f"Error: {str(e)}")
        
        # Add Context-Aware Generation Scorers
        try:
            results["context_faithfulness_pp"] = await self.context_faithfulness_pp.evaluate(
                input_text=agent_data.ground_truth or "",
                output_text=agent_data.agent_response or "",
                context=agent_data.retrieved_context
            )
        except Exception as e:
            results["context_faithfulness_pp"] = ScoreResult(score=0.0, passed=False, reasoning=f"Error: {str(e)}")
        
        try:
            results["context_groundedness"] = await self.context_groundedness_scorer.evaluate(
                input_text=agent_data.ground_truth or "",
                output_text=agent_data.agent_response or "",
                context=agent_data.retrieved_context
            )
        except Exception as e:
            results["context_groundedness"] = ScoreResult(score=0.0, passed=False, reasoning=f"Error: {str(e)}")
        
        try:
            results["context_completeness"] = await self.context_completeness_scorer.evaluate(
                input_text=agent_data.ground_truth or "",
                output_text=agent_data.agent_response or "",
                context=agent_data.retrieved_context
            )
        except Exception as e:
            results["context_completeness"] = ScoreResult(score=0.0, passed=False, reasoning=f"Error: {str(e)}")
        
        try:
            results["context_consistency"] = await self.context_consistency_scorer.evaluate(
                input_text=agent_data.ground_truth or "",
                output_text=agent_data.agent_response or "",
                context=agent_data.retrieved_context
            )
        except Exception as e:
            results["context_consistency"] = ScoreResult(score=0.0, passed=False, reasoning=f"Error: {str(e)}")
        
        # Add Answer Quality Enhancement Scorers
        try:
            results["rag_answer_quality"] = await self.rag_answer_quality_scorer.evaluate(
                input_text=agent_data.ground_truth or "",
                output_text=agent_data.agent_response or "",
                context=agent_data.retrieved_context
            )
        except Exception as e:
            results["rag_answer_quality"] = ScoreResult(score=0.0, passed=False, reasoning=f"Error: {str(e)}")
        
        # Add Hallucination Detection Scorers
        try:
            results["hallucination_detection"] = await self.hallucination_detection_scorer.evaluate(
                input_text=agent_data.ground_truth or "",
                output_text=agent_data.agent_response or "",
                context=agent_data.retrieved_context
            )
        except Exception as e:
            results["hallucination_detection"] = ScoreResult(score=0.0, passed=False, reasoning=f"Error: {str(e)}")
        
        try:
            results["source_attribution"] = await self.source_attribution_scorer.evaluate(
                input_text=agent_data.ground_truth or "",
                output_text=agent_data.agent_response or "",
                context=agent_data.retrieved_context
            )
        except Exception as e:
            results["source_attribution"] = ScoreResult(score=0.0, passed=False, reasoning=f"Error: {str(e)}")
        
        try:
            results["factual_accuracy"] = await self.factual_accuracy_scorer.evaluate(
                input_text=agent_data.ground_truth or "",
                output_text=agent_data.agent_response or "",
                context=agent_data.retrieved_context
            )
        except Exception as e:
            results["factual_accuracy"] = ScoreResult(score=0.0, passed=False, reasoning=f"Error: {str(e)}")
        
        try:
            results["claim_verification"] = await self.claim_verification_scorer.evaluate(
                input_text=agent_data.ground_truth or "",
                output_text=agent_data.agent_response or "",
                context=agent_data.retrieved_context
            )
        except Exception as e:
            results["claim_verification"] = ScoreResult(score=0.0, passed=False, reasoning=f"Error: {str(e)}")
        
        # Add Answer Completeness and Relevance Scorers
        try:
            results["answer_completeness"] = await self.answer_completeness_scorer.evaluate(
                input_text=agent_data.ground_truth or "",
                output_text=agent_data.agent_response or "",
                context=agent_data.retrieved_context
            )
        except Exception as e:
            results["answer_completeness"] = ScoreResult(score=0.0, passed=False, reasoning=f"Error: {str(e)}")
        
        try:
            results["question_answer_alignment"] = await self.question_answer_alignment_scorer.evaluate(
                input_text=agent_data.ground_truth or "",
                output_text=agent_data.agent_response or "",
                context=agent_data.retrieved_context
            )
        except Exception as e:
            results["question_answer_alignment"] = ScoreResult(score=0.0, passed=False, reasoning=f"Error: {str(e)}")
        
        try:
            results["information_density"] = await self.information_density_scorer.evaluate(
                input_text=agent_data.ground_truth or "",
                output_text=agent_data.agent_response or "",
                context=agent_data.retrieved_context
            )
        except Exception as e:
            results["information_density"] = ScoreResult(score=0.0, passed=False, reasoning=f"Error: {str(e)}")
        
        try:
            results["clarity_coherence"] = await self.clarity_coherence_scorer.evaluate(
                input_text=agent_data.ground_truth or "",
                output_text=agent_data.agent_response or "",
                context=agent_data.retrieved_context
            )
        except Exception as e:
            results["clarity_coherence"] = ScoreResult(score=0.0, passed=False, reasoning=f"Error: {str(e)}")
        
        # Add Multi-Context Integration Scorers
        try:
            results["cross_context_synthesis"] = await self.cross_context_synthesis_scorer.evaluate(
                input_text=agent_data.ground_truth or "",
                output_text=agent_data.agent_response or "",
                context=agent_data.retrieved_context
            )
        except Exception as e:
            results["cross_context_synthesis"] = ScoreResult(score=0.0, passed=False, reasoning=f"Error: {str(e)}")
        
        try:
            results["conflict_resolution"] = await self.conflict_resolution_scorer.evaluate(
                input_text=agent_data.ground_truth or "",
                output_text=agent_data.agent_response or "",
                context=agent_data.retrieved_context
            )
        except Exception as e:
            results["conflict_resolution"] = ScoreResult(score=0.0, passed=False, reasoning=f"Error: {str(e)}")
        
        try:
            results["context_prioritization"] = await self.context_prioritization_scorer.evaluate(
                input_text=agent_data.ground_truth or "",
                output_text=agent_data.agent_response or "",
                context=agent_data.retrieved_context
            )
        except Exception as e:
            results["context_prioritization"] = ScoreResult(score=0.0, passed=False, reasoning=f"Error: {str(e)}")
        
        try:
            results["citation_quality"] = await self.citation_quality_scorer.evaluate(
                input_text=agent_data.ground_truth or "",
                output_text=agent_data.agent_response or "",
                context=agent_data.retrieved_context
            )
        except Exception as e:
            results["citation_quality"] = ScoreResult(score=0.0, passed=False, reasoning=f"Error: {str(e)}")
        
        # Add Domain-Specific Evaluation Scorers
        try:
            results["technical_accuracy"] = await self.technical_accuracy_scorer.evaluate(
                input_text=agent_data.ground_truth or "",
                output_text=agent_data.agent_response or "",
                context=agent_data.retrieved_context
            )
        except Exception as e:
            results["technical_accuracy"] = ScoreResult(score=0.0, passed=False, reasoning=f"Error: {str(e)}")
        
        try:
            results["bias_detection"] = await self.bias_detection_scorer.evaluate(
                input_text=agent_data.ground_truth or "",
                output_text=agent_data.agent_response or "",
                context=agent_data.retrieved_context
            )
        except Exception as e:
            results["bias_detection"] = ScoreResult(score=0.0, passed=False, reasoning=f"Error: {str(e)}")
        
        try:
            results["tone_consistency"] = await self.tone_consistency_scorer.evaluate(
                input_text=agent_data.ground_truth or "",
                output_text=agent_data.agent_response or "",
                context=agent_data.retrieved_context
            )
        except Exception as e:
            results["tone_consistency"] = ScoreResult(score=0.0, passed=False, reasoning=f"Error: {str(e)}")
        
        try:
            results["terminology_consistency"] = await self.terminology_consistency_scorer.evaluate(
                input_text=agent_data.ground_truth or "",
                output_text=agent_data.agent_response or "",
                context=agent_data.retrieved_context
            )
        except Exception as e:
            results["terminology_consistency"] = ScoreResult(score=0.0, passed=False, reasoning=f"Error: {str(e)}")
        
        # Aggregate score
        try:
            results["aggregate"] = self.aggregate_scorer.score(
                prediction=agent_data.agent_response or "",
                ground_truth=agent_data.ground_truth or "",
                context=context_dict
            )
        except Exception as e:
            results["aggregate"] = {"error": str(e)}
        
        return results
    
    async def evaluate_batch(self, agent_data_list: List[AgentData]) -> List[dict[str, Any]]:
        """
        Evaluate multiple AgentData instances.
        
        Args:
            agent_data_list: List of AgentData instances to evaluate
            
        Returns:
            List of evaluation results for each AgentData instance
        """
        results = []
        for agent_data in agent_data_list:
            result = await self.evaluate(agent_data)
            results.append(result)
        return results