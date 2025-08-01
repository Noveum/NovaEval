"""
RAG Assessment Engine for NovaEval.

This module provides a comprehensive evaluation engine for RAG systems,
combining multiple evaluation metrics and scorers.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
from pydantic import BaseModel

from novaeval.scorers.base import BaseScorer, ScoreResult
from novaeval.scorers.rag import (
    AnswerRelevancyScorer,
    FaithfulnessScorer,
    ContextualPrecisionScorer,
    ContextualRecallScorer,
    RAGASScorer,
)
from novaeval.scorers.basic_rag_scorers import (
    ContextualPrecisionScorerPP,
    ContextualRecallScorerPP,
    RetrievalRankingScorer,
    SemanticSimilarityScorer,
    RetrievalDiversityScorer,
    ContextualF1Scorer,
    AggregateRetrievalScorer,
)
from novaeval.scorers.g_eval import GEvalScorer, CommonGEvalCriteria
from novaeval.scorers.advanced_generation_scorers import (
    ContextFaithfulnessScorerPP,
    ContextGroundednessScorer,
    ContextCompletenessScorer,
    ContextConsistencyScorer,
    RAGAnswerQualityScorer,
    HallucinationDetectionScorer,
    SourceAttributionScorer,
    FactualAccuracyScorer,
    ClaimVerificationScorer,
    AnswerCompletenessScorer,
    QuestionAnswerAlignmentScorer,
    InformationDensityScorer,
    ClarityAndCoherenceScorer,
    CrossContextSynthesisScorer,
    ConflictResolutionScorer,
    ContextPrioritizationScorer,
    CitationQualityScorer,
    TechnicalAccuracyScorer,
    BiasDetectionScorer,
    ToneConsistencyScorer,
    TerminologyConsistencyScorer,
)
from novaeval.utils.llm import call_llm


class ScorerConfig(BaseModel):
    """Configuration for a single scorer."""
    name: str
    class_name: str
    weight: float = 1.0
    enabled: bool = True
    params: Dict[str, Any] = {}


class RAGAssessmentConfig(BaseModel):
    """Configuration for the RAG Assessment Engine."""
    threshold: float = 0.7
    scorers: List[ScorerConfig] = []
    
    @classmethod
    def get_default_config(cls) -> "RAGAssessmentConfig":
        """Get the default configuration with all scorers."""
        return cls(
            threshold=0.7,
            scorers=[
                # Basic RAG scorers
                ScorerConfig(name="answer_relevancy", class_name="AnswerRelevancyScorer", weight=0.06),
                ScorerConfig(name="faithfulness", class_name="FaithfulnessScorer", weight=0.06),
                ScorerConfig(name="contextual_precision", class_name="ContextualPrecisionScorer", weight=0.06),
                ScorerConfig(name="contextual_recall", class_name="ContextualRecallScorer", weight=0.06),
                ScorerConfig(name="ragas", class_name="RAGASScorer", weight=0.0),  # Disabled by default
                
                # Advanced retrieval scorers
                ScorerConfig(name="contextual_precision_pp", class_name="ContextualPrecisionScorerPP", weight=0.0),  # Disabled by default
                ScorerConfig(name="contextual_recall_pp", class_name="ContextualRecallScorerPP", weight=0.0),  # Disabled by default
                ScorerConfig(name="retrieval_ranking", class_name="RetrievalRankingScorer", weight=0.0),  # Disabled by default
                ScorerConfig(name="semantic_similarity", class_name="SemanticSimilarityScorer", weight=0.05),
                ScorerConfig(name="retrieval_diversity", class_name="RetrievalDiversityScorer", weight=0.04),
                
                # G-Eval scorers
                ScorerConfig(name="helpfulness", class_name="GEvalScorer", weight=0.06, 
                           params={"criteria": "helpfulness"}),
                ScorerConfig(name="correctness", class_name="GEvalScorer", weight=0.06,
                           params={"criteria": "correctness"}),
                
                # Context-Aware Generation Scorers
                ScorerConfig(name="context_faithfulness_pp", class_name="ContextFaithfulnessScorerPP", weight=0.06),
                ScorerConfig(name="context_groundedness", class_name="ContextGroundednessScorer", weight=0.0),  # Disabled by default
                ScorerConfig(name="context_completeness", class_name="ContextCompletenessScorer", weight=0.0),  # Disabled by default
                ScorerConfig(name="context_consistency", class_name="ContextConsistencyScorer", weight=0.0),  # Disabled by default
                
                # Answer Quality Enhancement Scorers
                ScorerConfig(name="rag_answer_quality", class_name="RAGAnswerQualityScorer", weight=0.06),
                
                # Hallucination Detection Scorers
                ScorerConfig(name="hallucination_detection", class_name="HallucinationDetectionScorer", weight=0.06),
                ScorerConfig(name="source_attribution", class_name="SourceAttributionScorer", weight=0.0),  # Disabled by default
                ScorerConfig(name="factual_accuracy", class_name="FactualAccuracyScorer", weight=0.06),
                ScorerConfig(name="claim_verification", class_name="ClaimVerificationScorer", weight=0.0),  # Disabled by default
                
                # Answer Completeness and Relevance Scorers
                ScorerConfig(name="answer_completeness", class_name="AnswerCompletenessScorer", weight=0.06),
                ScorerConfig(name="question_answer_alignment", class_name="QuestionAnswerAlignmentScorer", weight=0.0),  # Disabled by default
                ScorerConfig(name="information_density", class_name="InformationDensityScorer", weight=0.04),
                ScorerConfig(name="clarity_coherence", class_name="ClarityAndCoherenceScorer", weight=0.05),
                
                # Multi-Context Integration Scorers
                ScorerConfig(name="cross_context_synthesis", class_name="CrossContextSynthesisScorer", weight=0.06),
                ScorerConfig(name="conflict_resolution", class_name="ConflictResolutionScorer", weight=0.0),  # Disabled by default
                ScorerConfig(name="context_prioritization", class_name="ContextPrioritizationScorer", weight=0.0),  # Disabled by default
                ScorerConfig(name="citation_quality", class_name="CitationQualityScorer", weight=0.0),  # Disabled by default
                
                # Domain-Specific Evaluation Scorers
                ScorerConfig(name="technical_accuracy", class_name="TechnicalAccuracyScorer", weight=0.04),
                ScorerConfig(name="bias_detection", class_name="BiasDetectionScorer", weight=0.06),
                ScorerConfig(name="tone_consistency", class_name="ToneConsistencyScorer", weight=0.0),  # Disabled by default
                ScorerConfig(name="terminology_consistency", class_name="TerminologyConsistencyScorer", weight=0.0),  # Disabled by default
            ]
        )


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
    trace: Optional[list[dict[str, Any]]] = []
    tools_available: list[ToolSchema] = []
    tool_calls: list[ToolCall] = []
    parameters_passed: dict[str, Any] = {}
    tool_call_results: Optional[list[ToolResult]] = None
    retrieval_query: Optional[str] = None
    retrieved_context: Optional[str] = None
    metadata: Optional[str] = None


class RAGAssessmentEngine:
    """
    Comprehensive RAG assessment engine that combines multiple evaluation metrics.
    
    This engine evaluates RAG systems using a configurable set of scorers,
    allowing for flexible evaluation strategies and easy A/B testing.
    """

    def __init__(
        self, 
        model: Any, 
        threshold: float = 0.7, 
        config: Optional[RAGAssessmentConfig] = None,
        **kwargs: Any
    ) -> None:
        """
        Initialize the RAG Assessment Engine.
        
        Args:
            model: The language model to use for evaluation
            threshold: Default threshold for scorers
            config: Configuration for scorers and weights. If None, uses default config
            **kwargs: Additional arguments passed to scorers
        """
        self.model = model
        self.threshold = threshold
        self.config = config or RAGAssessmentConfig.get_default_config()
        self.kwargs = kwargs
        
        # Initialize scorers based on configuration
        self._initialize_scorers()
        
        # Initialize aggregate scorer
        self._initialize_aggregate_scorer()

    def _get_scorer_class(self, class_name: str) -> type:
        """Get the scorer class by name."""
        scorer_classes = {
            # Basic RAG scorers
            "AnswerRelevancyScorer": AnswerRelevancyScorer,
            "FaithfulnessScorer": FaithfulnessScorer,
            "ContextualPrecisionScorer": ContextualPrecisionScorer,
            "ContextualRecallScorer": ContextualRecallScorer,
            "RAGASScorer": RAGASScorer,
            
            # Advanced retrieval scorers
            "ContextualPrecisionScorerPP": ContextualPrecisionScorerPP,
            "ContextualRecallScorerPP": ContextualRecallScorerPP,
            "RetrievalRankingScorer": RetrievalRankingScorer,
            "SemanticSimilarityScorer": SemanticSimilarityScorer,
            "RetrievalDiversityScorer": RetrievalDiversityScorer,
            "ContextualF1Scorer": ContextualF1Scorer,
            
            # G-Eval scorers
            "GEvalScorer": GEvalScorer,
            
            # Advanced generation scorers
            "ContextFaithfulnessScorerPP": ContextFaithfulnessScorerPP,
            "ContextGroundednessScorer": ContextGroundednessScorer,
            "ContextCompletenessScorer": ContextCompletenessScorer,
            "ContextConsistencyScorer": ContextConsistencyScorer,
            "RAGAnswerQualityScorer": RAGAnswerQualityScorer,
            "HallucinationDetectionScorer": HallucinationDetectionScorer,
            "SourceAttributionScorer": SourceAttributionScorer,
            "FactualAccuracyScorer": FactualAccuracyScorer,
            "ClaimVerificationScorer": ClaimVerificationScorer,
            "AnswerCompletenessScorer": AnswerCompletenessScorer,
            "QuestionAnswerAlignmentScorer": QuestionAnswerAlignmentScorer,
            "InformationDensityScorer": InformationDensityScorer,
            "ClarityAndCoherenceScorer": ClarityAndCoherenceScorer,
            "CrossContextSynthesisScorer": CrossContextSynthesisScorer,
            "ConflictResolutionScorer": ConflictResolutionScorer,
            "ContextPrioritizationScorer": ContextPrioritizationScorer,
            "CitationQualityScorer": CitationQualityScorer,
            "TechnicalAccuracyScorer": TechnicalAccuracyScorer,
            "BiasDetectionScorer": BiasDetectionScorer,
            "ToneConsistencyScorer": ToneConsistencyScorer,
            "TerminologyConsistencyScorer": TerminologyConsistencyScorer,
        }
        
        if class_name not in scorer_classes:
            raise ValueError(f"Unknown scorer class: {class_name}")
        
        return scorer_classes[class_name]

    def _create_scorer(self, config: ScorerConfig) -> BaseScorer:
        """Create a scorer instance from configuration."""
        scorer_class = self._get_scorer_class(config.class_name)
        
        # Handle special cases
        if config.class_name == "GEvalScorer":
            criteria_name = config.params.get("criteria", "helpfulness")
            if criteria_name == "helpfulness":
                criteria = CommonGEvalCriteria.helpfulness()
            elif criteria_name == "correctness":
                criteria = CommonGEvalCriteria.correctness()
            else:
                raise ValueError(f"Unknown G-Eval criteria: {criteria_name}")
            
            return scorer_class(model=self.model, criteria=criteria, threshold=self.threshold, **self.kwargs)
        
        elif config.class_name == "ContextualF1Scorer":
            # This requires special handling as it needs precision and recall scorers
            precision_scorer = getattr(self, "contextual_precision_pp", None)
            recall_scorer = getattr(self, "contextual_recall_pp", None)
            
            if precision_scorer is None or recall_scorer is None:
                raise ValueError("ContextualF1Scorer requires contextual_precision_pp and contextual_recall_pp scorers")
            
            return scorer_class(
                precision_scorer=precision_scorer,
                recall_scorer=recall_scorer,
                threshold=self.threshold,
                **self.kwargs
            )
        
        else:
            # Standard scorer creation
            return scorer_class(model=self.model, threshold=self.threshold, **self.kwargs)

    def _initialize_scorers(self) -> None:
        """Initialize all scorers based on configuration."""
        for config in self.config.scorers:
            if not config.enabled:
                continue
                
            try:
                scorer = self._create_scorer(config)
                setattr(self, f"{config.name}_scorer", scorer)
            except Exception as e:
                print(f"Warning: Failed to initialize scorer {config.name}: {e}")

    def _initialize_aggregate_scorer(self) -> None:
        """Initialize the aggregate scorer with enabled scorers and their weights."""
        enabled_scorers = {}
        weights = {}
        
        for config in self.config.scorers:
            if not config.enabled:
                continue
                
            scorer_attr = f"{config.name}_scorer"
            if hasattr(self, scorer_attr):
                enabled_scorers[config.name] = getattr(self, scorer_attr)
                weights[config.name] = config.weight
        
        if enabled_scorers:
            self.aggregate_scorer = AggregateRetrievalScorer(
                scorers=enabled_scorers,
                weights=weights
            )
        else:
            self.aggregate_scorer = None

    def _prepare_context_dict(self, agent_data: AgentData) -> dict[str, Any]:
        """Prepare context dictionary for scorers."""
        return {
            "context": agent_data.retrieved_context,
            "retrieved_context": agent_data.retrieved_context.split("\n\n") if agent_data.retrieved_context else [],
            "relevant_indices": []  # You can populate this based on your relevance criteria
        }

    async def _evaluate_single_scorer(self, scorer: BaseScorer, agent_data: AgentData, context_dict: dict[str, Any]) -> ScoreResult:
        """
        Evaluate a single scorer with unified error handling.
        
        Args:
            scorer: The scorer to evaluate
            agent_data: AgentData instance
            context_dict: Prepared context dictionary
            
        Returns:
            ScoreResult with evaluation result or error
        """
        try:
            if hasattr(scorer, 'evaluate'):
                # Async scorers
                result = await scorer.evaluate(
                    input_text=agent_data.ground_truth or "",
                    output_text=agent_data.agent_response or "",
                    context=agent_data.retrieved_context
                )
            else:
                # Sync scorers
                result = scorer.score(
                    prediction=agent_data.agent_response or "",
                    ground_truth=agent_data.ground_truth or "",
                    context=context_dict
                )
            
            return result
            
        except Exception as e:
            return ScoreResult(
                score=0.0, 
                passed=False, 
                reasoning=f"Error evaluating {scorer.__class__.__name__}: {str(e)}"
            )

    def _get_scorer_by_category(self) -> dict[str, list[tuple[str, BaseScorer]]]:
        """
        Group enabled scorers by logical categories.
        
        Returns:
            Dictionary mapping category names to lists of (scorer_name, scorer) tuples
        """
        categories = {
            "Basic RAG": [],
            "Advanced Retrieval": [],
            "G-Eval": [],
            "Context-Aware Generation": [],
            "Answer Quality": [],
            "Hallucination Detection": [],
            "Answer Completeness": [],
            "Multi-Context Integration": [],
            "Domain-Specific": [],
            "Other": []
        }
        
        # Define category mappings
        category_mappings = {
            # Basic RAG scorers
            "answer_relevancy": "Basic RAG",
            "faithfulness": "Basic RAG", 
            "contextual_precision": "Basic RAG",
            "contextual_recall": "Basic RAG",
            "ragas": "Basic RAG",
            
            # Advanced retrieval scorers
            "contextual_precision_pp": "Advanced Retrieval",
            "contextual_recall_pp": "Advanced Retrieval",
            "retrieval_ranking": "Advanced Retrieval",
            "semantic_similarity": "Advanced Retrieval",
            "retrieval_diversity": "Advanced Retrieval",
            
            # G-Eval scorers
            "helpfulness": "G-Eval",
            "correctness": "G-Eval",
            
            # Context-Aware Generation Scorers
            "context_faithfulness_pp": "Context-Aware Generation",
            "context_groundedness": "Context-Aware Generation",
            "context_completeness": "Context-Aware Generation",
            "context_consistency": "Context-Aware Generation",
            
            # Answer Quality Enhancement Scorers
            "rag_answer_quality": "Answer Quality",
            
            # Hallucination Detection Scorers
            "hallucination_detection": "Hallucination Detection",
            "source_attribution": "Hallucination Detection",
            "factual_accuracy": "Hallucination Detection",
            "claim_verification": "Hallucination Detection",
            
            # Answer Completeness and Relevance Scorers
            "answer_completeness": "Answer Completeness",
            "question_answer_alignment": "Answer Completeness",
            "information_density": "Answer Completeness",
            "clarity_coherence": "Answer Completeness",
            
            # Multi-Context Integration Scorers
            "cross_context_synthesis": "Multi-Context Integration",
            "conflict_resolution": "Multi-Context Integration",
            "context_prioritization": "Multi-Context Integration",
            "citation_quality": "Multi-Context Integration",
            
            # Domain-Specific Evaluation Scorers
            "technical_accuracy": "Domain-Specific",
            "bias_detection": "Domain-Specific",
            "tone_consistency": "Domain-Specific",
            "terminology_consistency": "Domain-Specific",
        }
        
        # Group enabled scorers by category
        for config in self.config.scorers:
            if not config.enabled:
                continue
                
            scorer_attr = f"{config.name}_scorer"
            if not hasattr(self, scorer_attr):
                continue
                
            scorer = getattr(self, scorer_attr)
            category = category_mappings.get(config.name, "Other")
            categories[category].append((config.name, scorer))
        
        return categories

    async def _evaluate_category(self, category_name: str, scorers: list[tuple[str, BaseScorer]], 
                                agent_data: AgentData, context_dict: dict[str, Any]) -> dict[str, ScoreResult]:
        """
        Evaluate all scorers in a category.
        
        Args:
            category_name: Name of the category
            scorers: List of (scorer_name, scorer) tuples
            agent_data: AgentData instance
            context_dict: Prepared context dictionary
            
        Returns:
            Dictionary mapping scorer names to ScoreResults
        """
        results = {}
        
        for scorer_name, scorer in scorers:
            result = await self._evaluate_single_scorer(scorer, agent_data, context_dict)
            results[scorer_name] = result
        
        return results

    async def _evaluate_aggregate_scorer(self, agent_data: AgentData, context_dict: dict[str, Any]) -> ScoreResult:
        """
        Evaluate the aggregate scorer with error handling.
        
        Args:
            agent_data: AgentData instance
            context_dict: Prepared context dictionary
            
        Returns:
            ScoreResult for aggregate evaluation
        """
        if not self.aggregate_scorer:
            return ScoreResult(score=0.0, passed=False, reasoning="No aggregate scorer available")
        
        try:
            result = self.aggregate_scorer.score(
                prediction=agent_data.agent_response or "",
                ground_truth=agent_data.ground_truth or "",
                context=context_dict
            )
            return result
        except Exception as e:
            return ScoreResult(
                score=0.0, 
                passed=False, 
                reasoning=f"Error evaluating aggregate scorer: {str(e)}"
            )

    async def evaluate(self, agent_data: AgentData) -> dict[str, Any]:
        """
        Evaluate a single AgentData instance using all enabled metrics.
        
        Args:
            agent_data: AgentData instance to evaluate
            
        Returns:
            Dictionary containing all evaluation results
        """
        if not agent_data.retrieved_context:
            return {"error": "No retrieved context available for evaluation"}
        
        # Prepare context for scorers
        context_dict = self._prepare_context_dict(agent_data)
        
        # Group scorers by category
        categorized_scorers = self._get_scorer_by_category()
        
        # Evaluate all categories
        results = {}
        for category_name, scorers in categorized_scorers.items():
            if scorers:  # Only evaluate categories that have enabled scorers
                category_results = await self._evaluate_category(
                    category_name, scorers, agent_data, context_dict
                )
                results.update(category_results)
        
        # Add aggregate score
        aggregate_result = await self._evaluate_aggregate_scorer(agent_data, context_dict)
        results["aggregate"] = aggregate_result
        
        return results

    async def evaluate_batch(self, agent_data_list: List[AgentData]) -> List[dict[str, Any]]:
        """
        Evaluate a batch of AgentData instances.
        
        Args:
            agent_data_list: List of AgentData instances to evaluate
            
        Returns:
            List of evaluation results
        """
        results = []
        for agent_data in agent_data_list:
            result = await self.evaluate(agent_data)
            results.append(result)
        return results

    def update_config(self, new_config: RAGAssessmentConfig) -> None:
        """
        Update the configuration and reinitialize scorers.
        
        Args:
            new_config: New configuration to apply
        """
        self.config = new_config
        self._initialize_scorers()
        self._initialize_aggregate_scorer()

    def get_enabled_scorers(self) -> List[str]:
        """Get list of enabled scorer names."""
        return [config.name for config in self.config.scorers if config.enabled]

    def get_scorer_weights(self) -> Dict[str, float]:
        """Get current scorer weights."""
        return {config.name: config.weight for config in self.config.scorers if config.enabled}