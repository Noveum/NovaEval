"""
Panel of LLMs as Judge scorer for NovaEval.

This module implements a panel-based evaluation system where multiple LLMs
act as judges to evaluate AI model outputs. This approach provides more robust
and nuanced assessments by leveraging diverse perspectives from different models.
"""

import asyncio
import statistics
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, validator

from novaeval.models.base import BaseModel as LLMModel
from novaeval.scorers.base import BaseScorer, ScoreResult


class AggregationMethod(str, Enum):
    """Methods for aggregating scores from multiple judges."""

    MEAN = "mean"
    MEDIAN = "median"
    WEIGHTED_MEAN = "weighted_mean"
    MAJORITY_VOTE = "majority_vote"
    CONSENSUS = "consensus"
    MIN = "min"
    MAX = "max"


class JudgeConfig(BaseModel):
    """Configuration for a single judge in the panel."""

    model: LLMModel = Field(description="The LLM model acting as judge")
    weight: float = Field(default=1.0, description="Weight of this judge's opinion")
    name: Optional[str] = Field(
        default=None, description="Name identifier for the judge"
    )
    specialty: Optional[str] = Field(
        default=None, description="Judge's area of expertise"
    )
    temperature: float = Field(
        default=0.0, description="Temperature for judge's responses"
    )

    @validator("weight")
    def validate_weight(cls, v):
        if v < 0.0:
            raise ValueError("Judge weight must be non-negative")
        return v


class PanelResult(BaseModel):
    """Result from a panel of judges evaluation."""

    individual_scores: list[float] = Field(description="Scores from individual judges")
    individual_reasonings: list[str] = Field(
        description="Reasoning from individual judges"
    )
    judge_names: list[str] = Field(description="Names of the judges")
    aggregated_score: float = Field(description="Final aggregated score")
    aggregation_method: AggregationMethod = Field(
        description="Method used for aggregation"
    )
    consensus_level: float = Field(description="Level of consensus among judges (0-1)")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )


class PanelOfJudgesScorer(BaseScorer):
    """
    Panel of LLMs as Judge scorer.

    This scorer uses multiple LLM models as judges to evaluate outputs,
    providing more robust and diverse assessments than single-model evaluation.
    """

    def __init__(
        self,
        judges: list[JudgeConfig],
        aggregation_method: AggregationMethod = AggregationMethod.WEIGHTED_MEAN,
        threshold: float = 0.7,
        require_consensus: bool = False,
        consensus_threshold: float = 0.8,
        evaluation_criteria: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(threshold=threshold, **kwargs)

        if not judges:
            raise ValueError("At least one judge must be provided")

        self.judges = judges
        self.aggregation_method = aggregation_method
        self.require_consensus = require_consensus
        self.consensus_threshold = consensus_threshold
        self.evaluation_criteria = (
            evaluation_criteria or "overall quality and correctness"
        )

        # Normalize weights for weighted aggregation
        if aggregation_method == AggregationMethod.WEIGHTED_MEAN:
            total_weight = sum(judge.weight for judge in judges)
            if total_weight > 0:
                for judge in self.judges:
                    judge.weight = judge.weight / total_weight

    async def evaluate(
        self,
        input_text: str,
        output_text: str,
        expected_output: Optional[str] = None,
        context: Optional[str] = None,
        **kwargs,
    ) -> ScoreResult:
        """Evaluate using a panel of LLM judges."""

        try:
            # Prepare evaluation prompt
            evaluation_prompt = self._build_evaluation_prompt(
                input_text, output_text, expected_output, context
            )

            # Get evaluations from all judges in parallel
            judge_tasks = [
                self._evaluate_with_judge(judge, evaluation_prompt)
                for judge in self.judges
            ]

            judge_results = await asyncio.gather(*judge_tasks, return_exceptions=True)

            # Process results and handle exceptions
            valid_results = []
            failed_judges = []

            for i, result in enumerate(judge_results):
                if isinstance(result, Exception):
                    failed_judges.append(
                        f"{self.judges[i].name or f'Judge_{i}'}: {result!s}"
                    )
                else:
                    valid_results.append((self.judges[i], result))

            if not valid_results:
                return ScoreResult(
                    score=0.0,
                    passed=False,
                    reasoning="All judges failed to evaluate",
                    metadata={"failed_judges": failed_judges},
                )

            # Extract scores and reasonings
            individual_scores = [result[1]["score"] for result in valid_results]
            individual_reasonings = [result[1]["reasoning"] for result in valid_results]
            judge_names = [
                result[0].name or f"Judge_{i}" for i, result in enumerate(valid_results)
            ]
            judge_weights = [result[0].weight for result in valid_results]

            # Calculate consensus level
            consensus_level = self._calculate_consensus(individual_scores)

            # Check consensus requirement
            if self.require_consensus and consensus_level < self.consensus_threshold:
                return ScoreResult(
                    score=0.0,
                    passed=False,
                    reasoning=f"Insufficient consensus among judges (consensus: {consensus_level:.3f}, required: {self.consensus_threshold})",
                    metadata={
                        "individual_scores": individual_scores,
                        "consensus_level": consensus_level,
                        "failed_consensus": True,
                    },
                )

            # Aggregate scores
            aggregated_score = self._aggregate_scores(
                individual_scores, judge_weights, self.aggregation_method
            )

            # Create panel result
            panel_result = PanelResult(
                individual_scores=individual_scores,
                individual_reasonings=individual_reasonings,
                judge_names=judge_names,
                aggregated_score=aggregated_score,
                aggregation_method=self.aggregation_method,
                consensus_level=consensus_level,
                metadata={
                    "failed_judges": failed_judges,
                    "judge_weights": judge_weights,
                },
            )

            # Generate comprehensive reasoning
            reasoning = self._generate_panel_reasoning(panel_result)

            return ScoreResult(
                score=aggregated_score,
                passed=aggregated_score >= self.threshold,
                reasoning=reasoning,
                metadata=panel_result.dict(),
            )

        except Exception as e:
            return ScoreResult(
                score=0.0,
                passed=False,
                reasoning=f"Panel evaluation failed: {e!s}",
                metadata={"error": str(e)},
            )

    def _build_evaluation_prompt(
        self,
        input_text: str,
        output_text: str,
        expected_output: Optional[str] = None,
        context: Optional[str] = None,
    ) -> str:
        """Build evaluation prompt for judges."""

        prompt_parts = [
            "You are an expert evaluator tasked with assessing the quality of an AI model's response.",
            f"\nEvaluation Criteria: {self.evaluation_criteria}",
            f"\nInput/Question: {input_text}",
            f"\nAI Response to Evaluate: {output_text}",
        ]

        if expected_output:
            prompt_parts.append(f"\nExpected/Reference Answer: {expected_output}")

        if context:
            prompt_parts.append(f"\nAdditional Context: {context}")

        prompt_parts.extend(
            [
                "\nPlease evaluate the AI response based on the following criteria:",
                "1. Accuracy and correctness",
                "2. Completeness and thoroughness",
                "3. Clarity and coherence",
                "4. Relevance to the input",
                "5. Overall helpfulness",
                "",
                "Provide your evaluation in the following JSON format:",
                "{",
                '  "score": <numeric_score_from_1_to_5>,',
                '  "reasoning": "<detailed_explanation_of_your_evaluation>",',
                '  "strengths": "<what_the_response_does_well>",',
                '  "weaknesses": "<areas_for_improvement>",',
                '  "confidence": <confidence_level_from_1_to_5>',
                "}",
                "",
                "Score Guidelines:",
                "1 = Poor (major issues, incorrect or unhelpful)",
                "2 = Below Average (some issues, partially correct)",
                "3 = Average (acceptable, meets basic requirements)",
                "4 = Good (high quality, minor issues)",
                "5 = Excellent (outstanding, comprehensive, accurate)",
                "",
                "Ensure your response is valid JSON and provide detailed reasoning for your score.",
            ]
        )

        return "\n".join(prompt_parts)

    async def _evaluate_with_judge(
        self, judge: JudgeConfig, prompt: str
    ) -> dict[str, Any]:
        """Evaluate with a single judge."""

        try:
            # Set judge-specific temperature if different from default
            original_temp = getattr(judge.model, "temperature", None)
            if hasattr(judge.model, "temperature"):
                judge.model.temperature = judge.temperature

            # Get evaluation from judge
            response = await judge.model.generate(prompt)

            # Restore original temperature
            if original_temp is not None and hasattr(judge.model, "temperature"):
                judge.model.temperature = original_temp

            # Parse JSON response
            import json
            import re

            # Extract JSON from response
            json_match = re.search(r"\{.*\}", response, re.DOTALL)
            if not json_match:
                raise ValueError("No JSON found in judge response")

            result = json.loads(json_match.group())

            # Validate required fields
            if "score" not in result or "reasoning" not in result:
                raise ValueError("Judge response missing required fields")

            # Normalize score to 0-1 range
            score = float(result["score"])
            if score < 1 or score > 5:
                raise ValueError(f"Score must be between 1 and 5, got {score}")

            normalized_score = (score - 1) / 4  # Convert 1-5 to 0-1

            return {
                "score": normalized_score,
                "reasoning": result["reasoning"],
                "raw_score": score,
                "strengths": result.get("strengths", ""),
                "weaknesses": result.get("weaknesses", ""),
                "confidence": result.get("confidence", 3),
            }

        except Exception as e:
            raise Exception(f"Judge evaluation failed: {e!s}")

    def _calculate_consensus(self, scores: list[float]) -> float:
        """Calculate consensus level among judges (0-1)."""

        if len(scores) <= 1:
            return 1.0

        # Calculate standard deviation as a measure of disagreement
        statistics.mean(scores)
        variance = statistics.variance(scores)
        std_dev = variance**0.5

        # Convert to consensus level (lower std_dev = higher consensus)
        # Normalize by maximum possible std_dev for 0-1 scores
        max_std_dev = 0.5  # Maximum std_dev for scores in [0,1]
        consensus = max(0.0, 1.0 - (std_dev / max_std_dev))

        return consensus

    def _aggregate_scores(
        self, scores: list[float], weights: list[float], method: AggregationMethod
    ) -> float:
        """Aggregate scores from multiple judges."""

        if method == AggregationMethod.MEAN:
            return statistics.mean(scores)

        elif method == AggregationMethod.MEDIAN:
            return statistics.median(scores)

        elif method == AggregationMethod.WEIGHTED_MEAN:
            if len(scores) != len(weights):
                return statistics.mean(scores)  # Fallback to mean
            return sum(score * weight for score, weight in zip(scores, weights))

        elif method == AggregationMethod.MAJORITY_VOTE:
            # Convert to binary (pass/fail) and take majority
            binary_votes = [1 if score >= self.threshold else 0 for score in scores]
            majority = sum(binary_votes) > len(binary_votes) / 2
            return 1.0 if majority else 0.0

        elif method == AggregationMethod.CONSENSUS:
            # Only pass if all judges agree (all above threshold)
            return (
                min(scores) if all(score >= self.threshold for score in scores) else 0.0
            )

        elif method == AggregationMethod.MIN:
            return min(scores)

        elif method == AggregationMethod.MAX:
            return max(scores)

        else:
            return statistics.mean(scores)  # Default fallback

    def _generate_panel_reasoning(self, panel_result: PanelResult) -> str:
        """Generate comprehensive reasoning from panel results."""

        reasoning_parts = [
            f"Panel of {len(panel_result.individual_scores)} LLM Judges Evaluation",
            f"Aggregation Method: {panel_result.aggregation_method.value}",
            f"Final Score: {panel_result.aggregated_score:.3f}",
            f"Consensus Level: {panel_result.consensus_level:.3f}",
            "",
            "Individual Judge Scores:",
        ]

        for i, (name, score, reasoning) in enumerate(
            zip(
                panel_result.judge_names,
                panel_result.individual_scores,
                panel_result.individual_reasonings,
            )
        ):
            reasoning_parts.extend(
                [
                    f"{i+1}. {name}: {score:.3f}",
                    f"   Reasoning: {reasoning[:200]}{'...' if len(reasoning) > 200 else ''}",
                    "",
                ]
            )

        # Add summary statistics
        scores = panel_result.individual_scores
        reasoning_parts.extend(
            [
                "Summary Statistics:",
                f"• Mean Score: {statistics.mean(scores):.3f}",
                f"• Median Score: {statistics.median(scores):.3f}",
                f"• Score Range: {min(scores):.3f} - {max(scores):.3f}",
                f"• Standard Deviation: {statistics.stdev(scores) if len(scores) > 1 else 0:.3f}",
                "",
            ]
        )

        # Add consensus analysis
        if panel_result.consensus_level >= 0.8:
            consensus_desc = "High consensus among judges"
        elif panel_result.consensus_level >= 0.6:
            consensus_desc = "Moderate consensus among judges"
        else:
            consensus_desc = "Low consensus among judges - diverse opinions"

        reasoning_parts.append(f"Consensus Analysis: {consensus_desc}")

        return "\n".join(reasoning_parts)


class SpecializedPanelScorer(PanelOfJudgesScorer):
    """
    Specialized panel scorer with predefined judge configurations.

    This class provides common panel configurations for different evaluation scenarios.
    """

    @classmethod
    def create_diverse_panel(
        cls,
        models: list[LLMModel],
        evaluation_criteria: str = "overall quality and correctness",
        **kwargs,
    ) -> "SpecializedPanelScorer":
        """Create a diverse panel with different model types."""

        judges = []
        specialties = [
            "accuracy",
            "clarity",
            "completeness",
            "relevance",
            "helpfulness",
        ]

        for i, model in enumerate(models):
            specialty = specialties[i % len(specialties)]
            judges.append(
                JudgeConfig(
                    model=model,
                    weight=1.0,
                    name=f"{model.__class__.__name__}_{specialty}",
                    specialty=specialty,
                    temperature=0.0,
                )
            )

        return cls(judges=judges, evaluation_criteria=evaluation_criteria, **kwargs)

    @classmethod
    def create_consensus_panel(
        cls, models: list[LLMModel], consensus_threshold: float = 0.8, **kwargs
    ) -> "SpecializedPanelScorer":
        """Create a panel that requires high consensus."""

        judges = [
            JudgeConfig(
                model=model, weight=1.0, name=f"ConsensusJudge_{i+1}", temperature=0.0
            )
            for i, model in enumerate(models)
        ]

        return cls(
            judges=judges,
            aggregation_method=AggregationMethod.CONSENSUS,
            require_consensus=True,
            consensus_threshold=consensus_threshold,
            **kwargs,
        )

    @classmethod
    def create_weighted_expert_panel(
        cls,
        expert_models: list[tuple[LLMModel, float]],  # (model, expertise_weight)
        **kwargs,
    ) -> "SpecializedPanelScorer":
        """Create a panel with weighted expert judges."""

        judges = [
            JudgeConfig(
                model=model,
                weight=weight,
                name=f"Expert_{i+1}",
                specialty="domain_expert",
                temperature=0.0,
            )
            for i, (model, weight) in enumerate(expert_models)
        ]

        return cls(
            judges=judges, aggregation_method=AggregationMethod.WEIGHTED_MEAN, **kwargs
        )
