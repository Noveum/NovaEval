"""
G-Eval scorer implementation for NovaEval.

G-Eval is a framework that uses LLMs with chain-of-thought (CoT) reasoning
to evaluate LLM outputs based on custom criteria.
"""

from typing import Any, Optional, Union

from pydantic import BaseModel, Field

from novaeval.models.base import BaseModel as LLMModel
from novaeval.scorers.base import BaseScorer, ScoreResult


class GEvalCriteria(BaseModel):
    """G-Eval evaluation criteria."""

    name: str = Field(description="Name of the evaluation criteria")
    description: str = Field(description="Detailed description of what to evaluate")
    steps: list[str] = Field(description="Step-by-step evaluation instructions")
    score_range: tuple[int, int] = Field(
        default=(1, 5), description="Score range (min, max)"
    )


class GEvalScorer(BaseScorer):
    """
    G-Eval scorer that uses LLMs with chain-of-thought reasoning
    to evaluate outputs based on custom criteria.

    Based on the paper: "G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment"
    """

    def __init__(
        self,
        model: LLMModel,
        criteria: Union[str, GEvalCriteria],
        threshold: float = 0.5,
        use_cot: bool = True,
        num_iterations: int = 1,
        **kwargs: Any,
    ):
        """
        Initialize G-Eval scorer.

        Args:
            model: LLM model to use for evaluation
            criteria: Evaluation criteria (string or GEvalCriteria object)
            threshold: Score threshold for pass/fail
            use_cot: Whether to use chain-of-thought reasoning
            num_iterations: Number of evaluation iterations for consistency
        """
        super().__init__(name="g_eval", threshold=threshold, **kwargs)
        self.threshold = threshold
        self.model = model
        self.use_cot = use_cot
        self.num_iterations = num_iterations

        if isinstance(criteria, str):
            self.criteria = GEvalCriteria(
                name="Custom Evaluation",
                description=criteria,
                steps=[
                    "Read the task description and the text to evaluate",
                    "Analyze the text based on the given criteria",
                    "Provide a score and detailed reasoning",
                ],
            )
        else:
            self.criteria = criteria

    def _build_prompt(
        self, input_text: str, output_text: str, context: Optional[str] = None
    ) -> str:
        """Build the evaluation prompt for G-Eval."""

        prompt_parts = [
            f"# Evaluation Task: {self.criteria.name}",
            f"\n## Criteria:\n{self.criteria.description}",
            f"\n## Score Range: {self.criteria.score_range[0]} to {self.criteria.score_range[1]}",
        ]

        if self.use_cot:
            prompt_parts.extend(
                [
                    "\n## Evaluation Steps:",
                    *[f"{i+1}. {step}" for i, step in enumerate(self.criteria.steps)],
                ]
            )

        prompt_parts.extend(
            [f"\n## Input:\n{input_text}", f"\n## Output to Evaluate:\n{output_text}"]
        )

        if context:
            prompt_parts.append(f"\n## Context:\n{context}")

        if self.use_cot:
            prompt_parts.extend(
                [
                    "\n## Instructions:",
                    "Please evaluate the output step by step following the evaluation steps above.",
                    "Provide your reasoning for each step, then give a final score.",
                    "\nFormat your response as:",
                    "**Step-by-step Analysis:**",
                    "[Your detailed analysis following each step]",
                    "\n**Final Score:** [score]",
                    "**Reasoning:** [brief explanation of the score]",
                ]
            )
        else:
            prompt_parts.extend(
                [
                    "\n## Instructions:",
                    f"Evaluate the output based on the criteria and provide a score from {self.criteria.score_range[0]} to {self.criteria.score_range[1]}.",
                    "\nFormat your response as:",
                    "**Score:** [score]",
                    "**Reasoning:** [explanation]",
                ]
            )

        return "\n".join(prompt_parts)

    def _parse_response(self, response: str) -> tuple[float, str]:
        """Parse the LLM response to extract score and reasoning."""
        lines = response.strip().split("\n")
        score = None
        reasoning = ""

        for line in lines:
            line = line.strip()
            if line.startswith("**Final Score:**") or line.startswith("**Score:**"):
                # Extract score from the line
                score_text = line.split(":")[-1].strip()
                try:
                    # Handle various score formats
                    if "/" in score_text:
                        score = float(score_text.split("/")[0])
                    else:
                        # Extract first number found
                        import re

                        numbers = re.findall(r"\d+\.?\d*", score_text)
                        if numbers:
                            score = float(numbers[0])
                except (ValueError, IndexError):
                    continue
            elif line.startswith("**Reasoning:**"):
                reasoning = line.split(":", 1)[-1].strip()

        # If no score found, try to extract from the entire response
        if score is None:
            import re

            numbers = re.findall(r"\b(\d+\.?\d*)\b", response)
            if numbers:
                # Take the last number as it's likely the final score
                score = float(numbers[-1])

        # Normalize score to 0-1 range
        if score is not None:
            min_score, max_score = self.criteria.score_range
            normalized_score = (score - min_score) / (max_score - min_score)
            normalized_score = max(0.0, min(1.0, normalized_score))
        else:
            normalized_score = 0.0
            reasoning = f"Failed to parse score from response: {response[:200]}..."

        return normalized_score, reasoning or response.strip()

    async def _evaluate_single(
        self, input_text: str, output_text: str, context: Optional[str] = None
    ) -> tuple[float, str]:
        """Perform a single evaluation iteration."""
        prompt = self._build_prompt(input_text, output_text, context)

        try:
            response = await self.model.generate(prompt)  # type: ignore
            score, reasoning = self._parse_response(response)
            return score, reasoning
        except Exception as e:
            return 0.0, f"Evaluation failed: {e!s}"

    async def evaluate(
        self,
        input_text: str,
        output_text: str,
        expected_output: Optional[str] = None,
        context: Optional[str] = None,
        **kwargs: Any,
    ) -> ScoreResult:
        """
        Evaluate the output using G-Eval methodology.

        Args:
            input_text: The input prompt or question
            output_text: The output to evaluate
            expected_output: Expected output (optional, can be used in criteria)
            context: Additional context for evaluation

        Returns:
            ScoreResult with score, reasoning, and metadata
        """
        scores = []
        reasonings = []

        # Run multiple iterations for consistency
        for i in range(self.num_iterations):
            score, reasoning = await self._evaluate_single(
                input_text, output_text, context
            )
            scores.append(score)
            reasonings.append(f"Iteration {i+1}: {reasoning}")

        # Calculate final score (average of iterations)
        final_score = sum(scores) / len(scores) if scores else 0.0

        # Combine reasonings
        combined_reasoning = "\n\n".join(reasonings)
        if self.num_iterations > 1:
            combined_reasoning += f"\n\nFinal Score (average): {final_score:.3f}"

        # Calculate confidence based on score variance
        if len(scores) > 1:
            variance = sum((s - final_score) ** 2 for s in scores) / len(scores)
            confidence = max(0.0, 1.0 - variance)
        else:
            confidence = 0.8  # Default confidence for single iteration

        return ScoreResult(
            score=final_score,
            passed=final_score >= self.threshold,
            reasoning=combined_reasoning,
            metadata={
                "criteria": self.criteria.name,
                "iterations": self.num_iterations,
                "individual_scores": scores,
                "confidence": confidence,
                "use_cot": self.use_cot,
                "score_range": self.criteria.score_range,
            },
        )


# Predefined G-Eval criteria for common use cases
class CommonGEvalCriteria:
    """Common G-Eval criteria for typical evaluation scenarios."""

    @staticmethod
    def correctness() -> GEvalCriteria:
        """Criteria for evaluating correctness of responses."""
        return GEvalCriteria(
            name="Correctness",
            description="Evaluate whether the response is factually correct and accurate",
            steps=[
                "Check if the response contains factual errors",
                "Verify if the information provided is accurate",
                "Assess if the response directly answers the question",
                "Consider the completeness of the answer",
            ],
        )

    @staticmethod
    def relevance() -> GEvalCriteria:
        """Criteria for evaluating relevance of responses."""
        return GEvalCriteria(
            name="Relevance",
            description="Evaluate how relevant the response is to the input question or prompt",
            steps=[
                "Identify the main topic and intent of the input",
                "Check if the response addresses the main topic",
                "Assess if the response stays on topic throughout",
                "Evaluate if the response provides useful information for the query",
            ],
        )

    @staticmethod
    def coherence() -> GEvalCriteria:
        """Criteria for evaluating coherence and logical flow."""
        return GEvalCriteria(
            name="Coherence",
            description="Evaluate the logical flow and coherence of the response",
            steps=[
                "Check if the response has a clear structure",
                "Assess if ideas flow logically from one to another",
                "Verify if the response maintains consistency throughout",
                "Evaluate if the conclusion follows from the premises",
            ],
        )

    @staticmethod
    def helpfulness() -> GEvalCriteria:
        """Criteria for evaluating helpfulness of responses."""
        return GEvalCriteria(
            name="Helpfulness",
            description="Evaluate how helpful the response is to the user",
            steps=[
                "Assess if the response provides actionable information",
                "Check if the response anticipates follow-up questions",
                "Evaluate if the response is practical and applicable",
                "Consider if the response adds value beyond the obvious",
            ],
        )
