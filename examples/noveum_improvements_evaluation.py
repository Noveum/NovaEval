#!/usr/bin/env python3
"""
Noveum.ai Improvement Evaluation System

This system evaluates LLMs to find models that produce BETTER outputs than
the original dataset responses, even when the dataset's "expected" answers
may not be optimal. Uses Panel of Judges and comparative evaluation.
"""

import json
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

from novaeval.models import AnthropicModel, OpenAIModel
from novaeval.scorers.panel_judge import (
    AggregationMethod,
    JudgeConfig,
    PanelOfJudgesScorer,
)


class ImprovementType(Enum):
    """Types of improvements we can detect."""

    ACCURACY = "accuracy"
    COMPLETENESS = "completeness"
    CLARITY = "clarity"
    HELPFULNESS = "helpfulness"
    FACTUAL_CORRECTNESS = "factual_correctness"
    RELEVANCE = "relevance"
    OVERALL_QUALITY = "overall_quality"


@dataclass
class ImprovementResult:
    """Result of comparing candidate vs baseline response."""

    winner: str  # "candidate", "baseline", or "tie"
    improvement_score: float  # How much better (0-1 scale)
    improvement_types: list[ImprovementType]
    reasoning: str
    confidence: float


class NoveumImprovementEvaluator:
    """
    Advanced evaluator that finds LLMs producing better outputs than dataset baselines.

    This evaluator:
    1. Treats dataset responses as baselines (not ground truth)
    2. Compares candidate model outputs against baselines
    3. Rewards models that produce better responses
    4. Uses Panel of Judges for objective assessment
    """

    def __init__(self):
        self.comparison_panel = self._create_comparison_panel()
        self.quality_panel = self._create_quality_assessment_panel()
        self.improvement_panel = self._create_improvement_detection_panel()

    def _create_comparison_panel(self) -> PanelOfJudgesScorer:
        """Create panel specialized in comparing two responses."""

        judges = [
            JudgeConfig(
                model=OpenAIModel(model_name="gpt-4o", temperature=0.0),
                weight=1.5,
                name="Comparison Expert",
                specialty="comparative_analysis",
            ),
            JudgeConfig(
                model=AnthropicModel(
                    model_name="claude-3-5-sonnet-20241022", temperature=0.0
                ),
                weight=1.5,
                name="Quality Assessor",
                specialty="quality_evaluation",
            ),
            JudgeConfig(
                model=OpenAIModel(model_name="o1-preview", temperature=0.0),
                weight=2.0,
                name="Reasoning Evaluator",
                specialty="logical_assessment",
            ),
        ]

        return PanelOfJudgesScorer(
            judges=judges,
            aggregation_method=AggregationMethod.WEIGHTED_MEAN,
            threshold=0.6,  # Lower threshold for detecting improvements
            require_consensus=True,
            consensus_threshold=0.7,
            evaluation_criteria="comparative quality, improvement detection, and objective assessment",
        )

    def _create_quality_assessment_panel(self) -> PanelOfJudgesScorer:
        """Create panel for absolute quality assessment."""

        judges = [
            JudgeConfig(
                model=OpenAIModel(model_name="gpt-4o", temperature=0.0),
                weight=1.0,
                name="Accuracy Judge",
                specialty="factual_accuracy",
            ),
            JudgeConfig(
                model=AnthropicModel(
                    model_name="claude-3-5-sonnet-20241022", temperature=0.0
                ),
                weight=1.0,
                name="Helpfulness Judge",
                specialty="user_helpfulness",
            ),
            JudgeConfig(
                model=OpenAIModel(model_name="gpt-4o-mini", temperature=0.1),
                weight=0.8,
                name="Completeness Judge",
                specialty="response_completeness",
            ),
        ]

        return PanelOfJudgesScorer(
            judges=judges,
            aggregation_method=AggregationMethod.MEAN,
            threshold=0.7,
            evaluation_criteria="absolute quality assessment independent of comparison",
        )

    def _create_improvement_detection_panel(self) -> PanelOfJudgesScorer:
        """Create panel specialized in detecting specific improvements."""

        judges = [
            JudgeConfig(
                model=OpenAIModel(model_name="o1-preview", temperature=0.0),
                weight=2.0,
                name="Improvement Detector",
                specialty="improvement_identification",
            ),
            JudgeConfig(
                model=AnthropicModel(
                    model_name="claude-3-5-sonnet-20241022", temperature=0.0
                ),
                weight=1.5,
                name="Enhancement Evaluator",
                specialty="enhancement_assessment",
            ),
        ]

        return PanelOfJudgesScorer(
            judges=judges,
            aggregation_method=AggregationMethod.WEIGHTED_MEAN,
            threshold=0.65,
            evaluation_criteria="specific improvement detection and categorization",
        )

    async def compare_responses(
        self,
        input_text: str,
        candidate_response: str,
        baseline_response: str,
        context: Optional[str] = None,
    ) -> ImprovementResult:
        """Compare candidate response against baseline response."""

        comparison_prompt = self._build_comparison_prompt(
            input_text, candidate_response, baseline_response, context
        )

        # Get comparison assessment
        comparison_result = await self.comparison_panel.evaluate(
            input_text=comparison_prompt,
            output_text="",  # Not used in this context
            expected_output=None,
        )

        # Parse the comparison result
        return self._parse_comparison_result(comparison_result)

    def _build_comparison_prompt(
        self,
        input_text: str,
        candidate_response: str,
        baseline_response: str,
        context: Optional[str] = None,
    ) -> str:
        """Build prompt for comparing two responses."""

        prompt_parts = [
            "You are an expert evaluator comparing two AI responses to determine which is better.",
            "",
            f"Original Question/Input: {input_text}",
            "",
            f"Response A (Baseline): {baseline_response}",
            "",
            f"Response B (Candidate): {candidate_response}",
            "",
        ]

        if context:
            prompt_parts.extend([f"Additional Context: {context}", ""])

        prompt_parts.extend(
            [
                "Compare these responses and determine:",
                "1. Which response is better overall (A, B, or Tie)",
                "2. How much better (if any) on a scale of 0-1",
                "3. What specific improvements the better response has",
                "4. Your confidence in this assessment (0-1)",
                "",
                "Evaluation Criteria:",
                "- Factual accuracy and correctness",
                "- Completeness and thoroughness",
                "- Clarity and understandability",
                "- Helpfulness to the user",
                "- Relevance to the question",
                "- Overall quality and usefulness",
                "",
                "Respond in JSON format:",
                "{",
                '  "winner": "A" | "B" | "Tie",',
                '  "improvement_score": 0.0-1.0,',
                '  "improvement_types": ["accuracy", "completeness", "clarity", "helpfulness", "factual_correctness", "relevance"],',
                '  "reasoning": "Detailed explanation of your assessment",',
                '  "confidence": 0.0-1.0',
                "}",
                "",
                "Be objective and focus on which response better serves the user's needs.",
            ]
        )

        return "\n".join(prompt_parts)

    def _parse_comparison_result(self, result) -> ImprovementResult:
        """Parse the comparison result from the panel."""

        # Extract JSON from the reasoning
        import json
        import re

        try:
            json_match = re.search(r"\{.*\}", result.reasoning, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())

                # Map winner to our format
                winner_map = {"A": "baseline", "B": "candidate", "Tie": "tie"}
                winner = winner_map.get(parsed.get("winner", "tie"), "tie")

                # Convert improvement types to enum
                improvement_types = [
                    ImprovementType(t)
                    for t in parsed.get("improvement_types", [])
                    if t in [e.value for e in ImprovementType]
                ]

                return ImprovementResult(
                    winner=winner,
                    improvement_score=parsed.get("improvement_score", 0.0),
                    improvement_types=improvement_types,
                    reasoning=parsed.get("reasoning", ""),
                    confidence=parsed.get("confidence", 0.5),
                )
        except Exception as e:
            print(f"Error parsing comparison result: {e}")

        # Fallback result
        return ImprovementResult(
            winner="tie",
            improvement_score=0.0,
            improvement_types=[],
            reasoning="Failed to parse comparison result",
            confidence=0.0,
        )


class NoveumDatasetProcessor:
    """Process datasets from Noveum.ai ai-gateway logs."""

    def __init__(self, ai_gateway_logs_path: str):
        self.logs_path = ai_gateway_logs_path

    def create_evaluation_dataset(
        self, selected_log_ids: list[str]
    ) -> list[dict[str, Any]]:
        """Create evaluation dataset from selected ai-gateway logs."""
        # In a real implementation, this would fetch from ai-gateway
        # For demo purposes, we'll create synthetic data
        selected_logs = [
            log for log in self._get_sample_logs() if log["id"] in selected_log_ids
        ]

        # Use list comprehension for better performance
        evaluation_data = [
            {
                "id": log.get("id"),
                "input": log.get("request", {}).get("prompt", ""),
                "baseline_response": log.get("response", {}).get("content", ""),
                "metadata": {
                    "model_used": log.get("model"),
                    "timestamp": log.get("timestamp"),
                    "cost": log.get("cost", 0),
                    "latency": log.get("latency", 0),
                    "tokens": log.get("tokens", {}),
                    "provider": log.get("provider"),
                    "original_log_id": log.get("id"),
                },
            }
            for log in selected_logs
        ]

        return evaluation_data

    def _get_sample_logs(self) -> list[dict[str, Any]]:
        """Returns a list of sample logs for demonstration purposes."""
        return [
            {
                "id": "log_001",
                "request": {"prompt": "What is machine learning?"},
                "response": {
                    "content": "Machine learning is AI that learns from data."
                },
                "model": "gpt-3.5-turbo",
                "provider": "openai",
                "timestamp": "2024-01-01T10:00:00Z",
                "cost": 0.002,
                "latency": 1200,
                "tokens": {"input": 15, "output": 25},
            },
            {
                "id": "log_002",
                "request": {"prompt": "Explain quantum computing"},
                "response": {
                    "content": "Quantum computing uses quantum mechanics for computation."
                },
                "model": "claude-3-sonnet",
                "provider": "anthropic",
                "timestamp": "2024-01-01T10:05:00Z",
                "cost": 0.015,
                "latency": 2100,
                "tokens": {"input": 20, "output": 30},
            },
            {
                "id": "log_003",
                "request": {"prompt": "What is the capital of France?"},
                "response": {"content": "The capital of France is Paris."},
                "model": "gpt-4o",
                "provider": "openai",
                "timestamp": "2024-01-01T10:10:00Z",
                "cost": 0.005,
                "latency": 800,
                "tokens": {"input": 10, "output": 15},
            },
            {
                "id": "log_004",
                "request": {
                    "prompt": "What is the largest planet in our solar system?"
                },
                "response": {
                    "content": "The largest planet in our solar system is Jupiter."
                },
                "model": "claude-3-5-sonnet-20241022",
                "provider": "anthropic",
                "timestamp": "2024-01-01T10:15:00Z",
                "cost": 0.020,
                "latency": 2500,
                "tokens": {"input": 25, "output": 35},
            },
            {
                "id": "log_005",
                "request": {"prompt": "What is the smallest country in Europe?"},
                "response": {
                    "content": "The smallest country in Europe is Vatican City."
                },
                "model": "o1-preview",
                "provider": "openai",
                "timestamp": "2024-01-01T10:20:00Z",
                "cost": 0.008,
                "latency": 1500,
                "tokens": {"input": 18, "output": 22},
            },
        ]


class NoveumImprovementRunner:
    """Main runner for Noveum improvement evaluation."""

    def __init__(self):
        self.evaluator = NoveumImprovementEvaluator()
        self.candidate_models = [
            OpenAIModel(model_name="gpt-4o", temperature=0.7),
            AnthropicModel(model_name="claude-3-5-sonnet-20241022", temperature=0.7),
            OpenAIModel(model_name="o1-preview", temperature=0.0),
            OpenAIModel(model_name="gpt-4o-mini", temperature=0.7),
            AnthropicModel(model_name="claude-3-5-haiku-20241022", temperature=0.7),
        ]

    async def run_improvement_evaluation(
        self,
        evaluation_data: list[dict[str, Any]],
        output_dir: str = "./results/noveum_improvement",
    ) -> dict[str, Any]:
        """Run comprehensive improvement evaluation."""

        results = {
            "model_improvements": {},
            "detailed_comparisons": [],
            "summary_metrics": {},
            "recommendations": [],
        }

        print("🚀 Starting Noveum Improvement Evaluation")
        print(
            f"📊 Evaluating {len(self.candidate_models)} models on {len(evaluation_data)} samples"
        )

        for model in self.candidate_models:
            print(f"\n🤖 Evaluating {model.name}...")

            model_results = {
                "wins": 0,
                "ties": 0,
                "losses": 0,
                "total_improvement_score": 0.0,
                "improvement_types": {},
                "detailed_results": [],
            }

            for i, sample in enumerate(evaluation_data):
                print(
                    f"  Sample {i+1}/{len(evaluation_data)}: {sample['input'][:50]}..."
                )

                # Generate candidate response
                candidate_response = model.generate(sample["input"])

                # Compare against baseline
                comparison = await self.evaluator.compare_responses(
                    input_text=sample["input"],
                    candidate_response=candidate_response,
                    baseline_response=sample["baseline_response"],
                )

                # Update model results
                if comparison.winner == "candidate":
                    model_results["wins"] += 1
                elif comparison.winner == "tie":
                    model_results["ties"] += 1
                else:
                    model_results["losses"] += 1

                model_results["total_improvement_score"] += comparison.improvement_score

                # Track improvement types
                for imp_type in comparison.improvement_types:
                    if imp_type.value not in model_results["improvement_types"]:
                        model_results["improvement_types"][imp_type.value] = 0
                    model_results["improvement_types"][imp_type.value] += 1

                # Store detailed result
                detailed_result = {
                    "sample_id": sample.get("id"),
                    "input": sample["input"],
                    "baseline_response": sample["baseline_response"],
                    "candidate_response": candidate_response,
                    "comparison": {
                        "winner": comparison.winner,
                        "improvement_score": comparison.improvement_score,
                        "improvement_types": [
                            t.value for t in comparison.improvement_types
                        ],
                        "reasoning": comparison.reasoning,
                        "confidence": comparison.confidence,
                    },
                    "metadata": sample.get("metadata", {}),
                }

                model_results["detailed_results"].append(detailed_result)
                results["detailed_comparisons"].append(
                    {"model": model.name, **detailed_result}
                )

            # Calculate summary metrics for this model
            total_samples = len(evaluation_data)
            model_results["win_rate"] = model_results["wins"] / total_samples
            model_results["improvement_rate"] = (
                model_results["wins"] + model_results["ties"]
            ) / total_samples
            model_results["average_improvement_score"] = (
                model_results["total_improvement_score"] / total_samples
            )

            results["model_improvements"][model.name] = model_results

            print(f"    Win Rate: {model_results['win_rate']:.2%}")
            print(f"    Improvement Rate: {model_results['improvement_rate']:.2%}")
            print(
                f"    Avg Improvement Score: {model_results['average_improvement_score']:.3f}"
            )

        # Generate recommendations
        results["recommendations"] = self._generate_recommendations(
            results["model_improvements"]
        )

        # Save results
        self._save_results(results, output_dir)

        return results

    def _generate_recommendations(
        self, model_improvements: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Generate recommendations based on evaluation results."""

        recommendations = []

        # Find best overall model
        best_model = max(model_improvements.items(), key=lambda x: x[1]["win_rate"])

        recommendations.append(
            {
                "type": "best_overall",
                "model": best_model[0],
                "win_rate": best_model[1]["win_rate"],
                "reasoning": f"Highest win rate of {best_model[1]['win_rate']:.2%} against baseline responses",
            }
        )

        # Find model with highest improvement scores
        best_improvement = max(
            model_improvements.items(), key=lambda x: x[1]["average_improvement_score"]
        )

        recommendations.append(
            {
                "type": "highest_improvement",
                "model": best_improvement[0],
                "improvement_score": best_improvement[1]["average_improvement_score"],
                "reasoning": f"Highest average improvement score of {best_improvement[1]['average_improvement_score']:.3f}",
            }
        )

        # Find models with specific strengths
        for model_name, results in model_improvements.items():
            for imp_type, count in results["improvement_types"].items():
                if count >= len(results["detailed_results"]) * 0.3:  # 30% threshold
                    recommendations.append(
                        {
                            "type": "specialized_strength",
                            "model": model_name,
                            "strength": imp_type,
                            "frequency": count / len(results["detailed_results"]),
                            "reasoning": f"Shows consistent improvement in {imp_type} ({count}/{len(results['detailed_results'])} cases)",
                        }
                    )

        return recommendations

    def _save_results(self, results: dict[str, Any], output_dir: str):
        """Save evaluation results to files."""

        import os

        os.makedirs(output_dir, exist_ok=True)

        # Save full results
        with open(f"{output_dir}/improvement_evaluation_results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)

        # Save summary report
        summary = {
            "model_rankings": sorted(
                [
                    (name, data["win_rate"])
                    for name, data in results["model_improvements"].items()
                ],
                key=lambda x: x[1],
                reverse=True,
            ),
            "recommendations": results["recommendations"],
            "total_evaluations": len(results["detailed_comparisons"]),
        }

        with open(f"{output_dir}/summary_report.json", "w") as f:
            json.dump(summary, f, indent=2)

        print(f"\n💾 Results saved to {output_dir}/")


# Example usage
async def main():
    """Example of how to use the Noveum improvement evaluation system."""

    # 1. Process ai-gateway logs to create evaluation dataset
    processor = NoveumDatasetProcessor("./ai_gateway_logs.jsonl")

    # Selected log IDs from Noveum.ai interface
    selected_log_ids = ["log_001", "log_002", "log_003", "log_004", "log_005"]

    evaluation_data = processor.create_evaluation_dataset(selected_log_ids)

    # 2. Run improvement evaluation
    runner = NoveumImprovementRunner()
    results = await runner.run_improvement_evaluation(evaluation_data)

    # 3. Display recommendations
    print("\n🎯 Recommendations:")
    for rec in results["recommendations"]:
        print(f"  {rec['type']}: {rec['model']} - {rec['reasoning']}")


if __name__ == "__main__":
    import asyncio

    # Create sample ai-gateway logs for testing
    sample_logs = [
        {
            "id": "log_001",
            "request": {"prompt": "What is machine learning?"},
            "response": {"content": "Machine learning is AI that learns from data."},
            "model": "gpt-3.5-turbo",
            "provider": "openai",
            "timestamp": "2024-01-01T10:00:00Z",
            "cost": 0.002,
            "latency": 1200,
            "tokens": {"input": 15, "output": 25},
        },
        {
            "id": "log_002",
            "request": {"prompt": "Explain quantum computing"},
            "response": {
                "content": "Quantum computing uses quantum mechanics for computation."
            },
            "model": "claude-3-sonnet",
            "provider": "anthropic",
            "timestamp": "2024-01-01T10:05:00Z",
            "cost": 0.015,
            "latency": 2100,
            "tokens": {"input": 20, "output": 30},
        },
    ]

    # Save sample logs
    with open("./ai_gateway_logs.jsonl", "w") as f:
        for log in sample_logs:
            f.write(json.dumps(log) + "\n")

    print("📝 Created sample ai-gateway logs")
    print("🚀 Run with: python noveum_improvement_evaluation.py")

    # Uncomment to run the evaluation
    asyncio.run(main())
