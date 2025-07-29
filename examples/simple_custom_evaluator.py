"""
Simple Custom Evaluator Example for NovaEval.

This example shows how to create a basic custom evaluator by extending BaseEvaluator.
"""

import json
import logging
import time
from pathlib import Path
from typing import Any, Optional, Union

from novaeval.datasets.base import BaseDataset
from novaeval.evaluators.base import BaseEvaluator
from novaeval.models.base import BaseModel
from novaeval.scorers.base import BaseScorer

logger = logging.getLogger(__name__)


class SimpleCustomEvaluator(BaseEvaluator):
    """
    A simple custom evaluator that adds logging and custom result formatting.
    
    This example demonstrates the basic structure needed to extend BaseEvaluator.
    """

    def __init__(
        self,
        dataset: BaseDataset,
        models: list[BaseModel],
        scorers: list[BaseScorer],
        output_dir: Optional[Union[str, Path]] = None,
        config: Optional[dict[str, Any]] = None,
        verbose: bool = True,
    ):
        """
        Initialize the custom evaluator.

        Args:
            dataset: The dataset to evaluate on
            models: List of models to evaluate
            scorers: List of scorers to use for evaluation
            output_dir: Directory to save results
            config: Additional configuration options
            verbose: Whether to log detailed progress
        """
        super().__init__(dataset, models, scorers, output_dir, config)
        self.verbose = verbose

    def run(self) -> dict[str, Any]:
        """
        Run the complete evaluation process.
        
        This is the main method that orchestrates the evaluation.
        """
        if self.verbose:
            logger.info("Starting custom evaluation process")
        
        start_time = time.time()

        # Step 1: Validate inputs
        self.validate_inputs()

        # Step 2: Initialize results structure
        results = {
            "metadata": {
                "evaluator_type": "simple_custom",
                "start_time": start_time,
                "dataset": self.dataset.get_info(),
                "models": [model.get_info() for model in self.models],
                "scorers": [scorer.get_info() for scorer in self.scorers],
                "config": self.config,
            },
            "model_results": {},
            "summary": {},
        }

        # Step 3: Evaluate each model
        for i, model in enumerate(self.models):
            if self.verbose:
                logger.info(f"Evaluating model {i+1}/{len(self.models)}: {model.name}")
            
            model_results = self._evaluate_model(model)
            results["model_results"][model.name] = model_results

        # Step 4: Calculate summary statistics
        results["summary"] = self._calculate_custom_summary(results["model_results"])

        # Step 5: Add timing information
        end_time = time.time()
        results["metadata"]["end_time"] = end_time
        results["metadata"]["duration"] = end_time - start_time

        # Step 6: Save results
        self.save_results(results)

        if self.verbose:
            logger.info(f"Custom evaluation completed in {end_time - start_time:.2f} seconds")
        
        return results

    def _evaluate_model(self, model: BaseModel) -> dict[str, Any]:
        """
        Evaluate a single model on all dataset samples.
        
        You can customize this method to change how models are evaluated.
        """
        model_results = {
            "samples": [],
            "scores": {},
            "errors": [],
            "sample_count": 0,
        }

        # Get all samples from the dataset
        samples = list(self.dataset)
        total_samples = len(samples)

        if self.verbose:
            logger.info(f"Processing {total_samples} samples for {model.name}")

        # Evaluate each sample
        for i, sample in enumerate(samples):
            try:
                # Evaluate the sample
                sample_result = self.evaluate_sample(sample, model, self.scorers)
                model_results["samples"].append(sample_result)
                model_results["sample_count"] += 1
                
                # Log progress every 25%
                if self.verbose and (i + 1) % max(1, total_samples // 4) == 0:
                    progress = (i + 1) / total_samples * 100
                    logger.info(f"Progress: {progress:.1f}% ({i + 1}/{total_samples})")
                    
            except Exception as e:
                error_info = {
                    "sample_id": sample.get("id", "unknown"),
                    "error": str(e),
                    "error_type": type(e).__name__,
                }
                model_results["errors"].append(error_info)
                logger.error(f"Error evaluating sample {sample.get('id')}: {e}")

        # Aggregate scores across all samples
        model_results["scores"] = self._aggregate_scores(model_results["samples"])

        return model_results

    def evaluate_sample(
        self, sample: dict[str, Any], model: BaseModel, scorers: list[BaseScorer]
    ) -> dict[str, Any]:
        """
        Evaluate a single sample with a model.
        
        This method defines how individual samples are processed.
        You can customize this to add your own processing logic.
        """
        sample_result = {
            "sample_id": sample.get("id", "unknown"),
            "input": sample.get("input", ""),
            "expected": sample.get("expected", ""),
            "prediction": None,
            "scores": {},
            "metadata": {},
            "error": None,
        }

        try:
            # Step 1: Generate prediction from the model
            prediction = model.generate(
                sample["input"], 
                **sample.get("generation_kwargs", {})
            )
            sample_result["prediction"] = prediction

            # Step 2: Apply all scorers to the prediction
            for scorer in scorers:
                try:
                    score = scorer.score(
                        prediction=prediction,
                        ground_truth=sample.get("expected", ""),
                        context=sample,
                    )
                    sample_result["scores"][scorer.name] = score
                    
                except Exception as e:
                    logger.warning(
                        f"Scorer {scorer.name} failed on sample {sample.get('id')}: {e}"
                    )
                    sample_result["scores"][scorer.name] = None

            # Step 3: Add metadata
            sample_result["metadata"] = {
                "model_name": model.name,
                "timestamp": time.time(),
                "input_length": len(sample.get("input", "")),
                "prediction_length": len(prediction) if prediction else 0,
            }

        except Exception as e:
            logger.error(f"Failed to evaluate sample {sample.get('id')}: {e}")
            sample_result["error"] = str(e)

        return sample_result

    def _aggregate_scores(self, sample_results: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Aggregate scores across all samples for a model.
        
        Customize this method to change how scores are aggregated.
        """
        aggregated = {}

        # Collect all scorer names
        scorer_names = set()
        for result in sample_results:
            scorer_names.update(result.get("scores", {}).keys())

        # Calculate statistics for each scorer
        for scorer_name in scorer_names:
            scores = []
            
            # Collect valid scores
            for result in sample_results:
                score = result.get("scores", {}).get(scorer_name)
                if score is not None:
                    # Handle different score types
                    if isinstance(score, (int, float)):
                        scores.append(float(score))
                    elif isinstance(score, dict) and "score" in score:
                        scores.append(float(score["score"]))

            # Calculate statistics if we have scores
            if scores:
                mean = sum(scores) / len(scores)
                std = 0.0
                if len(scores) > 1:
                    variance = sum((x - mean) ** 2 for x in scores) / (len(scores) - 1)
                    std = variance ** 0.5

                aggregated[scorer_name] = {
                    "mean": mean,
                    "std": std,
                    "min": min(scores),
                    "max": max(scores),
                    "count": len(scores),
                }

        return aggregated

    def _calculate_custom_summary(self, model_results: dict[str, Any]) -> dict[str, Any]:
        """
        Calculate custom summary statistics.
        
        This adds some custom metrics beyond the standard summary.
        """
        summary = {
            "total_models": len(model_results),
            "total_samples": 0,
            "total_errors": 0,
            "average_error_rate": 0.0,
            "best_model": {},
            "model_comparison": {},
        }

        # Calculate basic totals
        for model_name, results in model_results.items():
            if isinstance(results, dict):
                sample_count = results.get("sample_count", 0)
                error_count = len(results.get("errors", []))
                
                summary["total_samples"] += sample_count
                summary["total_errors"] += error_count
                
                # Calculate error rate for this model
                error_rate = error_count / max(1, sample_count)
                summary["model_comparison"][model_name] = {
                    "samples_processed": sample_count,
                    "errors": error_count,
                    "error_rate": error_rate,
                }

        # Calculate average error rate
        if summary["total_samples"] > 0:
            summary["average_error_rate"] = summary["total_errors"] / summary["total_samples"]

        # Find best model for each scorer
        for model_name, results in model_results.items():
            if isinstance(results, dict):
                for scorer_name, score_info in results.get("scores", {}).items():
                    if isinstance(score_info, dict) and "mean" in score_info:
                        if (
                            scorer_name not in summary["best_model"]
                            or score_info["mean"] > summary["best_model"][scorer_name]["score"]
                        ):
                            summary["best_model"][scorer_name] = {
                                "model": model_name,
                                "score": score_info["mean"],
                            }

        return summary

    def save_results(self, results: dict[str, Any]) -> None:
        """
        Save evaluation results to disk.
        
        Customize this method to change how results are saved.
        """
        # Save main results as JSON
        results_file = self.output_dir / "results.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2, default=str)

        # Save a custom summary report
        self._save_custom_report(results)

        if self.verbose:
            logger.info(f"Results saved to {self.output_dir}")

    def _save_custom_report(self, results: dict[str, Any]) -> None:
        """Save a human-readable custom report."""
        report_file = self.output_dir / "custom_report.txt"
        
        with open(report_file, "w") as f:
            f.write("Custom Evaluation Report\n")
            f.write("=" * 25 + "\n\n")
            
            # Basic info
            metadata = results.get("metadata", {})
            f.write(f"Duration: {metadata.get('duration', 0):.2f} seconds\n")
            f.write(f"Models: {len(self.models)}\n")
            f.write(f"Scorers: {len(self.scorers)}\n\n")
            
            # Summary statistics
            summary = results.get("summary", {})
            f.write(f"Total Samples: {summary.get('total_samples', 0)}\n")
            f.write(f"Total Errors: {summary.get('total_errors', 0)}\n")
            f.write(f"Average Error Rate: {summary.get('average_error_rate', 0):.2%}\n\n")
            
            # Model comparison
            f.write("Model Performance:\n")
            model_comparison = summary.get("model_comparison", {})
            for model_name, stats in model_comparison.items():
                f.write(f"  {model_name}:\n")
                f.write(f"    Samples: {stats.get('samples_processed', 0)}\n")
                f.write(f"    Errors: {stats.get('errors', 0)}\n")
                f.write(f"    Error Rate: {stats.get('error_rate', 0):.2%}\n")
            
            # Best models
            f.write("\nBest Models by Scorer:\n")
            best_models = summary.get("best_model", {})
            for scorer, info in best_models.items():
                f.write(f"  {scorer}: {info.get('model', 'Unknown')} ")
                f.write(f"(score: {info.get('score', 0):.3f})\n")


# Example usage
if __name__ == "__main__":
    """
    Example of how to use the custom evaluator:
    
    from novaeval.datasets.custom import CustomDataset
    from novaeval.models.openai import OpenAIModel  
    from novaeval.scorers.accuracy import ExactMatchScorer
    
    # Setup components
    dataset = CustomDataset("path/to/data.jsonl")
    models = [OpenAIModel("gpt-4", api_key="your-key")]
    scorers = [ExactMatchScorer()]
    
    # Create and run custom evaluator
    evaluator = SimpleCustomEvaluator(
        dataset=dataset,
        models=models,
        scorers=scorers,
        output_dir="./custom_results",
        verbose=True
    )
    
    results = evaluator.run()
    print(f"Evaluation completed!")
    """
    print("This is a template file. See the docstring for usage examples.") 