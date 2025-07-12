"""
Standard evaluator implementation for NovaEval.

This module provides the main evaluator class that orchestrates
the evaluation process.
"""

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Optional, Union

import pandas as pd
from tqdm import tqdm

from novaeval.datasets.base import BaseDataset
from novaeval.evaluators.base import BaseEvaluator
from novaeval.models.base import BaseModel
from novaeval.scorers.base import BaseScorer
from novaeval.utils.config import Config
from novaeval.utils.logging import setup_logging

logger = logging.getLogger(__name__)


class Evaluator(BaseEvaluator):
    """
    Standard evaluator implementation.

    This class provides the main evaluation logic for running
    evaluations across datasets, models, and scorers.
    """

    def __init__(
        self,
        dataset: BaseDataset,
        models: list[BaseModel],
        scorers: list[BaseScorer],
        output_dir: Optional[Union[str, Path]] = None,
        config: Optional[dict[str, Any]] = None,
        max_workers: int = 4,
        batch_size: int = 1,
    ):
        """
        Initialize the standard evaluator.

        Args:
            dataset: The dataset to evaluate on
            models: List of models to evaluate
            scorers: List of scorers to use for evaluation
            output_dir: Directory to save results
            config: Additional configuration options
            max_workers: Maximum number of worker threads
            batch_size: Batch size for processing samples
        """
        super().__init__(dataset, models, scorers, output_dir, config)
        self.max_workers = max_workers
        self.batch_size = batch_size
        # TODO: Implement ReportGenerator
        # self.report_generator = ReportGenerator(self.output_dir)

        # Setup logging
        setup_logging(
            level=self.config.get("log_level", "INFO"),
            log_file=self.output_dir / "evaluation.log",
        )

    def run(self) -> dict[str, Any]:
        """
        Run the complete evaluation process.

        Returns:
            Dictionary containing aggregated evaluation results
        """
        logger.info("Starting evaluation process")
        start_time = time.time()

        # Validate inputs
        self.validate_inputs()

        # Initialize results structure
        results = {
            "metadata": {
                "start_time": start_time,
                "dataset": self.dataset.get_info(),
                "models": [model.get_info() for model in self.models],
                "scorers": [scorer.get_info() for scorer in self.scorers],
                "config": self.config,
            },
            "model_results": {},
            "summary": {},
        }

        # Evaluate each model
        for model in self.models:
            logger.info(f"Evaluating model: {model.name}")
            model_results = self._evaluate_model(model)
            results["model_results"][model.name] = model_results

        # Calculate summary statistics
        results["summary"] = self._calculate_summary(results["model_results"])

        # Add timing information
        end_time = time.time()
        results["metadata"]["end_time"] = end_time
        results["metadata"]["duration"] = end_time - start_time

        # Save results
        self.save_results(results)

        # TODO: Generate reports when ReportGenerator is implemented
        # self.report_generator.generate_all_reports(results)

        logger.info(f"Evaluation completed in {end_time - start_time:.2f} seconds")
        return results

    def _evaluate_model(self, model: BaseModel) -> dict[str, Any]:
        """
        Evaluate a single model on the dataset.

        Args:
            model: The model to evaluate

        Returns:
            Dictionary containing model evaluation results
        """
        model_results: dict[str, Any] = {
            "samples": [],
            "scores": {},
            "errors": [],
        }

        # Get dataset samples
        samples = list(self.dataset)

        # Process samples in batches
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit evaluation tasks
            future_to_sample = {
                executor.submit(
                    self.evaluate_sample, sample, model, self.scorers
                ): sample
                for sample in samples
            }

            # Collect results with progress bar
            for future in tqdm(
                as_completed(future_to_sample),
                total=len(samples),
                desc=f"Evaluating {model.name}",
            ):
                sample = future_to_sample[future]
                try:
                    sample_result = future.result()
                    model_results["samples"].append(sample_result)
                except Exception as e:
                    error_info = {
                        "sample_id": sample.get("id", "unknown"),
                        "error": str(e),
                        "error_type": type(e).__name__,
                    }
                    model_results["errors"].append(error_info)
                    logger.error(f"Error evaluating sample {sample.get('id')}: {e}")

        # Aggregate scores
        model_results["scores"] = self._aggregate_scores(model_results["samples"])

        return model_results

    def evaluate_sample(
        self, sample: dict[str, Any], model: BaseModel, scorers: list[BaseScorer]
    ) -> dict[str, Any]:
        """
        Evaluate a single sample with a model.

        Args:
            sample: The sample to evaluate
            model: The model to use for evaluation
            scorers: List of scorers to apply

        Returns:
            Dictionary containing sample evaluation results
        """
        sample_result = {
            "sample_id": sample.get("id", "unknown"),
            "input": sample.get("input", ""),
            "expected": sample.get("expected", ""),
            "prediction": None,
            "scores": {},
            "metadata": {},
        }

        try:
            # Generate prediction
            prediction = model.generate(
                sample["input"], **sample.get("generation_kwargs", {})
            )
            sample_result["prediction"] = prediction

            # Apply scorers
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

            # Add metadata
            sample_result["metadata"] = {
                "model_name": model.name,
                "timestamp": time.time(),
            }

        except Exception as e:
            logger.error(f"Failed to evaluate sample {sample.get('id')}: {e}")
            sample_result["error"] = str(e)

        return sample_result

    def _aggregate_scores(self, sample_results: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Aggregate scores across all samples.

        Args:
            sample_results: List of sample evaluation results

        Returns:
            Dictionary containing aggregated scores
        """
        aggregated = {}

        # Collect all scorer names
        scorer_names = set()
        for result in sample_results:
            scorer_names.update(result.get("scores", {}).keys())

        # Aggregate scores for each scorer
        for scorer_name in scorer_names:
            scores = []
            for result in sample_results:
                score = result.get("scores", {}).get(scorer_name)
                if score is not None:
                    scores.append(score)

            if scores:
                # Handle different score types
                numeric_scores = []
                detailed_scores = {}

                for score in scores:
                    if isinstance(score, (int, float)):
                        # Simple numeric score
                        numeric_scores.append(float(score))
                    elif isinstance(score, dict):
                        # Dict score (e.g., from F1Scorer)
                        if "score" in score:
                            # Use the main score for aggregation
                            numeric_scores.append(float(score["score"]))
                        else:
                            # Use the first numeric value
                            numeric_values = [
                                v for v in score.values() if isinstance(v, (int, float))
                            ]
                            if numeric_values:
                                numeric_scores.append(float(numeric_values[0]))
                            else:
                                numeric_scores.append(0.0)

                        # Collect detailed scores for dict-based scorers
                        for key, value in score.items():
                            if isinstance(value, (int, float)):
                                if key not in detailed_scores:
                                    detailed_scores[key] = []
                                detailed_scores[key].append(float(value))
                    else:
                        # Unknown score type, convert to float or use 0
                        try:
                            numeric_scores.append(float(score))
                        except (ValueError, TypeError):
                            numeric_scores.append(0.0)

                if numeric_scores:
                    # Calculate basic statistics
                    result = {
                        "mean": sum(numeric_scores) / len(numeric_scores),
                        "count": len(numeric_scores),
                        "min": min(numeric_scores),
                        "max": max(numeric_scores),
                    }

                    # Add detailed scores if available (for dict-based scorers)
                    if detailed_scores:
                        for key, values in detailed_scores.items():
                            result[f"{key}_mean"] = sum(values) / len(values)
                            result[f"{key}_min"] = min(values)
                            result[f"{key}_max"] = max(values)

                    aggregated[scorer_name] = result

        return aggregated

    def _calculate_summary(self, model_results: dict[str, Any]) -> dict[str, Any]:
        """
        Calculate summary statistics across all models.

        Args:
            model_results: Results for all models

        Returns:
            Dictionary containing summary statistics
        """
        summary: dict[str, Any] = {
            "total_models": len(model_results),
            "total_samples": 0,
            "total_errors": 0,
            "best_model": {},
        }

        # Calculate totals
        for _model_name, results in model_results.items():
            if isinstance(results, dict):
                summary["total_samples"] += len(results.get("samples", []))
                summary["total_errors"] += len(results.get("errors", []))

        # Find best model for each scorer
        for model_name, results in model_results.items():
            if isinstance(results, dict):
                for scorer_name, score_info in results.get("scores", {}).items():
                    if (
                        isinstance(score_info, dict)
                        and "mean" in score_info
                        and (
                            scorer_name not in summary["best_model"]
                            or score_info["mean"]
                            > summary["best_model"][scorer_name]["score"]
                        )
                    ):
                        summary["best_model"][scorer_name] = {
                            "model": model_name,
                            "score": score_info["mean"],
                        }

        return summary

    def save_results(self, results: dict[str, Any]) -> None:
        """
        Save evaluation results to disk.

        Args:
            results: The results to save
        """
        # Save JSON results
        results_file = self.output_dir / "results.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2, default=str)

        # Save CSV results for easy analysis
        self._save_csv_results(results)

        logger.info(f"Results saved to {self.output_dir}")

    def _save_csv_results(self, results: dict[str, Any]) -> None:
        """
        Save results in CSV format for easy analysis.

        Args:
            results: The results to save
        """
        # Flatten sample results for CSV
        rows = []
        for model_name, model_results in results["model_results"].items():
            for sample in model_results["samples"]:
                row = {
                    "model": model_name,
                    "sample_id": sample["sample_id"],
                    "input": sample["input"],
                    "expected": sample["expected"],
                    "prediction": sample["prediction"],
                }
                # Add scores as columns
                for scorer_name, score in sample.get("scores", {}).items():
                    row[f"score_{scorer_name}"] = score
                rows.append(row)

        if rows:
            df = pd.DataFrame(rows)
            csv_file = self.output_dir / "detailed_results.csv"
            df.to_csv(csv_file, index=False)

    @classmethod
    def from_config(cls, config_path: Union[str, Path]) -> "Evaluator":
        """
        Create an evaluator from a configuration file.

        Args:
            config_path: Path to the configuration file

        Returns:
            Configured evaluator instance
        """
        Config.load(config_path)

        # This would be implemented to parse the config and create
        # the appropriate dataset, models, and scorers
        # For now, this is a placeholder
        raise NotImplementedError(
            "Configuration-based initialization not yet implemented"
        )
