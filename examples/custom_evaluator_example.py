"""
Custom Evaluator Example for NovaEval.

This example demonstrates how to create custom evaluators by extending BaseEvaluator.
We'll implement several different evaluation strategies to show the flexibility.
"""

import json
import logging
import time
from pathlib import Path
from typing import Any, Optional, Union
import statistics

from novaeval.datasets.base import BaseDataset
from novaeval.evaluators.base import BaseEvaluator
from novaeval.models.base import BaseModel
from novaeval.scorers.base import BaseScorer
from novaeval.utils.logging import setup_logging

logger = logging.getLogger(__name__)


class SequentialEvaluator(BaseEvaluator):
    """
    Custom evaluator that processes samples sequentially instead of in parallel.
    
    Useful when you need:
    - Deterministic ordering
    - Memory-constrained environments
    - State management between samples
    """

    def __init__(
        self,
        dataset: BaseDataset,
        models: list[BaseModel],
        scorers: list[BaseScorer],
        output_dir: Optional[Union[str, Path]] = None,
        config: Optional[dict[str, Any]] = None,
        early_stopping_threshold: Optional[float] = None,
        sample_limit: Optional[int] = None,
    ):
        """
        Initialize the sequential evaluator.

        Args:
            dataset: The dataset to evaluate on
            models: List of models to evaluate
            scorers: List of scorers to use for evaluation
            output_dir: Directory to save results
            config: Additional configuration options
            early_stopping_threshold: Stop if score falls below this threshold
            sample_limit: Maximum number of samples to evaluate (for testing)
        """
        super().__init__(dataset, models, scorers, output_dir, config)
        self.early_stopping_threshold = early_stopping_threshold
        self.sample_limit = sample_limit
        
        # Setup logging
        setup_logging(
            level=self.config.get("log_level", "INFO"),
            log_file=self.output_dir / "evaluation.log",
        )

    def run(self) -> dict[str, Any]:
        """Run the complete evaluation process sequentially."""
        logger.info("Starting sequential evaluation process")
        start_time = time.time()

        # Validate inputs
        self.validate_inputs()

        # Initialize results structure
        results = {
            "metadata": {
                "evaluator_type": "sequential",
                "start_time": start_time,
                "dataset": self.dataset.get_info(),
                "models": [model.get_info() for model in self.models],
                "scorers": [scorer.get_info() for scorer in self.scorers],
                "config": self.config,
                "early_stopping_threshold": self.early_stopping_threshold,
                "sample_limit": self.sample_limit,
            },
            "model_results": {},
            "summary": {},
        }

        # Evaluate each model
        for model in self.models:
            logger.info(f"Evaluating model: {model.name}")
            model_results = self._evaluate_model_sequential(model)
            results["model_results"][model.name] = model_results
            
            # Early stopping check across models if needed
            if self._should_stop_evaluation(model_results):
                logger.info(f"Early stopping triggered for model {model.name}")
                break

        # Calculate summary statistics
        results["summary"] = self._calculate_summary(results["model_results"])

        # Add timing information
        end_time = time.time()
        results["metadata"]["end_time"] = end_time
        results["metadata"]["duration"] = end_time - start_time

        # Save results
        self.save_results(results)

        logger.info(f"Sequential evaluation completed in {end_time - start_time:.2f} seconds")
        return results

    def _evaluate_model_sequential(self, model: BaseModel) -> dict[str, Any]:
        """Evaluate a single model sequentially through all samples."""
        model_results: dict[str, Any] = {
            "samples": [],
            "scores": {},
            "errors": [],
            "stopped_early": False,
            "total_samples_processed": 0,
        }

        # Get dataset samples
        samples = list(self.dataset)
        if self.sample_limit:
            samples = samples[:self.sample_limit]
            
        logger.info(f"Processing {len(samples)} samples for {model.name}")

        # Process samples one by one
        for i, sample in enumerate(samples):
            try:
                # Evaluate single sample
                sample_result = self.evaluate_sample(sample, model, self.scorers)
                model_results["samples"].append(sample_result)
                model_results["total_samples_processed"] = i + 1
                
                # Log progress every 10 samples
                if (i + 1) % 10 == 0:
                    logger.info(f"Processed {i + 1}/{len(samples)} samples")
                
                # Early stopping check
                if self.early_stopping_threshold is not None:
                    current_scores = [s.get("scores", {}) for s in model_results["samples"]]
                    if self._should_stop_early(current_scores):
                        logger.info(f"Early stopping triggered after {i + 1} samples")
                        model_results["stopped_early"] = True
                        break
                        
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
        """Evaluate a single sample with enhanced logging and error handling."""
        sample_result = {
            "sample_id": sample.get("id", "unknown"),
            "input": sample.get("input", ""),
            "expected": sample.get("expected", ""),
            "prediction": None,
            "scores": {},
            "metadata": {},
            "error": None,
            "processing_time": 0.0,
        }

        start_time = time.time()
        
        try:
            # Generate prediction with timing
            prediction_start = time.time()
            prediction = model.generate(
                sample["input"], **sample.get("generation_kwargs", {})
            )
            prediction_time = time.time() - prediction_start
            
            sample_result["prediction"] = prediction
            sample_result["metadata"]["prediction_time"] = prediction_time

            # Apply scorers with individual timing
            scorer_times = {}
            for scorer in scorers:
                scorer_start = time.time()
                try:
                    score = scorer.score(
                        prediction=prediction,
                        ground_truth=sample.get("expected", ""),
                        context=sample,
                    )
                    sample_result["scores"][scorer.name] = score
                    scorer_times[scorer.name] = time.time() - scorer_start
                except Exception as e:
                    logger.warning(
                        f"Scorer {scorer.name} failed on sample {sample.get('id')}: {e}"
                    )
                    sample_result["scores"][scorer.name] = None
                    scorer_times[scorer.name] = time.time() - scorer_start

            # Add comprehensive metadata
            sample_result["metadata"].update({
                "model_name": model.name,
                "timestamp": time.time(),
                "scorer_times": scorer_times,
            })

        except Exception as e:
            logger.error(f"Failed to evaluate sample {sample.get('id')}: {e}")
            sample_result["error"] = str(e)

        sample_result["processing_time"] = time.time() - start_time
        return sample_result

    def _should_stop_early(self, sample_scores: list[dict]) -> bool:
        """Check if early stopping criteria are met."""
        if not sample_scores or len(sample_scores) < 5:  # Need at least 5 samples
            return False
            
        # Get the primary scorer (first one) average score
        primary_scorer = self.scorers[0].name
        scores = []
        
        for sample_score in sample_scores:
            score_value = sample_score.get(primary_scorer)
            if score_value is not None:
                if isinstance(score_value, dict) and "score" in score_value:
                    scores.append(score_value["score"])
                elif isinstance(score_value, (int, float)):
                    scores.append(score_value)
        
        if scores:
            avg_score = statistics.mean(scores)
            return avg_score < self.early_stopping_threshold
            
        return False

    def _should_stop_evaluation(self, model_results: dict[str, Any]) -> bool:
        """Check if we should stop evaluating additional models."""
        # This is a placeholder - implement your own logic
        # For example, stop if error rate is too high
        error_rate = len(model_results.get("errors", [])) / max(1, model_results.get("total_samples_processed", 1))
        return error_rate > 0.5  # Stop if more than 50% errors

    def _aggregate_scores(self, sample_results: list[dict[str, Any]]) -> dict[str, Any]:
        """Enhanced score aggregation with additional statistics."""
        aggregated = {}

        # Collect all scorer names
        scorer_names = set()
        for result in sample_results:
            scorer_names.update(result.get("scores", {}).keys())

        # Aggregate scores for each scorer
        for scorer_name in scorer_names:
            scores = []
            processing_times = []
            
            for result in sample_results:
                score = result.get("scores", {}).get(scorer_name)
                if score is not None:
                    # Handle different score types
                    if isinstance(score, (int, float)):
                        scores.append(float(score))
                    elif isinstance(score, dict) and "score" in score:
                        scores.append(float(score["score"]))
                
                # Collect processing times if available
                scorer_time = result.get("metadata", {}).get("scorer_times", {}).get(scorer_name)
                if scorer_time is not None:
                    processing_times.append(scorer_time)

            if scores:
                # Calculate comprehensive statistics
                mean = statistics.mean(scores)
                result = {
                    "mean": mean,
                    "median": statistics.median(scores),
                    "count": len(scores),
                    "min": min(scores),
                    "max": max(scores),
                    "std": statistics.stdev(scores) if len(scores) > 1 else 0.0,
                }
                
                # Add timing information
                if processing_times:
                    result["avg_processing_time"] = statistics.mean(processing_times)
                    result["total_processing_time"] = sum(processing_times)
                
                aggregated[scorer_name] = result

        return aggregated

    def _calculate_summary(self, model_results: dict[str, Any]) -> dict[str, Any]:
        """Calculate enhanced summary statistics."""
        summary = {
            "total_models": len(model_results),
            "total_samples": 0,
            "total_errors": 0,
            "models_stopped_early": 0,
            "average_processing_time": 0.0,
            "best_model": {},
        }

        total_processing_time = 0.0
        sample_count = 0

        for model_name, results in model_results.items():
            if isinstance(results, dict):
                summary["total_samples"] += results.get("total_samples_processed", 0)
                summary["total_errors"] += len(results.get("errors", []))
                
                if results.get("stopped_early", False):
                    summary["models_stopped_early"] += 1
                
                # Calculate processing times
                for sample in results.get("samples", []):
                    total_processing_time += sample.get("processing_time", 0.0)
                    sample_count += 1

        if sample_count > 0:
            summary["average_processing_time"] = total_processing_time / sample_count

        # Find best model for each scorer (same logic as standard evaluator)
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
        """Save results with enhanced format."""
        # Save JSON results
        results_file = self.output_dir / "results.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2, default=str)

        # Save detailed CSV
        self._save_detailed_csv(results)
        
        # Save summary report
        self._save_summary_report(results)

        logger.info(f"Results saved to {self.output_dir}")

    def _save_detailed_csv(self, results: dict[str, Any]) -> None:
        """Save detailed CSV with processing times."""
        try:
            import pandas as pd
            
            rows = []
            for model_name, model_results in results["model_results"].items():
                for sample in model_results.get("samples", []):
                    row = {
                        "model": model_name,
                        "sample_id": sample.get("sample_id", "unknown"),
                        "input": sample.get("input", ""),
                        "expected": sample.get("expected", ""),
                        "prediction": sample.get("prediction", ""),
                        "processing_time": sample.get("processing_time", 0.0),
                        "prediction_time": sample.get("metadata", {}).get("prediction_time", 0.0),
                    }
                    
                    # Add scores
                    for scorer_name, score in sample.get("scores", {}).items():
                        if isinstance(score, dict) and "score" in score:
                            row[f"score_{scorer_name}"] = score["score"]
                        else:
                            row[f"score_{scorer_name}"] = score
                    
                    # Add scorer processing times
                    scorer_times = sample.get("metadata", {}).get("scorer_times", {})
                    for scorer_name, scorer_time in scorer_times.items():
                        row[f"time_{scorer_name}"] = scorer_time
                    
                    rows.append(row)

            if rows:
                df = pd.DataFrame(rows)
                csv_file = self.output_dir / "detailed_results.csv"
                df.to_csv(csv_file, index=False)
                
        except ImportError:
            logger.warning("Pandas not available, skipping CSV export")

    def _save_summary_report(self, results: dict[str, Any]) -> None:
        """Save a human-readable summary report."""
        summary_file = self.output_dir / "summary_report.txt"
        
        with open(summary_file, "w") as f:
            f.write("NovaEval Sequential Evaluation Summary\n")
            f.write("=" * 40 + "\n\n")
            
            metadata = results.get("metadata", {})
            f.write(f"Evaluation Type: {metadata.get('evaluator_type', 'Unknown')}\n")
            f.write(f"Duration: {metadata.get('duration', 0):.2f} seconds\n")
            f.write(f"Dataset: {metadata.get('dataset', {}).get('name', 'Unknown')}\n\n")
            
            summary = results.get("summary", {})
            f.write(f"Models Evaluated: {summary.get('total_models', 0)}\n")
            f.write(f"Total Samples: {summary.get('total_samples', 0)}\n")
            f.write(f"Total Errors: {summary.get('total_errors', 0)}\n")
            f.write(f"Models Stopped Early: {summary.get('models_stopped_early', 0)}\n")
            f.write(f"Average Processing Time: {summary.get('average_processing_time', 0):.3f}s per sample\n\n")
            
            # Best models section
            best_models = summary.get("best_model", {})
            if best_models:
                f.write("Best Models by Scorer:\n")
                for scorer, info in best_models.items():
                    f.write(f"  {scorer}: {info.get('model', 'Unknown')} "
                           f"(score: {info.get('score', 0):.3f})\n")


class BatchEvaluator(BaseEvaluator):
    """
    Custom evaluator that processes samples in batches.
    
    Useful for:
    - Models that support batch inference
    - Optimizing API calls
    - Memory management with large datasets
    """
    
    def __init__(
        self,
        dataset: BaseDataset,
        models: list[BaseModel],
        scorers: list[BaseScorer],
        output_dir: Optional[Union[str, Path]] = None,
        config: Optional[dict[str, Any]] = None,
        batch_size: int = 10,
    ):
        super().__init__(dataset, models, scorers, output_dir, config)
        self.batch_size = batch_size
        
        setup_logging(
            level=self.config.get("log_level", "INFO"),
            log_file=self.output_dir / "evaluation.log",
        )

    def run(self) -> dict[str, Any]:
        """Run evaluation processing samples in batches."""
        logger.info(f"Starting batch evaluation (batch_size={self.batch_size})")
        start_time = time.time()
        
        self.validate_inputs()
        
        results = {
            "metadata": {
                "evaluator_type": "batch",
                "batch_size": self.batch_size,
                "start_time": start_time,
                "dataset": self.dataset.get_info(),
                "models": [model.get_info() for model in self.models],
                "scorers": [scorer.get_info() for scorer in self.scorers],
            },
            "model_results": {},
            "summary": {},
        }
        
        # Evaluate each model
        for model in self.models:
            logger.info(f"Evaluating model: {model.name}")
            model_results = self._evaluate_model_batched(model)
            results["model_results"][model.name] = model_results
        
        results["summary"] = self._calculate_summary(results["model_results"])
        
        end_time = time.time()
        results["metadata"]["end_time"] = end_time
        results["metadata"]["duration"] = end_time - start_time
        
        self.save_results(results)
        
        logger.info(f"Batch evaluation completed in {end_time - start_time:.2f} seconds")
        return results

    def _evaluate_model_batched(self, model: BaseModel) -> dict[str, Any]:
        """Evaluate model using batch processing."""
        model_results = {
            "samples": [],
            "scores": {},
            "errors": [],
            "batches_processed": 0,
        }
        
        samples = list(self.dataset)
        
        # Process in batches
        for i in range(0, len(samples), self.batch_size):
            batch = samples[i:i + self.batch_size]
            logger.info(f"Processing batch {i//self.batch_size + 1}/{(len(samples) + self.batch_size - 1)//self.batch_size}")
            
            batch_results = self._process_batch(batch, model)
            model_results["samples"].extend(batch_results)
            model_results["batches_processed"] += 1
        
        model_results["scores"] = self._aggregate_scores(model_results["samples"])
        return model_results

    def _process_batch(self, batch: list[dict], model: BaseModel) -> list[dict]:
        """Process a batch of samples."""
        batch_results = []
        
        for sample in batch:
            try:
                result = self.evaluate_sample(sample, model, self.scorers)
                batch_results.append(result)
            except Exception as e:
                logger.error(f"Error processing sample {sample.get('id')}: {e}")
                # Add error result
                batch_results.append({
                    "sample_id": sample.get("id", "unknown"),
                    "error": str(e),
                    "scores": {},
                })
        
        return batch_results

    def evaluate_sample(self, sample: dict[str, Any], model: BaseModel, scorers: list[BaseScorer]) -> dict[str, Any]:
        """Standard sample evaluation (same as sequential evaluator)."""
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
                    logger.warning(f"Scorer {scorer.name} failed: {e}")
                    sample_result["scores"][scorer.name] = None

            sample_result["metadata"] = {
                "model_name": model.name,
                "timestamp": time.time(),
            }

        except Exception as e:
            logger.error(f"Failed to evaluate sample {sample.get('id')}: {e}")
            sample_result["error"] = str(e)

        return sample_result

    def _aggregate_scores(self, sample_results: list[dict[str, Any]]) -> dict[str, Any]:
        """Same aggregation logic as sequential evaluator."""
        # Implementation would be the same as SequentialEvaluator._aggregate_scores
        # Omitted for brevity - copy from above
        return {}

    def _calculate_summary(self, model_results: dict[str, Any]) -> dict[str, Any]:
        """Calculate summary with batch-specific metrics."""
        return {
            "total_models": len(model_results),
            "batch_size_used": self.batch_size,
            # Add other summary statistics
        }

    def save_results(self, results: dict[str, Any]) -> None:
        """Save batch evaluation results."""
        results_file = self.output_dir / "results.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2, default=str)


# Example usage
if __name__ == "__main__":
    # This is just a template - you'd need to import and instantiate
    # your actual dataset, models, and scorers
    
    # Example of how to use the custom evaluators:
    """
    from novaeval.datasets.custom import CustomDataset
    from novaeval.models.openai import OpenAIModel
    from novaeval.scorers.accuracy import ExactMatchScorer
    
    # Setup components
    dataset = CustomDataset("path/to/data.jsonl")
    models = [OpenAIModel("gpt-4", api_key="your-key")]
    scorers = [ExactMatchScorer()]
    
    # Use sequential evaluator with early stopping
    evaluator = SequentialEvaluator(
        dataset=dataset,
        models=models,
        scorers=scorers,
        output_dir="./results",
        early_stopping_threshold=0.5,
        sample_limit=100
    )
    
    results = evaluator.run()
    print(f"Evaluation completed: {results['summary']}")
    
    # Or use batch evaluator
    batch_evaluator = BatchEvaluator(
        dataset=dataset,
        models=models,
        scorers=scorers,
        batch_size=5
    )
    
    batch_results = batch_evaluator.run()
    """ 