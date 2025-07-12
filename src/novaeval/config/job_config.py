"""
Job configuration and runner for YAML-based evaluation jobs.

This module provides the core functionality for running evaluation jobs
defined in YAML configuration files, making NovaEval suitable for CI/CD pipelines.
"""

import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional, Union

import yaml
from pydantic import ValidationError

from novaeval.config.schema import (
    DatasetType,
    EvaluationJobConfig,
    ModelProvider,
    ScorerType,
)
from novaeval.datasets.base import BaseDataset
from novaeval.evaluators.standard import Evaluator
from novaeval.models.base import BaseModel
from novaeval.scorers.base import BaseScorer
from novaeval.utils.logging import get_logger

logger = get_logger(__name__)


class JobConfigLoader:
    """Loads and validates evaluation job configurations from YAML files."""

    @staticmethod
    def load_from_file(config_path: Union[str, Path]) -> EvaluationJobConfig:
        """Load configuration from YAML file."""

        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        try:
            with open(config_path, encoding="utf-8") as f:
                config_dict = yaml.safe_load(f)

            # Validate and parse configuration
            return EvaluationJobConfig(**config_dict)

        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in configuration file: {e}")
        except ValidationError as e:
            raise ValueError(f"Invalid configuration: {e}")

    @staticmethod
    def load_from_dict(config_dict: dict[str, Any]) -> EvaluationJobConfig:
        """Load configuration from dictionary."""
        try:
            return EvaluationJobConfig(**config_dict)
        except ValidationError as e:
            raise ValueError(f"Invalid configuration: {e}")


class ModelFactory:
    """Factory for creating model instances from configuration."""

    @staticmethod
    def create_model(model_config: Any) -> BaseModel:
        """Create model instance from configuration."""

        provider = model_config.provider

        if provider == ModelProvider.OPENAI:
            from novaeval.models.openai import OpenAIModel

            return OpenAIModel(
                model_name=model_config.model_name,
                api_key=model_config.api_key or os.getenv("OPENAI_API_KEY"),
                api_base=model_config.api_base,
                temperature=model_config.temperature,
                max_tokens=model_config.max_tokens,
                timeout=model_config.timeout,
                retry_attempts=model_config.retry_attempts,
                **model_config.additional_params,
            )

        elif provider == ModelProvider.ANTHROPIC:
            from novaeval.models.anthropic import AnthropicModel

            return AnthropicModel(
                model_name=model_config.model_name,
                api_key=model_config.api_key or os.getenv("ANTHROPIC_API_KEY"),
                temperature=model_config.temperature,
                max_tokens=model_config.max_tokens,
                timeout=model_config.timeout,
                retry_attempts=model_config.retry_attempts,
                **model_config.additional_params,
            )

        elif provider == ModelProvider.NOVEUM:
            # Note: NoveumModel not implemented yet, falling back to OpenAI model
            from novaeval.models.openai import OpenAIModel

            return OpenAIModel(
                model_name=model_config.model_name,
                api_key=model_config.api_key or os.getenv("NOVEUM_API_KEY"),
                api_base=model_config.api_base or os.getenv("NOVEUM_API_BASE"),
                temperature=model_config.temperature,
                max_tokens=model_config.max_tokens,
                timeout=model_config.timeout,
                retry_attempts=model_config.retry_attempts,
                **model_config.additional_params,
            )

        else:
            raise ValueError(f"Unsupported model provider: {provider}")


class DatasetFactory:
    """Factory for creating dataset instances from configuration."""

    @staticmethod
    def create_dataset(dataset_config: Any) -> BaseDataset:
        """Create dataset instance from configuration."""

        dataset_type = dataset_config.type

        if dataset_type == DatasetType.MMLU:
            from novaeval.datasets.mmlu import MMLUDataset

            return MMLUDataset(
                subset=dataset_config.subset,
                split=dataset_config.split,
                limit=dataset_config.limit,
                shuffle=dataset_config.shuffle,
                seed=dataset_config.seed,
            )

        elif dataset_type == DatasetType.HUGGINGFACE:
            from novaeval.datasets.huggingface import HuggingFaceDataset

            return HuggingFaceDataset(
                dataset_name=dataset_config.name,
                subset=dataset_config.subset,
                split=dataset_config.split,
                limit=dataset_config.limit,
                shuffle=dataset_config.shuffle,
                seed=dataset_config.seed,
            )

        elif dataset_type in [
            DatasetType.CUSTOM,
            DatasetType.JSON,
            DatasetType.CSV,
            DatasetType.JSONL,
        ]:
            from novaeval.datasets.custom import CustomDataset

            return CustomDataset(
                data_source=dataset_config.path,
                format=dataset_type.value,
                limit=dataset_config.limit,
                shuffle=dataset_config.shuffle,
                seed=dataset_config.seed,
                preprocessing=dataset_config.preprocessing,
            )

        else:
            raise ValueError(f"Unsupported dataset type: {dataset_type}")


class ScorerFactory:
    """Factory for creating scorer instances from configuration."""

    @staticmethod
    def create_scorer(scorer_config: Any, model: BaseModel) -> BaseScorer:
        """Create scorer instance from configuration."""

        scorer_type = scorer_config.type

        if scorer_type == ScorerType.ACCURACY:
            from novaeval.scorers.accuracy import AccuracyScorer

            return AccuracyScorer(
                threshold=scorer_config.threshold, **scorer_config.parameters
            )

        elif scorer_type == ScorerType.G_EVAL:
            from novaeval.scorers.g_eval import GEvalScorer

            return GEvalScorer(  # type: ignore
                model=model,
                threshold=scorer_config.threshold,
                **scorer_config.parameters,
            )

        elif scorer_type == ScorerType.RAG_ANSWER_RELEVANCY:
            from novaeval.scorers.rag import AnswerRelevancyScorer

            return AnswerRelevancyScorer(
                model=model,
                threshold=scorer_config.threshold,
                **scorer_config.parameters,
            )

        elif scorer_type == ScorerType.RAG_FAITHFULNESS:
            from novaeval.scorers.rag import FaithfulnessScorer

            return FaithfulnessScorer(
                model=model,
                threshold=scorer_config.threshold,
                **scorer_config.parameters,
            )

        elif scorer_type == ScorerType.RAGAS:
            from novaeval.scorers.rag import RAGASScorer

            return RAGASScorer(  # type: ignore
                model=model,
                threshold=scorer_config.threshold,
                **scorer_config.parameters,
            )

        elif scorer_type == ScorerType.CONVERSATIONAL_METRICS:
            from novaeval.scorers.conversational import ConversationalMetricsScorer

            return ConversationalMetricsScorer(  # type: ignore
                model=model,
                threshold=scorer_config.threshold,
                **scorer_config.parameters,
            )

        else:
            raise ValueError(f"Unsupported scorer type: {scorer_type}")


class JobRunner:
    """Runs evaluation jobs based on YAML configuration."""

    def __init__(self, config: EvaluationJobConfig) -> None:
        self.config = config
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.results: dict[str, Any] = {}

        # Set up environment variables
        for key, value in config.environment.items():
            os.environ[key] = value

    async def run(self) -> dict[str, Any]:
        """Run the evaluation job."""

        logger.info(f"Starting evaluation job: {self.config.name}")
        self.start_time = time.time()

        try:
            # Create models
            models = []
            for model_config in self.config.models:
                model = ModelFactory.create_model(model_config)
                models.append((model_config, model))

            # Create datasets
            datasets = []
            for dataset_config in self.config.datasets:
                dataset = DatasetFactory.create_dataset(dataset_config)
                datasets.append((dataset_config, dataset))

            # Run evaluations
            evaluation_results = []

            if self.config.parallel_models:
                # Run models in parallel
                tasks = []
                for model_config, model in models:
                    for dataset_config, dataset in datasets:
                        task = self._evaluate_model_dataset(
                            model_config, model, dataset_config, dataset
                        )
                        tasks.append(task)

                results = await asyncio.gather(*tasks, return_exceptions=True)

                for result in results:
                    if isinstance(result, Exception):
                        logger.error(f"Evaluation failed: {result}")
                    else:
                        evaluation_results.append(result)
            else:
                # Run models sequentially
                for model_config, model in models:
                    for dataset_config, dataset in datasets:
                        try:
                            result = await self._evaluate_model_dataset(
                                model_config, model, dataset_config, dataset
                            )
                            evaluation_results.append(result)
                        except Exception as e:
                            logger.error(f"Evaluation failed: {e}")

            # Compile final results
            self.end_time = time.time()
            final_results = self._compile_results(evaluation_results)  # type: ignore

            # Generate outputs
            await self._generate_outputs(final_results)

            # Check CI/CD requirements
            ci_status = self._check_ci_requirements(final_results)

            logger.info(f"Evaluation job completed: {self.config.name}")

            return {
                "job_name": self.config.name,
                "status": "completed",
                "duration": self.end_time - self.start_time,
                "results": final_results,
                "ci_status": ci_status,
            }

        except Exception as e:
            self.end_time = time.time()
            logger.error(f"Evaluation job failed: {e}")

            return {
                "job_name": self.config.name,
                "status": "failed",
                "duration": (self.end_time - self.start_time) if self.start_time else 0,
                "error": str(e),
                "ci_status": {"passed": False, "reason": f"Job failed: {e}"},
            }

    async def _evaluate_model_dataset(
        self,
        model_config: Any,
        model: BaseModel,
        dataset_config: Any,
        dataset: BaseDataset,
    ) -> dict[str, Any]:
        """Evaluate a single model-dataset combination."""

        logger.info(f"Evaluating {model_config.model_name} on {dataset_config.type}")

        # Create scorers
        scorers = []
        for scorer_config in self.config.scorers:
            scorer = ScorerFactory.create_scorer(scorer_config, model)
            scorers.append((scorer_config, scorer))

        # Create evaluator
        evaluator = Evaluator(
            models=[model], dataset=dataset, scorers=[scorer for _, scorer in scorers]
        )

        # Run evaluation
        evaluation_result = await evaluator.run_evaluation()  # type: ignore

        return {
            "model": {
                "provider": model_config.provider,
                "name": model_config.model_name,
            },
            "dataset": {
                "type": dataset_config.type,
                "name": getattr(dataset_config, "name", None) or dataset_config.type,
            },
            "scorers": [
                {
                    "type": scorer_config.type,
                    "name": scorer_config.name or scorer_config.type,
                    "threshold": scorer_config.threshold,
                    "weight": scorer_config.weight,
                }
                for scorer_config, _ in scorers
            ],
            "results": evaluation_result.dict(),
            "timestamp": time.time(),
        }

    def _compile_results(
        self, evaluation_results: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Compile evaluation results into final format."""

        # Group results by model and dataset
        grouped_results: dict[str, Any] = {}
        overall_scores: list[float] = []

        for result in evaluation_results:
            model_key = f"{result['model']['provider']}:{result['model']['name']}"
            dataset_key = result["dataset"]["name"]

            if model_key not in grouped_results:
                grouped_results[model_key] = {}

            grouped_results[model_key][dataset_key] = result

            # Collect overall scores for summary
            if "overall_score" in result["results"]:
                overall_scores.append(result["results"]["overall_score"])

        # Calculate summary statistics
        summary = {
            "total_evaluations": len(evaluation_results),
            "models_evaluated": len({r["model"]["name"] for r in evaluation_results}),
            "datasets_used": len({r["dataset"]["name"] for r in evaluation_results}),
            "average_score": (
                sum(overall_scores) / len(overall_scores) if overall_scores else 0.0
            ),
            "min_score": min(overall_scores) if overall_scores else 0.0,
            "max_score": max(overall_scores) if overall_scores else 0.0,
        }

        return {
            "summary": summary,
            "detailed_results": grouped_results,
            "raw_results": evaluation_results,
            "job_config": self.config.dict(),
            "execution_metadata": {
                "start_time": self.start_time,
                "end_time": self.end_time,
                "duration": (
                    (self.end_time - self.start_time)
                    if self.end_time and self.start_time
                    else None
                ),
            },
        }

    async def _generate_outputs(self, results: dict[str, Any]) -> None:
        """Generate output files in specified formats."""

        output_dir = Path(self.config.output.directory)
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = int(time.time())
        base_filename = f"{self.config.output.filename_prefix}_{timestamp}"

        for format_type in self.config.output.formats:
            if format_type == "json":
                output_file = output_dir / f"{base_filename}.json"
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(results, f, indent=2, default=str)

            elif format_type == "html":
                output_file = output_dir / f"{base_filename}.html"
                html_content = self._generate_html_report(results)
                with open(output_file, "w", encoding="utf-8") as f:
                    f.write(html_content)

            elif format_type == "junit_xml":
                output_file = output_dir / f"{base_filename}_junit.xml"
                xml_content = self._generate_junit_xml(results)
                with open(output_file, "w", encoding="utf-8") as f:
                    f.write(xml_content)

        logger.info(f"Output files generated in: {output_dir}")

    def _check_ci_requirements(self, results: dict[str, Any]) -> dict[str, Any]:
        """Check CI/CD requirements and determine pass/fail status."""

        ci_config = self.config.ci

        # Check if any evaluation failed the threshold
        failed_evaluations = []
        passed_evaluations = []

        for result in results["raw_results"]:
            overall_score = result["results"].get("overall_score", 0.0)

            if overall_score < ci_config.fail_threshold:
                failed_evaluations.append(
                    {
                        "model": result["model"]["name"],
                        "dataset": result["dataset"]["name"],
                        "score": overall_score,
                        "threshold": ci_config.fail_threshold,
                    }
                )
            else:
                passed_evaluations.append(
                    {
                        "model": result["model"]["name"],
                        "dataset": result["dataset"]["name"],
                        "score": overall_score,
                    }
                )

        # Determine overall CI status
        ci_passed = (
            len(failed_evaluations) == 0 if ci_config.fail_on_threshold else True
        )

        return {
            "passed": ci_passed,
            "fail_threshold": ci_config.fail_threshold,
            "passed_evaluations": len(passed_evaluations),
            "failed_evaluations": len(failed_evaluations),
            "failed_details": failed_evaluations,
            "average_score": results["summary"]["average_score"],
            "recommendation": (
                "Deploy" if ci_passed else "Do not deploy - evaluations below threshold"
            ),
        }

    def _generate_html_report(self, results: dict[str, Any]) -> str:
        """Generate HTML report."""

        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>NovaEval Report - {job_name}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .summary {{ margin: 20px 0; }}
                .results {{ margin: 20px 0; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .pass {{ color: green; }}
                .fail {{ color: red; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>NovaEval Evaluation Report</h1>
                <h2>{job_name}</h2>
                <p>Generated on: {timestamp}</p>
            </div>

            <div class="summary">
                <h3>Summary</h3>
                <p>Total Evaluations: {total_evaluations}</p>
                <p>Models Evaluated: {models_evaluated}</p>
                <p>Datasets Used: {datasets_used}</p>
                <p>Average Score: {average_score:.3f}</p>
                <p>Score Range: {min_score:.3f} - {max_score:.3f}</p>
            </div>

            <div class="results">
                <h3>Detailed Results</h3>
                <!-- Results table would be generated here -->
            </div>
        </body>
        </html>
        """

        return html_template.format(
            job_name=self.config.name,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            **results["summary"],
        )

    def _generate_junit_xml(self, results: dict[str, Any]) -> str:
        """Generate JUnit XML for CI/CD integration."""

        xml_template = """<?xml version="1.0" encoding="UTF-8"?>
<testsuite name="NovaEval" tests="{total_tests}" failures="{failures}" time="{duration}">
{test_cases}
</testsuite>"""

        test_cases = []
        failures = 0

        for result in results["raw_results"]:
            model_name = result["model"]["name"]
            dataset_name = result["dataset"]["name"]
            score = result["results"].get("overall_score", 0.0)
            threshold = self.config.ci.fail_threshold

            test_name = f"{model_name}_on_{dataset_name}"

            if score >= threshold:
                test_case = f'  <testcase name="{test_name}" time="0" />'
            else:
                failures += 1
                test_case = f"""  <testcase name="{test_name}" time="0">
    <failure message="Score {score:.3f} below threshold {threshold}">
      Model: {model_name}
      Dataset: {dataset_name}
      Score: {score:.3f}
      Threshold: {threshold}
    </failure>
  </testcase>"""

            test_cases.append(test_case)

        return xml_template.format(
            total_tests=len(results["raw_results"]),
            failures=failures,
            duration=results["execution_metadata"].get("duration", 0),
            test_cases="\n".join(test_cases),
        )


# CLI integration
def main() -> int:
    """Main CLI entry point for running evaluation jobs."""

    import argparse

    parser = argparse.ArgumentParser(description="Run NovaEval evaluation jobs")
    parser.add_argument("config", help="Path to YAML configuration file")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument(
        "--dry-run", action="store_true", help="Validate config without running"
    )

    args = parser.parse_args()

    try:
        # Load configuration
        config = JobConfigLoader.load_from_file(args.config)

        if args.dry_run:
            print(f"Configuration is valid: {config.name}")
            print(f"Models: {len(config.models)}")
            print(f"Datasets: {len(config.datasets)}")
            print(f"Scorers: {len(config.scorers)}")
            return 0

        # Run evaluation job
        runner = JobRunner(config)
        result = asyncio.run(runner.run())

        # Print results
        if result["status"] == "completed":
            print(f"✅ Evaluation completed: {result['job_name']}")
            print(f"Duration: {result['duration']:.2f} seconds")
            print(
                f"CI Status: {'✅ PASSED' if result['ci_status']['passed'] else '❌ FAILED'}"
            )

            if not result["ci_status"]["passed"]:
                print(f"Reason: {result['ci_status']['recommendation']}")
                return 1

            return 0
        else:
            print(f"❌ Evaluation failed: {result.get('error', 'Unknown error')}")
            return 1

    except Exception as e:
        print(f"❌ Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
