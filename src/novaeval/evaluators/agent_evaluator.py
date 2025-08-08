"""
Agent evaluator for NovaEval.

This module provides an evaluator specifically designed for agent evaluation tasks.
"""

import logging
from pathlib import Path
from typing import Any, Callable, Optional, Union

import pandas as pd
from tqdm import tqdm

from novaeval.datasets.agent_dataset import AgentDataset
from novaeval.evaluators.base import BaseEvaluator
from novaeval.models.base import BaseModel
from novaeval.scorers.base import BaseScorer
from novaeval.utils.logging import setup_logging

logger = logging.getLogger(__name__)


class AgentEvaluator(BaseEvaluator):
    """
    Evaluator for agent evaluation tasks.

    This evaluator is specifically designed to work with agent datasets and
    scoring functions that evaluate agent performance.
    """

    def __init__(
        self,
        agent_dataset: AgentDataset,
        models: list[BaseModel],
        scoring_functions: list[Callable],
        output_dir: Optional[Union[str, Path]] = None,
        config: Optional[dict[str, Any]] = None,
        stream: bool = False,
        include_reasoning: bool = True,
    ):
        """
        Initialize the agent evaluator.

        Args:
            agent_dataset: The agent dataset to evaluate on
            models: List of models to evaluate
            agent_scorers: List of agent scorers to use for evaluation
            output_dir: Directory to save results
            config: Additional configuration options (placeholder)
            stream: Whether to use streaming mode
            include_reasoning: Whether to include reasoning in results
        """
        # Convert agent_dataset to BaseDataset for compatibility
        # We'll store the original agent_dataset separately
        self.agent_dataset = agent_dataset
        self.scoring_functions = scoring_functions
        self.stream = stream
        self.include_reasoning = include_reasoning

        # Initialize with empty dataset for base class compatibility
        # We'll use agent_dataset directly in our methods
        from novaeval.datasets.base import BaseDataset

        # Create a dummy dataset for base class compatibility
        class DummyDataset(BaseDataset):
            def __init__(self) -> None:
                pass

            def get_data(self) -> list:
                return []

            def load_data(self) -> list[dict[str, Any]]:
                return []

        super().__init__(
            dataset=DummyDataset(),
            models=models,
            scorers=[],  # We'll use scoring_functions instead
            output_dir=output_dir,
            config=config or {},
        )

        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize DataFrame with required columns
        self._initialize_dataframe()

        # Setup logging
        setup_logging(
            level=self.config.get("log_level", "INFO"),
            log_file=self.output_dir / "agent_evaluation.log",
        )

    def _initialize_dataframe(self) -> None:
        """Initialize the pandas DataFrame with required columns."""
        # Base columns
        base_columns = ["user_id", "task_id", "turn_id", "agent_name"]

        # Add scorer columns
        scorer_columns = []
        reasoning_columns = []

        for _i, scoring_function in enumerate(self.scoring_functions):
            # Get scorer name from function name
            if hasattr(scoring_function, "__name__"):
                scorer_name = scoring_function.__name__.replace("_scorer", "")
            else:
                scorer_name = f"scorer_{_i}"
            scorer_columns.append(scorer_name)

            if self.include_reasoning:
                reasoning_columns.append(f"{scorer_name}_reasoning")

        # Combine all columns
        all_columns = base_columns + scorer_columns + reasoning_columns

        # Initialize empty DataFrame
        self.results_df = pd.DataFrame(columns=all_columns)

        # Store column information for later use
        self.scorer_columns = scorer_columns
        self.reasoning_columns = reasoning_columns if self.include_reasoning else []

    def run_all(
        self,
        save_every: int = 100,
        file_type: str = "csv",
        aggregate_by_task: bool = False,
        aggregate_by_user: bool = False,
        aggregate_by_agent_name: bool = False,
        aggregator_functions: Optional[list[Callable]] = None,
        aggregation_chunk_size: int = 1000,
    ) -> None:
        """
        Run the scorers on all samples in the dataset and store results.

        Args:
            save_every: Save results every N samples to avoid memory leaks
            file_type: Type of file to save ('csv' or 'json')
            aggregate_by_task: Whether to run task aggregation
            aggregate_by_user: Whether to run user aggregation
            aggregate_by_agent_name: Whether to run agent name aggregation
            aggregator_functions: List of callable functions for aggregation
            aggregation_chunk_size: Chunk size for streaming aggregation
        """
        logger.info("Starting agent evaluation process")

        # Get all samples from the agent dataset
        samples = list(self.agent_dataset.get_datapoint())

        logger.info(f"Processing {len(samples)} samples")

        # Process samples in batches
        for i, sample in enumerate(tqdm(samples, desc="Evaluating samples")):
            # Evaluate the sample
            model = self.models[0] if self.models else None
            if not model:
                logger.error("No model available for evaluation")
                continue
            sample_result = self.evaluate_sample(sample, model, [])

            # Add result to DataFrame
            self._add_result_to_dataframe(sample_result)

            # Save periodically to avoid memory leaks
            if (i + 1) % save_every == 0:
                logger.info(f"Saving intermediate results after {i + 1} samples")
                self._save_intermediate_results(file_type)

        # Save final results
        logger.info("Saving final results")
        self._save_intermediate_results(file_type)

        # Run aggregations if requested
        if any([aggregate_by_task, aggregate_by_user, aggregate_by_agent_name]):
            self._run_aggregations(
                file_type=file_type,
                aggregate_by_task=aggregate_by_task,
                aggregate_by_user=aggregate_by_user,
                aggregate_by_agent_name=aggregate_by_agent_name,
                aggregator_functions=aggregator_functions,
                aggregation_chunk_size=aggregation_chunk_size,
            )

        logger.info("Agent evaluation completed")

    def _run_aggregations(
        self,
        file_type: str,
        aggregate_by_task: bool,
        aggregate_by_user: bool,
        aggregate_by_agent_name: bool,
        aggregator_functions: Optional[list[Callable]],
        aggregation_chunk_size: int,
    ) -> None:
        """
        Run aggregations based on the provided flags.

        Args:
            file_type: Type of file to read ('csv' or 'json')
            aggregate_by_task: Whether to run task aggregation
            aggregate_by_user: Whether to run user aggregation
            aggregate_by_agent_name: Whether to run agent name aggregation
            aggregator_functions: List of callable functions for aggregation
            aggregation_chunk_size: Chunk size for streaming aggregation
        """
        from novaeval.evaluators.aggregators import mean_callable

        # Set default aggregator functions if none provided
        if aggregator_functions is None:
            aggregator_functions = [mean_callable]

        # Determine input file path
        input_file = self.output_dir / f"agent_evaluation_results.{file_type}"

        if not input_file.exists():
            logger.warning(
                f"Input file {input_file} does not exist. Skipping aggregations."
            )
            return

        # Run each requested aggregation
        if aggregate_by_task:
            output_file = self.output_dir / f"task_aggregation.{file_type}"
            self._run_single_aggregation(
                "task",
                input_file,
                output_file,
                aggregator_functions,
                aggregation_chunk_size,
            )

        if aggregate_by_user:
            output_file = self.output_dir / f"user_aggregation.{file_type}"
            self._run_single_aggregation(
                "user",
                input_file,
                output_file,
                aggregator_functions,
                aggregation_chunk_size,
            )

        if aggregate_by_agent_name:
            output_file = self.output_dir / f"agent_aggregation.{file_type}"
            self._run_single_aggregation(
                "agent",
                input_file,
                output_file,
                aggregator_functions,
                aggregation_chunk_size,
            )

    def _run_single_aggregation(
        self,
        aggregation_type: str,
        input_file: Path,
        output_file: Path,
        aggregator_functions: list[Callable],
        aggregation_chunk_size: int,
    ) -> None:
        """
        Run a single aggregation operation.

        Args:
            aggregation_type: Type of aggregation ('task', 'user', 'agent')
            input_file: Path to input file
            output_file: Path to output file
            aggregator_functions: List of callable functions for aggregation
            aggregation_chunk_size: Chunk size for streaming aggregation
        """
        from novaeval.evaluators.aggregators import (
            aggregate_by_agent_name,
            aggregate_by_task,
            aggregate_by_user,
        )

        logger.info(f"Running {aggregation_type} aggregation")

        try:
            if aggregation_type == "task":
                aggregate_by_task(
                    input_file=input_file,
                    output_filename=output_file,
                    callable_func=aggregator_functions,
                    streaming=True,
                    chunk_size=aggregation_chunk_size,
                )
            elif aggregation_type == "user":
                aggregate_by_user(
                    input_file=input_file,
                    output_filename=output_file,
                    callable_func=aggregator_functions,
                    streaming=True,
                    chunk_size=aggregation_chunk_size,
                )
            elif aggregation_type == "agent":
                aggregate_by_agent_name(
                    input_file=input_file,
                    output_filename=output_file,
                    callable_func=aggregator_functions,
                    streaming=True,
                    chunk_size=aggregation_chunk_size,
                )
            else:
                logger.error(f"Unknown aggregation type: {aggregation_type}")

            logger.info(
                f"{aggregation_type.capitalize()} aggregation completed: {output_file}"
            )

        except Exception as e:
            logger.error(f"Failed to run {aggregation_type} aggregation: {e}")

    def evaluate_sample(
        self, sample: dict[str, Any], model: BaseModel, scorers: list[BaseScorer]
    ) -> dict[str, Any]:
        """
        Evaluate a single sample using all scoring functions.

        Args:
            sample: The sample to evaluate
            model: The model to use for evaluation
            scorers: List of scorers to apply

        Returns:
            Dictionary containing evaluation results
        """
        # Initialize result structure
        sample_result: dict[str, Any] = {
            "user_id": getattr(sample, "user_id", ""),
            "task_id": getattr(sample, "task_id", ""),
            "turn_id": getattr(sample, "turn_id", ""),
            "agent_name": getattr(sample, "agent_name", ""),
            "scores": {},
            "reasoning": {},
        }

        # Ensure scores and reasoning are dictionaries
        if not isinstance(sample_result["scores"], dict):
            sample_result["scores"] = {}
        if not isinstance(sample_result["reasoning"], dict):
            sample_result["reasoning"] = {}

        try:
            # Run each scoring function on the sample
            if not model:
                logger.error("No model available for scoring")
                return sample_result

            for scoring_function in self.scoring_functions:
                if hasattr(scoring_function, "__name__"):
                    scorer_name = scoring_function.__name__.replace("_scorer", "")
                else:
                    scorer_name = "unknown_scorer"

                try:
                    # Call the scoring function directly
                    score_result = scoring_function(sample, model)

                    # Extract score and reasoning based on result type
                    if hasattr(score_result, "score"):
                        # Single score object
                        sample_result["scores"][scorer_name] = score_result.score
                        if self.include_reasoning and hasattr(
                            score_result, "reasoning"
                        ):
                            sample_result["reasoning"][
                                scorer_name
                            ] = score_result.reasoning
                    elif isinstance(score_result, list) and len(score_result) > 0:
                        # List of scores - take the first one
                        first_score = score_result[0]
                        if hasattr(first_score, "score"):
                            sample_result["scores"][scorer_name] = first_score.score
                            if self.include_reasoning and hasattr(
                                first_score, "reasoning"
                            ):
                                sample_result["reasoning"][
                                    scorer_name
                                ] = first_score.reasoning
                        else:
                            sample_result["scores"][scorer_name] = 0.0
                    elif isinstance(score_result, dict):
                        # Dict-based results (error or special format)
                        if "error" in score_result:
                            sample_result["scores"][scorer_name] = 0.0
                            if self.include_reasoning:
                                sample_result["reasoning"][
                                    scorer_name
                                ] = f"Error: {score_result.get('error', 'Unknown error')}"
                        elif "score" in score_result:
                            sample_result["scores"][scorer_name] = score_result["score"]
                            if self.include_reasoning and "reasoning" in score_result:
                                sample_result["reasoning"][scorer_name] = score_result[
                                    "reasoning"
                                ]
                        else:
                            sample_result["scores"][scorer_name] = 0.0
                    else:
                        # Fallback: try to extract numeric value
                        try:
                            sample_result["scores"][scorer_name] = float(score_result)
                        except (ValueError, TypeError):
                            sample_result["scores"][scorer_name] = 0.0

                except Exception as e:
                    logger.warning(
                        f"Scoring function {scorer_name} failed on sample: {e}"
                    )
                    sample_result["scores"][scorer_name] = 0.0
                    if self.include_reasoning:
                        sample_result["reasoning"][scorer_name] = f"Error: {e!s}"

        except Exception as e:
            logger.error(f"Failed to evaluate sample: {e}")
            sample_result["error"] = str(e)

        return sample_result

    def _add_result_to_dataframe(self, sample_result: dict[str, Any]) -> None:
        """
        Add a sample result to the DataFrame.

        Args:
            sample_result: The sample evaluation result
        """
        # Create a new row
        new_row = {
            "user_id": sample_result.get("user_id", ""),
            "task_id": sample_result.get("task_id", ""),
            "turn_id": sample_result.get("turn_id", ""),
            "agent_name": sample_result.get("agent_name", ""),
        }

        # Add scores
        new_row.update(sample_result.get("scores", {}))

        # Add reasoning if enabled
        if self.include_reasoning:
            for scorer_name, reasoning in sample_result.get("reasoning", {}).items():
                reasoning_col = f"{scorer_name}_reasoning"
                if reasoning_col in self.results_df.columns:
                    new_row[reasoning_col] = reasoning

        # Ensure all columns exist in the new row
        for col in self.results_df.columns:
            if col not in new_row:
                new_row[col] = ""

        # Append to DataFrame
        new_df = pd.DataFrame([new_row])

        # If DataFrame is empty, just set it to the new DataFrame
        if self.results_df.empty:
            self.results_df = new_df
        else:
            # Ensure all columns exist in both DataFrames
            for col in self.results_df.columns:
                if col not in new_df.columns:
                    new_df[col] = ""
            for col in new_df.columns:
                if col not in self.results_df.columns:
                    self.results_df[col] = ""

            self.results_df = pd.concat([self.results_df, new_df], ignore_index=True)

    def _save_intermediate_results(self, file_type: str) -> None:
        """
        Save intermediate results to disk.

        Args:
            file_type: Type of file to save ('csv' or 'json')
        """
        output_file = self.output_dir / f"agent_evaluation_results.{file_type}"

        if file_type.lower() == "json":
            self._convert_to_json()
            self.results_df.to_json(output_file, orient="records", indent=2)
        else:
            self.results_df.to_csv(output_file, index=False)

        logger.info(f"Intermediate results saved to {output_file}")

    def save_results(self, results: dict[str, Any]) -> None:
        """
        Save evaluation results to disk.

        Args:
            results: The results to save
        """
        # Convert results to DataFrame if needed
        if not hasattr(self, "results_df") or self.results_df.empty:
            if isinstance(results, list):
                self.results_df = pd.DataFrame(results)
            else:
                # Handle dict format
                self.results_df = pd.DataFrame([results])

        output_file = self.output_dir / "agent_evaluation_results.csv"

        self.results_df.to_csv(output_file, index=False)
        logger.info(f"Results saved to {output_file}")

    def _convert_to_json(self) -> None:
        """Convert DataFrame to JSON-compatible format."""
        # Convert any non-serializable types to strings
        for col in self.results_df.columns:
            if self.results_df[col].dtype == "object":
                self.results_df[col] = self.results_df[col].astype(str)

    def run(self) -> dict[str, Any]:
        """
        Run the evaluation.

        Returns:
            Dictionary containing evaluation results
        """
        # This method is required by the base class but not used in agent evaluation
        # We use run_all() instead for agent evaluation
        logger.warning("run() method called on AgentEvaluator. Use run_all() instead.")
        return {"status": "not_implemented", "message": "Use run_all() method instead"}
