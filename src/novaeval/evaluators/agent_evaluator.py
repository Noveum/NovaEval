"""
Agent evaluator implementation for NovaEval.

This module provides the agent evaluator class that orchestrates
the evaluation process for agent datasets using agent scorers.
"""

import logging
from pathlib import Path
from typing import Any, Optional, Union

import pandas as pd
from tqdm import tqdm

from novaeval.datasets.agent_dataset import AgentDataset
from novaeval.evaluators.base import BaseEvaluator
from novaeval.models.base import BaseModel
from novaeval.utils.logging import setup_logging

logger = logging.getLogger(__name__)


class AgentEvaluator(BaseEvaluator):
    """
    Agent evaluator implementation.

    This class provides the main evaluation logic for running
    evaluations on agent datasets using agent scorers.
    """

    def __init__(
        self,
        agent_dataset: AgentDataset,
        models: list[BaseModel],
        scoring_functions: list[callable],
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
        super().__init__(
            dataset=None,  # We'll handle this differently
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
            scorer_name = scoring_function.__name__.replace("_scorer", "")
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
        aggregator_functions: Optional[list[callable]] = None,
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
            sample_result = self.evaluate_sample(sample)

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
        aggregator_functions: Optional[list[callable]],
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

        # Use default aggregator function if none provided
        if aggregator_functions is None:
            aggregator_functions = [mean_callable]

        # Determine input file
        if file_type.lower() == "json":
            input_file = self.output_dir / "agent_evaluation_results.json"
        else:
            input_file = self.output_dir / "agent_evaluation_results.csv"

        logger.info(f"Running aggregations on {input_file}")

        # Run task aggregation
        if aggregate_by_task:
            logger.info("Running task aggregation")
            output_file = self.output_dir / "task_aggregation.csv"
            self._run_single_aggregation(
                "task",
                input_file,
                output_file,
                aggregator_functions,
                aggregation_chunk_size,
            )

        # Run user aggregation
        if aggregate_by_user:
            logger.info("Running user aggregation")
            output_file = self.output_dir / "user_aggregation.csv"
            self._run_single_aggregation(
                "user",
                input_file,
                output_file,
                aggregator_functions,
                aggregation_chunk_size,
            )

        # Run agent name aggregation
        if aggregate_by_agent_name:
            logger.info("Running agent name aggregation")
            output_file = self.output_dir / "agent_aggregation.csv"
            self._run_single_aggregation(
                "agent_name",
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
        aggregator_functions: list[callable],
        aggregation_chunk_size: int,
    ) -> None:
        """
        Run a single aggregation with multiple callable functions.

        Args:
            aggregation_type: Type of aggregation ('task', 'user', 'agent_name')
            input_file: Input file path
            output_file: Output file path
            aggregator_functions: List of callable functions
            aggregation_chunk_size: Chunk size for streaming
        """
        # Import aggregator functions
        from novaeval.evaluators.aggregators import (
            aggregate_by_agent_name,
            aggregate_by_task,
            aggregate_by_user,
        )

        # Map aggregation types to functions
        aggregator_map = {
            "task": aggregate_by_task,
            "user": aggregate_by_user,
            "agent_name": aggregate_by_agent_name,
        }

        # Call the modified aggregator function
        aggregator_func = aggregator_map[aggregation_type]
        aggregator_func(
            input_file=input_file,
            output_filename=output_file,
            callable_func=aggregator_functions,  # Pass list of functions
            streaming=self.stream,
            chunk_size=aggregation_chunk_size,
        )

    def evaluate_sample(self, sample: Any) -> dict[str, Any]:
        """
        Evaluate a single sample against the evaluators.

        Args:
            sample: The sample to evaluate (AgentData object)

        Returns:
            Dictionary containing sample evaluation results
        """
        sample_result = {
            "user_id": sample.user_id,
            "task_id": sample.task_id,
            "turn_id": sample.turn_id,
            "agent_name": sample.agent_name,
            "scores": {},
            "reasoning": {},
            "error": None,
        }

        try:
            # Run each scoring function on the sample
            model = self.models[0] if self.models else None
            if not model:
                logger.error("No model available for scoring")
                return sample_result

            for scoring_function in self.scoring_functions:
                scorer_name = scoring_function.__name__.replace("_scorer", "")

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
                new_row[col] = None

        # Append to DataFrame using loc to avoid concatenation warnings
        if len(self.results_df) == 0:
            # If DataFrame is empty, create it with the new row
            self.results_df = pd.DataFrame([new_row])
        else:
            # Append using loc
            self.results_df.loc[len(self.results_df)] = new_row

    def _save_intermediate_results(self, file_type: str) -> None:
        """
        Save intermediate results to file.

        Args:
            file_type: Type of file to save ('csv' or 'json')
        """
        if file_type.lower() == "json":
            self._convert_to_json()
        else:
            # Save as CSV
            csv_file = self.output_dir / "agent_evaluation_results.csv"
            self.results_df.to_csv(csv_file, index=False)
            logger.info(f"Results saved to {csv_file}")

    def save_results(
        self, results: list[dict[str, Any]], file_type: str = "csv"
    ) -> None:
        """
        Save a list of evaluation results.

        Args:
            results: List of evaluation result dictionaries
            file_type: Type of file to save ('csv' or 'json')
        """
        # Convert results to DataFrame
        temp_df = pd.DataFrame(results)

        # Save based on file type
        if file_type.lower() == "json":
            json_file = self.output_dir / "agent_evaluation_results.json"
            temp_df.to_json(json_file, orient="records", indent=2)
            logger.info(f"Results saved to {json_file}")
        else:
            csv_file = self.output_dir / "agent_evaluation_results.csv"
            temp_df.to_csv(csv_file, index=False)
            logger.info(f"Results saved to {csv_file}")

    def _convert_to_json(self) -> None:
        """
        Convert the DataFrame to JSON and save it.
        Uses streaming if the stream flag is set.
        """
        json_file = self.output_dir / "agent_evaluation_results.json"

        if self.stream:
            # Streaming approach for large datasets
            with open(json_file, "w") as f:
                f.write("[\n")
                for i, row in self.results_df.iterrows():
                    if i > 0:
                        f.write(",\n")
                    f.write(row.to_json())
                f.write("\n]")
        else:
            # Non-streaming approach
            self.results_df.to_json(json_file, orient="records", indent=2)

        logger.info(f"Results saved to {json_file}")

    def run(self) -> dict[str, Any]:
        """
        Run the complete evaluation process (compatibility with BaseEvaluator).

        Returns:
            Dictionary containing evaluation results summary
        """
        # Run the evaluation
        self.run_all()

        # Return summary information
        return {
            "total_samples": len(self.results_df),
            "scorers_used": len(self.scoring_functions),
            "models_used": len(self.models),
            "output_directory": str(self.output_dir),
            "include_reasoning": self.include_reasoning,
            "streaming_mode": self.stream,
        }
