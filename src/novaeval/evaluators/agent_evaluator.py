"""
Agent evaluator for NovaEval.

This module provides an evaluator specifically designed for agent evaluation tasks.
"""

import json
import logging
from pathlib import Path
from typing import Any, Callable, Optional, Union

import pandas as pd
from tqdm import tqdm

from novaeval.datasets.agent_dataset import AgentDataset
from novaeval.models.base import BaseModel
from novaeval.utils.logging import setup_logging

logger = logging.getLogger(__name__)


class AgentEvaluator:
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
            scoring_functions: List of scoring functions to use for evaluation
            output_dir: Directory to save results
            config: Additional configuration options
            stream: Whether to use streaming mode
            include_reasoning: Whether to include reasoning in results
        """
        self.agent_dataset = agent_dataset
        self.models = models
        self.scoring_functions = scoring_functions
        self.stream = stream
        self.include_reasoning = include_reasoning
        self.config = config or {}

        # Set output directory
        if output_dir is None:
            self.output_dir = Path("results")
        else:
            self.output_dir = Path(output_dir)

        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize DataFrame with required columns
        self._initialize_dataframe()

        # Setup logging
        setup_logging(
            level=self.config.get("log_level", "INFO"),
            log_file=self.output_dir / "agent_evaluation.log",
        )

        # Track if headers have been written for CSV/JSONL
        self._headers_written = False

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

    def _reset_evaluation_state(self, file_type: str) -> None:
        """
        Reset evaluation state for a new run.

        Args:
            file_type: Type of file being used
        """
        # Reset headers written flag
        self._headers_written = False

        # Clear any existing results file
        output_file = self.output_dir / f"agent_evaluation_results.{file_type}"
        if output_file.exists():
            output_file.unlink()

        # Clear final JSON file if it exists
        if file_type.lower() == "json":
            final_json_file = self.output_dir / "agent_evaluation_results_final.json"
            if final_json_file.exists():
                final_json_file.unlink()

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

        # Validate file_type
        if file_type.lower() not in ["csv", "json"]:
            logger.error(f"Unsupported file type: {file_type}")
            return

        # Reset evaluation state
        self._reset_evaluation_state(file_type)

        # Get the generator from the agent dataset
        samples_generator = self.agent_dataset.get_datapoint()

        # Process samples directly from the generator to preserve streaming
        # Note: We can't get the total count without consuming the generator,
        # so we'll use tqdm without a total count for streaming behavior
        for i, sample in enumerate(tqdm(samples_generator, desc="Evaluating samples")):
            # Evaluate the sample
            model = self.models[0] if self.models else None
            if not model:
                logger.error("No model available for evaluation")
                continue
            sample_result = self.evaluate_sample(sample, model)

            # Add result to DataFrame
            self._add_result_to_dataframe(sample_result)

            # Save periodically to avoid memory leaks
            if (i + 1) % save_every == 0:
                logger.info(f"Saving intermediate results after {i + 1} samples")
                # For intermediate saves, save to file but keep DataFrame for final save
                self._save_intermediate_results(file_type, is_final=False)

        # Save final results (flush any remaining data in DataFrame)
        logger.info("Saving final results")
        # Only save final results if there's data in the DataFrame
        if not self.results_df.empty:
            self._save_intermediate_results(file_type, is_final=True)

        # Reload all results into DataFrame for testing/accessibility
        self._reload_results_to_dataframe(file_type)

        # Finalize results (convert JSONL to JSON if needed)
        self.finalize_results(file_type)

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
        # For JSON, use the final JSON file if it exists, otherwise use JSONL
        if file_type.lower() == "json":
            input_file = self.output_dir / "agent_evaluation_results_final.json"
            if not input_file.exists():
                input_file = self.output_dir / f"agent_evaluation_results.{file_type}"
        else:
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

    def _read_jsonl_for_aggregation(self, file_path: Path) -> pd.DataFrame:
        """
        Read JSONL or JSON file and convert to DataFrame for aggregation.

        Args:
            file_path: Path to JSONL or JSON file

        Returns:
            DataFrame containing the data
        """
        try:
            # First try to read as JSONL (one JSON object per line)
            records: list[dict[str, Any]] = []
            with open(file_path, encoding="utf-8") as f:
                records.extend(json.loads(line) for line in f if line.strip())

            if records:
                return pd.DataFrame(records)

        except json.JSONDecodeError:
            # If JSONL fails, try reading as proper JSON
            try:
                with open(file_path, encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        return pd.DataFrame(data)
                    else:
                        logger.error(f"Unexpected JSON format in {file_path}")
                        return pd.DataFrame()
            except Exception as e:
                logger.error(f"Failed to read JSON file {file_path}: {e}")
                return pd.DataFrame()

        except Exception as e:
            logger.error(f"Failed to read file {file_path}: {e}")
            return pd.DataFrame()

        return pd.DataFrame()

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
            # The aggregation functions support both CSV and JSON files directly
            # No need to convert JSON files to CSV

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
        self, sample: dict[str, Any], model: BaseModel
    ) -> dict[str, Any]:
        """
        Evaluate a single sample using all scoring functions.

        Args:
            sample: The sample to evaluate
            model: The model to use for evaluation

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

    def _save_intermediate_results(
        self, file_type: str, is_final: bool = False
    ) -> None:
        """
        Save intermediate results to disk using streaming/append mode to avoid memory bloat.

        Args:
            file_type: Type of file to save ('csv' or 'json')
            is_final: Whether this is the final save (don't clear DataFrame)
        """
        if self.results_df.empty:
            logger.debug("No results to save")
            return

        output_file = self.output_dir / f"agent_evaluation_results.{file_type}"

        try:
            if file_type.lower() == "json":
                # Use JSONL format for streaming JSON
                self._save_jsonl_append(output_file)
            else:
                # Use append mode for CSV
                self._save_csv_append(output_file)

            # Clear DataFrame to free memory (except on final save)
            # Only clear if this is an intermediate save and we have data
            if not is_final and len(self.results_df) > 0:
                self.results_df = pd.DataFrame(columns=self.results_df.columns)
            logger.info(f"Intermediate results saved to {output_file}")

        except Exception as e:
            logger.error(f"Failed to save intermediate results: {e}")

    def _save_csv_append(self, output_file: Path) -> None:
        """
        Save DataFrame to CSV in append mode.

        Args:
            output_file: Path to output file
        """
        # Write headers only if file doesn't exist or headers haven't been written
        header = not output_file.exists() or not self._headers_written

        self.results_df.to_csv(
            output_file, mode="a" if not header else "w", header=header, index=False
        )

        if header:
            self._headers_written = True

    def _save_jsonl_append(self, output_file: Path) -> None:
        """
        Save DataFrame to JSONL format in append mode.

        Args:
            output_file: Path to output file
        """
        # Convert DataFrame to JSONL format (one JSON object per line)
        records = self.results_df.to_dict("records")

        # Convert any non-serializable types to strings
        for record in records:
            for key, value in record.items():
                if not isinstance(value, (str, int, float, bool, type(None))):
                    record[key] = str(value)

        # Append to file
        with open(output_file, "a", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record) + "\n")

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

    def finalize_results(self, file_type: str = "csv") -> None:
        """
        Finalize results by converting streaming format to final format if needed.

        This method should be called after all processing is complete to convert
        JSONL files to proper JSON format if desired.

        Args:
            file_type: Type of file to finalize ('csv' or 'json')
        """
        if file_type.lower() == "json":
            self._convert_jsonl_to_json()
        # CSV files are already in final format, no conversion needed

    def _reload_results_to_dataframe(self, file_type: str) -> None:
        """
        Reload all results from the saved file into the DataFrame.

        This method is called after processing is complete to ensure the DataFrame
        contains all results for testing and accessibility purposes.

        Args:
            file_type: Type of file to reload ('csv' or 'json')
        """
        try:
            if file_type.lower() == "csv":
                output_file = self.output_dir / "agent_evaluation_results.csv"
                if output_file.exists():
                    self.results_df = pd.read_csv(output_file)
                    logger.info(f"Reloaded {len(self.results_df)} results from CSV")
            elif file_type.lower() == "json":
                # Try to read from final JSON file first, then JSONL
                json_file = self.output_dir / "agent_evaluation_results_final.json"
                jsonl_file = self.output_dir / "agent_evaluation_results.json"

                if json_file.exists():
                    with open(json_file, encoding="utf-8") as f:
                        data = json.load(f)
                    self.results_df = pd.DataFrame(data)
                    logger.info(f"Reloaded {len(self.results_df)} results from JSON")
                elif jsonl_file.exists():
                    records = []
                    with open(jsonl_file, encoding="utf-8") as f:
                        records = [json.loads(line) for line in f if line.strip()]
                    self.results_df = pd.DataFrame(records)
                    logger.info(f"Reloaded {len(self.results_df)} results from JSONL")
        except Exception as e:
            logger.error(f"Failed to reload results to DataFrame: {e}")

    def _convert_jsonl_to_json(self) -> None:
        """
        Convert JSONL file to proper JSON format for final output.
        """
        jsonl_file = self.output_dir / "agent_evaluation_results.json"
        json_file = self.output_dir / "agent_evaluation_results_final.json"

        if not jsonl_file.exists():
            logger.warning("JSONL file not found for conversion")
            return

        try:
            records = []
            with open(jsonl_file, encoding="utf-8") as f:
                records = [json.loads(line) for line in f if line.strip()]

            with open(json_file, "w", encoding="utf-8") as f:
                json.dump(records, f, indent=2)

            logger.info(f"Converted JSONL to JSON: {json_file}")

        except Exception as e:
            logger.error(f"Failed to convert JSONL to JSON: {e}")
