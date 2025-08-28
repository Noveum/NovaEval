"""
Aggregator functions for NovaEval evaluation results.

This module provides functions to aggregate evaluation results by different
grouping criteria (task, user, agent) with streaming support for memory efficiency.
"""

import json
import logging
from pathlib import Path
from typing import Callable, Optional, Union

import ijson
import pandas as pd

logger = logging.getLogger(__name__)


def mean_callable(scores: list[float]) -> float:
    """Default aggregation function - calculates mean."""
    if not scores:
        return 0.0
    return sum(scores) / len(scores)


def aggregate_by_task(
    input_file: Union[str, Path],
    output_filename: Union[str, Path],
    callable_func: Optional[
        Union[Callable[[list[float]], float], list[Callable[[list[float]], float]]]
    ] = None,
    streaming: bool = False,
    chunk_size: int = 1000,
) -> None:
    """
    Aggregate scores by task_id.

    Args:
        input_file: Path to CSV/JSON from run_all
        output_filename: Where to save aggregated results
        callable_func: Function(s) to apply to scores (default: mean). Can be single function or list of functions.
        streaming: Whether to use streaming mode (processes column by column)
        chunk_size: How many rows to process at once in streaming mode
    """
    input_file = Path(input_file)
    output_filename = Path(output_filename)

    # Handle single callable or list of callables
    if callable_func is None:
        callable_funcs: list[Callable[[list[float]], float]] = [mean_callable]
    elif not isinstance(callable_func, list):
        callable_funcs = [callable_func]
    else:
        callable_funcs = callable_func

    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    # Ensure output directory exists
    output_filename.parent.mkdir(parents=True, exist_ok=True)

    if streaming:
        _aggregate_by_task_streaming(
            input_file, output_filename, callable_funcs, chunk_size
        )
    else:
        _aggregate_by_task_memory(input_file, output_filename, callable_funcs)


def aggregate_by_user(
    input_file: Union[str, Path],
    output_filename: Union[str, Path],
    callable_func: Optional[
        Union[Callable[[list[float]], float], list[Callable[[list[float]], float]]]
    ] = None,
    streaming: bool = False,
    chunk_size: int = 1000,
) -> None:
    """
    Aggregate scores by user_id.

    Args:
        input_file: Path to CSV/JSON from run_all
        output_filename: Where to save aggregated results
        callable_func: Function(s) to apply to scores (default: mean). Can be single function or list of functions.
        streaming: Whether to use streaming mode (processes column by column)
        chunk_size: How many rows to process at once in streaming mode
    """
    input_file = Path(input_file)
    output_filename = Path(output_filename)

    # Handle single callable or list of callables
    if callable_func is None:
        callable_funcs: list[Callable[[list[float]], float]] = [mean_callable]
    elif not isinstance(callable_func, list):
        callable_funcs = [callable_func]
    else:
        callable_funcs = callable_func

    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    # Ensure output directory exists
    output_filename.parent.mkdir(parents=True, exist_ok=True)

    if streaming:
        _aggregate_by_user_streaming(
            input_file, output_filename, callable_funcs, chunk_size
        )
    else:
        _aggregate_by_user_memory(input_file, output_filename, callable_funcs)


def aggregate_by_agent_name(
    input_file: Union[str, Path],
    output_filename: Union[str, Path],
    callable_func: Optional[
        Union[Callable[[list[float]], float], list[Callable[[list[float]], float]]]
    ] = None,
    streaming: bool = False,
    chunk_size: int = 1000,
) -> None:
    """
    Aggregate scores by agent_name.

    Args:
        input_file: Path to CSV/JSON from run_all
        output_filename: Where to save aggregated results
        callable_func: Function(s) to apply to scores (default: mean). Can be single function or list of functions.
        streaming: Whether to use streaming mode (processes column by column)
        chunk_size: How many rows to process at once in streaming mode
    """
    input_file = Path(input_file)
    output_filename = Path(output_filename)

    # Handle single callable or list of callables
    if callable_func is None:
        callable_funcs: list[Callable[[list[float]], float]] = [mean_callable]
    elif not isinstance(callable_func, list):
        callable_funcs = [callable_func]
    else:
        callable_funcs = callable_func

    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    # Ensure output directory exists
    output_filename.parent.mkdir(parents=True, exist_ok=True)

    if streaming:
        _aggregate_by_agent_streaming(
            input_file, output_filename, callable_funcs, chunk_size
        )
    else:
        _aggregate_by_agent_memory(input_file, output_filename, callable_funcs)


def _aggregate_by_task_streaming(
    input_file: Path,
    output_filename: Path,
    callable_funcs: list[Callable[[list[float]], float]],
    chunk_size: int,
) -> None:
    """Streaming aggregation by task_id - processes column by column."""
    logger.info(f"Starting streaming aggregation by task_id from {input_file}")

    # Read the file to get column names
    if input_file.suffix.lower() == ".json":
        # Use ijson to get column names from first item
        try:
            with open(input_file, "rb") as f:
                parser = ijson.items(f, "item")
                first_item = next(parser)
                if first_item:
                    # Get all columns from the first item
                    all_columns = list(first_item.keys())
                else:
                    raise ValueError("Empty JSON array")
        except Exception as e:
            logger.error(f"Error reading column names from JSON file {input_file}: {e}")
            raise
    else:
        df_sample = pd.read_csv(input_file, nrows=1)
        all_columns = list(df_sample.columns)

    # Get base columns and scorer columns
    base_columns = ["user_id", "task_id", "turn_id", "agent_name"]
    scorer_columns = [
        col
        for col in all_columns
        if col not in base_columns and not col.endswith("_reasoning")
    ]

    # Initialize results dictionary
    results: dict[str, dict[str, float]] = {}

    # Process each scorer column separately
    for scorer_col in scorer_columns:
        logger.info(f"Processing column: {scorer_col}")

        # Read the file in chunks, processing only the grouping column and current scorer column
        columns_to_read = ["task_id", scorer_col]

        if input_file.suffix.lower() == ".json":
            # Handle JSON files with true streaming using ijson
            task_scores_json: dict[str, list[float]] = {}

            try:
                with open(input_file, "rb") as f:
                    # Stream through JSON array items
                    parser = ijson.items(f, "item")

                    for item in parser:
                        try:
                            task_id = item.get("task_id", "unknown")
                            score = item.get(scorer_col)

                            if score is not None and not (
                                isinstance(score, float) and pd.isna(score)
                            ):
                                if task_id not in task_scores_json:
                                    task_scores_json[task_id] = []
                                task_scores_json[task_id].append(float(score))
                        except (KeyError, ValueError, TypeError) as e:
                            logger.warning(f"Error processing item in JSON stream: {e}")
                            continue
            except Exception as e:
                logger.error(f"Error reading JSON file {input_file}: {e}")
                raise
        else:
            # Handle CSV files
            task_scores_csv: dict[str, list[float]] = {}

            for chunk in pd.read_csv(
                input_file, usecols=columns_to_read, chunksize=chunk_size
            ):
                for _, row in chunk.iterrows():
                    task_id = row.get("task_id", "unknown")
                    score = row.get(scorer_col)
                    if pd.notna(score) and score is not None:
                        if task_id not in task_scores_csv:
                            task_scores_csv[task_id] = []
                        task_scores_csv[task_id].append(float(score))

        # Combine scores from both sources
        task_scores = (
            task_scores_json
            if input_file.suffix.lower() == ".json"
            else task_scores_csv
        )

        # Apply each callable to each task's scores
        for task_id, scores in task_scores.items():
            if task_id not in results:
                results[task_id] = {}
            # Apply each callable function
            for func in callable_funcs:
                func_name = func.__name__
                column_name = f"{func_name}_{scorer_col}"
                results[task_id][column_name] = func(scores)

    # Write results
    if output_filename.suffix.lower() == ".json":
        with open(output_filename, "w") as f:
            json.dump(results, f, indent=2)
    else:
        # Convert to DataFrame and save as CSV
        df_results = pd.DataFrame.from_dict(results, orient="index")
        df_results.index.name = "task_id"
        df_results.reset_index(inplace=True)
        df_results.to_csv(output_filename, index=False)

    logger.info(f"Streaming aggregation completed. Results saved to {output_filename}")


def _aggregate_by_task_memory(
    input_file: Path,
    output_filename: Path,
    callable_funcs: list[Callable[[list[float]], float]],
) -> None:
    """Memory-based aggregation by task_id - loads entire file into memory."""
    logger.info(f"Starting memory-based aggregation by task_id from {input_file}")

    # Read the entire file
    if input_file.suffix.lower() == ".json":
        with open(input_file) as f:
            if f.read(1) == "[":
                # JSON array format
                f.seek(0)
                df = pd.read_json(input_file)
            else:
                # JSONL format
                f.seek(0)
                df = pd.read_json(input_file, lines=True)
    else:
        df = pd.read_csv(input_file)

    # Get scorer columns (exclude base columns and reasoning columns)
    base_columns = ["user_id", "task_id", "turn_id", "agent_name"]
    scorer_columns = [
        col
        for col in df.columns
        if col not in base_columns and not col.endswith("_reasoning")
    ]

    # Group by task_id and aggregate each scorer column with multiple callables
    # Convert pandas Series to list for each group before applying callables
    results: dict[str, dict[str, float]] = {}
    for task_id, group in df.groupby("task_id"):
        results[task_id] = {}
        for col in scorer_columns:
            scores = group[col].dropna().tolist()  # Convert to list, remove NaN
            # Apply each callable function
            for func in callable_funcs:
                func_name = func.__name__
                column_name = f"{func_name}_{col}"
                results[task_id][column_name] = func(scores)

    # Convert to DataFrame
    results_df = pd.DataFrame.from_dict(results, orient="index")

    # Save results
    if output_filename.suffix.lower() == ".json":
        results_df.to_json(output_filename, orient="index", indent=2)
    else:
        results_df.reset_index().to_csv(output_filename, index=False)

    logger.info(
        f"Memory-based aggregation completed. Results saved to {output_filename}"
    )


def _aggregate_by_user_streaming(
    input_file: Path,
    output_filename: Path,
    callable_funcs: list[Callable[[list[float]], float]],
    chunk_size: int,
) -> None:
    """Streaming aggregation by user_id - processes column by column."""
    logger.info(f"Starting streaming aggregation by user_id from {input_file}")

    # Read the file to get column names
    if input_file.suffix.lower() == ".json":
        # Use ijson to get column names from first item
        try:
            with open(input_file, "rb") as f:
                parser = ijson.items(f, "item")
                first_item = next(parser)
                if first_item:
                    # Get all columns from the first item
                    all_columns = list(first_item.keys())
                else:
                    raise ValueError("Empty JSON array")
        except Exception as e:
            logger.error(f"Error reading column names from JSON file {input_file}: {e}")
            raise
    else:
        df_sample = pd.read_csv(input_file, nrows=1)
        all_columns = list(df_sample.columns)

    # Get base columns and scorer columns
    base_columns = ["user_id", "task_id", "turn_id", "agent_name"]
    scorer_columns = [
        col
        for col in all_columns
        if col not in base_columns and not col.endswith("_reasoning")
    ]

    # Initialize results dictionary
    results: dict[str, dict[str, float]] = {}

    # Process each scorer column separately
    for scorer_col in scorer_columns:
        logger.info(f"Processing column: {scorer_col}")

        # Read the file in chunks, processing only the grouping column and current scorer column
        columns_to_read = ["user_id", scorer_col]

        if input_file.suffix.lower() == ".json":
            # Handle JSON files with true streaming using ijson
            user_scores_json: dict[str, list[float]] = {}

            try:
                with open(input_file, "rb") as f:
                    # Stream through JSON array items
                    parser = ijson.items(f, "item")

                    for item in parser:
                        try:
                            user_id = item.get("user_id", "unknown")
                            score = item.get(scorer_col)

                            if score is not None and not (
                                isinstance(score, float) and pd.isna(score)
                            ):
                                if user_id not in user_scores_json:
                                    user_scores_json[user_id] = []
                                user_scores_json[user_id].append(float(score))
                        except (KeyError, ValueError, TypeError) as e:
                            logger.warning(f"Error processing item in JSON stream: {e}")
                            continue
            except Exception as e:
                logger.error(f"Error reading JSON file {input_file}: {e}")
                raise
        else:
            # Handle CSV files
            user_scores_csv: dict[str, list[float]] = {}

            for chunk in pd.read_csv(
                input_file, usecols=columns_to_read, chunksize=chunk_size
            ):
                for _, row in chunk.iterrows():
                    user_id = row.get("user_id", "unknown")
                    score = row.get(scorer_col)
                    if pd.notna(score) and score is not None:
                        if user_id not in user_scores_csv:
                            user_scores_csv[user_id] = []
                        user_scores_csv[user_id].append(float(score))

        # Combine scores from both sources
        user_scores = (
            user_scores_json
            if input_file.suffix.lower() == ".json"
            else user_scores_csv
        )

        # Apply callable to each user's scores
        for user_id, scores in user_scores.items():
            if user_id not in results:
                results[user_id] = {}
            for func in callable_funcs:
                func_name = func.__name__
                column_name = f"{func_name}_{scorer_col}"
                results[user_id][column_name] = func(scores)

    # Write results
    if output_filename.suffix.lower() == ".json":
        with open(output_filename, "w") as f:
            json.dump(results, f, indent=2)
    else:
        # Convert to DataFrame and save as CSV
        df_results = pd.DataFrame.from_dict(results, orient="index")
        df_results.index.name = "user_id"
        df_results.reset_index(inplace=True)
        df_results.to_csv(output_filename, index=False)

    logger.info(f"Streaming aggregation completed. Results saved to {output_filename}")


def _aggregate_by_user_memory(
    input_file: Path,
    output_filename: Path,
    callable_funcs: list[Callable[[list[float]], float]],
) -> None:
    """Memory-based aggregation by user_id - loads entire file into memory."""
    logger.info(f"Starting memory-based aggregation by user_id from {input_file}")

    # Read the entire file
    if input_file.suffix.lower() == ".json":
        with open(input_file) as f:
            if f.read(1) == "[":
                # JSON array format
                f.seek(0)
                df = pd.read_json(input_file)
            else:
                # JSONL format
                f.seek(0)
                df = pd.read_json(input_file, lines=True)
    else:
        df = pd.read_csv(input_file)

    # Get scorer columns (exclude base columns and reasoning columns)
    base_columns = ["user_id", "task_id", "turn_id", "agent_name"]
    scorer_columns = [
        col
        for col in df.columns
        if col not in base_columns and not col.endswith("_reasoning")
    ]

    # Group by user_id and aggregate each scorer column with multiple callables
    # Convert pandas Series to list for each group before applying callables
    results: dict[str, dict[str, float]] = {}
    for user_id, group in df.groupby("user_id"):
        results[user_id] = {}
        for col in scorer_columns:
            scores = group[col].dropna().tolist()  # Convert to list, remove NaN
            # Apply each callable function
            for func in callable_funcs:
                func_name = func.__name__
                column_name = f"{func_name}_{col}"
                results[user_id][column_name] = func(scores)

    # Convert to DataFrame
    results_df = pd.DataFrame.from_dict(results, orient="index")

    # Save results
    if output_filename.suffix.lower() == ".json":
        results_df.to_json(output_filename, orient="index", indent=2)
    else:
        results_df.reset_index().to_csv(output_filename, index=False)

    logger.info(
        f"Memory-based aggregation completed. Results saved to {output_filename}"
    )


def _aggregate_by_agent_streaming(
    input_file: Path,
    output_filename: Path,
    callable_funcs: list[Callable[[list[float]], float]],
    chunk_size: int,
) -> None:
    """Streaming aggregation by agent_name - processes column by column."""
    logger.info(f"Starting streaming aggregation by agent_name from {input_file}")

    # Read the file to get column names
    if input_file.suffix.lower() == ".json":
        # Use ijson to get column names from first item
        try:
            with open(input_file, "rb") as f:
                parser = ijson.items(f, "item")
                first_item = next(parser)
                if first_item:
                    # Get all columns from the first item
                    all_columns = list(first_item.keys())
                else:
                    raise ValueError("Empty JSON array")
        except Exception as e:
            logger.error(f"Error reading column names from JSON file {input_file}: {e}")
            raise
    else:
        df_sample = pd.read_csv(input_file, nrows=1)
        all_columns = list(df_sample.columns)

    # Get base columns and scorer columns
    base_columns = ["user_id", "task_id", "turn_id", "agent_name"]
    scorer_columns = [
        col
        for col in all_columns
        if col not in base_columns and not col.endswith("_reasoning")
    ]

    # Initialize results dictionary
    results: dict[str, dict[str, float]] = {}

    # Process each scorer column separately
    for scorer_col in scorer_columns:
        logger.info(f"Processing column: {scorer_col}")

        # Read the file in chunks, processing only the grouping column and current scorer column
        columns_to_read = ["agent_name", scorer_col]

        if input_file.suffix.lower() == ".json":
            # Handle JSON files with true streaming using ijson
            agent_scores_json: dict[str, list[float]] = {}

            try:
                with open(input_file, "rb") as f:
                    # Stream through JSON array items
                    parser = ijson.items(f, "item")

                    for item in parser:
                        try:
                            agent_name = item.get("agent_name", "unknown")
                            score = item.get(scorer_col)

                            if score is not None and not (
                                isinstance(score, float) and pd.isna(score)
                            ):
                                if agent_name not in agent_scores_json:
                                    agent_scores_json[agent_name] = []
                                agent_scores_json[agent_name].append(float(score))
                        except (KeyError, ValueError, TypeError) as e:
                            logger.warning(f"Error processing item in JSON stream: {e}")
                            continue
            except Exception as e:
                logger.error(f"Error reading JSON file {input_file}: {e}")
                raise
        else:
            # Handle CSV files
            agent_scores_csv: dict[str, list[float]] = {}

            for chunk in pd.read_csv(
                input_file, usecols=columns_to_read, chunksize=chunk_size
            ):
                for _, row in chunk.iterrows():
                    agent_name = row.get("agent_name", "unknown")
                    score = row.get(scorer_col)
                    if pd.notna(score) and score is not None:
                        if agent_name not in agent_scores_csv:
                            agent_scores_csv[agent_name] = []
                        agent_scores_csv[agent_name].append(float(score))

        # Combine scores from both sources
        agent_scores = (
            agent_scores_json
            if input_file.suffix.lower() == ".json"
            else agent_scores_csv
        )

        # Apply callable to each agent's scores
        for agent_name, scores in agent_scores.items():
            if agent_name not in results:
                results[agent_name] = {}
            for func in callable_funcs:
                func_name = func.__name__
                column_name = f"{func_name}_{scorer_col}"
                results[agent_name][column_name] = func(scores)

    # Write results
    if output_filename.suffix.lower() == ".json":
        with open(output_filename, "w") as f:
            json.dump(results, f, indent=2)
    else:
        # Convert to DataFrame and save as CSV
        df_results = pd.DataFrame.from_dict(results, orient="index")
        df_results.index.name = "agent_name"
        df_results.reset_index(inplace=True)
        df_results.to_csv(output_filename, index=False)

    logger.info(f"Streaming aggregation completed. Results saved to {output_filename}")


def _aggregate_by_agent_memory(
    input_file: Path,
    output_filename: Path,
    callable_funcs: list[Callable[[list[float]], float]],
) -> None:
    """Memory-based aggregation by agent_name - loads entire file into memory."""
    logger.info(f"Starting memory-based aggregation by agent_name from {input_file}")

    # Read the entire file
    if input_file.suffix.lower() == ".json":
        with open(input_file) as f:
            if f.read(1) == "[":
                # JSON array format
                f.seek(0)
                df = pd.read_json(input_file)
            else:
                # JSONL format
                f.seek(0)
                df = pd.read_json(input_file, lines=True)
    else:
        df = pd.read_csv(input_file)

    # Get scorer columns (exclude base columns and reasoning columns)
    base_columns = ["user_id", "task_id", "turn_id", "agent_name"]
    scorer_columns = [
        col
        for col in df.columns
        if col not in base_columns and not col.endswith("_reasoning")
    ]

    # Group by agent_name and aggregate each scorer column with multiple callables
    # Convert pandas Series to list for each group before applying callables
    results: dict[str, dict[str, float]] = {}
    for agent_name, group in df.groupby("agent_name"):
        results[agent_name] = {}
        for col in scorer_columns:
            scores = group[col].dropna().tolist()  # Convert to list, remove NaN
            # Apply each callable function
            for func in callable_funcs:
                func_name = func.__name__
                column_name = f"{func_name}_{col}"
                results[agent_name][column_name] = func(scores)

    # Convert to DataFrame
    results_df = pd.DataFrame.from_dict(results, orient="index")

    # Save results
    if output_filename.suffix.lower() == ".json":
        results_df.to_json(output_filename, orient="index", indent=2)
    else:
        results_df.reset_index().to_csv(output_filename, index=False)

    logger.info(
        f"Memory-based aggregation completed. Results saved to {output_filename}"
    )
