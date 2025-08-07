"""
Aggregator functions for NovaEval evaluation results.

This module provides functions to aggregate evaluation results by different
grouping criteria (task, user, agent) with streaming support for memory efficiency.
"""

import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import pandas as pd

logger = logging.getLogger(__name__)


def mean_callable(scores: List[float]) -> float:
    """Default aggregation function - calculates mean."""
    if not scores:
        return 0.0
    return sum(scores) / len(scores)


def aggregate_by_task(
    input_file: Union[str, Path],
    output_filename: Union[str, Path],
    callable_func: Optional[Union[Callable[[List[float]], float], List[Callable[[List[float]], float]]]] = None,
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
        callable_funcs = [mean_callable]
    elif not isinstance(callable_func, list):
        callable_funcs = [callable_func]
    else:
        callable_funcs = callable_func
    
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    # Ensure output directory exists
    output_filename.parent.mkdir(parents=True, exist_ok=True)
    
    if streaming:
        _aggregate_by_task_streaming(input_file, output_filename, callable_funcs, chunk_size)
    else:
        _aggregate_by_task_memory(input_file, output_filename, callable_funcs)


def aggregate_by_user(
    input_file: Union[str, Path],
    output_filename: Union[str, Path],
    callable_func: Optional[Union[Callable[[List[float]], float], List[Callable[[List[float]], float]]]] = None,
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
        callable_funcs = [mean_callable]
    elif not isinstance(callable_func, list):
        callable_funcs = [callable_func]
    else:
        callable_funcs = callable_func
    
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    # Ensure output directory exists
    output_filename.parent.mkdir(parents=True, exist_ok=True)
    
    if streaming:
        _aggregate_by_user_streaming(input_file, output_filename, callable_funcs, chunk_size)
    else:
        _aggregate_by_user_memory(input_file, output_filename, callable_funcs)


def aggregate_by_agent_name(
    input_file: Union[str, Path],
    output_filename: Union[str, Path],
    callable_func: Optional[Union[Callable[[List[float]], float], List[Callable[[List[float]], float]]]] = None,
    streaming: bool = False,
    chunk_size: int = 1000,
) -> None:
    """
    Aggregate scores by agent_name.
    
    Args:
        input_file: Path to CSV/JSON from run_all
        output_filename: Where to save aggregated results
        callable_func: Function to apply to scores (default: mean)
        streaming: Whether to use streaming mode (processes column by column)
        chunk_size: How many rows to process at once in streaming mode
    """
    input_file = Path(input_file)
    output_filename = Path(output_filename)
    
    if callable_func is None:
        callable_func = mean_callable
    
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    # Ensure output directory exists
    output_filename.parent.mkdir(parents=True, exist_ok=True)
    
    if streaming:
        _aggregate_by_agent_streaming(input_file, output_filename, callable_func, chunk_size)
    else:
        _aggregate_by_agent_memory(input_file, output_filename, callable_func)


def _aggregate_by_task_streaming(
    input_file: Path,
    output_filename: Path,
    callable_funcs: List[Callable[[List[float]], float]],
    chunk_size: int,
) -> None:
    """Streaming aggregation by task_id - processes column by column."""
    logger.info(f"Starting streaming aggregation by task_id from {input_file}")
    
    # Read the file to get column names
    if input_file.suffix.lower() == '.json':
        with open(input_file, 'r') as f:
            first_line = f.readline().strip()
            if first_line.startswith('['):
                # JSON array format
                df_sample = pd.read_json(input_file, nrows=1)
            else:
                # JSONL format
                df_sample = pd.read_json(input_file, lines=True, nrows=1)
    else:
        df_sample = pd.read_csv(input_file, nrows=1)
    
    # Get base columns and scorer columns
    base_columns = ['user_id', 'task_id', 'turn_id', 'agent_name']
    scorer_columns = [col for col in df_sample.columns if col not in base_columns and not col.endswith('_reasoning')]
    
    # Initialize results dictionary
    results = {}
    
    # Process each scorer column separately
    for scorer_col in scorer_columns:
        logger.info(f"Processing column: {scorer_col}")
        
        # Read the file in chunks, processing only the grouping column and current scorer column
        columns_to_read = ['task_id', scorer_col]
        
        if input_file.suffix.lower() == '.json':
            # Handle JSON files
            task_scores = {}
            
            with open(input_file, 'r') as f:
                if f.read(1) == '[':
                    # JSON array format
                    f.seek(0)
                    for chunk in pd.read_json(input_file, chunksize=chunk_size):
                        for _, row in chunk.iterrows():
                            task_id = row.get('task_id', 'unknown')
                            score = row.get(scorer_col)
                            if pd.notna(score) and score is not None:
                                if task_id not in task_scores:
                                    task_scores[task_id] = []
                                task_scores[task_id].append(float(score))
                else:
                    # JSONL format
                    f.seek(0)
                    for chunk in pd.read_json(input_file, lines=True, chunksize=chunk_size):
                        for _, row in chunk.iterrows():
                            task_id = row.get('task_id', 'unknown')
                            score = row.get(scorer_col)
                            if pd.notna(score) and score is not None:
                                if task_id not in task_scores:
                                    task_scores[task_id] = []
                                task_scores[task_id].append(float(score))
        else:
            # Handle CSV files
            task_scores = {}
            
            for chunk in pd.read_csv(input_file, usecols=columns_to_read, chunksize=chunk_size):
                for _, row in chunk.iterrows():
                    task_id = row.get('task_id', 'unknown')
                    score = row.get(scorer_col)
                    if pd.notna(score) and score is not None:
                        if task_id not in task_scores:
                            task_scores[task_id] = []
                        task_scores[task_id].append(float(score))
        
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
    if output_filename.suffix.lower() == '.json':
        with open(output_filename, 'w') as f:
            json.dump(results, f, indent=2)
    else:
        # Convert to DataFrame and save as CSV
        df_results = pd.DataFrame.from_dict(results, orient='index')
        df_results.index.name = 'task_id'
        df_results.reset_index(inplace=True)
        df_results.to_csv(output_filename, index=False)
    
    logger.info(f"Streaming aggregation completed. Results saved to {output_filename}")


def _aggregate_by_task_memory(
    input_file: Path,
    output_filename: Path,
    callable_funcs: List[Callable[[List[float]], float]],
) -> None:
    """Memory-based aggregation by task_id - loads entire file into memory."""
    logger.info(f"Starting memory-based aggregation by task_id from {input_file}")
    
    # Read the entire file
    if input_file.suffix.lower() == '.json':
        with open(input_file, 'r') as f:
            if f.read(1) == '[':
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
    base_columns = ['user_id', 'task_id', 'turn_id', 'agent_name']
    scorer_columns = [col for col in df.columns if col not in base_columns and not col.endswith('_reasoning')]
    
    # Group by task_id and aggregate each scorer column with multiple callables
    # Convert pandas Series to list for each group before applying callables
    results = {}
    for task_id, group in df.groupby('task_id'):
        results[task_id] = {}
        for col in scorer_columns:
            scores = group[col].dropna().tolist()  # Convert to list, remove NaN
            # Apply each callable function
            for func in callable_funcs:
                func_name = func.__name__
                column_name = f"{func_name}_{col}"
                results[task_id][column_name] = func(scores)
    
    # Convert to DataFrame
    results = pd.DataFrame.from_dict(results, orient='index')
    
    # Save results
    if output_filename.suffix.lower() == '.json':
        results.to_json(output_filename, orient='index', indent=2)
    else:
        results.reset_index().to_csv(output_filename, index=False)
    
    logger.info(f"Memory-based aggregation completed. Results saved to {output_filename}")


def _aggregate_by_user_streaming(
    input_file: Path,
    output_filename: Path,
    callable_func: Callable[[List[float]], float],
    chunk_size: int,
) -> None:
    """Streaming aggregation by user_id - processes column by column."""
    logger.info(f"Starting streaming aggregation by user_id from {input_file}")
    
    # Read the file to get column names
    if input_file.suffix.lower() == '.json':
        with open(input_file, 'r') as f:
            first_line = f.readline().strip()
            if first_line.startswith('['):
                # JSON array format
                df_sample = pd.read_json(input_file, nrows=1)
            else:
                # JSONL format
                df_sample = pd.read_json(input_file, lines=True, nrows=1)
    else:
        df_sample = pd.read_csv(input_file, nrows=1)
    
    # Get base columns and scorer columns
    base_columns = ['user_id', 'task_id', 'turn_id', 'agent_name']
    scorer_columns = [col for col in df_sample.columns if col not in base_columns and not col.endswith('_reasoning')]
    
    # Initialize results dictionary
    results = {}
    
    # Process each scorer column separately
    for scorer_col in scorer_columns:
        logger.info(f"Processing column: {scorer_col}")
        
        # Read the file in chunks, processing only the grouping column and current scorer column
        columns_to_read = ['user_id', scorer_col]
        
        if input_file.suffix.lower() == '.json':
            # Handle JSON files
            user_scores = {}
            
            with open(input_file, 'r') as f:
                if f.read(1) == '[':
                    # JSON array format
                    f.seek(0)
                    for chunk in pd.read_json(input_file, chunksize=chunk_size):
                        for _, row in chunk.iterrows():
                            user_id = row.get('user_id', 'unknown')
                            score = row.get(scorer_col)
                            if pd.notna(score) and score is not None:
                                if user_id not in user_scores:
                                    user_scores[user_id] = []
                                user_scores[user_id].append(float(score))
                else:
                    # JSONL format
                    f.seek(0)
                    for chunk in pd.read_json(input_file, lines=True, chunksize=chunk_size):
                        for _, row in chunk.iterrows():
                            user_id = row.get('user_id', 'unknown')
                            score = row.get(scorer_col)
                            if pd.notna(score) and score is not None:
                                if user_id not in user_scores:
                                    user_scores[user_id] = []
                                user_scores[user_id].append(float(score))
        else:
            # Handle CSV files
            user_scores = {}
            
            for chunk in pd.read_csv(input_file, usecols=columns_to_read, chunksize=chunk_size):
                for _, row in chunk.iterrows():
                    user_id = row.get('user_id', 'unknown')
                    score = row.get(scorer_col)
                    if pd.notna(score) and score is not None:
                        if user_id not in user_scores:
                            user_scores[user_id] = []
                        user_scores[user_id].append(float(score))
        
        # Apply callable to each user's scores
        for user_id, scores in user_scores.items():
            if user_id not in results:
                results[user_id] = {}
            results[user_id][scorer_col] = callable_func(scores)
    
    # Write results
    if output_filename.suffix.lower() == '.json':
        with open(output_filename, 'w') as f:
            json.dump(results, f, indent=2)
    else:
        # Convert to DataFrame and save as CSV
        df_results = pd.DataFrame.from_dict(results, orient='index')
        df_results.index.name = 'user_id'
        df_results.reset_index(inplace=True)
        df_results.to_csv(output_filename, index=False)
    
    logger.info(f"Streaming aggregation completed. Results saved to {output_filename}")


def _aggregate_by_user_memory(
    input_file: Path,
    output_filename: Path,
    callable_funcs: List[Callable[[List[float]], float]],
) -> None:
    """Memory-based aggregation by user_id - loads entire file into memory."""
    logger.info(f"Starting memory-based aggregation by user_id from {input_file}")
    
    # Read the entire file
    if input_file.suffix.lower() == '.json':
        with open(input_file, 'r') as f:
            if f.read(1) == '[':
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
    base_columns = ['user_id', 'task_id', 'turn_id', 'agent_name']
    scorer_columns = [col for col in df.columns if col not in base_columns and not col.endswith('_reasoning')]
    
    # Group by user_id and aggregate each scorer column with multiple callables
    # Convert pandas Series to list for each group before applying callables
    results = {}
    for user_id, group in df.groupby('user_id'):
        results[user_id] = {}
        for col in scorer_columns:
            scores = group[col].dropna().tolist()  # Convert to list, remove NaN
            # Apply each callable function
            for func in callable_funcs:
                func_name = func.__name__
                column_name = f"{func_name}_{col}"
                results[user_id][column_name] = func(scores)
    
    # Convert to DataFrame
    results = pd.DataFrame.from_dict(results, orient='index')
    
    # Save results
    if output_filename.suffix.lower() == '.json':
        results.to_json(output_filename, orient='index', indent=2)
    else:
        results.reset_index().to_csv(output_filename, index=False)
    
    logger.info(f"Memory-based aggregation completed. Results saved to {output_filename}")


def _aggregate_by_agent_streaming(
    input_file: Path,
    output_filename: Path,
    callable_func: Callable[[List[float]], float],
    chunk_size: int,
) -> None:
    """Streaming aggregation by agent_name - processes column by column."""
    logger.info(f"Starting streaming aggregation by agent_name from {input_file}")
    
    # Read the file to get column names
    if input_file.suffix.lower() == '.json':
        with open(input_file, 'r') as f:
            first_line = f.readline().strip()
            if first_line.startswith('['):
                # JSON array format
                df_sample = pd.read_json(input_file, nrows=1)
            else:
                # JSONL format
                df_sample = pd.read_json(input_file, lines=True, nrows=1)
    else:
        df_sample = pd.read_csv(input_file, nrows=1)
    
    # Get base columns and scorer columns
    base_columns = ['user_id', 'task_id', 'turn_id', 'agent_name']
    scorer_columns = [col for col in df_sample.columns if col not in base_columns and not col.endswith('_reasoning')]
    
    # Initialize results dictionary
    results = {}
    
    # Process each scorer column separately
    for scorer_col in scorer_columns:
        logger.info(f"Processing column: {scorer_col}")
        
        # Read the file in chunks, processing only the grouping column and current scorer column
        columns_to_read = ['agent_name', scorer_col]
        
        if input_file.suffix.lower() == '.json':
            # Handle JSON files
            agent_scores = {}
            
            with open(input_file, 'r') as f:
                if f.read(1) == '[':
                    # JSON array format
                    f.seek(0)
                    for chunk in pd.read_json(input_file, chunksize=chunk_size):
                        for _, row in chunk.iterrows():
                            agent_name = row.get('agent_name', 'unknown')
                            score = row.get(scorer_col)
                            if pd.notna(score) and score is not None:
                                if agent_name not in agent_scores:
                                    agent_scores[agent_name] = []
                                agent_scores[agent_name].append(float(score))
                else:
                    # JSONL format
                    f.seek(0)
                    for chunk in pd.read_json(input_file, lines=True, chunksize=chunk_size):
                        for _, row in chunk.iterrows():
                            agent_name = row.get('agent_name', 'unknown')
                            score = row.get(scorer_col)
                            if pd.notna(score) and score is not None:
                                if agent_name not in agent_scores:
                                    agent_scores[agent_name] = []
                                agent_scores[agent_name].append(float(score))
        else:
            # Handle CSV files
            agent_scores = {}
            
            for chunk in pd.read_csv(input_file, usecols=columns_to_read, chunksize=chunk_size):
                for _, row in chunk.iterrows():
                    agent_name = row.get('agent_name', 'unknown')
                    score = row.get(scorer_col)
                    if pd.notna(score) and score is not None:
                        if agent_name not in agent_scores:
                            agent_scores[agent_name] = []
                        agent_scores[agent_name].append(float(score))
        
        # Apply callable to each agent's scores
        for agent_name, scores in agent_scores.items():
            if agent_name not in results:
                results[agent_name] = {}
            results[agent_name][scorer_col] = callable_func(scores)
    
    # Write results
    if output_filename.suffix.lower() == '.json':
        with open(output_filename, 'w') as f:
            json.dump(results, f, indent=2)
    else:
        # Convert to DataFrame and save as CSV
        df_results = pd.DataFrame.from_dict(results, orient='index')
        df_results.index.name = 'agent_name'
        df_results.reset_index(inplace=True)
        df_results.to_csv(output_filename, index=False)
    
    logger.info(f"Streaming aggregation completed. Results saved to {output_filename}")


def _aggregate_by_agent_memory(
    input_file: Path,
    output_filename: Path,
    callable_func: Callable[[List[float]], float],
) -> None:
    """Memory-based aggregation by agent_name - loads entire file into memory."""
    logger.info(f"Starting memory-based aggregation by agent_name from {input_file}")
    
    # Read the entire file
    if input_file.suffix.lower() == '.json':
        with open(input_file, 'r') as f:
            if f.read(1) == '[':
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
    base_columns = ['user_id', 'task_id', 'turn_id', 'agent_name']
    scorer_columns = [col for col in df.columns if col not in base_columns and not col.endswith('_reasoning')]
    
    # Group by agent_name and aggregate each scorer column
    # Convert pandas Series to list for each group before applying callable
    results = {}
    for agent_name, group in df.groupby('agent_name'):
        results[agent_name] = {}
        for col in scorer_columns:
            scores = group[col].dropna().tolist()  # Convert to list, remove NaN
            results[agent_name][col] = callable_func(scores)
    
    # Convert to DataFrame
    results = pd.DataFrame.from_dict(results, orient='index')
    
    # Save results
    if output_filename.suffix.lower() == '.json':
        results.to_json(output_filename, orient='index', indent=2)
    else:
        results.reset_index().to_csv(output_filename, index=False)
    
    logger.info(f"Memory-based aggregation completed. Results saved to {output_filename}") 