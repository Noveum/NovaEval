import json
import logging
import os
from collections.abc import Iterator
from typing import Optional

import pandas as pd

from novaeval.agents.agent_data import AgentData
from novaeval.datasets.agent_dataset import AgentDataset

# Configure logger for this module
logger = logging.getLogger(__name__)


def _transform_dataframe_rows(df: pd.DataFrame) -> list[dict]:
    """
    Transform DataFrame rows into the required format for AgentDataset.

    Args:
        df (pd.DataFrame): Input DataFrame with swe agent trajectory data

    Returns:
        list[dict]: List of transformed row dictionaries
    """
    rows = []
    for _, row in df.iterrows():
        instance_id = row.get("instance_id")
        generated_patch = row.get("generated_patch")
        exit_status = row.get("exit_status")
        agent_exit = row.get("agent_exit", False)
        mask = row.get("mask")
        # Build tool_call_results as a list of dicts
        tool_call_results = [
            {
                "call_id": instance_id,
                "result": generated_patch,
                "success": True,
                "error_message": None,
            }
        ]
        # Build metadata as JSON string
        metadata = json.dumps(
            {"exit_status": exit_status, "mask": mask, "agent_exit": agent_exit}
        )
        mapped = {
            "turn_id": instance_id,
            "agent_name": row.get("model_name"),
            "agent_task": row.get("target"),
            "agent_role": row.get("role"),
            "system_prompt": row.get("system_prompt"),
            "agent_response": row.get("text"),
            "tool_call_results": json.dumps(tool_call_results),
            "metadata": metadata,
            "exit_status": exit_status,
            "agent_exit": agent_exit,
        }
        rows.append(mapped)
    return rows


def swe_agent_trajectories_preprocessing(
    parquet_dir: Optional[str] = None,
    parquet_files: Optional[list] = None,
    output_csv: str = "output.csv",
) -> None:
    """
    Preprocesses swe agent trajectories from parquet files, expanding the trajectory column.
    Args:
        parquet_dir (str): Directory containing only parquet files.
        parquet_files (list): List of parquet file paths.
        output_csv (str): Path to save the output CSV.
    Raises:
        ValueError: If both or neither of parquet_dir and parquet_files are provided, or if non-parquet files are in the directory.
    """
    if (parquet_dir is not None and parquet_files is not None) or (
        parquet_dir is None and parquet_files is None
    ):
        raise ValueError("Provide either parquet_dir or parquet_files, but not both.")

    if parquet_dir is not None:
        if not os.path.isdir(parquet_dir):
            raise ValueError(f"{parquet_dir} is not a valid directory.")
        files = [os.path.join(parquet_dir, f) for f in os.listdir(parquet_dir)]
        parquet_files = [f for f in files if f.endswith(".parquet")]
        non_parquet = [f for f in files if not f.endswith(".parquet")]
        if non_parquet:
            raise ValueError(f"Directory contains non-parquet files: {non_parquet}")
        if not parquet_files:
            raise ValueError("No parquet files found in the directory.")
    else:
        if not isinstance(parquet_files, list) or not parquet_files:
            raise ValueError("parquet_files must be a non-empty list of file paths.")
        for f in parquet_files:
            if not f.endswith(".parquet"):
                raise ValueError(f"File {f} is not a parquet file.")

    # Process files in chunks to save memory
    chunk_size = 1000  # Adjust this based on your memory constraints
    required_cols = [
        "instance_id",
        "model_name",
        "target",
        "trajectory",
        "exit_status",
        "generated_patch",
        "eval_logs",
    ]

    # Open output file in append mode
    try:
        with open(output_csv, "w", encoding="utf-8") as f:
            header_written = False

            for parquet_file in parquet_files:
                logger.info(f"Processing {parquet_file}")

                try:
                    # First read the file to check columns
                    df = pd.read_parquet(parquet_file, columns=required_cols)
                    missing = [col for col in required_cols if col not in df.columns]
                    if missing:
                        raise ValueError(f"Missing required columns: {missing}")

                    # Process in chunks
                    total_rows = len(df)
                    for start_idx in range(0, total_rows, chunk_size):
                        end_idx = min(start_idx + chunk_size, total_rows)
                        chunk_df = df.iloc[start_idx:end_idx]

                        # Process each chunk
                        rows = []
                        for _, row in chunk_df.iterrows():
                            traj = row["trajectory"]
                            # Handle both list and pandas Series
                            if hasattr(traj, "tolist"):
                                traj = traj.tolist()
                            if not isinstance(traj, list):
                                continue

                            # Create base row once
                            base = {
                                col: row[col]
                                for col in required_cols
                                if col != "trajectory"
                            }

                            # Process each step
                            for i, step in enumerate(traj):
                                if not isinstance(step, dict):
                                    continue
                                expanded = {**base, **step}

                                # Set agent_exit=True for the last item in the trajectory
                                expanded["agent_exit"] = i == len(traj) - 1

                                rows.append(expanded)

                        # Convert chunk to DataFrame and save
                        if rows:
                            chunk_output_df = pd.DataFrame(rows)
                            chunk_output_df.to_csv(
                                f,
                                index=False,
                                header=not header_written,
                                escapechar="\\",
                                encoding="utf-8",
                                quoting=1,
                            )
                            header_written = True

                        # Clear memory
                        del rows
                        if "chunk_output_df" in locals():
                            del chunk_output_df

                    # Clear memory after processing each file
                    del df
                except Exception as e:
                    logger.error(f"Error processing file {parquet_file}: {e}")
                    continue

            # Ensure header is written even if no data was processed
            if not header_written:
                # Write just the header row
                empty_df = pd.DataFrame(
                    columns=[
                        "instance_id",
                        "model_name",
                        "target",
                        "exit_status",
                        "generated_patch",
                        "eval_logs",
                        "agent_exit",
                    ]
                )
                empty_df.to_csv(
                    f,
                    index=False,
                    header=True,
                    escapechar="\\",
                    encoding="utf-8",
                    quoting=1,
                )
    except OSError as e:
        error_msg = f"Failed to open or write to output file '{output_csv}': {e}"
        logger.error(error_msg)
        raise RuntimeError(error_msg) from e


def create_dataset(csv_path: str) -> AgentDataset:
    import csv
    import sys

    maxInt = sys.maxsize

    while True:
        # decrease the maxInt value by factor 10
        # as long as the OverflowError occurs.
        try:
            csv.field_size_limit(maxInt)
            break
        except OverflowError:
            maxInt = int(maxInt / 10)

    df = pd.read_csv(csv_path)
    # Build new DataFrame with only the mapped columns
    rows = _transform_dataframe_rows(df)
    # Write to a temp CSV for ingest_from_csv
    import tempfile

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".csv") as tmp:
            pd.DataFrame(rows).to_csv(tmp.name, index=False)
            tmp.flush()
            tmp_path = tmp.name

        dataset = AgentDataset()
        dataset.ingest_from_csv(
            file_path=tmp_path,
            turn_id="turn_id",
            agent_name="agent_name",
            agent_task="agent_task",
            agent_role="agent_role",
            system_prompt="system_prompt",
            agent_response="agent_response",
            tool_call_results="tool_call_results",
            metadata="metadata",
            exit_status="exit_status",
            agent_exit="agent_exit",
        )
        return dataset
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                # Log the error but don't raise it to avoid masking the original exception
                logger.warning(f"Failed to clean up temporary file {tmp_path}")


def stream_dataset(csv_path: str, chunk_size: int = 1000) -> Iterator[list[AgentData]]:
    """
    Creates an iterator that yields chunks of AgentData objects from the preprocessed CSV.
    Similar to create_dataset but streams data instead of loading it all at once.

    Args:
        csv_path (str): Path to the preprocessed CSV file
        chunk_size (int): Number of rows to process at a time

    Returns:
        Iterator[list[AgentData]]: Iterator yielding lists of AgentData objects
    """
    import tempfile

    # First preprocess the CSV like in create_dataset
    df = pd.read_csv(csv_path)
    rows = _transform_dataframe_rows(df)

    # Use context manager only to get a temp file path
    with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".csv") as tmp:
        tmp_path = tmp.name

    pd.DataFrame(rows).to_csv(tmp_path, index=False)

    try:
        dataset = AgentDataset()
        yield from dataset.stream_from_csv(
            file_path=tmp_path,
            chunk_size=chunk_size,
            turn_id="turn_id",
            agent_name="agent_name",
            agent_task="agent_task",
            agent_role="agent_role",
            system_prompt="system_prompt",
            agent_response="agent_response",
            tool_call_results="tool_call_results",
            metadata="metadata",
            exit_status="exit_status",
            agent_exit="agent_exit",
        )
    finally:
        os.unlink(tmp_path)
