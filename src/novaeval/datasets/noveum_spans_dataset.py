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


def noveum_spans_preprocessing(
    json_dir: Optional[str] = None,
    json_files: Optional[list] = None,
    output_csv: str = "output.csv",
) -> None:
    """
    Preprocesses Noveum spans from JSON files, extracting spans data.
    Args:
        json_dir (str): Directory containing only JSON files.
        json_files (list): List of JSON file paths.
        output_csv (str): Path to save the output CSV.
    Raises:
        ValueError: If both or neither of json_dir and json_files are provided, or if non-JSON files are in the directory.
    """
    if (json_dir is not None and json_files is not None) or (
        json_dir is None and json_files is None
    ):
        raise ValueError("Provide either json_dir or json_files, but not both.")

    if json_dir is not None:
        if not os.path.isdir(json_dir):
            raise ValueError(f"{json_dir} is not a valid directory.")
        files = [os.path.join(json_dir, f) for f in os.listdir(json_dir)]
        json_files = [f for f in files if f.endswith(".json")]
        non_json = [f for f in files if not f.endswith(".json")]
        if non_json:
            raise ValueError(f"Directory contains non-JSON files: {non_json}")
        if not json_files:
            raise ValueError("No JSON files found in the directory.")
    else:
        if not isinstance(json_files, list) or not json_files:
            raise ValueError("json_files must be a non-empty list of file paths.")
        for f in json_files:
            if not f.endswith(".json"):
                raise ValueError(f"File {f} is not a JSON file.")

    # Process files
    rows = []

    for json_file in json_files:
        logger.info(f"Processing {json_file}")

        try:
            with open(json_file, encoding="utf-8") as f:
                data = json.load(f)

            # Extract spans from the trace
            spans = data.get("spans", [])
            trace_id = data.get("trace_id")

            # Sort spans by start_time to determine the last span
            spans_with_times = []
            for span in spans:
                start_time = span.get("start_time", "")
                spans_with_times.append((span, start_time))

            # Sort by start_time (assuming ISO format or similar)
            spans_with_times.sort(key=lambda x: x[1])

            for i, (span, _) in enumerate(spans_with_times):
                # Extract basic span information
                span_id = span.get("span_id")
                parent_span_id = span.get("parent_span_id")
                duration_ms = span.get("duration_ms")
                status_message = span.get("status_message")
                status = span.get("status", "")

                # Extract attributes
                attributes = span.get("attributes", {})

                # Extract agent_name from function.name
                agent_name = attributes.get("function.name", "")

                # Extract agent_task from agent.input.*, tool.input.*, or llm.prompts (ignore agent.input.topic)
                agent_task = ""
                input_fields = []
                for key, value in attributes.items():
                    if (
                        key.startswith("agent.input.") and key != "agent.input.topic"
                    ) or key.startswith("tool.input."):
                        input_fields.append((key, value))

                # For LLM calls, use llm.prompts as input if no other input fields found
                if not input_fields and "llm.prompts" in attributes:
                    input_fields.append(("llm.prompts", attributes["llm.prompts"]))

                # Raise exception if more than 1 input field
                if len(input_fields) > 1:
                    field_names = [field[0] for field in input_fields]
                    raise ValueError(
                        f"Expected only one input field (agent.input.* excluding topic, tool.input.*, or llm.prompts), but found: {field_names}"
                    )

                # Set agent_task from the single field
                if input_fields:
                    agent_task = str(input_fields[0][1])

                # Extract agent_response from agent.output.result, tool.output.result, or llm.completion
                agent_response = (
                    attributes.get("agent.output.result", "")
                    or attributes.get("tool.output.result", "")
                    or attributes.get("llm.completion", "")
                )

                # Extract agent_type for metadata (from agent.type, tool.type, or llm.provider)
                agent_type = (
                    attributes.get("agent.type", "")
                    or attributes.get("tool.type", "")
                    or attributes.get("llm.provider", "")
                )

                # Determine agent_exit - set True only for the last span in the trace
                agent_exit = i == len(spans_with_times) - 1

                # Set exit_status to None/blank for Noveum dataset as requested
                exit_status = None

                # Build metadata
                metadata = {
                    "trace_id": trace_id,
                    "duration_ms": duration_ms,
                    "parent_span_id": parent_span_id,
                    "status_message": status_message,
                    "agent_type": agent_type,
                    "exit_status": exit_status,
                    "agent_exit": agent_exit,
                }

                # Create row
                row = {
                    "turn_id": span_id,
                    "agent_name": agent_name,
                    "agent_task": agent_task,
                    "agent_response": agent_response,
                    "metadata": json.dumps(metadata),
                    "trace_id": trace_id,
                    "span_name": span.get("name", ""),
                    "status": status,
                    "start_time": span.get("start_time", ""),
                    "end_time": span.get("end_time", ""),
                    "attributes": json.dumps(attributes),
                    "exit_status": exit_status,
                    "agent_exit": agent_exit,
                }
                rows.append(row)

        except json.JSONDecodeError as e:
            logger.error(f"Error parsing JSON file {json_file}: {e}")
            continue
        except Exception as e:
            logger.error(f"Error processing file {json_file}: {e}")
            continue

    # Convert to DataFrame and save
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(output_csv, index=False, encoding="utf-8", quoting=1)
        logger.info(f"Processed {len(rows)} spans and saved to {output_csv}")
    else:
        logger.info("No spans found to process.")


def _prepare_dataset_csv(csv_path: str) -> str:
    """
    Helper function that reads the input CSV, maps and formats the data,
    writes it to a temporary CSV, and returns its path.

    Args:
        csv_path (str): Path to the input preprocessed CSV file

    Returns:
        str: Path to the temporary CSV file with processed data
    """
    import csv
    import sys
    import tempfile

    # Handle large CSV fields
    maxInt = sys.maxsize
    while True:
        try:
            csv.field_size_limit(maxInt)
            break
        except OverflowError:
            maxInt = int(maxInt / 10)

    df = pd.read_csv(csv_path)

    # Build new DataFrame with only the mapped columns
    rows = []
    for _, row in df.iterrows():
        turn_id = row.get("turn_id")
        agent_response = row.get("agent_response", "")
        exit_status = row.get("exit_status")
        agent_exit = row.get("agent_exit", False)

        # Build tool_call_results - for spans, we might not have traditional tool calls
        # but we can use the agent_response as a result
        tool_call_results = [
            {
                "call_id": turn_id,
                "result": agent_response,
                "success": True,
                "error_message": None,
            }
        ]

        # Get existing metadata and ensure it's properly formatted
        metadata_str = row.get("metadata", "{}")
        try:
            metadata = (
                json.loads(metadata_str)
                if isinstance(metadata_str, str)
                else metadata_str
            )
        except (json.JSONDecodeError, TypeError):
            metadata = {}

        # Add additional metadata from other columns
        metadata.update(
            {
                "span_name": row.get("span_name", ""),
                "status": row.get("status", ""),
                "start_time": row.get("start_time", ""),
                "end_time": row.get("end_time", ""),
                "trace_id": row.get("trace_id", ""),
                "exit_status": exit_status,
                "agent_exit": agent_exit,
            }
        )

        mapped = {
            "turn_id": turn_id,
            "agent_name": row.get("agent_name", ""),
            "agent_task": row.get("agent_task", ""),
            "agent_role": metadata.get(
                "agent_type", ""
            ),  # Use agent_type from metadata as role
            "system_prompt": "",  # Noveum spans don't typically have system prompts
            "agent_response": agent_response,
            "tool_call_results": json.dumps(tool_call_results),
            "metadata": json.dumps(metadata),
            "exit_status": exit_status,
            "agent_exit": agent_exit,
        }
        rows.append(mapped)

    # Write to a temp CSV and return its path
    with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".csv") as tmp:
        pd.DataFrame(rows).to_csv(tmp.name, index=False)
        tmp.flush()
        return tmp.name


def create_dataset(csv_path: str) -> AgentDataset:
    """
    Creates an AgentDataset from the preprocessed CSV file.

    Args:
        csv_path (str): Path to the preprocessed CSV file

    Returns:
        AgentDataset: Dataset ready for evaluation
    """
    # Use helper function to prepare the dataset CSV
    temp_csv_path = _prepare_dataset_csv(csv_path)

    try:
        dataset = AgentDataset()
        dataset.ingest_from_csv(
            file_path=temp_csv_path,
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
        # Clean up temp file
        os.unlink(temp_csv_path)


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
    # Use helper function to prepare the dataset CSV
    temp_csv_path = _prepare_dataset_csv(csv_path)

    try:
        # Now use AgentDataset's stream_from_csv
        dataset = AgentDataset()
        yield from dataset.stream_from_csv(
            file_path=temp_csv_path,
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
        # Clean up temp file
        os.unlink(temp_csv_path)
