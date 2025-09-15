"""
Unit tests for novaeval.agents.noveum_spans_dataset module.

Tests all functions for processing Noveum spans data.
"""

import contextlib
import json
import os
import tempfile
from unittest.mock import mock_open, patch

import pandas as pd
import pytest

from novaeval.agents.agent_data import AgentData
from novaeval.datasets.noveum_spans_dataset import (
    create_dataset,
    noveum_spans_preprocessing,
    stream_dataset,
)


class TestNoveumSpansPreprocessing:
    """Test the noveum_spans_preprocessing function."""

    def test_both_parameters_provided_raises_error(self):
        """Test that providing both json_dir and json_files raises ValueError."""
        with pytest.raises(
            ValueError, match="Provide either json_dir or json_files, but not both"
        ):
            noveum_spans_preprocessing(json_dir="/path", json_files=["file.json"])

    def test_neither_parameter_provided_raises_error(self):
        """Test that providing neither json_dir nor json_files raises ValueError."""
        with pytest.raises(
            ValueError, match="Provide either json_dir or json_files, but not both"
        ):
            noveum_spans_preprocessing()

    def test_invalid_directory_raises_error(self):
        """Test that invalid directory path raises ValueError."""
        with pytest.raises(ValueError, match="is not a valid directory"):
            noveum_spans_preprocessing(json_dir="/nonexistent/path")

    def test_non_json_files_in_directory_raises_error(self):
        """Test that non-JSON files in directory raise ValueError."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a non-JSON file
            with open(os.path.join(temp_dir, "test.txt"), "w") as f:
                f.write("not json")

            with pytest.raises(ValueError, match="Directory contains non-JSON files"):
                noveum_spans_preprocessing(json_dir=temp_dir)

    def test_no_json_files_in_directory_raises_error(self):
        """Test that directory with no JSON files raises ValueError."""
        with (
            tempfile.TemporaryDirectory() as temp_dir,
            pytest.raises(ValueError, match="No JSON files found in the directory"),
        ):
            noveum_spans_preprocessing(json_dir=temp_dir)

    def test_invalid_json_files_list_raises_error(self):
        """Test that invalid json_files parameter raises ValueError."""
        with pytest.raises(ValueError, match="json_files must be a non-empty list"):
            noveum_spans_preprocessing(json_files="not_a_list")

        with pytest.raises(ValueError, match="json_files must be a non-empty list"):
            noveum_spans_preprocessing(json_files=[])

    def test_non_json_file_in_list_raises_error(self):
        """Test that non-JSON file in list raises ValueError."""
        with pytest.raises(ValueError, match="is not a JSON file"):
            noveum_spans_preprocessing(json_files=["test.txt"])

    @patch("builtins.open", new_callable=mock_open)
    @patch("os.path.exists")
    def test_json_file_not_found_continues(self, mock_exists, mock_file):
        """Test that FileNotFoundError is handled gracefully."""
        mock_exists.return_value = True
        mock_file.side_effect = FileNotFoundError("File not found")

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as temp_csv:
            temp_csv_path = temp_csv.name

        try:
            # Should not raise an error, just continue
            noveum_spans_preprocessing(
                json_files=["test.json"], output_csv=temp_csv_path
            )
        finally:
            os.unlink(temp_csv_path)

    @patch("builtins.open")
    @patch("os.path.exists")
    def test_invalid_json_continues(self, mock_exists, mock_open_func):
        """Test that invalid JSON is handled gracefully."""
        mock_exists.return_value = True
        mock_open_func.return_value.__enter__.return_value.read.return_value = (
            "invalid json"
        )

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as temp_csv:
            temp_csv_path = temp_csv.name

        try:
            # Should not raise an error, just continue
            noveum_spans_preprocessing(
                json_files=["test.json"], output_csv=temp_csv_path
            )
        finally:
            os.unlink(temp_csv_path)

    def test_successful_preprocessing_with_valid_json_files(self):
        """Test successful preprocessing with valid JSON files."""
        # Create sample JSON data
        sample_data = {
            "trace_id": "trace123",
            "spans": [
                {
                    "span_id": "span1",
                    "name": "test_span",
                    "start_time": "2023-01-01T00:00:00Z",
                    "end_time": "2023-01-01T00:01:00Z",
                    "attributes": {"key": "value"},
                    "events": [{"name": "event1", "timestamp": "2023-01-01T00:00:30Z"}],
                }
            ],
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            # Create JSON file
            json_file = os.path.join(temp_dir, "test.json")
            with open(json_file, "w") as f:
                json.dump(sample_data, f)

            # Create output CSV path
            output_csv = os.path.join(temp_dir, "output.csv")

            # Test preprocessing
            noveum_spans_preprocessing(json_files=[json_file], output_csv=output_csv)

            # Verify CSV was created
            assert os.path.exists(output_csv)

            # Verify CSV contents
            df = pd.read_csv(output_csv)
            assert len(df) == 1
            assert df.iloc[0]["trace_id"] == "trace123"
            assert df.iloc[0]["turn_id"] == "span1"  # span_id is mapped to turn_id

    def test_successful_preprocessing_with_json_directory(self):
        """Test successful preprocessing with JSON directory."""
        sample_data = {
            "trace_id": "trace456",
            "spans": [
                {
                    "span_id": "span2",
                    "name": "another_span",
                    "start_time": "2023-01-02T00:00:00Z",
                    "end_time": "2023-01-02T00:01:00Z",
                    "attributes": {},
                    "events": [],
                }
            ],
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            # Create JSON file in directory
            json_file = os.path.join(temp_dir, "test.json")
            with open(json_file, "w") as f:
                json.dump(sample_data, f)

            # Create output CSV path
            output_csv = os.path.join(temp_dir, "output.csv")

            # Test preprocessing with directory
            noveum_spans_preprocessing(json_dir=temp_dir, output_csv=output_csv)

            # Verify CSV was created and has content
            assert os.path.exists(output_csv)
            df = pd.read_csv(output_csv)
            assert len(df) == 1
            assert df.iloc[0]["trace_id"] == "trace456"
            assert df.iloc[0]["turn_id"] == "span2"  # span_id is mapped to turn_id


class TestNoveumSpansDatasetFunctions:
    """Test the dataset creation and streaming functions."""

    def test_create_dataset_success(self):
        """Test create_dataset with valid CSV file."""
        # Create sample CSV data matching expected format
        sample_data = {
            "turn_id": ["turn1", "turn2"],
            "agent_name": ["agent1", "agent2"],
            "agent_task": ["task1", "task2"],
            "agent_response": ["response1", "response2"],
            "trace_id": ["trace1", "trace2"],
            "agent_exit": [False, True],
        }

        temp_file_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".csv", delete=False
            ) as temp_file:
                temp_file_path = temp_file.name
                df = pd.DataFrame(sample_data)
                df.to_csv(temp_file_path, index=False)

            dataset = create_dataset(temp_file_path)
            assert hasattr(dataset, "data")
            assert len(dataset.data) == 2

            # Verify each item is an AgentData instance
            for item in dataset.data:
                assert isinstance(item, AgentData)
        finally:
            if temp_file_path and os.path.exists(temp_file_path):
                os.unlink(temp_file_path)

    def test_stream_dataset_success(self):
        """Test stream_dataset with valid CSV file."""
        # Create sample CSV data
        sample_data = {
            "turn_id": ["turn1", "turn2", "turn3"],
            "agent_name": ["agent1", "agent2", "agent3"],
            "agent_task": ["task1", "task2", "task3"],
            "agent_response": ["response1", "response2", "response3"],
        }

        temp_file_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".csv", delete=False
            ) as temp_file:
                temp_file_path = temp_file.name
                df = pd.DataFrame(sample_data)
                df.to_csv(temp_file_path, index=False)

            chunks = list(stream_dataset(temp_file_path, chunk_size=2))
            assert len(chunks) >= 1

            # Check that each chunk contains AgentData objects
            for chunk in chunks:
                assert isinstance(chunk, list)
                for item in chunk:
                    assert isinstance(item, AgentData)
        finally:
            if temp_file_path and os.path.exists(temp_file_path):
                os.unlink(temp_file_path)

    def test_create_dataset_file_not_found(self):
        """Test create_dataset with non-existent file."""
        with contextlib.suppress(FileNotFoundError, Exception):
            create_dataset("/nonexistent/file.csv")

    def test_stream_dataset_file_not_found(self):
        """Test stream_dataset with non-existent file."""
        with contextlib.suppress(FileNotFoundError, Exception):
            list(stream_dataset("/nonexistent/file.csv"))

    def test_preprocessing_with_tool_input_fields(self):
        """Test preprocessing with tool.input.* fields."""
        sample_data = {
            "trace_id": "trace123",
            "spans": [
                {
                    "span_id": "span1",
                    "name": "tool_span",
                    "start_time": "2023-01-01T00:00:00Z",
                    "end_time": "2023-01-01T00:01:00Z",
                    "attributes": {
                        "tool.input.query": "search query",
                        "tool.output.result": "search results",
                    },
                    "events": [],
                }
            ],
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            json_file = os.path.join(temp_dir, "test.json")
            with open(json_file, "w") as f:
                json.dump(sample_data, f)

            output_csv = os.path.join(temp_dir, "output.csv")

            noveum_spans_preprocessing(json_files=[json_file], output_csv=output_csv)

            assert os.path.exists(output_csv)
            df = pd.read_csv(output_csv)
            assert len(df) == 1
            assert df.iloc[0]["agent_task"] == "search query"

    def test_preprocessing_with_llm_prompts_field(self):
        """Test preprocessing with llm.prompts field when no other input fields exist."""
        sample_data = {
            "trace_id": "trace123",
            "spans": [
                {
                    "span_id": "span1",
                    "name": "llm_span",
                    "start_time": "2023-01-01T00:00:00Z",
                    "end_time": "2023-01-01T00:01:00Z",
                    "attributes": {
                        "llm.prompts": "prompt text",
                        "llm.completion": "completion text",
                    },
                    "events": [],
                }
            ],
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            json_file = os.path.join(temp_dir, "test.json")
            with open(json_file, "w") as f:
                json.dump(sample_data, f)

            output_csv = os.path.join(temp_dir, "output.csv")

            noveum_spans_preprocessing(json_files=[json_file], output_csv=output_csv)

            assert os.path.exists(output_csv)
            df = pd.read_csv(output_csv)
            assert len(df) == 1
            assert df.iloc[0]["agent_task"] == "prompt text"

    def test_preprocessing_multiple_input_fields_error(self):
        """Test preprocessing handles error when multiple input fields are found."""
        sample_data = {
            "trace_id": "trace123",
            "spans": [
                {
                    "span_id": "span1",
                    "name": "multi_input_span",
                    "start_time": "2023-01-01T00:00:00Z",
                    "end_time": "2023-01-01T00:01:00Z",
                    "attributes": {
                        "agent.input.query": "agent query",
                        "tool.input.search": "tool search",  # This creates multiple input fields
                    },
                    "events": [],
                }
            ],
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            json_file = os.path.join(temp_dir, "test.json")
            with open(json_file, "w") as f:
                json.dump(sample_data, f)

            output_csv = os.path.join(temp_dir, "output.csv")

            # Should handle the error gracefully and continue processing
            noveum_spans_preprocessing(json_files=[json_file], output_csv=output_csv)

            # When no spans are processed due to error, CSV file is not created
            # This tests the error handling path

    @patch("csv.field_size_limit")
    def test_create_dataset_overflow_error_handling(self, mock_field_size_limit):
        """Test create_dataset with OverflowError in field_size_limit."""
        # Mock field_size_limit to raise OverflowError initially
        mock_field_size_limit.side_effect = [OverflowError(), None]

        sample_dict = {
            "turn_id": ["turn1"],
            "agent_name": ["agent1"],
            "agent_task": ["task1"],
            "agent_response": ["response1"],
        }

        temp_file_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".csv", delete=False
            ) as temp_file:
                temp_file_path = temp_file.name
                df = pd.DataFrame(sample_dict)
                df.to_csv(temp_file_path, index=False)

            dataset = create_dataset(temp_file_path)
            assert hasattr(dataset, "data")

            # Verify that field_size_limit was called multiple times due to OverflowError
            assert mock_field_size_limit.call_count >= 2

        finally:
            if temp_file_path and os.path.exists(temp_file_path):
                os.unlink(temp_file_path)

    @patch("csv.field_size_limit")
    def test_stream_dataset_overflow_error_handling(self, mock_field_size_limit):
        """Test stream_dataset with OverflowError in field_size_limit."""
        # Mock field_size_limit to raise OverflowError initially
        mock_field_size_limit.side_effect = [OverflowError(), None]

        sample_data = {
            "turn_id": ["turn1"],
            "agent_name": ["agent1"],
            "agent_task": ["task1"],
            "agent_response": ["response1"],
        }

        temp_file_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".csv", delete=False
            ) as temp_file:
                temp_file_path = temp_file.name
                df = pd.DataFrame(sample_data)
                df.to_csv(temp_file_path, index=False)

            chunks = list(stream_dataset(temp_file_path, chunk_size=10))
            assert len(chunks) >= 1

            # Verify that field_size_limit was called multiple times due to OverflowError
            assert mock_field_size_limit.call_count >= 2

        finally:
            if temp_file_path and os.path.exists(temp_file_path):
                os.unlink(temp_file_path)

    def test_create_dataset_json_decode_error_handling(self):
        """Test create_dataset with invalid JSON in metadata field."""
        sample_data = {
            "turn_id": ["turn1"],
            "agent_name": ["agent1"],
            "agent_task": ["task1"],
            "agent_response": ["response1"],
            "metadata": ["invalid json {"],  # Malformed JSON
        }

        temp_file_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".csv", delete=False
            ) as temp_file:
                temp_file_path = temp_file.name
                df = pd.DataFrame(sample_data)
                df.to_csv(temp_file_path, index=False)

            dataset = create_dataset(temp_file_path)
            assert hasattr(dataset, "data")
            assert len(dataset.data) == 1

            # The agent should still be created with empty metadata
            agent = dataset.data[0]
            assert isinstance(agent, AgentData)

        finally:
            if temp_file_path and os.path.exists(temp_file_path):
                os.unlink(temp_file_path)

    def test_stream_dataset_json_decode_error_handling(self):
        """Test stream_dataset with invalid JSON in metadata field."""
        sample_data = {
            "turn_id": ["turn1"],
            "agent_name": ["agent1"],
            "agent_task": ["task1"],
            "agent_response": ["response1"],
            "metadata": ["invalid json {"],  # Malformed JSON
        }

        temp_file_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".csv", delete=False
            ) as temp_file:
                temp_file_path = temp_file.name
                df = pd.DataFrame(sample_data)
                df.to_csv(temp_file_path, index=False)

            chunks = list(stream_dataset(temp_file_path, chunk_size=10))
            assert len(chunks) >= 1

            # The agent should still be created with empty metadata
            for chunk in chunks:
                for agent in chunk:
                    assert isinstance(agent, AgentData)

        finally:
            if temp_file_path and os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
