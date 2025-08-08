"""
Tests for the aggregators module.
"""

import json

import pandas as pd
import pytest

from novaeval.evaluators.aggregators import (
    _aggregate_by_agent_memory,
    _aggregate_by_agent_streaming,
    _aggregate_by_task_memory,
    _aggregate_by_task_streaming,
    _aggregate_by_user_memory,
    _aggregate_by_user_streaming,
    aggregate_by_agent_name,
    aggregate_by_task,
    aggregate_by_user,
    mean_callable,
)

pytestmark = pytest.mark.unit


class TestAggregators:
    """Test cases for aggregators module."""

    def test_mean_callable(self):
        """Test the mean_callable function."""
        # Test with normal list
        assert mean_callable([1.0, 2.0, 3.0]) == 2.0

        # Test with empty list
        assert mean_callable([]) == 0.0

        # Test with single value
        assert mean_callable([5.0]) == 5.0

        # Test with mixed values
        assert mean_callable([0.5, 1.5, 2.0]) == pytest.approx(1.33333, rel=1e-4)

    def test_aggregate_by_task_csv_memory(self, tmp_path):
        """Test aggregate_by_task with CSV file in memory mode."""
        # Create test CSV file
        test_data = pd.DataFrame(
            {
                "user_id": ["user1", "user2", "user1", "user2"],
                "task_id": ["task1", "task1", "task2", "task2"],
                "turn_id": ["turn1", "turn1", "turn1", "turn1"],
                "agent_name": ["agent1", "agent2", "agent1", "agent2"],
                "score1": [0.8, 0.9, 0.7, 0.6],
                "score2": [0.75, 0.85, 0.65, 0.55],
            }
        )

        input_file = tmp_path / "test_input.csv"
        test_data.to_csv(input_file, index=False)

        output_file = tmp_path / "task_aggregation.csv"

        # Run aggregation
        aggregate_by_task(
            input_file=input_file,
            output_filename=output_file,
            callable_func=mean_callable,
            streaming=False,
        )

        # Check output
        assert output_file.exists()
        result = pd.read_csv(output_file)

        # Should have 2 tasks
        assert len(result) == 2
        # The index becomes the task_id when reset_index is called
        assert (
            "task_id" in result.columns
            or result.index.name == "task_id"
            or "index" in result.columns
        )
        assert "mean_callable_score1" in result.columns
        assert "mean_callable_score2" in result.columns

    def test_aggregate_by_task_csv_streaming(self, tmp_path):
        """Test aggregate_by_task with CSV file in streaming mode."""
        # Create test CSV file
        test_data = pd.DataFrame(
            {
                "user_id": ["user1", "user2"],
                "task_id": ["task1", "task1"],
                "turn_id": ["turn1", "turn1"],
                "agent_name": ["agent1", "agent2"],
                "score1": [0.8, 0.9],
            }
        )

        input_file = tmp_path / "test_input.csv"
        test_data.to_csv(input_file, index=False)

        output_file = tmp_path / "task_aggregation.csv"

        # Run streaming aggregation
        aggregate_by_task(
            input_file=input_file,
            output_filename=output_file,
            callable_func=mean_callable,
            streaming=True,
            chunk_size=1,
        )

        # Check output
        assert output_file.exists()
        result = pd.read_csv(output_file)
        assert len(result) == 1  # One task

    def test_aggregate_by_task_json_array_memory(self, tmp_path):
        """Test aggregate_by_task with JSON array file in memory mode."""
        # Create test JSON array file
        test_data = [
            {
                "user_id": "user1",
                "task_id": "task1",
                "turn_id": "turn1",
                "agent_name": "agent1",
                "score1": 0.8,
            },
            {
                "user_id": "user2",
                "task_id": "task1",
                "turn_id": "turn1",
                "agent_name": "agent2",
                "score1": 0.9,
            },
        ]

        input_file = tmp_path / "test_input.json"
        with open(input_file, "w") as f:
            json.dump(test_data, f)

        output_file = tmp_path / "task_aggregation.json"

        # Run aggregation
        aggregate_by_task(
            input_file=input_file,
            output_filename=output_file,
            callable_func=mean_callable,
            streaming=False,
        )

        # Check output
        assert output_file.exists()
        with open(output_file) as f:
            result = json.load(f)

        assert "task1" in result

    def test_aggregate_by_task_json_array_streaming(self, tmp_path, monkeypatch):
        """Test aggregate_by_task with JSON array file in streaming mode using monkeypatched read_json."""
        from novaeval.evaluators import aggregators as ag

        # Create JSON array input
        data = [
            {
                "user_id": "u1",
                "task_id": "t1",
                "turn_id": "x",
                "agent_name": "a1",
                "score1": 1.0,
            },
            {
                "user_id": "u1",
                "task_id": "t1",
                "turn_id": "y",
                "agent_name": "a1",
                "score1": 0.0,
            },
        ]
        json_path = tmp_path / "in.json"
        with open(json_path, "w") as f:
            json.dump(data, f)

        df_full = pd.DataFrame(data)
        df_sample = df_full.head(1)

        def fake_read_json(path, *args, **kwargs):
            if kwargs.get("nrows") == 1:
                return df_sample
            if "chunksize" in kwargs:

                class _Chunks:
                    def __iter__(self):
                        yield df_full

                return _Chunks()
            return df_full

        monkeypatch.setattr(ag.pd, "read_json", fake_read_json)

        out = tmp_path / "task_stream_arr.csv"
        aggregate_by_task(
            json_path, out, callable_func=[mean_callable], streaming=True, chunk_size=10
        )
        assert out.exists()
        res = pd.read_csv(out)
        assert "task_id" in res.columns
        assert "mean_callable_score1" in res.columns

    def test_aggregate_by_task_jsonl_memory(self, tmp_path):
        """Test aggregate_by_task with JSONL file in memory mode."""
        # Create test JSONL file
        test_data = [
            {
                "user_id": "user1",
                "task_id": "task1",
                "turn_id": "turn1",
                "agent_name": "agent1",
                "score1": 0.8,
            },
            {
                "user_id": "user2",
                "task_id": "task1",
                "turn_id": "turn1",
                "agent_name": "agent2",
                "score1": 0.9,
            },
        ]

        input_file = tmp_path / "test_input.json"
        with open(input_file, "w") as f:
            for item in test_data:
                f.write(json.dumps(item) + "\n")

        output_file = tmp_path / "task_aggregation.csv"

        # Run aggregation
        aggregate_by_task(
            input_file=input_file,
            output_filename=output_file,
            callable_func=mean_callable,
            streaming=False,
        )

        # Check output
        assert output_file.exists()

    def test_aggregate_by_task_jsonl_streaming(self, tmp_path):
        """Test aggregate_by_task with JSON array file in streaming mode."""
        # Create test JSON array file
        test_data = [
            {
                "user_id": "user1",
                "task_id": "task1",
                "turn_id": "turn1",
                "agent_name": "agent1",
                "score1": 0.8,
            }
        ]

        input_file = tmp_path / "test_input.json"
        with open(input_file, "w") as f:
            json.dump(test_data, f)

        output_file = tmp_path / "task_aggregation.csv"

        # Run streaming aggregation
        aggregate_by_task(
            input_file=input_file,
            output_filename=output_file,
            callable_func=mean_callable,
            streaming=True,
        )

        # Check output
        assert output_file.exists()

    def test_aggregate_by_task_multiple_functions(self, tmp_path):
        """Test aggregate_by_task with multiple callable functions."""

        # Create custom aggregation functions
        def max_callable(scores):
            return max(scores) if scores else 0.0

        def min_callable(scores):
            return min(scores) if scores else 0.0

        # Create test CSV file
        test_data = pd.DataFrame(
            {
                "user_id": ["user1", "user2"],
                "task_id": ["task1", "task1"],
                "turn_id": ["turn1", "turn1"],
                "agent_name": ["agent1", "agent2"],
                "score1": [0.8, 0.9],
            }
        )

        input_file = tmp_path / "test_input.csv"
        test_data.to_csv(input_file, index=False)

        output_file = tmp_path / "task_aggregation.csv"

        # Run aggregation with multiple functions
        aggregate_by_task(
            input_file=input_file,
            output_filename=output_file,
            callable_func=[mean_callable, max_callable, min_callable],
            streaming=False,
        )

        # Check output
        assert output_file.exists()
        result = pd.read_csv(output_file)

        # Should have columns for all functions
        assert "mean_callable_score1" in result.columns
        assert "max_callable_score1" in result.columns
        assert "min_callable_score1" in result.columns

    def test_aggregate_by_task_default_function(self, tmp_path):
        """Test aggregate_by_task with default function (None)."""
        # Create test CSV file
        test_data = pd.DataFrame(
            {
                "user_id": ["user1"],
                "task_id": ["task1"],
                "turn_id": ["turn1"],
                "agent_name": ["agent1"],
                "score1": [0.8],
            }
        )

        input_file = tmp_path / "test_input.csv"
        test_data.to_csv(input_file, index=False)

        output_file = tmp_path / "task_aggregation.csv"

        # Run aggregation with None (should use default mean_callable)
        aggregate_by_task(
            input_file=input_file,
            output_filename=output_file,
            callable_func=None,
            streaming=False,
        )

        # Check output
        assert output_file.exists()
        result = pd.read_csv(output_file)
        assert "mean_callable_score1" in result.columns

    def test_aggregate_by_task_file_not_found(self, tmp_path):
        """Test aggregate_by_task with non-existent input file."""
        input_file = tmp_path / "nonexistent.csv"
        output_file = tmp_path / "output.csv"

        with pytest.raises(FileNotFoundError):
            aggregate_by_task(
                input_file=input_file,
                output_filename=output_file,
                callable_func=mean_callable,
            )

    def test_aggregate_by_user_csv_memory(self, tmp_path):
        """Test aggregate_by_user with CSV file in memory mode."""
        # Create test CSV file
        test_data = pd.DataFrame(
            {
                "user_id": ["user1", "user2", "user1", "user2"],
                "task_id": ["task1", "task1", "task2", "task2"],
                "turn_id": ["turn1", "turn1", "turn1", "turn1"],
                "agent_name": ["agent1", "agent2", "agent1", "agent2"],
                "score1": [0.8, 0.9, 0.7, 0.6],
            }
        )

        input_file = tmp_path / "test_input.csv"
        test_data.to_csv(input_file, index=False)

        output_file = tmp_path / "user_aggregation.csv"

        # Run aggregation
        aggregate_by_user(
            input_file=input_file,
            output_filename=output_file,
            callable_func=[mean_callable],
            streaming=False,
        )

        # Check output
        assert output_file.exists()
        result = pd.read_csv(output_file)

        # Should have 2 users
        assert len(result) == 2
        # Check for user_id column or index
        assert "user_id" in result.columns or "index" in result.columns

    def test_aggregate_by_user_streaming(self, tmp_path):
        """Test aggregate_by_user in streaming mode."""
        # Create test CSV file
        test_data = pd.DataFrame(
            {
                "user_id": ["user1", "user2", "user1", "user2"],
                "task_id": ["task1", "task1", "task2", "task2"],
                "turn_id": ["turn1", "turn1", "turn1", "turn1"],
                "agent_name": ["agent1", "agent2", "agent1", "agent2"],
                "score1": [0.8, 0.9, 0.7, 0.6],
            }
        )

        input_file = tmp_path / "test_input.csv"
        test_data.to_csv(input_file, index=False)

        output_file = tmp_path / "user_aggregation.csv"

        # Run streaming aggregation
        aggregate_by_user(
            input_file=input_file,
            output_filename=output_file,
            callable_func=[mean_callable],
            streaming=True,
            chunk_size=1,
        )

        # Check output
        assert output_file.exists()
        result = pd.read_csv(output_file)
        assert len(result) == 2
        assert "mean_callable_score1" in result.columns

    def test_aggregate_by_agent_name_csv_memory(self, tmp_path):
        """Test aggregate_by_agent_name with CSV file in memory mode."""
        # Create test CSV file
        test_data = pd.DataFrame(
            {
                "user_id": ["user1", "user2", "user1", "user2"],
                "task_id": ["task1", "task1", "task2", "task2"],
                "turn_id": ["turn1", "turn1", "turn1", "turn1"],
                "agent_name": ["agent1", "agent2", "agent1", "agent2"],
                "score1": [0.8, 0.9, 0.7, 0.6],
            }
        )

        input_file = tmp_path / "test_input.csv"
        test_data.to_csv(input_file, index=False)

        output_file = tmp_path / "agent_aggregation.csv"

        # Run aggregation
        aggregate_by_agent_name(
            input_file=input_file,
            output_filename=output_file,
            callable_func=[mean_callable],
            streaming=False,
        )

        # Check output
        assert output_file.exists()
        result = pd.read_csv(output_file)

        # Should have 2 agents
        assert len(result) == 2
        # Check for agent_name column or index
        assert "agent_name" in result.columns or "index" in result.columns

    def test_aggregate_by_agent_name_streaming(self, tmp_path):
        """Test aggregate_by_agent_name in streaming mode."""
        # Create test CSV file
        test_data = pd.DataFrame(
            {
                "user_id": ["user1", "user2"],
                "task_id": ["task1", "task1"],
                "turn_id": ["turn1", "turn1"],
                "agent_name": ["agent1", "agent2"],
                "score1": [0.8, 0.9],
            }
        )

        input_file = tmp_path / "test_input.csv"
        test_data.to_csv(input_file, index=False)

        output_file = tmp_path / "agent_aggregation.csv"

        # Run streaming aggregation
        aggregate_by_agent_name(
            input_file=input_file,
            output_filename=output_file,
            callable_func=[mean_callable],
            streaming=True,
            chunk_size=1,
        )

        # Check output
        assert output_file.exists()
        result = pd.read_csv(output_file)
        assert len(result) == 2
        assert "mean_callable_score1" in result.columns

    def test_streaming_user_aggregation_json_array(self, tmp_path, monkeypatch):
        """Test streaming user aggregation with JSON array."""
        # Use monkeypatch-based test defined elsewhere to cover JSON array streaming for user
        from novaeval.evaluators import aggregators as ag

        data = [
            {
                "user_id": "u1",
                "task_id": "t1",
                "turn_id": "x",
                "agent_name": "a1",
                "score1": 1.0,
            },
            {
                "user_id": "u1",
                "task_id": "t1",
                "turn_id": "y",
                "agent_name": "a1",
                "score1": 0.0,
            },
        ]
        json_path = tmp_path / "user_arr.json"
        with open(json_path, "w") as f:
            json.dump(data, f)

        df_full = pd.DataFrame(data)
        df_sample = df_full.head(1)

        def fake_read_json(path, *args, **kwargs):
            if kwargs.get("nrows") == 1:
                return df_sample
            if "chunksize" in kwargs:

                class _Chunks:
                    def __iter__(self):
                        yield df_full

                return _Chunks()
            return df_full

        monkeypatch.setattr(ag.pd, "read_json", fake_read_json)

        out = tmp_path / "user_stream_arr.csv"
        aggregate_by_user(
            json_path, out, callable_func=[mean_callable], streaming=True, chunk_size=10
        )
        assert out.exists()
        res = pd.read_csv(out)
        assert "user_id" in res.columns
        assert "mean_callable_score1" in res.columns

    def test_streaming_agent_aggregation_json_array(self, tmp_path, monkeypatch):
        """Test streaming agent aggregation with JSON array."""
        from novaeval.evaluators import aggregators as ag

        data = [
            {
                "user_id": "u1",
                "task_id": "t1",
                "turn_id": "x",
                "agent_name": "a1",
                "score1": 0.2,
            },
            {
                "user_id": "u2",
                "task_id": "t1",
                "turn_id": "y",
                "agent_name": "a1",
                "score1": 0.6,
            },
        ]
        json_path = tmp_path / "agent_arr.json"
        with open(json_path, "w") as f:
            json.dump(data, f)

        df_full = pd.DataFrame(data)
        df_sample = df_full.head(1)

        def fake_read_json(path, *args, **kwargs):
            if kwargs.get("nrows") == 1:
                return df_sample
            if "chunksize" in kwargs:

                class _Chunks:
                    def __iter__(self):
                        yield df_full

                return _Chunks()
            return df_full

        monkeypatch.setattr(ag.pd, "read_json", fake_read_json)

        out = tmp_path / "agent_stream_arr.csv"
        aggregate_by_agent_name(
            json_path, out, callable_func=[mean_callable], streaming=True, chunk_size=10
        )
        assert out.exists()
        res = pd.read_csv(out)
        assert "agent_name" in res.columns
        assert "mean_callable_score1" in res.columns

    def test_streaming_aggregation_jsonl(self, tmp_path):
        """Test streaming aggregation with JSON array format."""
        # Create test JSON array file
        test_data = [
            {
                "user_id": "user1",
                "task_id": "task1",
                "turn_id": "turn1",
                "agent_name": "agent1",
                "score1": 0.8,
            }
        ]

        input_file = tmp_path / "test_input.json"
        with open(input_file, "w") as f:
            json.dump(test_data, f)

        output_file = tmp_path / "task_aggregation.csv"

        # Run streaming aggregation
        aggregate_by_task(
            input_file=input_file,
            output_filename=output_file,
            callable_func=[mean_callable],
            streaming=True,
        )

        # Check output
        assert output_file.exists()

    def test_output_directory_creation(self, tmp_path):
        """Test that output directories are created if they don't exist."""
        # Create test CSV file
        test_data = pd.DataFrame(
            {
                "user_id": ["user1"],
                "task_id": ["task1"],
                "turn_id": ["turn1"],
                "agent_name": ["agent1"],
                "score1": [0.8],
            }
        )

        input_file = tmp_path / "test_input.csv"
        test_data.to_csv(input_file, index=False)

        # Output file in non-existent directory
        output_dir = tmp_path / "new_dir" / "subdir"
        output_file = output_dir / "task_aggregation.csv"

        # Run aggregation
        aggregate_by_task(
            input_file=input_file,
            output_filename=output_file,
            callable_func=mean_callable,
            streaming=False,
        )

        # Check that directory was created and output exists
        assert output_file.exists()
        assert output_dir.exists()

    def test_single_callable_function_conversion(self, tmp_path):
        """Test that single callable function is converted to list."""
        # Create test CSV file
        test_data = pd.DataFrame(
            {
                "user_id": ["user1"],
                "task_id": ["task1"],
                "turn_id": ["turn1"],
                "agent_name": ["agent1"],
                "score1": [0.8],
            }
        )

        input_file = tmp_path / "test_input.csv"
        test_data.to_csv(input_file, index=False)

        output_file = tmp_path / "task_aggregation.csv"

        # Run aggregation with single function (not in list)
        aggregate_by_task(
            input_file=input_file,
            output_filename=output_file,
            callable_func=mean_callable,  # Single function, not list
            streaming=False,
        )

        # Check output
        assert output_file.exists()
        result = pd.read_csv(output_file)
        assert "mean_callable_score1" in result.columns

    def test_private_streaming_functions_directly(self, tmp_path):
        """Test private streaming functions directly for better coverage."""
        # Create test CSV file
        test_data = pd.DataFrame(
            {
                "user_id": ["user1"],
                "task_id": ["task1"],
                "turn_id": ["turn1"],
                "agent_name": ["agent1"],
                "score1": [0.8],
            }
        )

        input_file = tmp_path / "test_input.csv"
        test_data.to_csv(input_file, index=False)

        output_file = tmp_path / "test_output.csv"

        # Test _aggregate_by_task_streaming directly
        _aggregate_by_task_streaming(
            input_file=input_file,
            output_filename=output_file,
            callable_funcs=[mean_callable],
            chunk_size=1000,
        )

        assert output_file.exists()

    def test_private_memory_functions_directly(self, tmp_path):
        """Test private memory functions directly for better coverage."""
        # Create test CSV file
        test_data = pd.DataFrame(
            {
                "user_id": ["user1"],
                "task_id": ["task1"],
                "turn_id": ["turn1"],
                "agent_name": ["agent1"],
                "score1": [0.8],
            }
        )

        input_file = tmp_path / "test_input.csv"
        test_data.to_csv(input_file, index=False)

        output_file = tmp_path / "test_output.json"

        # Test _aggregate_by_task_memory directly
        _aggregate_by_task_memory(
            input_file=input_file,
            output_filename=output_file,
            callable_funcs=[mean_callable],
        )

        assert output_file.exists()

    def test_streaming_with_multiple_callable_functions(self, tmp_path):
        """Test streaming aggregation with multiple callable functions."""

        def max_callable(scores):
            return max(scores) if scores else 0.0

        def min_callable(scores):
            return min(scores) if scores else 0.0

        # Create test CSV file
        test_data = pd.DataFrame(
            {
                "user_id": ["user1", "user2"],
                "task_id": ["task1", "task1"],
                "turn_id": ["turn1", "turn1"],
                "agent_name": ["agent1", "agent2"],
                "score1": [0.8, 0.9],
            }
        )

        input_file = tmp_path / "test_input.csv"
        test_data.to_csv(input_file, index=False)

        output_file = tmp_path / "task_aggregation.csv"

        # Test streaming with multiple functions
        _aggregate_by_task_streaming(
            input_file=input_file,
            output_filename=output_file,
            callable_funcs=[mean_callable, max_callable, min_callable],
            chunk_size=1000,
        )

        # Check output
        assert output_file.exists()
        result = pd.read_csv(output_file)

        # Should have columns for all functions
        assert "mean_callable_score1" in result.columns
        assert "max_callable_score1" in result.columns
        assert "min_callable_score1" in result.columns

    def test_aggregate_user_with_default_callable_none(self, tmp_path):
        """Test aggregate_by_user with None callable to cover line 87."""
        # Create test CSV file
        test_data = pd.DataFrame(
            {
                "user_id": ["user1"],
                "task_id": ["task1"],
                "turn_id": ["turn1"],
                "agent_name": ["agent1"],
                "score1": [0.8],
            }
        )

        input_file = tmp_path / "test_input.csv"
        test_data.to_csv(input_file, index=False)

        output_file = tmp_path / "user_aggregation.csv"

        # Run aggregation with None (should use default mean_callable)
        aggregate_by_user(
            input_file=input_file,
            output_filename=output_file,
            callable_func=None,  # This should trigger line 87
            streaming=False,
        )

        # Check output
        assert output_file.exists()
        result = pd.read_csv(output_file)
        assert "mean_callable_score1" in result.columns

    def test_aggregate_agent_with_single_callable(self, tmp_path):
        """Test aggregate_by_agent_name with single callable to cover line 89."""

        def custom_callable(scores):
            return sum(scores) if scores else 0.0

        # Create test CSV file
        test_data = pd.DataFrame(
            {
                "user_id": ["user1"],
                "task_id": ["task1"],
                "turn_id": ["turn1"],
                "agent_name": ["agent1"],
                "score1": [0.8],
            }
        )

        input_file = tmp_path / "test_input.csv"
        test_data.to_csv(input_file, index=False)

        output_file = tmp_path / "agent_aggregation.csv"

        # Run aggregation with single callable (not in list) - should trigger line 89
        aggregate_by_agent_name(
            input_file=input_file,
            output_filename=output_file,
            callable_func=custom_callable,  # Single function, not list
            streaming=False,
        )

        # Check output
        assert output_file.exists()
        result = pd.read_csv(output_file)
        assert "custom_callable_score1" in result.columns

    def test_aggregate_task_file_not_found_error(self, tmp_path):
        """Test FileNotFoundError to cover line 94."""
        nonexistent_file = tmp_path / "nonexistent.csv"
        output_file = tmp_path / "output.csv"

        # This should trigger line 94
        with pytest.raises(FileNotFoundError, match="Input file not found"):
            aggregate_by_task(
                input_file=nonexistent_file,
                output_filename=output_file,
                callable_func=mean_callable,
            )

    def test_aggregate_user_file_not_found_error(self, tmp_path):
        """Test FileNotFoundError for user aggregation."""
        nonexistent_file = tmp_path / "nonexistent.csv"
        output_file = tmp_path / "output.csv"

        with pytest.raises(FileNotFoundError, match="Input file not found"):
            aggregate_by_user(
                input_file=nonexistent_file,
                output_filename=output_file,
                callable_func=mean_callable,
            )

    def test_aggregate_agent_file_not_found_error(self, tmp_path):
        """Test FileNotFoundError for agent aggregation."""
        nonexistent_file = tmp_path / "nonexistent.csv"
        output_file = tmp_path / "output.csv"

        with pytest.raises(FileNotFoundError, match="Input file not found"):
            aggregate_by_agent_name(
                input_file=nonexistent_file,
                output_filename=output_file,
                callable_func=mean_callable,
            )

    def test_memory_aggregation_with_list_of_callables(self, tmp_path):
        """Test memory aggregation with multiple callables."""

        def sum_callable(scores):
            return sum(scores) if scores else 0.0

        def count_callable(scores):
            return len(scores) if scores else 0

        # Create test CSV file
        test_data = pd.DataFrame(
            {
                "user_id": ["user1", "user1"],
                "task_id": ["task1", "task1"],
                "turn_id": ["turn1", "turn2"],
                "agent_name": ["agent1", "agent1"],
                "score1": [0.8, 0.9],
                "score2": [0.7, 0.8],
            }
        )

        input_file = tmp_path / "test_input.csv"
        test_data.to_csv(input_file, index=False)

        output_file = tmp_path / "user_aggregation.csv"

        # Test with list of callables - should trigger line 91
        aggregate_by_user(
            input_file=input_file,
            output_filename=output_file,
            callable_func=[mean_callable, sum_callable, count_callable],
            streaming=False,
        )

        # Check output
        assert output_file.exists()
        result = pd.read_csv(output_file)

        # Should have columns for all functions and both score columns
        assert "mean_callable_score1" in result.columns
        assert "sum_callable_score1" in result.columns
        assert "count_callable_score1" in result.columns
        assert "mean_callable_score2" in result.columns

    def test_memory_aggregation_jsonl_format(self, tmp_path):
        """Test memory aggregation with JSONL format."""
        # Create test JSONL file
        test_data = [
            {
                "user_id": "user1",
                "task_id": "task1",
                "turn_id": "turn1",
                "agent_name": "agent1",
                "score1": 0.8,
                "score2": 0.7,
            },
            {
                "user_id": "user1",
                "task_id": "task1",
                "turn_id": "turn2",
                "agent_name": "agent1",
                "score1": 0.9,
                "score2": 0.8,
            },
        ]

        input_file = tmp_path / "test_input.json"
        with open(input_file, "w") as f:
            for item in test_data:
                f.write(json.dumps(item) + "\n")

        output_file = tmp_path / "task_aggregation.json"

        # Run aggregation
        aggregate_by_task(
            input_file=input_file,
            output_filename=output_file,
            callable_func=[mean_callable],
            streaming=False,
        )

        # Check output
        assert output_file.exists()
        with open(output_file) as f:
            result = json.load(f)

        assert "task1" in result

    def test_memory_aggregation_json_array_format(self, tmp_path):
        """Test memory aggregation with JSON array format."""
        # Create test JSON array file
        test_data = [
            {
                "user_id": "user1",
                "task_id": "task1",
                "turn_id": "turn1",
                "agent_name": "agent1",
                "score1": 0.8,
            },
            {
                "user_id": "user2",
                "task_id": "task1",
                "turn_id": "turn1",
                "agent_name": "agent2",
                "score1": 0.9,
            },
        ]

        input_file = tmp_path / "test_input.json"
        with open(input_file, "w") as f:
            json.dump(test_data, f)

        output_file = tmp_path / "agent_aggregation.csv"

        # Run aggregation
        aggregate_by_agent_name(
            input_file=input_file,
            output_filename=output_file,
            callable_func=[mean_callable],
            streaming=False,
        )

        # Check output
        assert output_file.exists()

    def test_private_memory_functions_with_json_output(self, tmp_path):
        """Test private memory functions with JSON output."""
        # Create test CSV file
        test_data = pd.DataFrame(
            {
                "user_id": ["user1", "user2"],
                "task_id": ["task1", "task1"],
                "turn_id": ["turn1", "turn1"],
                "agent_name": ["agent1", "agent2"],
                "score1": [0.8, 0.9],
            }
        )

        input_file = tmp_path / "test_input.csv"
        test_data.to_csv(input_file, index=False)

        # Test user aggregation with JSON output
        output_file_user = tmp_path / "user_test_output.json"
        _aggregate_by_user_memory(
            input_file=input_file,
            output_filename=output_file_user,
            callable_funcs=[mean_callable],
        )
        assert output_file_user.exists()

        # Test agent aggregation with JSON output
        output_file_agent = tmp_path / "agent_test_output.json"
        _aggregate_by_agent_memory(
            input_file=input_file,
            output_filename=output_file_agent,
            callable_funcs=[mean_callable],
        )
        assert output_file_agent.exists()

    def test_memory_functions_with_csv_output(self, tmp_path):
        """Test memory functions with CSV output."""
        # Create test CSV file
        test_data = pd.DataFrame(
            {
                "user_id": ["user1", "user2"],
                "task_id": ["task1", "task1"],
                "turn_id": ["turn1", "turn1"],
                "agent_name": ["agent1", "agent2"],
                "score1": [0.8, 0.9],
            }
        )

        input_file = tmp_path / "test_input.csv"
        test_data.to_csv(input_file, index=False)

        # Test user aggregation with CSV output
        output_file_user = tmp_path / "user_test_output.csv"
        _aggregate_by_user_memory(
            input_file=input_file,
            output_filename=output_file_user,
            callable_funcs=[mean_callable],
        )
        assert output_file_user.exists()
        result = pd.read_csv(output_file_user)
        # Check for user_id column or index (pandas may name it differently)
        assert "user_id" in result.columns or "index" in result.columns

        # Test agent aggregation with CSV output
        output_file_agent = tmp_path / "agent_test_output.csv"
        _aggregate_by_agent_memory(
            input_file=input_file,
            output_filename=output_file_agent,
            callable_funcs=[mean_callable],
        )
        assert output_file_agent.exists()
        result = pd.read_csv(output_file_agent)
        # Check for agent_name column or index (pandas may name it differently)
        assert "agent_name" in result.columns or "index" in result.columns

    def test_aggregation_with_complex_data(self, tmp_path):
        """Test aggregation with more complex data scenarios."""
        # Create test data with multiple tasks, users, and agents
        test_data = pd.DataFrame(
            {
                "user_id": ["user1", "user1", "user2", "user2", "user3"],
                "task_id": ["task1", "task2", "task1", "task2", "task1"],
                "turn_id": ["turn1", "turn1", "turn1", "turn1", "turn1"],
                "agent_name": ["agent1", "agent1", "agent2", "agent2", "agent3"],
                "accuracy": [0.8, 0.9, 0.7, 0.85, 0.95],
                "precision": [0.75, 0.88, 0.72, 0.80, 0.92],
                "recall": [0.85, 0.91, 0.68, 0.87, 0.97],
            }
        )

        input_file = tmp_path / "complex_input.csv"
        test_data.to_csv(input_file, index=False)

        # Test task aggregation
        task_output = tmp_path / "complex_task_aggregation.csv"
        aggregate_by_task(
            input_file=input_file,
            output_filename=task_output,
            callable_func=[mean_callable],
            streaming=False,
        )

        assert task_output.exists()
        task_result = pd.read_csv(task_output)
        assert len(task_result) == 2  # 2 tasks
        assert "mean_callable_accuracy" in task_result.columns
        assert "mean_callable_precision" in task_result.columns
        assert "mean_callable_recall" in task_result.columns

        # Test user aggregation
        user_output = tmp_path / "complex_user_aggregation.csv"
        aggregate_by_user(
            input_file=input_file,
            output_filename=user_output,
            callable_func=[mean_callable],
            streaming=False,
        )

        assert user_output.exists()
        user_result = pd.read_csv(user_output)
        assert len(user_result) == 3  # 3 users

        # Test agent aggregation
        agent_output = tmp_path / "complex_agent_aggregation.csv"
        aggregate_by_agent_name(
            input_file=input_file,
            output_filename=agent_output,
            callable_func=[mean_callable],
            streaming=False,
        )

        assert agent_output.exists()
        agent_result = pd.read_csv(agent_output)
        assert len(agent_result) == 3  # 3 agents

    def test_aggregate_by_task_memory_with_empty_file(self, tmp_path):
        """Test memory-based task aggregation with an empty file."""
        # Create empty CSV file
        input_file = tmp_path / "empty_input.csv"
        pd.DataFrame(
            columns=["user_id", "task_id", "turn_id", "agent_name", "score1"]
        ).to_csv(input_file, index=False)

        output_file = tmp_path / "empty_task_aggregation.csv"

        # Run aggregation
        _aggregate_by_task_memory(
            input_file=input_file,
            output_filename=output_file,
            callable_funcs=[mean_callable],
        )

        # Check output
        assert output_file.exists()
        result = pd.read_csv(output_file)
        assert len(result) == 0

    def test_aggregate_by_task_memory_with_all_nan_values(self, tmp_path):
        """Test memory-based task aggregation with all NaN values."""
        # Create test CSV file with NaN values
        test_data = pd.DataFrame(
            {
                "user_id": ["user1", "user2"],
                "task_id": ["task1", "task1"],
                "turn_id": ["turn1", "turn1"],
                "agent_name": ["agent1", "agent2"],
                "score1": [float("nan"), float("nan")],
            }
        )

        input_file = tmp_path / "nan_input.csv"
        test_data.to_csv(input_file, index=False)

        output_file = tmp_path / "nan_task_aggregation.csv"

        # Run aggregation
        _aggregate_by_task_memory(
            input_file=input_file,
            output_filename=output_file,
            callable_funcs=[mean_callable],
        )

        # Check output
        assert output_file.exists()
        result = pd.read_csv(output_file)
        assert len(result) == 1  # Should have one row for task1
        # Since mean_callable returns 0.0 for empty lists, and NaN values are filtered out
        assert result.iloc[0]["mean_callable_score1"] == 0.0

    def test_aggregate_by_task_memory_with_mixed_values(self, tmp_path):
        """Test memory-based task aggregation with mix of valid and NaN values."""
        # Create test CSV file with mixed values
        test_data = pd.DataFrame(
            {
                "user_id": ["user1", "user2", "user3"],
                "task_id": ["task1", "task1", "task1"],
                "turn_id": ["turn1", "turn1", "turn1"],
                "agent_name": ["agent1", "agent2", "agent3"],
                "score1": [0.8, float("nan"), 0.9],
            }
        )

        input_file = tmp_path / "mixed_input.csv"
        test_data.to_csv(input_file, index=False)

        output_file = tmp_path / "mixed_task_aggregation.csv"

        # Run aggregation
        _aggregate_by_task_memory(
            input_file=input_file,
            output_filename=output_file,
            callable_funcs=[mean_callable],
        )

        # Check output
        assert output_file.exists()
        result = pd.read_csv(output_file)
        assert len(result) == 1  # Should have one row for task1
        assert result.iloc[0]["mean_callable_score1"] == pytest.approx(
            0.85
        )  # Mean of 0.8 and 0.9

    def test_aggregate_by_task_memory_with_invalid_json(self, tmp_path):
        """Test memory-based task aggregation with invalid JSON file."""
        # Create invalid JSON file
        input_file = tmp_path / "invalid.json"
        with open(input_file, "w") as f:
            f.write('{"invalid": "json"')  # Missing closing brace

        output_file = tmp_path / "output.csv"

        # Should raise a JSON parsing error
        with pytest.raises(ValueError):
            _aggregate_by_task_memory(
                input_file=input_file,
                output_filename=output_file,
                callable_funcs=[mean_callable],
            )

    def test_aggregate_by_task_memory_with_invalid_csv(self, tmp_path):
        """Test memory-based task aggregation with invalid CSV file."""
        # Create invalid CSV file
        input_file = tmp_path / "invalid.csv"
        with open(input_file, "w") as f:
            f.write("invalid,csv\ndata")  # Invalid CSV format

        output_file = tmp_path / "output.csv"

        # Should raise KeyError for missing task_id column
        with pytest.raises(KeyError):
            _aggregate_by_task_memory(
                input_file=input_file,
                output_filename=output_file,
                callable_funcs=[mean_callable],
            )

    def test_aggregate_by_task_memory_with_missing_columns(self, tmp_path):
        """Test memory-based task aggregation with missing required columns."""
        # Create CSV file without required columns
        test_data = pd.DataFrame(
            {"user_id": ["user1"], "agent_name": ["agent1"], "score1": [0.8]}
        )

        input_file = tmp_path / "missing_columns.csv"
        test_data.to_csv(input_file, index=False)

        output_file = tmp_path / "output.csv"

        # Should raise KeyError for missing task_id
        with pytest.raises(KeyError):
            _aggregate_by_task_memory(
                input_file=input_file,
                output_filename=output_file,
                callable_funcs=[mean_callable],
            )

    def test_aggregate_by_task_memory_with_custom_callable(self, tmp_path):
        """Test memory-based task aggregation with custom callable function."""

        def custom_callable(scores):
            """Return sum of valid scores."""
            valid_scores = [s for s in scores if pd.notna(s)]
            return sum(valid_scores) if valid_scores else 0.0

        # Create test CSV file
        test_data = pd.DataFrame(
            {
                "user_id": ["user1", "user2", "user3"],
                "task_id": ["task1", "task1", "task1"],
                "turn_id": ["turn1", "turn1", "turn1"],
                "agent_name": ["agent1", "agent2", "agent3"],
                "score1": [0.8, float("nan"), 0.9],
            }
        )

        input_file = tmp_path / "custom_input.csv"
        test_data.to_csv(input_file, index=False)

        output_file = tmp_path / "custom_task_aggregation.csv"

        # Run aggregation with custom callable
        _aggregate_by_task_memory(
            input_file=input_file,
            output_filename=output_file,
            callable_funcs=[custom_callable],
        )

        # Check output
        assert output_file.exists()
        result = pd.read_csv(output_file)
        assert len(result) == 1  # Should have one row for task1
        assert result.iloc[0]["custom_callable_score1"] == pytest.approx(
            1.7
        )  # Sum of 0.8 and 0.9

    def test_aggregate_by_task_memory_with_multiple_score_columns(self, tmp_path):
        """Test memory-based task aggregation with multiple score columns."""
        # Create test CSV file with multiple score columns
        test_data = pd.DataFrame(
            {
                "user_id": ["user1", "user2"],
                "task_id": ["task1", "task1"],
                "turn_id": ["turn1", "turn1"],
                "agent_name": ["agent1", "agent2"],
                "accuracy": [0.8, 0.9],
                "precision": [0.75, 0.85],
                "recall": [0.85, 0.95],
            }
        )

        input_file = tmp_path / "multi_score_input.csv"
        test_data.to_csv(input_file, index=False)

        output_file = tmp_path / "multi_score_task_aggregation.csv"

        # Run aggregation
        _aggregate_by_task_memory(
            input_file=input_file,
            output_filename=output_file,
            callable_funcs=[mean_callable],
        )

        # Check output
        assert output_file.exists()
        result = pd.read_csv(output_file)
        assert len(result) == 1  # Should have one row for task1
        assert "mean_callable_accuracy" in result.columns
        assert "mean_callable_precision" in result.columns
        assert "mean_callable_recall" in result.columns
        assert result.iloc[0]["mean_callable_accuracy"] == pytest.approx(0.85)
        assert result.iloc[0]["mean_callable_precision"] == 0.80
        assert result.iloc[0]["mean_callable_recall"] == pytest.approx(0.90)

    def test_aggregate_by_task_memory_with_reasoning_columns(self, tmp_path):
        """Test memory-based task aggregation with reasoning columns."""
        # Create test CSV file with reasoning columns
        test_data = pd.DataFrame(
            {
                "user_id": ["user1", "user2"],
                "task_id": ["task1", "task1"],
                "turn_id": ["turn1", "turn1"],
                "agent_name": ["agent1", "agent2"],
                "score1": [0.8, 0.9],
                "score1_reasoning": ["Good", "Excellent"],  # Should be ignored
            }
        )

        input_file = tmp_path / "reasoning_input.csv"
        test_data.to_csv(input_file, index=False)

        output_file = tmp_path / "reasoning_task_aggregation.csv"

        # Run aggregation
        _aggregate_by_task_memory(
            input_file=input_file,
            output_filename=output_file,
            callable_funcs=[mean_callable],
        )

        # Check output
        assert output_file.exists()
        result = pd.read_csv(output_file)
        assert len(result) == 1  # Should have one row for task1
        assert "mean_callable_score1" in result.columns
        assert (
            "score1_reasoning" not in result.columns
        )  # Reasoning column should be ignored
        assert result.iloc[0]["mean_callable_score1"] == pytest.approx(0.85)

    def test_aggregate_by_task_memory_with_output_directory_creation(self, tmp_path):
        """Test memory-based task aggregation with output directory creation."""
        # Create test CSV file
        test_data = pd.DataFrame(
            {
                "user_id": ["user1"],
                "task_id": ["task1"],
                "turn_id": ["turn1"],
                "agent_name": ["agent1"],
                "score1": [0.8],
            }
        )

        input_file = tmp_path / "input.csv"
        test_data.to_csv(input_file, index=False)

        # Create output path with non-existent directory
        output_dir = tmp_path / "new_dir"
        output_file = output_dir / "task_aggregation.csv"

        # Run aggregation - this function does not create directories; ensure parent exists first
        output_dir.mkdir(parents=True, exist_ok=True)
        _aggregate_by_task_memory(
            input_file=input_file,
            output_filename=output_file,
            callable_funcs=[mean_callable],
        )

        # Check output
        assert output_file.exists()
        result = pd.read_csv(output_file)
        assert len(result) == 1

    def test_aggregate_by_task_memory_with_json_array_input(self, tmp_path):
        """Test memory-based task aggregation with JSON array input."""
        # Create test JSON array file
        test_data = [
            {
                "user_id": "user1",
                "task_id": "task1",
                "turn_id": "turn1",
                "agent_name": "agent1",
                "score1": 0.8,
            },
            {
                "user_id": "user2",
                "task_id": "task1",
                "turn_id": "turn1",
                "agent_name": "agent2",
                "score1": 0.9,
            },
        ]

        input_file = tmp_path / "input.json"
        with open(input_file, "w") as f:
            json.dump(test_data, f)

        output_file = tmp_path / "task_aggregation.csv"

        # Run aggregation
        _aggregate_by_task_memory(
            input_file=input_file,
            output_filename=output_file,
            callable_funcs=[mean_callable],
        )

        # Check output
        assert output_file.exists()
        result = pd.read_csv(output_file)
        assert len(result) == 1  # Should have one row for task1
        assert result.iloc[0]["mean_callable_score1"] == pytest.approx(0.85)

    def test_aggregate_by_task_memory_with_jsonl_input(self, tmp_path):
        """Test memory-based task aggregation with JSONL input."""
        # Create test JSONL file
        test_data = [
            {
                "user_id": "user1",
                "task_id": "task1",
                "turn_id": "turn1",
                "agent_name": "agent1",
                "score1": 0.8,
            },
            {
                "user_id": "user2",
                "task_id": "task1",
                "turn_id": "turn1",
                "agent_name": "agent2",
                "score1": 0.9,
            },
        ]

        input_file = tmp_path / "input.json"
        with open(input_file, "w") as f:
            for item in test_data:
                f.write(json.dumps(item) + "\n")

        output_file = tmp_path / "task_aggregation.csv"

        # Run aggregation
        _aggregate_by_task_memory(
            input_file=input_file,
            output_filename=output_file,
            callable_funcs=[mean_callable],
        )

        # Check output
        assert output_file.exists()
        result = pd.read_csv(output_file)
        assert len(result) == 1  # Should have one row for task1
        assert result.iloc[0]["mean_callable_score1"] == pytest.approx(0.85)

    def test_aggregate_by_task_memory_with_json_output(self, tmp_path):
        """Test memory-based task aggregation with JSON output."""
        # Create test CSV file
        test_data = pd.DataFrame(
            {
                "user_id": ["user1", "user2"],
                "task_id": ["task1", "task1"],
                "turn_id": ["turn1", "turn1"],
                "agent_name": ["agent1", "agent2"],
                "score1": [0.8, 0.9],
            }
        )

        input_file = tmp_path / "input.csv"
        test_data.to_csv(input_file, index=False)

        output_file = tmp_path / "task_aggregation.json"

        # Run aggregation
        _aggregate_by_task_memory(
            input_file=input_file,
            output_filename=output_file,
            callable_funcs=[mean_callable],
        )

        # Check output
        assert output_file.exists()
        with open(output_file) as f:
            result = json.load(f)
            assert "task1" in result
            assert isinstance(result["task1"]["mean_callable_score1"], float)
            assert result["task1"]["mean_callable_score1"] == pytest.approx(0.85)

    def test_aggregate_by_task_memory_with_multiple_tasks(self, tmp_path):
        """Test memory-based task aggregation with multiple tasks."""
        # Create test CSV file with multiple tasks
        test_data = pd.DataFrame(
            {
                "user_id": ["user1", "user2", "user3", "user4"],
                "task_id": ["task1", "task1", "task2", "task2"],
                "turn_id": ["turn1", "turn1", "turn1", "turn1"],
                "agent_name": ["agent1", "agent1", "agent2", "agent2"],
                "score1": [0.8, 0.9, 0.7, 0.6],
            }
        )

        input_file = tmp_path / "input.csv"
        test_data.to_csv(input_file, index=False)

        output_file = tmp_path / "task_aggregation.csv"

        # Run aggregation
        _aggregate_by_task_memory(
            input_file=input_file,
            output_filename=output_file,
            callable_funcs=[mean_callable],
        )

        # Check output
        assert output_file.exists()
        result = pd.read_csv(output_file)
        assert len(result) == 2  # Should have two rows for two tasks
        # Pandas may reset index column as 'index' instead of original name
        key_col = "task_id" if "task_id" in result.columns else "index"
        task1_row = result[result[key_col] == "task1"].iloc[0]
        task2_row = result[result[key_col] == "task2"].iloc[0]
        assert task1_row["mean_callable_score1"] == pytest.approx(
            0.85
        )  # Mean of 0.8 and 0.9
        assert task2_row["mean_callable_score1"] == pytest.approx(
            0.65
        )  # Mean of 0.7 and 0.6

    def test_aggregate_by_task_memory_with_multiple_callables(self, tmp_path):
        """Test memory-based task aggregation with multiple callable functions."""

        def max_callable(scores):
            return max(scores) if scores else 0.0

        def min_callable(scores):
            return min(scores) if scores else 0.0

        # Create test CSV file
        test_data = pd.DataFrame(
            {
                "user_id": ["user1", "user2"],
                "task_id": ["task1", "task1"],
                "turn_id": ["turn1", "turn1"],
                "agent_name": ["agent1", "agent2"],
                "score1": [0.8, 0.9],
            }
        )

        input_file = tmp_path / "input.csv"
        test_data.to_csv(input_file, index=False)

        output_file = tmp_path / "task_aggregation.csv"

        # Run aggregation with multiple callables
        _aggregate_by_task_memory(
            input_file=input_file,
            output_filename=output_file,
            callable_funcs=[mean_callable, max_callable, min_callable],
        )

        # Check output
        assert output_file.exists()
        result = pd.read_csv(output_file)

        # Should have columns for all functions
        assert "mean_callable_score1" in result.columns
        assert "max_callable_score1" in result.columns
        assert "min_callable_score1" in result.columns

    def test_aggregate_by_task_memory_with_empty_scores(self, tmp_path):
        """Test memory-based task aggregation with empty score lists."""
        # Create test CSV file with no valid scores
        test_data = pd.DataFrame(
            {
                "user_id": ["user1", "user2"],
                "task_id": ["task1", "task1"],
                "turn_id": ["turn1", "turn1"],
                "agent_name": ["agent1", "agent2"],
                "score1": [None, None],
            }
        )

        input_file = tmp_path / "input.csv"
        test_data.to_csv(input_file, index=False)

        output_file = tmp_path / "task_aggregation.csv"

        # Run aggregation
        _aggregate_by_task_memory(
            input_file=input_file,
            output_filename=output_file,
            callable_funcs=[mean_callable],
        )

        # Check output
        assert output_file.exists()
        result = pd.read_csv(output_file)
        assert len(result) == 1  # Should have one row for task1
        # With empty valid scores list, mean_callable returns 0.0
        assert result.iloc[0]["mean_callable_score1"] == 0.0

    def test_aggregate_by_task_memory_with_mixed_score_types(self, tmp_path):
        """Test memory-based task aggregation with mixed score types."""
        # Create test CSV file with mixed score types
        test_data = pd.DataFrame(
            {
                "user_id": ["user1", "user2", "user3", "user4"],
                "task_id": ["task1", "task1", "task1", "task1"],
                "turn_id": ["turn1", "turn1", "turn1", "turn1"],
                "agent_name": ["agent1", "agent2", "agent3", "agent4"],
                "score1": [0.8, "0.9", None, 1],  # Mix of float, string, None, and int
            }
        )

        input_file = tmp_path / "input.csv"
        test_data.to_csv(input_file, index=False)

        output_file = tmp_path / "task_aggregation.csv"

        # Run aggregation
        _aggregate_by_task_memory(
            input_file=input_file,
            output_filename=output_file,
            callable_funcs=[mean_callable],
        )

        # Check output
        assert output_file.exists()
        result = pd.read_csv(output_file)
        assert len(result) == 1  # Should have one row for task1
        # None values are filtered out, and string/int values are converted to float
        assert (
            abs(result.iloc[0]["mean_callable_score1"] - 0.9) < 0.0001
        )  # Mean of 0.8, 0.9, and 1.0

    def test_private_user_streaming_csv_direct(self, tmp_path):
        """Directly test _aggregate_by_user_streaming with CSV input and single callable."""

        # Create CSV input
        df = pd.DataFrame(
            {
                "user_id": ["u1", "u1", "u2"],
                "task_id": ["t1", "t1", "t2"],
                "turn_id": ["x", "y", "z"],
                "agent_name": ["a1", "a1", "a2"],
                "score1": [0.5, 0.7, 0.9],
            }
        )
        input_file = tmp_path / "input.csv"
        df.to_csv(input_file, index=False)

        output_file = tmp_path / "user_streaming.csv"

        # Run streaming user aggregation with single callable
        _aggregate_by_user_streaming(
            input_file=input_file,
            output_filename=output_file,
            callable_funcs=[mean_callable],
            chunk_size=2,
        )

        assert output_file.exists()
        out = pd.read_csv(output_file)
        assert "user_id" in out.columns
        assert "mean_callable_score1" in out.columns
        # u1 average (0.5,0.7) = 0.6; u2 average = 0.9
        assert out.loc[out["user_id"] == "u1", "mean_callable_score1"].iloc[0] == 0.6
        assert out.loc[out["user_id"] == "u2", "mean_callable_score1"].iloc[0] == 0.9

    def test_private_agent_streaming_csv_direct(self, tmp_path):
        """Directly test _aggregate_by_agent_streaming with CSV input and single callable."""

        df = pd.DataFrame(
            {
                "user_id": ["u1", "u2", "u3"],
                "task_id": ["t1", "t1", "t2"],
                "turn_id": ["x", "y", "z"],
                "agent_name": ["a1", "a1", "a2"],
                "score1": [0.5, 1.0, 0.0],
            }
        )
        input_file = tmp_path / "input.csv"
        df.to_csv(input_file, index=False)

        output_file = tmp_path / "agent_streaming.csv"

        _aggregate_by_agent_streaming(
            input_file=input_file,
            output_filename=output_file,
            callable_funcs=[mean_callable],
            chunk_size=2,
        )

        assert output_file.exists()
        out = pd.read_csv(output_file)
        assert "agent_name" in out.columns
        assert "mean_callable_score1" in out.columns
        assert (
            out.loc[out["agent_name"] == "a1", "mean_callable_score1"].iloc[0] == 0.75
        )
        assert out.loc[out["agent_name"] == "a2", "mean_callable_score1"].iloc[0] == 0.0

    def test_user_memory_json_array_and_jsonl(self, tmp_path):
        """Exercise JSON array and JSONL branches in _aggregate_by_user_memory and JSON output."""
        from novaeval.evaluators.aggregators import _aggregate_by_user_memory

        # JSON array
        array_data = [
            {
                "user_id": "u1",
                "task_id": "t1",
                "turn_id": "x",
                "agent_name": "a1",
                "score1": 1.0,
            },
            {
                "user_id": "u1",
                "task_id": "t1",
                "turn_id": "y",
                "agent_name": "a1",
                "score1": 0.0,
            },
        ]
        arr_file = tmp_path / "arr.json"
        with open(arr_file, "w") as f:
            json.dump(array_data, f)
        out_arr = tmp_path / "user_mem_arr.json"
        _aggregate_by_user_memory(arr_file, out_arr, [mean_callable])
        assert out_arr.exists()
        with open(out_arr) as f:
            data = json.load(f)
        assert "u1" in data
        assert isinstance(data["u1"]["mean_callable_score1"], float)

        # JSONL
        jsonl_file = tmp_path / "file.json"
        with open(jsonl_file, "w") as f:
            for row in array_data:
                f.write(json.dumps(row) + "\n")
        out_jsonl = tmp_path / "user_mem_jsonl.json"
        _aggregate_by_user_memory(jsonl_file, out_jsonl, [mean_callable])
        assert out_jsonl.exists()
        with open(out_jsonl) as f:
            data2 = json.load(f)
        assert "u1" in data2

    def test_agent_memory_json_array_and_jsonl(self, tmp_path):
        """Exercise JSON array and JSONL branches in _aggregate_by_agent_memory and JSON output."""
        from novaeval.evaluators.aggregators import _aggregate_by_agent_memory

        array_data = [
            {
                "user_id": "u1",
                "task_id": "t1",
                "turn_id": "x",
                "agent_name": "a1",
                "score1": 0.2,
            },
            {
                "user_id": "u2",
                "task_id": "t1",
                "turn_id": "y",
                "agent_name": "a1",
                "score1": 0.6,
            },
        ]
        arr_file = tmp_path / "arr.json"
        with open(arr_file, "w") as f:
            json.dump(array_data, f)
        out_arr = tmp_path / "agent_mem_arr.json"
        _aggregate_by_agent_memory(arr_file, out_arr, [mean_callable])
        assert out_arr.exists()
        with open(out_arr) as f:
            data = json.load(f)
        assert "a1" in data

        jsonl_file = tmp_path / "file.json"
        with open(jsonl_file, "w") as f:
            for row in array_data:
                f.write(json.dumps(row) + "\n")
        out_jsonl = tmp_path / "agent_mem_jsonl.json"
        _aggregate_by_agent_memory(jsonl_file, out_jsonl, [mean_callable])
        assert out_jsonl.exists()
        with open(out_jsonl) as f:
            data2 = json.load(f)
        assert "a1" in data2

    def test_public_user_agent_streaming_dispatch(self, tmp_path, monkeypatch):
        """Ensure public functions execute streaming branch without signature mismatch by patching impl."""
        # Prepare minimal CSV
        df = pd.DataFrame(
            {
                "user_id": ["u1"],
                "task_id": ["t1"],
                "turn_id": ["x"],
                "agent_name": ["a1"],
                "score1": [1.0],
            }
        )
        inp = tmp_path / "in.csv"
        df.to_csv(inp, index=False)
        out1 = tmp_path / "u.csv"
        out2 = tmp_path / "a.csv"

        # Patch streaming implementations to accept list and do nothing
        def dummy_user(input_file, output_filename, callable_funcs, chunk_size):
            # Touch output to simulate work
            pd.DataFrame({"user_id": ["u1"], "score1": [1.0]}).to_csv(
                output_filename, index=False
            )

        def dummy_agent(input_file, output_filename, callable_funcs, chunk_size):
            pd.DataFrame({"agent_name": ["a1"], "score1": [1.0]}).to_csv(
                output_filename, index=False
            )

        from novaeval.evaluators import aggregators as ag

        monkeypatch.setattr(ag, "_aggregate_by_user_streaming", dummy_user)
        monkeypatch.setattr(ag, "_aggregate_by_agent_streaming", dummy_agent)

        # Call public wrappers with streaming=True and list of callables
        ag.aggregate_by_user(
            inp, out1, callable_func=[mean_callable], streaming=True, chunk_size=10
        )
        ag.aggregate_by_agent_name(
            inp, out2, callable_func=[mean_callable], streaming=True, chunk_size=10
        )

        # Verify outputs created
        assert out1.exists() and out2.exists()

    def test_task_streaming_json_array_with_read_json_patch(
        self, tmp_path, monkeypatch
    ):
        """Cover JSON array branch in task streaming by mocking pandas.read_json to support nrows and chunksize."""
        from novaeval.evaluators import aggregators as ag

        # Create a JSON array file so the code path detects '['
        data = [
            {
                "user_id": "u1",
                "task_id": "t1",
                "turn_id": "x",
                "agent_name": "a1",
                "score1": 1.0,
            },
            {
                "user_id": "u1",
                "task_id": "t1",
                "turn_id": "y",
                "agent_name": "a1",
                "score1": 0.0,
            },
        ]
        json_path = tmp_path / "in.json"
        with open(json_path, "w") as f:
            json.dump(data, f)

        # Prepare fake read_json
        df_full = pd.DataFrame(data)
        df_sample = df_full.head(1)

        def fake_read_json(path, *args, **kwargs):
            if kwargs.get("nrows") == 1:
                return df_sample
            if "chunksize" in kwargs:

                class _Chunks:
                    def __iter__(self):
                        yield df_full

                return _Chunks()
            # Fallback
            return df_full

        monkeypatch.setattr(ag.pd, "read_json", fake_read_json)

        out = tmp_path / "task_stream_arr.csv"
        ag._aggregate_by_task_streaming(json_path, out, [mean_callable], chunk_size=100)
        assert out.exists()
        res = pd.read_csv(out)
        assert "task_id" in res.columns
        assert "mean_callable_score1" in res.columns

    def test_user_streaming_json_array_with_read_json_patch(
        self, tmp_path, monkeypatch
    ):
        """Cover JSON array branch in user streaming by mocking pandas.read_json."""
        from novaeval.evaluators import aggregators as ag

        data = [
            {
                "user_id": "u1",
                "task_id": "t1",
                "turn_id": "x",
                "agent_name": "a1",
                "score1": 1.0,
            },
            {
                "user_id": "u1",
                "task_id": "t1",
                "turn_id": "y",
                "agent_name": "a1",
                "score1": 0.0,
            },
        ]
        json_path = tmp_path / "user_arr.json"
        with open(json_path, "w") as f:
            json.dump(data, f)

        df_full = pd.DataFrame(data)
        df_sample = df_full.head(1)

        def fake_read_json(path, *args, **kwargs):
            if kwargs.get("nrows") == 1:
                return df_sample
            if "chunksize" in kwargs:

                class _Chunks:
                    def __iter__(self):
                        yield df_full

                return _Chunks()
            return df_full

        monkeypatch.setattr(ag.pd, "read_json", fake_read_json)

        out = tmp_path / "user_stream_arr.csv"
        ag._aggregate_by_user_streaming(json_path, out, [mean_callable], chunk_size=50)
        assert out.exists()
        res = pd.read_csv(out)
        assert "user_id" in res.columns
        assert "mean_callable_score1" in res.columns

    def test_agent_streaming_json_array_with_read_json_patch(
        self, tmp_path, monkeypatch
    ):
        """Cover JSON array branch in agent streaming by mocking pandas.read_json."""
        from novaeval.evaluators import aggregators as ag

        data = [
            {
                "user_id": "u1",
                "task_id": "t1",
                "turn_id": "x",
                "agent_name": "a1",
                "score1": 0.2,
            },
            {
                "user_id": "u2",
                "task_id": "t1",
                "turn_id": "y",
                "agent_name": "a1",
                "score1": 0.6,
            },
        ]
        json_path = tmp_path / "agent_arr.json"
        with open(json_path, "w") as f:
            json.dump(data, f)

        df_full = pd.DataFrame(data)
        df_sample = df_full.head(1)

        def fake_read_json(path, *args, **kwargs):
            if kwargs.get("nrows") == 1:
                return df_sample
            if "chunksize" in kwargs:

                class _Chunks:
                    def __iter__(self):
                        yield df_full

                return _Chunks()
            return df_full

        monkeypatch.setattr(ag.pd, "read_json", fake_read_json)

        out = tmp_path / "agent_stream_arr.csv"
        ag._aggregate_by_agent_streaming(json_path, out, [mean_callable], chunk_size=50)
        assert out.exists()
        res = pd.read_csv(out)
        assert "agent_name" in res.columns
        assert "mean_callable_score1" in res.columns

    def test_streaming_user_agent_json_output(self, tmp_path):
        """Ensure streaming write-to-JSON branch is covered for user and agent aggregations."""
        df = pd.DataFrame(
            {
                "user_id": ["u1", "u2"],
                "task_id": ["t1", "t1"],
                "turn_id": ["x", "y"],
                "agent_name": ["a1", "a2"],
                "score1": [1.0, 0.0],
            }
        )
        inp = tmp_path / "in.csv"
        df.to_csv(inp, index=False)

        out_user = tmp_path / "user_stream.json"
        aggregate_by_user(
            inp, out_user, callable_func=[mean_callable], streaming=True, chunk_size=1
        )
        assert out_user.exists()
        with open(out_user) as f:
            data_user = json.load(f)
        assert "u1" in data_user and "u2" in data_user

        out_agent = tmp_path / "agent_stream.json"
        aggregate_by_agent_name(
            inp, out_agent, callable_func=[mean_callable], streaming=True, chunk_size=1
        )
        assert out_agent.exists()
        with open(out_agent) as f:
            data_agent = json.load(f)
        assert "a1" in data_agent and "a2" in data_agent

    def test_streaming_task_jsonl_with_missing_task_id(self, tmp_path):
        """Task streaming should handle rows missing task_id by grouping under 'unknown'."""
        rows = [
            {
                "user_id": "u1",
                "turn_id": "x",
                "agent_name": "a1",
                "score1": 0.5,
            },  # missing task_id
            {
                "user_id": "u1",
                "task_id": "t1",
                "turn_id": "y",
                "agent_name": "a1",
                "score1": 0.5,
            },
        ]
        jsonl = tmp_path / "in.json"
        with open(jsonl, "w") as f:
            json.dump(rows, f)

        out = tmp_path / "task_stream.json"
        aggregate_by_task(
            jsonl, out, callable_func=[mean_callable], streaming=True, chunk_size=2
        )
        assert out.exists()
        with open(out) as f:
            res = json.load(f)
        # Should contain both a placeholder for missing task_id and 't1' keys
        assert ("unknown" in res or "NaN" in res) and "t1" in res

    def test_aggregate_by_task_json_array_streaming_ijson(self, tmp_path):
        """Test aggregate_by_task with JSON array file in streaming mode using ijson."""
        # Create JSON array input
        data = [
            {
                "user_id": "u1",
                "task_id": "t1",
                "turn_id": "x",
                "agent_name": "a1",
                "score1": 1.0,
            },
            {
                "user_id": "u1",
                "task_id": "t1",
                "turn_id": "y",
                "agent_name": "a1",
                "score1": 0.0,
            },
            {
                "user_id": "u2",
                "task_id": "t2",
                "turn_id": "z",
                "agent_name": "a2",
                "score1": 0.5,
            },
        ]
        json_path = tmp_path / "in.json"
        with open(json_path, "w") as f:
            json.dump(data, f)

        out = tmp_path / "task_stream_arr.csv"
        aggregate_by_task(
            json_path, out, callable_func=[mean_callable], streaming=True, chunk_size=10
        )
        assert out.exists()
        res = pd.read_csv(out)
        assert "task_id" in res.columns
        assert "mean_callable_score1" in res.columns

        # Check that we have 2 tasks
        assert len(res) == 2

        # Check that task t1 has mean of 0.5 (1.0 + 0.0) / 2
        task_t1 = res[res["task_id"] == "t1"]
        assert len(task_t1) == 1
        assert task_t1["mean_callable_score1"].iloc[0] == 0.5

        # Check that task t2 has mean of 0.5
        task_t2 = res[res["task_id"] == "t2"]
        assert len(task_t2) == 1
        assert task_t2["mean_callable_score1"].iloc[0] == 0.5


class TestAggregatorsAdditional:
    def test_agent_default_callable_none_memory(self, tmp_path):
        df = pd.DataFrame(
            {
                "user_id": ["u1"],
                "task_id": ["t1"],
                "turn_id": ["x"],
                "agent_name": ["a1"],
                "score1": [1.0],
            }
        )
        inp = tmp_path / "in.csv"
        df.to_csv(inp, index=False)
        out = tmp_path / "out.csv"
        aggregate_by_agent_name(inp, out, callable_func=None, streaming=False)
        res = pd.read_csv(out)
        assert "mean_callable_score1" in res.columns

    def test_user_streaming_jsonl_branch(self, tmp_path):
        # Ensure JSON array streaming branch is executed for user aggregation
        rows = [
            {
                "user_id": "u1",
                "task_id": "t1",
                "turn_id": "x",
                "agent_name": "a1",
                "score1": 0.5,
            },
            {
                "user_id": "u1",
                "task_id": "t1",
                "turn_id": "y",
                "agent_name": "a1",
                "score1": 1.0,
            },
        ]
        inp = tmp_path / "in.json"
        with open(inp, "w") as f:
            json.dump(rows, f)
        out = tmp_path / "user_jsonl_stream.csv"
        aggregate_by_user(
            inp, out, callable_func=[mean_callable], streaming=True, chunk_size=2
        )
        res = pd.read_csv(out)
        assert "user_id" in res.columns
        assert "mean_callable_score1" in res.columns

    def test_agent_streaming_jsonl_branch(self, tmp_path):
        # Ensure JSON array streaming branch is executed for agent aggregation
        rows = [
            {
                "user_id": "u1",
                "task_id": "t1",
                "turn_id": "x",
                "agent_name": "a1",
                "score1": 0.2,
            },
            {
                "user_id": "u2",
                "task_id": "t1",
                "turn_id": "y",
                "agent_name": "a1",
                "score1": 0.6,
            },
        ]
        inp = tmp_path / "in.json"
        with open(inp, "w") as f:
            json.dump(rows, f)
        out = tmp_path / "agent_jsonl_stream.csv"
        aggregate_by_agent_name(
            inp, out, callable_func=[mean_callable], streaming=True, chunk_size=2
        )
        res = pd.read_csv(out)
        assert "agent_name" in res.columns
        assert "mean_callable_score1" in res.columns

    def test_task_streaming_csv_nan_filtered_and_no_scorer_columns(self, tmp_path):
        # 1) CSV streaming with NaN to take the false branch of pd.notna(score)
        df = pd.DataFrame(
            {
                "user_id": ["u1", "u2"],
                "task_id": ["t1", "t1"],
                "turn_id": ["x", "y"],
                "agent_name": ["a1", "a1"],
                "score1": [1.0, None],
            }
        )
        inp1 = tmp_path / "in1.csv"
        df.to_csv(inp1, index=False)
        out1 = tmp_path / "task_stream_nan.csv"
        _aggregate_by_task_streaming(inp1, out1, [mean_callable], chunk_size=1)
        res1 = pd.read_csv(out1)
        assert "mean_callable_score1" in res1.columns

        # 2) No scorer columns: ensure empty results path still writes a CSV with just key column
        base_only = pd.DataFrame(
            {
                "user_id": ["u1"],
                "task_id": ["t1"],
                "turn_id": ["x"],
                "agent_name": ["a1"],
            }
        )
        inp2 = tmp_path / "in2.csv"
        base_only.to_csv(inp2, index=False)
        out2 = tmp_path / "task_stream_empty.csv"
        aggregate_by_task(
            inp2, out2, callable_func=[mean_callable], streaming=True, chunk_size=1
        )
        res2 = pd.read_csv(out2)
        assert "task_id" in res2.columns and len(res2) == 0

    def test_user_streaming_csv_nan_filtered(self, tmp_path):
        # CSV streaming with NaN to take the false branch
        df = pd.DataFrame(
            {
                "user_id": ["u1", "u2"],
                "task_id": ["t1", "t2"],
                "turn_id": ["x", "y"],
                "agent_name": ["a1", "a2"],
                "score1": [None, 0.9],
            }
        )
        inp = tmp_path / "users_nan.csv"
        df.to_csv(inp, index=False)
        out = tmp_path / "users_stream.csv"
        _aggregate_by_user_streaming(inp, out, [mean_callable], chunk_size=2)
        res = pd.read_csv(out)
        assert "user_id" in res.columns
        assert "mean_callable_score1" in res.columns

    def test_user_streaming_no_scorer_columns_json_output(self, tmp_path):
        # No scorer columns with JSON output (empty results dict path)
        base_only = pd.DataFrame(
            {
                "user_id": ["u1"],
                "task_id": ["t1"],
                "turn_id": ["x"],
                "agent_name": ["a1"],
            }
        )
        inp = tmp_path / "in_base.csv"
        base_only.to_csv(inp, index=False)
        out = tmp_path / "users_stream.json"
        aggregate_by_user(
            inp, out, callable_func=[mean_callable], streaming=True, chunk_size=1
        )
        with open(out) as f:
            data = json.load(f)
        # Should be an empty dict since there are no scorer columns
        assert isinstance(data, dict) and len(data) == 0

    def test_aggregate_by_user_json_array_streaming_ijson(self, tmp_path):
        """Test aggregate_by_user with JSON array file in streaming mode using ijson."""
        # Create JSON array input
        data = [
            {
                "user_id": "u1",
                "task_id": "t1",
                "turn_id": "x",
                "agent_name": "a1",
                "score1": 1.0,
            },
            {
                "user_id": "u1",
                "task_id": "t2",
                "turn_id": "y",
                "agent_name": "a1",
                "score1": 0.0,
            },
            {
                "user_id": "u2",
                "task_id": "t3",
                "turn_id": "z",
                "agent_name": "a2",
                "score1": 0.5,
            },
        ]
        json_path = tmp_path / "in.json"
        with open(json_path, "w") as f:
            json.dump(data, f)

        out = tmp_path / "user_stream_arr.csv"
        aggregate_by_user(
            json_path, out, callable_func=[mean_callable], streaming=True, chunk_size=10
        )
        assert out.exists()
        res = pd.read_csv(out)
        assert "user_id" in res.columns
        assert "mean_callable_score1" in res.columns

        # Check that we have 2 users
        assert len(res) == 2

        # Check that user u1 has mean of 0.5 (1.0 + 0.0) / 2
        user_u1 = res[res["user_id"] == "u1"]
        assert len(user_u1) == 1
        assert user_u1["mean_callable_score1"].iloc[0] == 0.5

        # Check that user u2 has mean of 0.5
        user_u2 = res[res["user_id"] == "u2"]
        assert len(user_u2) == 1
        assert user_u2["mean_callable_score1"].iloc[0] == 0.5

    def test_aggregate_by_agent_json_array_streaming_ijson(self, tmp_path):
        """Test aggregate_by_agent_name with JSON array file in streaming mode using ijson."""
        # Create JSON array input
        data = [
            {
                "user_id": "u1",
                "task_id": "t1",
                "turn_id": "x",
                "agent_name": "a1",
                "score1": 1.0,
            },
            {
                "user_id": "u2",
                "task_id": "t2",
                "turn_id": "y",
                "agent_name": "a1",
                "score1": 0.0,
            },
            {
                "user_id": "u3",
                "task_id": "t3",
                "turn_id": "z",
                "agent_name": "a2",
                "score1": 0.5,
            },
        ]
        json_path = tmp_path / "in.json"
        with open(json_path, "w") as f:
            json.dump(data, f)

        out = tmp_path / "agent_stream_arr.csv"
        aggregate_by_agent_name(
            json_path, out, callable_func=[mean_callable], streaming=True, chunk_size=10
        )
        assert out.exists()
        res = pd.read_csv(out)
        assert "agent_name" in res.columns
        assert "mean_callable_score1" in res.columns

        # Check that we have 2 agents
        assert len(res) == 2

        # Check that agent a1 has mean of 0.5 (1.0 + 0.0) / 2
        agent_a1 = res[res["agent_name"] == "a1"]
        assert len(agent_a1) == 1
        assert agent_a1["mean_callable_score1"].iloc[0] == 0.5

        # Check that agent a2 has mean of 0.5
        agent_a2 = res[res["agent_name"] == "a2"]
        assert len(agent_a2) == 1
        assert agent_a2["mean_callable_score1"].iloc[0] == 0.5
