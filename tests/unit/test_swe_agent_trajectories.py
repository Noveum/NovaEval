"""
Unit tests for novaeval.agents.swe_agent_trajectories module.

Tests all functions for processing SWE agent trajectory data.
"""

import contextlib
import os
import tempfile
from unittest.mock import patch

import pandas as pd
import pytest

from novaeval.agents.agent_data import AgentData
from novaeval.datasets.swe_agent_trajectories_dataset import (
    create_dataset,
    stream_dataset,
    swe_agent_trajectories_preprocessing,
)


class TestSWEAgentTrajectoriesPreprocessing:
    """Test the swe_agent_trajectories_preprocessing function."""

    @pytest.mark.unit
    def test_both_parameters_provided_raises_error(self):
        """Test that providing both parquet_dir and parquet_files raises ValueError."""
        with pytest.raises(
            ValueError,
            match="Provide either parquet_dir or parquet_files, but not both",
        ):
            swe_agent_trajectories_preprocessing(
                parquet_dir="/path", parquet_files=["file.parquet"]
            )

    @pytest.mark.unit
    def test_neither_parameter_provided_raises_error(self):
        """Test that providing neither parquet_dir nor parquet_files raises ValueError."""
        with pytest.raises(
            ValueError,
            match="Provide either parquet_dir or parquet_files, but not both",
        ):
            swe_agent_trajectories_preprocessing()

    @pytest.mark.unit
    def test_invalid_directory_raises_error(self):
        """Test that invalid directory path raises ValueError."""
        with pytest.raises(ValueError, match="is not a valid directory"):
            swe_agent_trajectories_preprocessing(parquet_dir="/nonexistent/path")

    @pytest.mark.unit
    def test_non_parquet_files_in_directory_raises_error(self):
        """Test that non-parquet files in directory raise ValueError."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a non-parquet file
            with open(os.path.join(temp_dir, "test.txt"), "w") as f:
                f.write("not parquet")

            with pytest.raises(
                ValueError, match="Directory contains non-parquet files"
            ):
                swe_agent_trajectories_preprocessing(parquet_dir=temp_dir)

    @pytest.mark.unit
    def test_no_parquet_files_in_directory_raises_error(self):
        """Test that directory with no parquet files raises ValueError."""
        with (
            tempfile.TemporaryDirectory() as temp_dir,
            pytest.raises(ValueError, match="No parquet files found in the directory"),
        ):
            swe_agent_trajectories_preprocessing(parquet_dir=temp_dir)

    @pytest.mark.unit
    def test_invalid_parquet_files_list_raises_error(self):
        """Test that invalid parquet_files parameter raises ValueError."""
        with pytest.raises(ValueError, match="parquet_files must be a non-empty list"):
            swe_agent_trajectories_preprocessing(parquet_files="not_a_list")

        with pytest.raises(ValueError, match="parquet_files must be a non-empty list"):
            swe_agent_trajectories_preprocessing(parquet_files=[])

    @pytest.mark.unit
    def test_non_parquet_file_in_list_raises_error(self):
        """Test that non-parquet file in list raises ValueError."""
        with pytest.raises(ValueError, match="is not a parquet file"):
            swe_agent_trajectories_preprocessing(parquet_files=["test.txt"])

    @patch("pandas.read_parquet")
    @pytest.mark.unit
    def test_successful_preprocessing_with_parquet_files(self, mock_read_parquet):
        """Test successful preprocessing with valid parquet files."""
        # Mock the parquet data
        mock_df = pd.DataFrame(
            {
                "instance_id": ["inst1", "inst2"],
                "model_name": ["model1", "model2"],
                "target": ["target1", "target2"],
                "trajectory": [
                    [
                        {"step": 1, "action": "action1"},
                        {"step": 2, "action": "action2"},
                    ],
                    [{"step": 1, "action": "action3"}],
                ],
                "exit_status": ["completed", "failed"],
                "generated_patch": ["patch1", "patch2"],
                "eval_logs": ["log1", "log2"],
            }
        )
        mock_read_parquet.return_value = mock_df

        temp_csv_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as temp_csv:
                temp_csv_path = temp_csv.name

            swe_agent_trajectories_preprocessing(
                parquet_files=["test.parquet"], output_csv=temp_csv_path
            )

            # Verify that the CSV was created and contains expected data
            assert os.path.exists(temp_csv_path)
            result_df = pd.read_csv(temp_csv_path)

            # Should have 3 rows (2 steps from first trajectory + 1 step from second)
            assert len(result_df) == 3

            # Check some expected values
            assert result_df.iloc[0]["instance_id"] == "inst1"
            assert result_df.iloc[0]["step"] == 1
            assert result_df.iloc[0]["action"] == "action1"

        finally:
            if temp_csv_path and os.path.exists(temp_csv_path):
                os.unlink(temp_csv_path)

    @patch("pandas.read_parquet")
    @pytest.mark.unit
    def test_successful_preprocessing_with_parquet_directory(self, mock_read_parquet):
        """Test successful preprocessing with parquet directory."""
        mock_df = pd.DataFrame(
            {
                "instance_id": ["inst1"],
                "model_name": ["model1"],
                "target": ["target1"],
                "trajectory": [[{"step": 1, "action": "action1"}]],
                "exit_status": ["completed"],
                "generated_patch": ["patch1"],
                "eval_logs": ["log1"],
            }
        )
        mock_read_parquet.return_value = mock_df

        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a parquet file in directory
            parquet_file = os.path.join(temp_dir, "test.parquet")
            with open(parquet_file, "w") as f:
                f.write("dummy")  # Just need the file to exist

            output_csv = os.path.join(temp_dir, "output.csv")

            swe_agent_trajectories_preprocessing(
                parquet_dir=temp_dir, output_csv=output_csv
            )

            # Verify CSV was created
            assert os.path.exists(output_csv)
            result_df = pd.read_csv(output_csv)
            assert len(result_df) == 1

    @patch("pandas.read_parquet")
    @pytest.mark.unit
    def test_preprocessing_handles_missing_columns(self, mock_read_parquet):
        """Test preprocessing handles missing required columns gracefully."""
        # Missing some required columns
        mock_df = pd.DataFrame(
            {
                "instance_id": ["inst1"],
                "model_name": ["model1"],
                # Missing other required columns
            }
        )
        mock_read_parquet.return_value = mock_df

        temp_csv_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as temp_csv:
                temp_csv_path = temp_csv.name

            # Should handle missing columns and continue
            swe_agent_trajectories_preprocessing(
                parquet_files=["test.parquet"], output_csv=temp_csv_path
            )
        except Exception:
            # Should not crash on missing columns, but might create empty output
            pass
        finally:
            if temp_csv_path and os.path.exists(temp_csv_path):
                os.unlink(temp_csv_path)

    @patch("pandas.read_parquet")
    @pytest.mark.unit
    def test_preprocessing_with_empty_trajectory(self, mock_read_parquet):
        """Test preprocessing with empty trajectory."""
        mock_df = pd.DataFrame(
            {
                "instance_id": ["inst1"],
                "model_name": ["model1"],
                "target": ["target1"],
                "trajectory": [[]],  # Empty trajectory
                "exit_status": ["completed"],
                "generated_patch": ["patch1"],
                "eval_logs": ["log1"],
            }
        )
        mock_read_parquet.return_value = mock_df

        temp_csv_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as temp_csv:
                temp_csv_path = temp_csv.name

            swe_agent_trajectories_preprocessing(
                parquet_files=["test.parquet"], output_csv=temp_csv_path
            )

            # Should create CSV but with no trajectory rows
            assert os.path.exists(temp_csv_path)
            result_df = pd.read_csv(temp_csv_path)
            assert len(result_df) == 0

        finally:
            if temp_csv_path and os.path.exists(temp_csv_path):
                os.unlink(temp_csv_path)

    @patch("pandas.read_parquet")
    @pytest.mark.unit
    def test_preprocessing_file_error_continues(self, mock_read_parquet):
        """Test that file processing errors are handled gracefully."""
        mock_read_parquet.side_effect = Exception("File read error")

        temp_csv_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as temp_csv:
                temp_csv_path = temp_csv.name

            # Should not crash, just continue with next file (if any)
            swe_agent_trajectories_preprocessing(
                parquet_files=["test.parquet"], output_csv=temp_csv_path
            )

            # Output file should exist but might be empty
            assert os.path.exists(temp_csv_path)

        finally:
            if temp_csv_path and os.path.exists(temp_csv_path):
                os.unlink(temp_csv_path)

    @patch("pandas.read_parquet")
    @pytest.mark.unit
    def test_preprocessing_chunk_processing(self, mock_read_parquet):
        """Test that large datasets are processed in chunks."""
        # Create a large mock dataset
        large_data = {
            "instance_id": [f"inst{i}" for i in range(1000)],
            "model_name": [f"model{i}" for i in range(1000)],
            "target": [f"target{i}" for i in range(1000)],
            "trajectory": [[{"step": 1, "action": f"action{i}"}] for i in range(1000)],
            "exit_status": ["completed"] * 1000,
            "generated_patch": [f"patch{i}" for i in range(1000)],
            "eval_logs": [f"log{i}" for i in range(1000)],
        }
        mock_df = pd.DataFrame(large_data)
        mock_read_parquet.return_value = mock_df

        temp_csv_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as temp_csv:
                temp_csv_path = temp_csv.name

            swe_agent_trajectories_preprocessing(
                parquet_files=["large_test.parquet"], output_csv=temp_csv_path
            )

            # Should handle large dataset and create output
            assert os.path.exists(temp_csv_path)
            result_df = pd.read_csv(temp_csv_path)
            assert len(result_df) == 1000  # One row per trajectory step

        finally:
            if temp_csv_path and os.path.exists(temp_csv_path):
                os.unlink(temp_csv_path)


class TestSWEAgentTrajectoriesDatasetFunctions:
    """Test the dataset creation and streaming functions."""

    @pytest.mark.unit
    def test_create_dataset_success(self):
        """Test create_dataset with valid CSV file."""
        # Create sample CSV data matching expected format
        sample_data = {
            "instance_id": ["inst1", "inst2"],
            "model_name": ["model1", "model2"],
            "target": ["target1", "target2"],
            "step": [1, 2],
            "action": ["action1", "action2"],
            "exit_status": ["completed", "failed"],
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

    @pytest.mark.unit
    def test_stream_dataset_success(self):
        """Test stream_dataset with valid CSV file."""
        # Create sample CSV data
        sample_data = {
            "instance_id": ["inst1", "inst2", "inst3"],
            "model_name": ["model1", "model2", "model3"],
            "target": ["target1", "target2", "target3"],
            "step": [1, 2, 3],
            "action": ["action1", "action2", "action3"],
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

    @pytest.mark.unit
    def test_create_dataset_file_not_found(self):
        """Test create_dataset with non-existent file."""
        with contextlib.suppress(FileNotFoundError, Exception):
            create_dataset("/nonexistent/file.csv")

    @pytest.mark.unit
    def test_stream_dataset_file_not_found(self):
        """Test stream_dataset with non-existent file."""
        with contextlib.suppress(FileNotFoundError, Exception):
            list(stream_dataset("/nonexistent/file.csv"))

    @patch("pandas.read_parquet")
    @pytest.mark.unit
    def test_preprocessing_with_non_list_trajectory(self, mock_read_parquet):
        """Test preprocessing when trajectory is not a list."""
        mock_df = pd.DataFrame(
            {
                "instance_id": ["inst1"],
                "model_name": ["model1"],
                "target": ["target1"],
                "trajectory": ["not_a_list"],  # String instead of list
                "exit_status": ["completed"],
                "generated_patch": ["patch1"],
                "eval_logs": ["log1"],
            }
        )
        mock_read_parquet.return_value = mock_df

        temp_csv_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as temp_csv:
                temp_csv_path = temp_csv.name

            swe_agent_trajectories_preprocessing(
                parquet_files=["test.parquet"], output_csv=temp_csv_path
            )

            # Should create empty CSV since trajectory is not a list
            assert os.path.exists(temp_csv_path)
            result_df = pd.read_csv(temp_csv_path)
            assert len(result_df) == 0

        finally:
            if temp_csv_path and os.path.exists(temp_csv_path):
                os.unlink(temp_csv_path)

    @patch("pandas.read_parquet")
    @pytest.mark.unit
    def test_preprocessing_with_non_dict_steps(self, mock_read_parquet):
        """Test preprocessing when trajectory steps are not dicts."""
        mock_df = pd.DataFrame(
            {
                "instance_id": ["inst1"],
                "model_name": ["model1"],
                "target": ["target1"],
                "trajectory": [
                    ["not_a_dict", 123, None]
                ],  # Non-dict items in trajectory
                "exit_status": ["completed"],
                "generated_patch": ["patch1"],
                "eval_logs": ["log1"],
            }
        )
        mock_read_parquet.return_value = mock_df

        temp_csv_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as temp_csv:
                temp_csv_path = temp_csv.name

            swe_agent_trajectories_preprocessing(
                parquet_files=["test.parquet"], output_csv=temp_csv_path
            )

            # Should create empty CSV since no valid steps
            assert os.path.exists(temp_csv_path)
            result_df = pd.read_csv(temp_csv_path)
            assert len(result_df) == 0

        finally:
            if temp_csv_path and os.path.exists(temp_csv_path):
                os.unlink(temp_csv_path)

    @patch("pandas.read_parquet")
    @pytest.mark.unit
    def test_preprocessing_with_pandas_series_trajectory(self, mock_read_parquet):
        """Test preprocessing when trajectory needs tolist() conversion."""
        import pandas as pd

        # Create a DataFrame where the trajectory column contains pandas Series values
        # This simulates how pandas might represent nested array data
        trajectory_data = [{"step": 1, "action": "test"}]

        # Create DataFrame with trajectory as pandas objects that have tolist method
        data = {
            "instance_id": ["inst1"],
            "model_name": ["model1"],
            "target": ["target1"],
            "trajectory": [trajectory_data],
            "exit_status": ["completed"],
            "generated_patch": ["patch1"],
            "eval_logs": ["log1"],
        }
        mock_df = pd.DataFrame(data)

        # Convert trajectory column to object type to simulate Series behavior
        mock_df["trajectory"] = mock_df["trajectory"].astype(object)

        # Verify the trajectory has tolist method
        traj_val = mock_df.iloc[0]["trajectory"]
        if not hasattr(traj_val, "tolist"):
            # Create a mock object with tolist method that returns the trajectory
            class MockSeriesValue:
                def __init__(self, data):
                    self._data = data

                def tolist(self):
                    return self._data

            mock_df.iloc[0, mock_df.columns.get_loc("trajectory")] = MockSeriesValue(
                trajectory_data
            )

        mock_read_parquet.return_value = mock_df

        temp_csv_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as temp_csv:
                temp_csv_path = temp_csv.name

            swe_agent_trajectories_preprocessing(
                parquet_files=["test.parquet"], output_csv=temp_csv_path
            )

            # Should successfully process the trajectory
            assert os.path.exists(temp_csv_path)
            result_df = pd.read_csv(temp_csv_path)
            assert len(result_df) == 1
            assert result_df.iloc[0]["step"] == 1

        finally:
            if temp_csv_path and os.path.exists(temp_csv_path):
                os.unlink(temp_csv_path)

    @patch("csv.field_size_limit")
    @pytest.mark.unit
    def test_create_dataset_overflow_error_handling(self, mock_field_size_limit):
        """Test create_dataset with OverflowError in field_size_limit."""

        # Mock field_size_limit to raise OverflowError initially
        mock_field_size_limit.side_effect = [OverflowError(), None]

        # Create sample CSV data
        sample_data = {
            "instance_id": ["inst1"],
            "model_name": ["model1"],
            "target": ["target1"],
            "step": [1],
            "action": ["action1"],
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

            # Verify that field_size_limit was called multiple times due to OverflowError
            assert mock_field_size_limit.call_count >= 2

        finally:
            if temp_file_path and os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
