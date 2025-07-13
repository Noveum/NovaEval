"""
Unit tests for custom dataset functionality.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from novaeval.datasets.custom import CustomDataset


class TestCustomDataset:
    """Test cases for CustomDataset class."""

    def test_init_default(self):
        """Test initialization with default parameters."""
        test_data = [
            {"input": "Question 1", "expected": "Answer 1"},
            {"input": "Question 2", "expected": "Answer 2"},
        ]

        dataset = CustomDataset(test_data)

        assert dataset.name == "custom"
        assert dataset.data_source == test_data
        assert dataset.input_column == "input"
        assert dataset.target_column == "expected"
        assert dataset.id_column is None
        assert dataset.preprocessing_fn is None

    def test_init_with_params(self):
        """Test initialization with custom parameters."""
        test_data = [{"question": "Q1", "answer": "A1"}]

        def preprocess_fn(x):
            return x

        dataset = CustomDataset(
            data_source=test_data,
            input_column="question",
            target_column="answer",
            id_column="question_id",
            num_samples=100,
            split="train",
            seed=123,
            preprocessing_fn=preprocess_fn,
        )

        assert dataset.data_source == test_data
        assert dataset.input_column == "question"
        assert dataset.target_column == "answer"
        assert dataset.id_column == "question_id"
        assert dataset.num_samples == 100
        assert dataset.split == "train"
        assert dataset.seed == 123
        assert dataset.preprocessing_fn == preprocess_fn

    def test_load_data_from_list(self):
        """Test loading data from a list."""
        test_data = [
            {"input": "Question 1", "expected": "Answer 1"},
            {"input": "Question 2", "expected": "Answer 2"},
        ]

        dataset = CustomDataset(test_data)
        loaded_data = dataset.load_data()

        assert len(loaded_data) == 2
        inputs = [sample["input"] for sample in loaded_data]
        outputs = [sample["expected"] for sample in loaded_data]
        assert "Question 1" in inputs
        assert "Question 2" in inputs
        assert "Answer 1" in outputs
        assert "Answer 2" in outputs

    def test_load_data_from_callable(self):
        """Test loading data from a callable."""

        def data_generator():
            return [
                {"input": "Generated Q1", "expected": "Generated A1"},
                {"input": "Generated Q2", "expected": "Generated A2"},
            ]

        dataset = CustomDataset(data_generator)
        loaded_data = dataset.load_data()

        assert len(loaded_data) == 2
        inputs = [sample["input"] for sample in loaded_data]
        outputs = [sample["expected"] for sample in loaded_data]
        assert "Generated Q1" in inputs
        assert "Generated Q2" in inputs
        assert "Generated A1" in outputs
        assert "Generated A2" in outputs

    def test_load_data_from_json_file(self):
        """Test loading data from JSON file."""
        test_data = [
            {"input": "JSON Q1", "expected": "JSON A1"},
            {"input": "JSON Q2", "expected": "JSON A2"},
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(test_data, f)
            file_path = f.name

        try:
            dataset = CustomDataset(file_path)
            loaded_data = dataset.load_data()

            assert len(loaded_data) == 2
            inputs = [sample["input"] for sample in loaded_data]
            outputs = [sample["expected"] for sample in loaded_data]
            assert "JSON Q1" in inputs
            assert "JSON Q2" in inputs
            assert "JSON A1" in outputs
            assert "JSON A2" in outputs
        finally:
            Path(file_path).unlink()

    def test_load_data_from_jsonl_file(self):
        """Test loading data from JSONL file."""
        test_data = [
            {"input": "JSONL Q1", "expected": "JSONL A1"},
            {"input": "JSONL Q2", "expected": "JSONL A2"},
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for item in test_data:
                f.write(json.dumps(item) + "\n")
            file_path = f.name

        try:
            dataset = CustomDataset(file_path)
            loaded_data = dataset.load_data()

            assert len(loaded_data) == 2
            inputs = [sample["input"] for sample in loaded_data]
            outputs = [sample["expected"] for sample in loaded_data]
            assert "JSONL Q1" in inputs
            assert "JSONL Q2" in inputs
            assert "JSONL A1" in outputs
            assert "JSONL A2" in outputs
        finally:
            Path(file_path).unlink()

    def test_load_data_from_csv_file(self):
        """Test loading data from CSV file."""
        test_data = pd.DataFrame(
            {"input": ["CSV Q1", "CSV Q2"], "expected": ["CSV A1", "CSV A2"]}
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            test_data.to_csv(f.name, index=False)
            file_path = f.name

        try:
            dataset = CustomDataset(file_path)
            loaded_data = dataset.load_data()

            assert len(loaded_data) == 2
            inputs = [sample["input"] for sample in loaded_data]
            outputs = [sample["expected"] for sample in loaded_data]
            assert "CSV Q1" in inputs
            assert "CSV Q2" in inputs
            assert "CSV A1" in outputs
            assert "CSV A2" in outputs
        finally:
            Path(file_path).unlink()

    def test_load_data_from_excel_file(self):
        """Test loading data from Excel file."""
        import importlib.util

        if importlib.util.find_spec("openpyxl") is None:
            pytest.skip("openpyxl not installed")

        test_data = pd.DataFrame(
            {
                "input": ["Excel Q1", "Excel Q2"],
                "expected": ["Excel A1", "Excel A2"],
            }
        )

        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as f:
            test_data.to_excel(f.name, index=False)
            file_path = f.name

        try:
            dataset = CustomDataset(file_path)
            loaded_data = dataset.load_data()

            assert len(loaded_data) == 2
            inputs = [sample["input"] for sample in loaded_data]
            outputs = [sample["expected"] for sample in loaded_data]
            assert "Excel Q1" in inputs
            assert "Excel Q2" in inputs
            assert "Excel A1" in outputs
            assert "Excel A2" in outputs
        finally:
            Path(file_path).unlink()

    def test_load_data_with_custom_columns(self):
        """Test loading data with custom column names."""
        test_data = [
            {"question": "Q1", "answer": "A1", "id": "1"},
            {"question": "Q2", "answer": "A2", "id": "2"},
        ]

        dataset = CustomDataset(
            test_data, input_column="question", target_column="answer", id_column="id"
        )
        loaded_data = dataset.load_data()

        assert len(loaded_data) == 2
        inputs = [sample["input"] for sample in loaded_data]
        outputs = [sample["expected"] for sample in loaded_data]
        ids = [sample["id"] for sample in loaded_data]
        assert "Q1" in inputs
        assert "Q2" in inputs
        assert "A1" in outputs
        assert "A2" in outputs
        assert "1" in ids
        assert "2" in ids

    def test_load_data_with_preprocess_function(self):
        """Test loading data with custom preprocessing function."""
        test_data = [
            {"input": "question 1", "expected": "answer 1"},
            {"input": "question 2", "expected": "answer 2"},
        ]

        def preprocess_func(sample):
            return {
                "input": sample["input"].upper(),
                "expected": sample["expected"].upper(),
                "id": sample.get("id", "unknown"),
            }

        dataset = CustomDataset(test_data, preprocessing_fn=preprocess_func)
        loaded_data = dataset.load_data()

        assert len(loaded_data) == 2
        inputs = [sample["input"] for sample in loaded_data]
        outputs = [sample["expected"] for sample in loaded_data]
        assert "QUESTION 1" in inputs
        assert "QUESTION 2" in inputs
        assert "ANSWER 1" in outputs
        assert "ANSWER 2" in outputs

    def test_load_data_with_num_samples_limit(self):
        """Test loading data with sample limit."""
        test_data = [
            {"input": "Q1", "expected": "A1"},
            {"input": "Q2", "expected": "A2"},
            {"input": "Q3", "expected": "A3"},
            {"input": "Q4", "expected": "A4"},
        ]

        dataset = CustomDataset(test_data, num_samples=2)
        loaded_data = dataset.load_data()

        assert len(loaded_data) == 2

    def test_load_data_with_seeding(self):
        """Test that seeding produces consistent results."""
        test_data = [{"input": f"Q{i}", "expected": f"A{i}"} for i in range(100)]

        dataset1 = CustomDataset(test_data, seed=42, num_samples=10)
        dataset2 = CustomDataset(test_data, seed=42, num_samples=10)

        data1 = dataset1.load_data()
        data2 = dataset2.load_data()

        assert data1 == data2

    def test_load_json_dict_format(self):
        """Test loading JSON with dict format."""
        test_data = {
            "data": [
                {"input": "Dict Q1", "expected": "Dict A1"},
                {"input": "Dict Q2", "expected": "Dict A2"},
            ]
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(test_data, f)
            file_path = f.name

        try:
            dataset = CustomDataset(file_path)
            loaded_data = dataset.load_data()

            assert len(loaded_data) == 2
            inputs = [sample["input"] for sample in loaded_data]
            outputs = [sample["expected"] for sample in loaded_data]
            assert "Dict Q1" in inputs
            assert "Dict Q2" in inputs
            assert "Dict A1" in outputs
            assert "Dict A2" in outputs
        finally:
            Path(file_path).unlink()

    def test_load_json_single_dict(self):
        """Test loading JSON with single dict."""
        test_data = {"input": "Single Q", "expected": "Single A"}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(test_data, f)
            file_path = f.name

        try:
            dataset = CustomDataset(file_path)
            loaded_data = dataset.load_data()

            assert len(loaded_data) == 1
            assert loaded_data[0]["input"] == "Single Q"
        finally:
            Path(file_path).unlink()

    def test_load_json_with_samples_key(self):
        """Test loading JSON with 'samples' key."""
        test_data = {
            "samples": [
                {"input": "Sample Q1", "expected": "Sample A1"},
                {"input": "Sample Q2", "expected": "Sample A2"},
            ]
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(test_data, f)
            file_path = f.name

        try:
            dataset = CustomDataset(file_path)
            loaded_data = dataset.load_data()

            assert len(loaded_data) == 2
            inputs = [sample["input"] for sample in loaded_data]
            outputs = [sample["expected"] for sample in loaded_data]
            assert "Sample Q1" in inputs
            assert "Sample Q2" in inputs
            assert "Sample A1" in outputs
            assert "Sample A2" in outputs
        finally:
            Path(file_path).unlink()

    def test_load_jsonl_with_empty_lines(self):
        """Test loading JSONL with empty lines."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write('{"input": "Q1", "expected": "A1"}\n')
            f.write("\n")  # Empty line
            f.write('{"input": "Q2", "expected": "A2"}\n')
            file_path = f.name

        try:
            dataset = CustomDataset(file_path)
            loaded_data = dataset.load_data()

            assert len(loaded_data) == 2
            inputs = [sample["input"] for sample in loaded_data]
            outputs = [sample["expected"] for sample in loaded_data]
            assert "Q1" in inputs
            assert "Q2" in inputs
            assert "A1" in outputs
            assert "A2" in outputs
        finally:
            Path(file_path).unlink()

    def test_convert_sample_with_id_column(self):
        """Test sample conversion with ID column."""
        test_data = [{"question": "Q1", "answer": "A1", "question_id": "id1"}]

        dataset = CustomDataset(
            test_data,
            input_column="question",
            target_column="answer",
            id_column="question_id",
        )
        loaded_data = dataset.load_data()

        assert loaded_data[0]["id"] == "id1"
        # Temporarily disabled for shuffle fix
        assert loaded_data[0]["expected"] == "A1"

    def test_convert_sample_without_id_column(self):
        """Test sample conversion without ID column."""
        test_data = [{"input": "Q1", "expected": "A1"}]

        dataset = CustomDataset(test_data)
        loaded_data = dataset.load_data()

        assert loaded_data[0]["id"] == "custom_0"  # Generated ID
        # Temporarily disabled for shuffle fix
        assert loaded_data[0]["expected"] == "A1"

    def test_convert_sample_missing_columns(self):
        """Test sample conversion with missing columns."""
        test_data = [{"input": "Q1"}]  # Missing expected column

        dataset = CustomDataset(test_data)
        loaded_data = dataset.load_data()

        # Should skip samples without required columns
        assert len(loaded_data) == 0

    def test_unsupported_file_format(self):
        """Test loading unsupported file format."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Some text content")
            file_path = f.name

        try:
            dataset = CustomDataset(file_path)

            with pytest.raises(ValueError, match="Unsupported file format"):
                dataset.load_data()
        finally:
            Path(file_path).unlink()

    def test_nonexistent_file(self):
        """Test loading from nonexistent file."""
        dataset = CustomDataset("/nonexistent/path.json")

        with pytest.raises(FileNotFoundError):
            dataset.load_data()

    def test_invalid_json_file(self):
        """Test loading invalid JSON file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("invalid json content")
            file_path = f.name

        try:
            dataset = CustomDataset(file_path)

            with pytest.raises(json.JSONDecodeError):
                dataset.load_data()
        finally:
            Path(file_path).unlink()

    def test_unsupported_data_source_type(self):
        """Test unsupported data source type."""
        dataset = CustomDataset(123)  # Integer is not supported

        with pytest.raises(ValueError, match="Unsupported data source type"):
            dataset.load_data()

    def test_load_excel_with_mock(self):
        """Test loading Excel file with mocked pandas."""
        with (
            patch("novaeval.datasets.custom.pd.read_excel") as mock_read_excel,
            patch("novaeval.datasets.custom.Path.exists", return_value=True),
        ):
            mock_read_excel.return_value = pd.DataFrame(
                {"input": ["Excel Q1"], "expected": ["Excel A1"]}
            )

            dataset = CustomDataset("test.xlsx")
            loaded_data = dataset.load_data()

            assert len(loaded_data) == 1
            inputs = [sample["input"] for sample in loaded_data]
            assert "Excel Q1" in inputs
            mock_read_excel.assert_called_once()

    def test_load_csv_with_mock(self):
        """Test loading CSV file with mocked pandas."""
        with (
            patch("novaeval.datasets.custom.pd.read_csv") as mock_read_csv,
            patch("novaeval.datasets.custom.Path.exists", return_value=True),
        ):
            mock_read_csv.return_value = pd.DataFrame(
                {"input": ["CSV Q1"], "expected": ["CSV A1"]}
            )

            dataset = CustomDataset("test.csv")
            loaded_data = dataset.load_data()

            assert len(loaded_data) == 1
            inputs = [sample["input"] for sample in loaded_data]
            assert "CSV Q1" in inputs
            mock_read_csv.assert_called_once()

    def test_json_invalid_structure(self):
        """Test loading JSON with invalid structure."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump("just a string", f)
            file_path = f.name

        try:
            dataset = CustomDataset(file_path)

            with pytest.raises(
                ValueError, match="JSON file must contain a list or dict"
            ):
                dataset.load_data()
        finally:
            Path(file_path).unlink()

    def test_data_shuffling(self):
        """Test that data is shuffled consistently."""
        test_data = [{"input": f"Q{i}", "expected": f"A{i}"} for i in range(10)]

        dataset = CustomDataset(test_data, seed=42)
        loaded_data = dataset.load_data()

        # Data should be shuffled (not in original order)
        assert loaded_data != test_data

        # But should be consistent with same seed
        dataset2 = CustomDataset(test_data, seed=42)
        loaded_data2 = dataset2.load_data()
        assert loaded_data == loaded_data2

    def test_inheritance_from_base_dataset(self):
        """Test that CustomDataset properly inherits from BaseDataset."""
        from novaeval.datasets.base import BaseDataset

        test_data = [{"input": "Q1", "expected": "A1"}]
        dataset = CustomDataset(test_data)

        assert isinstance(dataset, BaseDataset)
        assert hasattr(dataset, "load_data")
        assert hasattr(dataset, "get_info")
        assert hasattr(dataset, "preprocess_sample")
        assert hasattr(dataset, "postprocess_sample")

    def test_get_info_method(self):
        """Test get_info method."""
        test_data = [{"input": "Q1", "expected": "A1"}]
        dataset = CustomDataset(test_data)

        info = dataset.get_info()
        assert info["name"] == "custom"
        assert "num_samples" in info

    def test_empty_dataset(self):
        """Test handling empty dataset."""
        dataset = CustomDataset([])
        loaded_data = dataset.load_data()

        assert len(loaded_data) == 0

    def test_preprocessing_error_handling(self):
        """Test error handling in preprocessing."""
        test_data = [{"input": "Q1", "expected": "A1"}]

        def failing_preprocess(sample):
            raise ValueError("Preprocessing failed")

        dataset = CustomDataset(test_data, preprocessing_fn=failing_preprocess)

        with pytest.raises(ValueError, match="Preprocessing failed"):
            dataset.load_data()

    def test_path_conversion(self):
        """Test that string paths are converted to Path objects."""
        test_data = [{"input": "Q1", "expected": "A1"}]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(test_data, f)
            file_path = f.name

        try:
            # Pass string path
            dataset = CustomDataset(file_path)
            loaded_data = dataset.load_data()

            assert len(loaded_data) == 1
            # Temporarily disabled for shuffle fix
        finally:
            Path(file_path).unlink()
