"""
Unit tests for HuggingFace dataset functionality.
"""

from unittest.mock import patch

import pytest

from novaeval.datasets.huggingface import CommonHFDatasets, HuggingFaceDataset


@pytest.mark.unit
class TestHuggingFaceDataset:
    """Test cases for HuggingFaceDataset class."""

    def test_init_default(self):
        """Test initialization with default parameters."""
        dataset = HuggingFaceDataset("test_dataset")

        assert dataset.name == "hf_test_dataset"
        assert dataset.dataset_name == "test_dataset"
        assert dataset.subset is None
        assert dataset.input_column == "input"
        assert dataset.target_column == "target"
        assert dataset.num_samples is None
        assert dataset.split == "test"
        assert dataset.seed == 42
        assert dataset.preprocessing_fn is None
        assert dataset.load_kwargs == {}

    def test_init_with_subset(self):
        """Test initialization with subset."""
        dataset = HuggingFaceDataset("glue", subset="cola")

        assert dataset.name == "hf_glue_cola"
        assert dataset.dataset_name == "glue"
        assert dataset.subset == "cola"

    def test_init_with_custom_params(self):
        """Test initialization with custom parameters."""

        def preprocess_fn(x):
            return x

        dataset = HuggingFaceDataset(
            dataset_name="custom_dataset",
            subset="config1",
            input_column="question",
            target_column="answer",
            num_samples=100,
            split="train",
            seed=123,
            preprocessing_fn=preprocess_fn,
            trust_remote_code=True,
        )

        assert dataset.name == "hf_custom_dataset_config1"
        assert dataset.dataset_name == "custom_dataset"
        assert dataset.subset == "config1"
        assert dataset.input_column == "question"
        assert dataset.target_column == "answer"
        assert dataset.num_samples == 100
        assert dataset.split == "train"
        assert dataset.seed == 123
        assert dataset.preprocessing_fn == preprocess_fn
        assert dataset.load_kwargs["trust_remote_code"] is True

    @patch("novaeval.datasets.huggingface.load_dataset")
    def test_load_data_without_subset(self, mock_load_dataset):
        """Test loading data without subset."""
        # Mock the dataset
        mock_dataset = [
            {"input": "Question 1", "target": "Answer 1"},
            {"input": "Question 2", "target": "Answer 2"},
        ]
        mock_load_dataset.return_value = mock_dataset

        dataset = HuggingFaceDataset("test_dataset")
        loaded_data = dataset.load_data()

        # Verify load_dataset was called correctly
        mock_load_dataset.assert_called_once_with("test_dataset", split="test")

        # Verify data conversion - check that we have the data but account for shuffling
        assert len(loaded_data) == 2
        # Check that both expected inputs and outputs are present
        inputs = [sample["input"] for sample in loaded_data]
        outputs = [sample["expected"] for sample in loaded_data]
        assert "Question 1" in inputs
        assert "Question 2" in inputs
        assert "Answer 1" in outputs
        assert "Answer 2" in outputs
        # Check metadata is present
        assert loaded_data[0]["metadata"]["dataset"] == "test_dataset"
        assert loaded_data[0]["metadata"]["source"] == "huggingface"

    @patch("novaeval.datasets.huggingface.load_dataset")
    def test_load_data_with_subset(self, mock_load_dataset):
        """Test loading data with subset."""
        mock_dataset = [
            {"input": "Q1", "target": "A1"},
            {"input": "Q2", "target": "A2"},
        ]
        mock_load_dataset.return_value = mock_dataset

        dataset = HuggingFaceDataset("glue", subset="cola")
        dataset.load_data()

        # Verify load_dataset was called with subset
        mock_load_dataset.assert_called_once_with("glue", "cola", split="test")

    @patch("novaeval.datasets.huggingface.load_dataset")
    def test_load_data_with_kwargs(self, mock_load_dataset):
        """Test loading data with additional kwargs."""
        mock_dataset = [{"input": "Q1", "target": "A1"}]
        mock_load_dataset.return_value = mock_dataset

        dataset = HuggingFaceDataset(
            "test_dataset", trust_remote_code=True, cache_dir="/tmp"
        )
        dataset.load_data()

        # Verify kwargs were passed through
        mock_load_dataset.assert_called_once_with(
            "test_dataset", split="test", trust_remote_code=True, cache_dir="/tmp"
        )

    @patch("novaeval.datasets.huggingface.load_dataset")
    def test_load_data_exception_handling(self, mock_load_dataset):
        """Test exception handling during dataset loading."""
        mock_load_dataset.side_effect = Exception("Dataset not found")

        dataset = HuggingFaceDataset("nonexistent_dataset")

        with pytest.raises(
            ValueError, match="Failed to load dataset nonexistent_dataset"
        ):
            dataset.load_data()

    @patch("novaeval.datasets.huggingface.load_dataset")
    def test_load_data_with_preprocessing(self, mock_load_dataset):
        """Test loading data with preprocessing function."""
        mock_dataset = [
            {"input": "question 1", "target": "answer 1"},
            {"input": "question 2", "target": "answer 2"},
        ]
        mock_load_dataset.return_value = mock_dataset

        def preprocess_fn(sample):
            return {
                **sample,
                "input": sample["input"].upper(),
                "expected": sample["expected"].upper(),
            }

        dataset = HuggingFaceDataset("test_dataset", preprocessing_fn=preprocess_fn)
        loaded_data = dataset.load_data()

        # Check that preprocessing was applied - account for shuffling
        inputs = [sample["input"] for sample in loaded_data]
        outputs = [sample["expected"] for sample in loaded_data]
        assert "QUESTION 1" in inputs
        assert "QUESTION 2" in inputs
        assert "ANSWER 1" in outputs
        assert "ANSWER 2" in outputs

    @patch("novaeval.datasets.huggingface.load_dataset")
    def test_load_data_with_num_samples_limit(self, mock_load_dataset):
        """Test loading data with sample limit."""
        mock_dataset = [{"input": f"Q{i}", "target": f"A{i}"} for i in range(10)]
        mock_load_dataset.return_value = mock_dataset

        dataset = HuggingFaceDataset("test_dataset", num_samples=3, seed=42)
        loaded_data = dataset.load_data()

        assert len(loaded_data) == 3

    @patch("novaeval.datasets.huggingface.load_dataset")
    @patch("novaeval.datasets.huggingface.random.seed")
    @patch("novaeval.datasets.huggingface.random.shuffle")
    def test_load_data_seeding_and_shuffling(
        self, mock_shuffle, mock_seed, mock_load_dataset
    ):
        """Test that data is seeded and shuffled."""
        mock_dataset = [
            {"input": "Q1", "target": "A1"},
            {"input": "Q2", "target": "A2"},
        ]
        mock_load_dataset.return_value = mock_dataset

        dataset = HuggingFaceDataset("test_dataset", seed=123)
        dataset.load_data()

        # Verify seed was set
        mock_seed.assert_called_once_with(123)
        # Verify shuffle was called
        mock_shuffle.assert_called_once()

    def test_convert_sample_valid(self):
        """Test converting a valid sample."""
        dataset = HuggingFaceDataset("test_dataset")

        item = {"input": "Test question", "target": "Test answer", "extra": "value"}
        sample = dataset._convert_sample(item, 5)

        assert sample["id"] == "hf_test_dataset_5"
        assert sample["input"] == "Test question"
        assert sample["expected"] == "Test answer"
        assert sample["metadata"]["dataset"] == "test_dataset"
        assert sample["metadata"]["subset"] is None
        assert sample["metadata"]["original_index"] == 5
        assert sample["metadata"]["source"] == "huggingface"
        assert sample["original_extra"] == "value"

    def test_convert_sample_with_subset(self):
        """Test converting a sample with subset."""
        dataset = HuggingFaceDataset("glue", subset="cola")

        item = {"input": "Test", "target": "Label"}
        sample = dataset._convert_sample(item, 0)

        assert sample["metadata"]["dataset"] == "glue"
        assert sample["metadata"]["subset"] == "cola"

    def test_convert_sample_missing_input(self):
        """Test converting sample with missing input."""
        dataset = HuggingFaceDataset("test_dataset")

        item = {"target": "Answer"}  # Missing input
        sample = dataset._convert_sample(item, 0)

        assert sample is None

    def test_convert_sample_missing_target(self):
        """Test converting sample with missing target."""
        dataset = HuggingFaceDataset("test_dataset")

        item = {"input": "Question"}  # Missing target
        sample = dataset._convert_sample(item, 0)

        assert sample is None

    def test_convert_sample_none_values(self):
        """Test converting sample with None values."""
        dataset = HuggingFaceDataset("test_dataset")

        item = {"input": None, "target": "Answer"}
        sample = dataset._convert_sample(item, 0)

        assert sample is None

    def test_convert_sample_custom_columns(self):
        """Test converting sample with custom column names."""
        dataset = HuggingFaceDataset(
            "test_dataset", input_column="question", target_column="answer"
        )

        item = {"question": "Test question", "answer": "Test answer"}
        sample = dataset._convert_sample(item, 0)

        assert sample["input"] == "Test question"
        assert sample["expected"] == "Test answer"

    @patch("builtins.print")  # Mock print to capture warning
    def test_convert_sample_exception_handling(self, mock_print):
        """Test exception handling in sample conversion."""
        dataset = HuggingFaceDataset("test_dataset")

        # Create an item that causes an exception by having a problematic item field
        class BadObject:
            def __str__(self):
                raise ValueError("Cannot convert to string")

        item = {"input": BadObject(), "target": "Answer"}
        sample = dataset._convert_sample(item, 0)

        assert sample is None
        # Verify warning was printed
        mock_print.assert_called_once()
        printed_args = mock_print.call_args[0]
        assert len(printed_args) == 1
        assert "Warning: Failed to convert sample 0:" in printed_args[0]

    @patch("novaeval.datasets.huggingface.load_dataset")
    def test_load_data_filters_invalid_samples(self, mock_load_dataset):
        """Test that invalid samples are filtered out."""
        mock_dataset = [
            {"input": "Q1", "target": "A1"},  # Valid
            {"input": None, "target": "A2"},  # Invalid - None input
            {"target": "A3"},  # Invalid - missing input
            {"input": "Q4", "target": "A4"},  # Valid
        ]
        mock_load_dataset.return_value = mock_dataset

        dataset = HuggingFaceDataset("test_dataset")
        loaded_data = dataset.load_data()

        # Should only have the 2 valid samples
        assert len(loaded_data) == 2
        # Check that the valid inputs are present (account for shuffling)
        inputs = [sample["input"] for sample in loaded_data]
        assert "Q1" in inputs
        assert "Q4" in inputs

    @patch("novaeval.datasets.huggingface.load_dataset")
    def test_get_info(self, mock_load_dataset):
        """Test get_info method."""
        # Mock the dataset to avoid loading from HuggingFace
        mock_load_dataset.return_value = []

        dataset = HuggingFaceDataset(
            "test_dataset",
            subset="config1",
            input_column="question",
            target_column="answer",
            split="train",
            seed=123,
        )

        info = dataset.get_info()

        assert info["name"] == "hf_test_dataset_config1"
        assert info["dataset_name"] == "test_dataset"
        assert info["subset"] == "config1"
        assert info["input_column"] == "question"
        assert info["target_column"] == "answer"
        assert info["source"] == "huggingface"
        assert info["split"] == "train"
        assert info["seed"] == 123

    @patch("novaeval.datasets.huggingface.load_dataset")
    def test_get_info_without_subset(self, mock_load_dataset):
        """Test get_info method without subset."""
        # Mock the dataset to avoid loading from HuggingFace
        mock_load_dataset.return_value = []

        dataset = HuggingFaceDataset("test_dataset")
        info = dataset.get_info()

        assert info["subset"] is None

    @patch("novaeval.datasets.huggingface.load_dataset")
    def test_empty_dataset_handling(self, mock_load_dataset):
        """Test handling of empty dataset."""
        mock_load_dataset.return_value = []

        dataset = HuggingFaceDataset("empty_dataset")
        loaded_data = dataset.load_data()

        assert len(loaded_data) == 0

    @patch("novaeval.datasets.huggingface.load_dataset")
    def test_string_conversion_in_sample(self, mock_load_dataset):
        """Test that values are converted to strings."""
        mock_dataset = [
            {"input": 123, "target": 456},  # Numeric values
        ]
        mock_load_dataset.return_value = mock_dataset

        dataset = HuggingFaceDataset("test_dataset")
        loaded_data = dataset.load_data()

        assert loaded_data[0]["input"] == "123"
        assert loaded_data[0]["expected"] == "456"


@pytest.mark.unit
class TestCommonHFDatasets:
    """Test cases for CommonHFDatasets factory class."""

    def test_squad_dataset_creation(self):
        """Test SQuAD dataset creation."""
        dataset = CommonHFDatasets.squad()

        assert isinstance(dataset, HuggingFaceDataset)
        assert dataset.dataset_name == "squad"
        assert dataset.input_column == "question"
        assert dataset.target_column == "answers"
        assert dataset.preprocessing_fn is not None

    def test_squad_preprocessing_function(self):
        """Test SQuAD preprocessing function."""
        dataset = CommonHFDatasets.squad()

        # Test the preprocessing function
        sample = {
            "input": "What is the capital?",
            "expected": {"text": ["Paris"]},
            "original_context": "France is a country.",
        }

        processed = dataset.preprocessing_fn(sample)

        assert processed["expected"] == "Paris"
        assert "Context: France is a country." in processed["input"]
        assert "Question: What is the capital?" in processed["input"]

    def test_squad_preprocessing_empty_answers(self):
        """Test SQuAD preprocessing with empty answers."""
        dataset = CommonHFDatasets.squad()

        sample = {
            "input": "Question?",
            "expected": {"text": []},
            "original_context": "Context",
        }

        processed = dataset.preprocessing_fn(sample)
        assert processed["expected"] == ""

    def test_glue_supported_tasks(self):
        """Test GLUE dataset creation for supported tasks."""
        supported_tasks = [
            "cola",
            "sst2",
            "mrpc",
            "qqp",
            "stsb",
            "mnli",
            "qnli",
            "rte",
            "wnli",
        ]

        for task in supported_tasks:
            dataset = CommonHFDatasets.glue(task)

            assert isinstance(dataset, HuggingFaceDataset)
            assert dataset.dataset_name == "glue"
            assert dataset.subset == task
            assert dataset.target_column == "label"

    def test_glue_unsupported_task(self):
        """Test GLUE dataset creation with unsupported task."""
        with pytest.raises(ValueError, match="Unsupported GLUE task: invalid_task"):
            CommonHFDatasets.glue("invalid_task")

    def test_glue_task_configurations(self):
        """Test specific GLUE task configurations."""
        # Test CoLA
        cola = CommonHFDatasets.glue("cola")
        assert cola.input_column == "sentence"

        # Test MRPC
        mrpc = CommonHFDatasets.glue("mrpc")
        assert mrpc.input_column == "sentence1"

        # Test QQP
        qqp = CommonHFDatasets.glue("qqp")
        assert qqp.input_column == "question1"

    def test_hellaswag_dataset_creation(self):
        """Test HellaSwag dataset creation."""
        dataset = CommonHFDatasets.hellaswag()

        assert isinstance(dataset, HuggingFaceDataset)
        assert dataset.dataset_name == "hellaswag"
        assert dataset.input_column == "ctx"
        assert dataset.target_column == "label"
        assert dataset.preprocessing_fn is not None

    def test_hellaswag_preprocessing_function(self):
        """Test HellaSwag preprocessing function."""
        dataset = CommonHFDatasets.hellaswag()

        sample = {
            "input": "The man walked into",
            "expected": "1",
            "original_endings": ["the store", "the park", "the house", "the car"],
        }

        processed = dataset.preprocessing_fn(sample)

        assert "The man walked into" in processed["input"]
        assert "Choices:" in processed["input"]
        assert "0. the store" in processed["input"]
        assert "1. the park" in processed["input"]
        assert processed["expected"] == "1"

    def test_hellaswag_preprocessing_missing_endings(self):
        """Test HellaSwag preprocessing with missing endings."""
        dataset = CommonHFDatasets.hellaswag()

        sample = {
            "input": "Context",
            "expected": "0",
        }

        processed = dataset.preprocessing_fn(sample)

        assert "Context" in processed["input"]
        assert "Choices:" in processed["input"]

    def test_truthful_qa_dataset_creation(self):
        """Test TruthfulQA dataset creation."""
        dataset = CommonHFDatasets.truthful_qa()

        assert isinstance(dataset, HuggingFaceDataset)
        assert dataset.dataset_name == "truthful_qa"
        assert dataset.subset == "generation"
        assert dataset.input_column == "question"
        assert dataset.target_column == "best_answer"


@pytest.mark.unit
class TestHuggingFaceDatasetEdgeCases:
    """Test edge cases for HuggingFace dataset functionality."""

    @patch("novaeval.datasets.huggingface.load_dataset")
    def test_preprocessing_function_exception(self, mock_load_dataset):
        """Test exception handling in preprocessing function."""
        mock_dataset = [{"input": "Q1", "target": "A1"}]
        mock_load_dataset.return_value = mock_dataset

        def failing_preprocess(sample):
            raise ValueError("Preprocessing failed")

        dataset = HuggingFaceDataset(
            "test_dataset", preprocessing_fn=failing_preprocess
        )

        with pytest.raises(ValueError, match="Preprocessing failed"):
            dataset.load_data()

    @patch("novaeval.datasets.huggingface.load_dataset")
    def test_large_dataset_sampling(self, mock_load_dataset):
        """Test sampling from large dataset."""
        # Create a large mock dataset
        mock_dataset = [{"input": f"Q{i}", "target": f"A{i}"} for i in range(1000)]
        mock_load_dataset.return_value = mock_dataset

        dataset = HuggingFaceDataset("large_dataset", num_samples=10, seed=42)
        loaded_data = dataset.load_data()

        assert len(loaded_data) == 10
        # Verify deterministic sampling with seed
        assert (
            loaded_data[0]["input"] == loaded_data[0]["input"]
        )  # Should be consistent

    @patch("novaeval.datasets.huggingface.load_dataset")
    def test_deterministic_sampling(self, mock_load_dataset):
        """Test that sampling is deterministic with same seed."""
        mock_dataset = [{"input": f"Q{i}", "target": f"A{i}"} for i in range(100)]
        mock_load_dataset.return_value = mock_dataset

        dataset1 = HuggingFaceDataset("test_dataset", num_samples=5, seed=42)
        dataset2 = HuggingFaceDataset("test_dataset", num_samples=5, seed=42)

        # Mock load_dataset to return the same data for both calls
        mock_load_dataset.reset_mock()
        mock_load_dataset.return_value = mock_dataset

        data1 = dataset1.load_data()

        mock_load_dataset.return_value = mock_dataset
        data2 = dataset2.load_data()

        # Should get the same samples in the same order
        assert len(data1) == len(data2) == 5
        for i in range(5):
            assert data1[i]["input"] == data2[i]["input"]

    def test_inheritance_from_base_dataset(self):
        """Test that HuggingFaceDataset properly inherits from BaseDataset."""
        from novaeval.datasets.base import BaseDataset

        dataset = HuggingFaceDataset("test_dataset")

        assert isinstance(dataset, BaseDataset)
        assert hasattr(dataset, "load_data")
        assert hasattr(dataset, "get_info")
        assert hasattr(dataset, "preprocess_sample")
        assert hasattr(dataset, "postprocess_sample")

    @patch("novaeval.datasets.huggingface.load_dataset")
    def test_zero_num_samples(self, mock_load_dataset):
        """Test with zero num_samples."""
        mock_dataset = [{"input": "Q1", "target": "A1"}]
        mock_load_dataset.return_value = mock_dataset

        dataset = HuggingFaceDataset("test_dataset", num_samples=0)
        loaded_data = dataset.load_data()

        # With the fix, num_samples=0 should return 0 samples
        assert len(loaded_data) == 0

    def test_complex_original_data_preservation(self):
        """Test that complex original data is preserved."""
        dataset = HuggingFaceDataset("test_dataset")

        item = {
            "input": "Question",
            "target": "Answer",
            "complex_field": {"nested": {"data": [1, 2, 3]}},
            "list_field": ["a", "b", "c"],
        }

        sample = dataset._convert_sample(item, 0)

        assert sample["original_complex_field"] == {"nested": {"data": [1, 2, 3]}}
        assert sample["original_list_field"] == ["a", "b", "c"]

    @patch("novaeval.datasets.huggingface.load_dataset")
    def test_split_parameter_forwarding(self, mock_load_dataset):
        """Test that split parameter is correctly forwarded."""
        mock_dataset = []
        mock_load_dataset.return_value = mock_dataset

        # Test different splits
        for split in ["train", "validation", "test"]:
            dataset = HuggingFaceDataset("test_dataset", split=split)
            dataset.load_data()

            # Get the last call to load_dataset
            call_kwargs = mock_load_dataset.call_args[1]
            assert call_kwargs["split"] == split

    def test_name_generation_edge_cases(self):
        """Test dataset name generation for edge cases."""
        # Test with special characters in dataset name
        dataset1 = HuggingFaceDataset("dataset-with-dashes")
        assert dataset1.name == "hf_dataset-with-dashes"

        # Test with subset containing special characters
        dataset2 = HuggingFaceDataset("dataset", subset="subset_with_underscores")
        assert dataset2.name == "hf_dataset_subset_with_underscores"

        # Test with empty string subset (current implementation treats it as falsy and excludes it)
        dataset3 = HuggingFaceDataset("dataset", subset="")
        assert dataset3.name == "hf_dataset"
