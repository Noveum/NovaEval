"""
Unit tests for base dataset functionality.
"""

import pytest

from novaeval.datasets.base import BaseDataset


class ConcreteDataset(BaseDataset):
    """Concrete implementation of BaseDataset for testing."""

    def __init__(self, data=None, name="test_dataset", **kwargs):
        super().__init__(name=name, **kwargs)
        if data is not None:
            self._test_data = data
        else:
            self._test_data = [
                {"id": "1", "input": "What is 2+2?", "expected": "4"},
                {
                    "id": "2",
                    "input": "What is the capital of France?",
                    "expected": "Paris",
                },
                {"id": "3", "input": "What is 3*3?", "expected": "9"},
            ]

    def load_data(self):
        """Mock implementation that returns test data."""
        if self.num_samples is not None:
            return self._test_data[: self.num_samples]
        return self._test_data


class TestBaseDataset:
    """Test cases for BaseDataset class."""

    def test_init_default(self):
        """Test initialization with default parameters."""
        dataset = ConcreteDataset()

        assert dataset.name == "test_dataset"
        assert dataset.num_samples is None
        assert dataset.split == "test"
        assert dataset.seed == 42
        assert dataset.kwargs == {}
        assert dataset._data is None
        assert dataset._loaded is False

    def test_init_with_params(self):
        """Test initialization with custom parameters."""
        dataset = ConcreteDataset(
            num_samples=100, split="train", seed=123, custom_param="value"
        )

        assert dataset.name == "test_dataset"
        assert dataset.num_samples == 100
        assert dataset.split == "train"
        assert dataset.seed == 123
        assert dataset.kwargs["custom_param"] == "value"

    def test_len_loads_data_once(self):
        """Test that __len__ loads data and caches it."""
        dataset = ConcreteDataset()

        # Initially not loaded
        assert dataset._loaded is False
        assert dataset._data is None

        # First call should load data
        length = len(dataset)
        assert length == 3
        assert dataset._loaded is True
        assert dataset._data is not None
        assert len(dataset._data) == 3

        # Second call should use cached data
        length2 = len(dataset)
        assert length2 == 3

    def test_len_with_num_samples_limit(self):
        """Test __len__ with num_samples limit."""
        dataset = ConcreteDataset(num_samples=2)

        length = len(dataset)
        assert length == 2
        assert len(dataset._data) == 2

    def test_len_empty_dataset(self):
        """Test __len__ with empty dataset."""
        dataset = ConcreteDataset(data=[])

        length = len(dataset)
        assert length == 0

    def test_getitem_loads_data(self):
        """Test that __getitem__ loads data if needed."""
        dataset = ConcreteDataset()

        # Initially not loaded
        assert dataset._loaded is False

        # Accessing item should load data
        item = dataset[0]
        assert item["id"] == "1"
        assert item["input"] == "What is 2+2?"
        assert item["expected"] == "4"
        assert dataset._loaded is True

    def test_getitem_valid_indices(self):
        """Test __getitem__ with valid indices."""
        dataset = ConcreteDataset()

        # Test positive indices
        assert dataset[0]["id"] == "1"
        assert dataset[1]["id"] == "2"
        assert dataset[2]["id"] == "3"

        # Test negative indices
        assert dataset[-1]["id"] == "3"
        assert dataset[-2]["id"] == "2"
        assert dataset[-3]["id"] == "1"

    def test_getitem_invalid_indices(self):
        """Test __getitem__ with invalid indices."""
        dataset = ConcreteDataset()

        with pytest.raises(IndexError):
            dataset[10]  # Out of range

        with pytest.raises(IndexError):
            dataset[-10]  # Out of range negative

    def test_getitem_with_slice(self):
        """Test __getitem__ with slice notation."""
        dataset = ConcreteDataset()

        # Test slice
        subset = dataset[0:2]
        assert len(subset) == 2
        assert subset[0]["id"] == "1"
        assert subset[1]["id"] == "2"

        # Test slice with step
        subset = dataset[::2]
        assert len(subset) == 2
        assert subset[0]["id"] == "1"
        assert subset[1]["id"] == "3"

    def test_iter_functionality(self):
        """Test iterator functionality."""
        dataset = ConcreteDataset()

        items = list(dataset)
        assert len(items) == 3
        assert items[0]["id"] == "1"
        assert items[1]["id"] == "2"
        assert items[2]["id"] == "3"

    def test_iter_with_num_samples(self):
        """Test iterator with num_samples limit."""
        dataset = ConcreteDataset(num_samples=2)

        items = list(dataset)
        assert len(items) == 2
        assert items[0]["id"] == "1"
        assert items[1]["id"] == "2"

    def test_contains_functionality(self):
        """Test membership testing."""
        dataset = ConcreteDataset()

        first_item = dataset[0]
        assert first_item in dataset

        fake_item = {"id": "999", "input": "fake", "expected": "fake"}
        assert fake_item not in dataset

    def test_get_sample(self):
        """Test get_sample method."""
        dataset = ConcreteDataset()

        # Test valid index
        sample = dataset.get_sample(1)
        assert sample["id"] == "2"
        assert sample["input"] == "What is the capital of France?"

        # Test invalid index
        with pytest.raises(IndexError):
            dataset.get_sample(10)

    def test_validate_sample(self):
        """Test sample validation method."""
        dataset = ConcreteDataset()

        # Valid sample
        valid_sample = {"input": "test", "expected": "result", "id": "1"}
        assert dataset.validate_sample(valid_sample) is True

        # Invalid sample - missing required keys
        invalid_sample = {"id": "1", "input": "test"}  # Missing 'expected'
        assert dataset.validate_sample(invalid_sample) is False

        invalid_sample2 = {"id": "1", "expected": "result"}  # Missing 'input'
        assert dataset.validate_sample(invalid_sample2) is False

    def test_get_info(self):
        """Test get_info method."""
        dataset = ConcreteDataset(split="train", seed=123, custom_param="value")

        info = dataset.get_info()

        assert info["name"] == "test_dataset"
        assert info["type"] == "ConcreteDataset"
        assert info["num_samples"] == 3  # Length of dataset
        assert info["split"] == "train"
        assert info["seed"] == 123

    def test_preprocess_sample(self):
        """Test sample preprocessing method."""
        dataset = ConcreteDataset()

        sample = {"input": "test", "expected": "result", "id": "1"}
        processed = dataset.preprocess_sample(sample)

        # Default implementation should return same sample
        assert processed == sample

    def test_postprocess_sample(self):
        """Test sample postprocessing method."""
        dataset = ConcreteDataset()

        sample = {"input": "test", "expected": "result", "id": "1", "score": 0.95}
        processed = dataset.postprocess_sample(sample)

        # Default implementation should return same sample
        assert processed == sample

    def test_save_to_file(self):
        """Test saving dataset to file."""
        import json
        import tempfile

        dataset = ConcreteDataset()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            file_path = f.name

        try:
            dataset.save_to_file(file_path)

            # Verify file was created and contains correct data
            with open(file_path) as f:
                saved_data = json.load(f)

            assert len(saved_data) == 3
            assert saved_data[0]["id"] == "1"
            assert saved_data[1]["id"] == "2"
            assert saved_data[2]["id"] == "3"
        finally:
            import os

            os.unlink(file_path)

    def test_load_from_file_not_implemented(self):
        """Test that load_from_file raises NotImplementedError."""
        with pytest.raises(NotImplementedError):
            ConcreteDataset.load_from_file("dummy_path.json")

    def test_abstract_load_data_method(self):
        """Test that abstract load_data method cannot be called directly."""
        with pytest.raises(TypeError):
            # Cannot instantiate abstract class
            BaseDataset("test")

    def test_from_config_classmethod(self):
        """Test creating dataset from configuration."""
        config = {
            "name": "config_dataset",
            "num_samples": 5,
            "split": "validation",
            "seed": 99,
            "custom_param": "test_value",
        }

        dataset = ConcreteDataset.from_config(config)

        assert dataset.name == "config_dataset"
        assert dataset.num_samples == 5
        assert dataset.split == "validation"
        assert dataset.seed == 99
        assert dataset.kwargs["custom_param"] == "test_value"

    def test_str_representation(self):
        """Test string representation."""
        dataset = ConcreteDataset()
        str_repr = str(dataset)

        # Default object string representation
        assert "ConcreteDataset" in str_repr
        assert "object at" in str_repr

    def test_repr_representation(self):
        """Test detailed string representation."""
        dataset = ConcreteDataset(num_samples=2, split="train")
        repr_str = repr(dataset)

        # Default object repr representation
        assert "ConcreteDataset" in repr_str
        assert "object at" in repr_str


class TestDatasetEdgeCases:
    """Test edge cases for dataset functionality."""

    def test_empty_dataset_operations(self):
        """Test operations on empty dataset."""
        dataset = ConcreteDataset(data=[])

        assert len(dataset) == 0
        assert list(dataset) == []

        with pytest.raises(IndexError):
            dataset[0]

        # Test iteration over empty dataset
        items = list(dataset)
        assert items == []

    def test_single_item_dataset(self):
        """Test dataset with single item."""
        single_data = [{"id": "1", "input": "test", "expected": "result"}]
        dataset = ConcreteDataset(data=single_data)

        assert len(dataset) == 1
        assert dataset[0]["id"] == "1"
        assert dataset[-1]["id"] == "1"  # Same item for negative index

        # Test iteration over single item
        items = list(dataset)
        assert len(items) == 1
        assert items[0]["id"] == "1"

    def test_num_samples_larger_than_data(self):
        """Test when num_samples is larger than actual data."""
        dataset = ConcreteDataset(num_samples=10)  # But only 3 items available

        length = len(dataset)
        assert length == 3  # Should return actual size, not requested size

    def test_num_samples_zero(self):
        """Test with num_samples set to zero."""
        dataset = ConcreteDataset(num_samples=0)

        length = len(dataset)
        assert length == 0
