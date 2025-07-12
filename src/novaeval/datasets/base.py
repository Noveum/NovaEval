"""
Base dataset class for NovaEval.

This module defines the abstract base class for all datasets.
"""

from abc import ABC, abstractmethod
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Optional, Union


class BaseDataset(ABC):
    """
    Abstract base class for all datasets.

    This class defines the interface that all datasets must implement.
    """

    def __init__(
        self,
        name: str,
        num_samples: Optional[int] = None,
        split: str = "test",
        seed: int = 42,
        **kwargs,
    ):
        """
        Initialize the dataset.

        Args:
            name: Name of the dataset
            num_samples: Maximum number of samples to load (None for all)
            split: Dataset split to use (train/validation/test)
            seed: Random seed for reproducibility
            **kwargs: Additional dataset-specific arguments
        """
        self.name = name
        self.num_samples = num_samples
        self.split = split
        self.seed = seed
        self.kwargs = kwargs
        self._data = None
        self._loaded = False

    @abstractmethod
    def load_data(self) -> list[dict[str, Any]]:
        """
        Load the dataset from its source.

        Returns:
            List of dataset samples, where each sample is a dictionary
            with keys like 'input', 'expected', 'id', etc.
        """
        pass

    def __len__(self) -> int:
        """
        Get the number of samples in the dataset.

        Returns:
            Number of samples
        """
        if not self._loaded:
            self._data = self.load_data()
            self._loaded = True
        return len(self._data)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """
        Iterate over dataset samples.

        Yields:
            Dataset samples as dictionaries
        """
        if not self._loaded:
            self._data = self.load_data()
            self._loaded = True

        yield from self._data

    def __getitem__(self, index: int) -> dict[str, Any]:
        """
        Get a specific sample by index.

        Args:
            index: Sample index

        Returns:
            Dataset sample as dictionary
        """
        if not self._loaded:
            self._data = self.load_data()
            self._loaded = True
        return self._data[index]

    def get_sample(self, index: int) -> dict[str, Any]:
        """
        Get a specific sample by index.

        Args:
            index: Sample index

        Returns:
            Dataset sample as dictionary
        """
        return self[index]

    def get_info(self) -> dict[str, Any]:
        """
        Get information about the dataset.

        Returns:
            Dictionary containing dataset metadata
        """
        return {
            "name": self.name,
            "split": self.split,
            "num_samples": len(self),
            "seed": self.seed,
            "type": self.__class__.__name__,
        }

    def validate_sample(self, sample: dict[str, Any]) -> bool:
        """
        Validate that a sample has the required format.

        Args:
            sample: Sample to validate

        Returns:
            True if sample is valid, False otherwise
        """
        required_keys = {"input", "expected"}
        return all(key in sample for key in required_keys)

    def preprocess_sample(self, sample: dict[str, Any]) -> dict[str, Any]:
        """
        Preprocess a sample before evaluation.

        Args:
            sample: Raw sample from the dataset

        Returns:
            Preprocessed sample
        """
        # Default implementation: no preprocessing
        return sample

    def postprocess_sample(self, sample: dict[str, Any]) -> dict[str, Any]:
        """
        Postprocess a sample after evaluation.

        Args:
            sample: Sample with evaluation results

        Returns:
            Postprocessed sample
        """
        # Default implementation: no postprocessing
        return sample

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "BaseDataset":
        """
        Create a dataset from configuration.

        Args:
            config: Configuration dictionary

        Returns:
            Configured dataset instance
        """
        return cls(**config)

    def save_to_file(self, file_path: Union[str, Path]) -> None:
        """
        Save the dataset to a file.

        Args:
            file_path: Path to save the dataset
        """
        import json

        if not self._loaded:
            self._data = self.load_data()
            self._loaded = True

        file_path = Path(file_path)
        with open(file_path, "w") as f:
            json.dump(self._data, f, indent=2)

    @classmethod
    def load_from_file(cls, file_path: Union[str, Path], **kwargs) -> "BaseDataset":
        """
        Load a dataset from a file.

        Args:
            file_path: Path to the dataset file
            **kwargs: Additional arguments for dataset initialization

        Returns:
            Dataset instance
        """
        # This would be implemented in a custom dataset class
        raise NotImplementedError("Subclasses should implement load_from_file")
