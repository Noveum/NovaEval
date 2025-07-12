"""
Custom dataset implementation for NovaEval.

This module provides dataset loaders for custom user-defined datasets
in various formats (JSON, CSV, etc.).
"""

import json
import random
from pathlib import Path
from typing import Any, Callable, Optional, Union

import pandas as pd

from novaeval.datasets.base import BaseDataset


class CustomDataset(BaseDataset):
    """
    Custom dataset loader for user-defined datasets.

    Supports loading from JSON, CSV, and programmatic data sources.
    """

    def __init__(
        self,
        data_source: Union[str, Path, list[dict[str, Any]], Callable[..., Any]],
        input_column: str = "input",
        target_column: str = "expected",
        id_column: Optional[str] = None,
        num_samples: Optional[int] = None,
        split: str = "test",
        seed: int = 42,
        preprocessing_fn: Optional[Callable[..., Any]] = None,
        **kwargs: Any,
    ):
        """
        Initialize the custom dataset.

        Args:
            data_source: Path to data file, list of samples, or callable that returns data
            input_column: Column name containing the input text
            target_column: Column name containing the target/expected output
            id_column: Column name containing sample IDs (optional)
            num_samples: Maximum number of samples to load
            split: Dataset split identifier
            seed: Random seed for reproducibility
            preprocessing_fn: Optional function to preprocess samples
            **kwargs: Additional arguments
        """
        super().__init__("custom", num_samples, split, seed, **kwargs)

        self.data_source = data_source
        self.input_column = input_column
        self.target_column = target_column
        self.id_column = id_column
        self.preprocessing_fn = preprocessing_fn

    def load_data(self) -> list[dict[str, Any]]:
        """
        Load data from the specified source.

        Returns:
            List of formatted samples
        """
        random.seed(self.seed)

        # Load raw data based on source type
        if isinstance(self.data_source, list):
            raw_data = self.data_source
        elif callable(self.data_source):
            raw_data = self.data_source()
        elif isinstance(self.data_source, (str, Path)):
            raw_data = self._load_from_file(self.data_source)
        else:
            raise ValueError(f"Unsupported data source type: {type(self.data_source)}")

        # Convert to our format
        samples = []
        for i, item in enumerate(raw_data):
            sample = self._convert_sample(item, i)
            if sample:
                samples.append(sample)

        # Apply preprocessing if provided
        if self.preprocessing_fn:
            samples = [self.preprocessing_fn(sample) for sample in samples]

        # Shuffle and limit samples
        random.shuffle(samples)
        if self.num_samples:
            samples = samples[: self.num_samples]

        return samples

    def _load_from_file(self, file_path: Union[str, Path]) -> list[dict[str, Any]]:
        """
        Load data from a file.

        Args:
            file_path: Path to the data file

        Returns:
            List of raw data samples
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")

        suffix = file_path.suffix.lower()

        if suffix == ".json":
            return self._load_json(file_path)
        elif suffix == ".jsonl":
            return self._load_jsonl(file_path)
        elif suffix == ".csv":
            return self._load_csv(file_path)
        elif suffix in [".xlsx", ".xls"]:
            return self._load_excel(file_path)
        else:
            raise ValueError(f"Unsupported file format: {suffix}")

    def _load_json(self, file_path: Path) -> list[dict[str, Any]]:
        """Load data from JSON file."""
        with open(file_path, encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            return data  # type: ignore
        elif isinstance(data, dict):
            # If it's a dict, try to find a list of samples
            for key in ["data", "samples", "examples", "items"]:
                if key in data and isinstance(data[key], list):
                    return data[key]  # type: ignore
            # If no list found, treat the dict as a single sample
            return [data]  # type: ignore
        else:
            raise ValueError("JSON file must contain a list or dict")

    def _load_jsonl(self, file_path: Path) -> list[dict[str, Any]]:
        """Load data from JSONL file."""
        data = []
        with open(file_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        return data

    def _load_csv(self, file_path: Path) -> list[dict[str, Any]]:
        """Load data from CSV file."""
        df = pd.read_csv(file_path)
        return df.to_dict("records")  # type: ignore

    def _load_excel(self, file_path: Path) -> list[dict[str, Any]]:
        """Load data from Excel file."""
        df = pd.read_excel(file_path)
        return df.to_dict("records")  # type: ignore

    def _convert_sample(
        self, item: dict[str, Any], index: int
    ) -> Optional[dict[str, Any]]:
        """
        Convert a raw sample to NovaEval format.

        Args:
            item: Raw sample
            index: Sample index

        Returns:
            Converted sample or None if conversion fails
        """
        try:
            # Extract input and target
            input_text = item.get(self.input_column)
            target_text = item.get(self.target_column)

            if input_text is None or target_text is None:
                return None

            # Generate ID
            if self.id_column and self.id_column in item:
                sample_id = str(item[self.id_column])
            else:
                sample_id = f"custom_{index}"

            sample = {
                "id": sample_id,
                "input": str(input_text),
                "expected": str(target_text),
                "metadata": {
                    "dataset": "custom",
                    "original_index": index,
                    "source": (
                        str(self.data_source)
                        if isinstance(self.data_source, (str, Path))
                        else "programmatic"
                    ),
                },
            }

            # Add any additional fields from the original item
            for key, value in item.items():
                if key not in [self.input_column, self.target_column, self.id_column]:
                    sample[f"original_{key}"] = value

            return sample

        except Exception as e:
            print(f"Warning: Failed to convert sample {index}: {e}")
            return None

    def get_info(self) -> dict[str, Any]:
        """
        Get information about the custom dataset.

        Returns:
            Dictionary containing dataset metadata
        """
        info = super().get_info()
        info.update(
            {
                "data_source": (
                    str(self.data_source)
                    if isinstance(self.data_source, (str, Path))
                    else "programmatic"
                ),
                "input_column": self.input_column,
                "target_column": self.target_column,
                "id_column": self.id_column,
                "source": "custom",
            }
        )
        return info

    @classmethod
    def from_samples(
        cls,
        samples: list[dict[str, Any]],
        input_column: str = "input",
        target_column: str = "expected",
        **kwargs: Any,
    ) -> "CustomDataset":
        """
        Create a dataset from a list of samples.

        Args:
            samples: List of sample dictionaries
            input_column: Column name for input text
            target_column: Column name for target text
            **kwargs: Additional arguments

        Returns:
            CustomDataset instance
        """
        return cls(
            data_source=samples,
            input_column=input_column,
            target_column=target_column,
            **kwargs,
        )

    @classmethod
    def from_generator(
        cls,
        generator_fn: Callable[[], list[dict[str, Any]]],
        input_column: str = "input",
        target_column: str = "expected",
        **kwargs: Any,
    ) -> "CustomDataset":
        """
        Create a dataset from a generator function.

        Args:
            generator_fn: Function that returns a list of samples
            input_column: Column name for input text
            target_column: Column name for target text
            **kwargs: Additional arguments

        Returns:
            CustomDataset instance
        """
        return cls(
            data_source=generator_fn,
            input_column=input_column,
            target_column=target_column,
            **kwargs,
        )
