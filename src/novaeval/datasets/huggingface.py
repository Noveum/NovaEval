"""
HuggingFace dataset implementation for NovaEval.

This module provides a generic dataset loader for any HuggingFace dataset.
"""

import random
from typing import Any, Callable, Optional

from datasets import load_dataset

from novaeval.datasets.base import BaseDataset


class HuggingFaceDataset(BaseDataset):
    """
    Generic HuggingFace dataset loader.

    This class can load any dataset from the HuggingFace Hub and format it
    for evaluation with NovaEval.
    """

    def __init__(
        self,
        dataset_name: str,
        subset: Optional[str] = None,
        input_column: str = "input",
        target_column: str = "target",
        num_samples: Optional[int] = None,
        split: str = "test",
        seed: int = 42,
        preprocessing_fn: Optional[Callable] = None,
        **kwargs,
    ):
        """
        Initialize the HuggingFace dataset.

        Args:
            dataset_name: Name of the dataset on HuggingFace Hub
            subset: Subset/configuration of the dataset
            input_column: Column name containing the input text
            target_column: Column name containing the target/expected output
            num_samples: Maximum number of samples to load
            split: Dataset split to use
            seed: Random seed for reproducibility
            preprocessing_fn: Optional function to preprocess samples
            **kwargs: Additional arguments for load_dataset
        """
        name = f"hf_{dataset_name}_{subset}" if subset else f"hf_{dataset_name}"
        super().__init__(name, num_samples, split, seed, **kwargs)

        self.dataset_name = dataset_name
        self.subset = subset
        self.input_column = input_column
        self.target_column = target_column
        self.preprocessing_fn = preprocessing_fn
        self.load_kwargs = kwargs

    def load_data(self) -> list[dict[str, Any]]:
        """
        Load data from HuggingFace dataset.

        Returns:
            List of formatted samples
        """
        random.seed(self.seed)

        # Load dataset from HuggingFace
        try:
            if self.subset:
                dataset = load_dataset(
                    self.dataset_name, self.subset, split=self.split, **self.load_kwargs
                )
            else:
                dataset = load_dataset(
                    self.dataset_name, split=self.split, **self.load_kwargs
                )
        except Exception as e:
            raise ValueError(f"Failed to load dataset {self.dataset_name}: {e}")

        # Convert to our format
        samples = []
        for i, item in enumerate(dataset):
            sample = self._convert_sample(item, i)
            if sample:  # Only add valid samples
                samples.append(sample)

        # Apply preprocessing if provided
        if self.preprocessing_fn:
            samples = [self.preprocessing_fn(sample) for sample in samples]

        # Shuffle and limit samples
        random.shuffle(samples)
        if self.num_samples:
            samples = samples[: self.num_samples]

        return samples

    def _convert_sample(
        self, item: dict[str, Any], index: int
    ) -> Optional[dict[str, Any]]:
        """
        Convert a HuggingFace sample to NovaEval format.

        Args:
            item: Raw sample from HuggingFace dataset
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

            sample = {
                "id": f"{self.name}_{index}",
                "input": str(input_text),
                "expected": str(target_text),
                "metadata": {
                    "dataset": self.dataset_name,
                    "subset": self.subset,
                    "original_index": index,
                    "source": "huggingface",
                },
            }

            # Add any additional fields from the original item
            for key, value in item.items():
                if key not in [self.input_column, self.target_column]:
                    sample[f"original_{key}"] = value

            return sample

        except Exception as e:
            print(f"Warning: Failed to convert sample {index}: {e}")
            return None

    def get_info(self) -> dict[str, Any]:
        """
        Get information about the HuggingFace dataset.

        Returns:
            Dictionary containing dataset metadata
        """
        info = super().get_info()
        info.update(
            {
                "dataset_name": self.dataset_name,
                "subset": self.subset,
                "input_column": self.input_column,
                "target_column": self.target_column,
                "source": "huggingface",
            }
        )
        return info


class CommonHFDatasets:
    """
    Factory class for common HuggingFace datasets with predefined configurations.
    """

    @staticmethod
    def squad() -> HuggingFaceDataset:
        """Create SQuAD dataset for reading comprehension."""
        return HuggingFaceDataset(
            dataset_name="squad",
            input_column="question",
            target_column="answers",
            preprocessing_fn=lambda x: {
                **x,
                "expected": x["expected"]["text"][0] if x["expected"]["text"] else "",
                "input": f"Context: {x.get('original_context', '')}\nQuestion: {x['input']}",
            },
        )

    @staticmethod
    def glue(task: str) -> HuggingFaceDataset:
        """Create GLUE benchmark dataset."""
        task_configs = {
            "cola": {"input_column": "sentence", "target_column": "label"},
            "sst2": {"input_column": "sentence", "target_column": "label"},
            "mrpc": {"input_column": "sentence1", "target_column": "label"},
            "qqp": {"input_column": "question1", "target_column": "label"},
            "stsb": {"input_column": "sentence1", "target_column": "label"},
            "mnli": {"input_column": "premise", "target_column": "label"},
            "qnli": {"input_column": "question", "target_column": "label"},
            "rte": {"input_column": "sentence1", "target_column": "label"},
            "wnli": {"input_column": "sentence1", "target_column": "label"},
        }

        if task not in task_configs:
            raise ValueError(f"Unsupported GLUE task: {task}")

        config = task_configs[task]
        return HuggingFaceDataset(dataset_name="glue", subset=task, **config)

    @staticmethod
    def hellaswag() -> HuggingFaceDataset:
        """Create HellaSwag dataset for commonsense reasoning."""
        return HuggingFaceDataset(
            dataset_name="hellaswag",
            input_column="ctx",
            target_column="label",
            preprocessing_fn=lambda x: {
                **x,
                "input": f"{x['input']}\nChoices:\n"
                + "\n".join(
                    [
                        f"{i}. {choice}"
                        for i, choice in enumerate(x.get("original_endings", []))
                    ]
                ),
                "expected": str(x["expected"]),
            },
        )

    @staticmethod
    def truthful_qa() -> HuggingFaceDataset:
        """Create TruthfulQA dataset."""
        return HuggingFaceDataset(
            dataset_name="truthful_qa",
            subset="generation",
            input_column="question",
            target_column="best_answer",
        )
