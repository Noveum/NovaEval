"""
Base evaluator class for NovaEval.

This module defines the abstract base class for all evaluators.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional, Union

from novaeval.datasets.base import BaseDataset
from novaeval.models.base import BaseModel
from novaeval.scorers.base import BaseScorer


class BaseEvaluator(ABC):
    """
    Abstract base class for all evaluators.

    This class defines the interface that all evaluators must implement.
    """

    def __init__(
        self,
        dataset: BaseDataset,
        models: list[BaseModel],
        scorers: list[BaseScorer],
        output_dir: Optional[Union[str, Path]] = None,
        config: Optional[dict[str, Any]] = None,
    ):
        """
        Initialize the evaluator.

        Args:
            dataset: The dataset to evaluate on
            models: List of models to evaluate
            scorers: List of scorers to use for evaluation
            output_dir: Directory to save results
            config: Additional configuration options
        """
        self.dataset = dataset
        self.models = models
        self.scorers = scorers
        self.output_dir = Path(output_dir) if output_dir else Path("./results")
        self.config = config or {}

        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def run(self) -> dict[str, Any]:
        """
        Run the evaluation.

        Returns:
            Dictionary containing evaluation results
        """
        pass

    @abstractmethod
    def evaluate_sample(
        self, sample: dict[str, Any], model: BaseModel, scorers: list[BaseScorer]
    ) -> dict[str, Any]:
        """
        Evaluate a single sample with a model.

        Args:
            sample: The sample to evaluate
            model: The model to use for evaluation
            scorers: List of scorers to apply

        Returns:
            Dictionary containing sample evaluation results
        """
        pass

    @abstractmethod
    def save_results(self, results: dict[str, Any]) -> None:
        """
        Save evaluation results to disk.

        Args:
            results: The results to save
        """
        pass

    def validate_inputs(self) -> None:
        """
        Validate that all inputs are properly configured.

        Raises:
            ValueError: If any inputs are invalid
        """
        if not self.dataset:
            raise ValueError("Dataset is required")

        if not self.models:
            raise ValueError("At least one model is required")

        if not self.scorers:
            raise ValueError("At least one scorer is required")

    @classmethod
    def from_config(cls, config_path: Union[str, Path]) -> "BaseEvaluator":
        """
        Create an evaluator from a configuration file.

        Args:
            config_path: Path to the configuration file

        Returns:
            Configured evaluator instance
        """
        # This will be implemented in subclasses
        raise NotImplementedError("Subclasses must implement from_config")
