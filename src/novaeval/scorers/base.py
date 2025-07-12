"""
Base scorer class for NovaEval.

This module defines the abstract base class for all scoring mechanisms.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional, Union

from pydantic import BaseModel, Field


class ScoreResult(BaseModel):
    """
    Result of a scoring operation.

    Contains the score, pass/fail status, reasoning, and additional metadata.
    """

    score: float = Field(description="Numerical score value")
    passed: bool = Field(description="Whether the score passes the threshold")
    reasoning: str = Field(description="Explanation of the scoring decision")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata about the scoring"
    )


class BaseScorer(ABC):
    """
    Abstract base class for all scorers.

    This class defines the interface that all scorers must implement.
    """

    def __init__(self, name: str, description: Optional[str] = None, **kwargs: Any):
        """
        Initialize the scorer.

        Args:
            name: Name of the scorer
            description: Description of what the scorer measures
            **kwargs: Additional scorer-specific parameters
        """
        self.name = name
        self.description = description or f"{name} scorer"
        self.kwargs = kwargs

        # Statistics tracking
        self.total_scores = 0
        self.score_sum = 0.0
        self.scores_history: list[Union[float, dict[str, float]]] = []

    @abstractmethod
    def score(
        self,
        prediction: str,
        ground_truth: str,
        context: Optional[dict[str, Any]] = None,
    ) -> Union[float, dict[str, float]]:
        """
        Score a prediction against ground truth.

        Args:
            prediction: Model's prediction/output
            ground_truth: Expected/correct output
            context: Additional context information (e.g., original sample)

        Returns:
            Score value (float) or dictionary of scores
        """
        pass

    def score_batch(
        self,
        predictions: list[str],
        ground_truths: list[str],
        contexts: Optional[list[dict[str, Any]]] = None,
    ) -> list[Union[float, dict[str, float]]]:
        """
        Score multiple predictions in batch.

        Args:
            predictions: List of model predictions
            ground_truths: List of expected outputs
            contexts: List of context information

        Returns:
            List of scores
        """
        if contexts is None:
            contexts = [None] * len(predictions)  # type: ignore

        scores = []
        for pred, truth, ctx in zip(predictions, ground_truths, contexts):
            try:
                score = self.score(pred, truth, ctx)
                scores.append(score)
                self._track_score(score)
            except Exception as e:
                # Handle scoring errors gracefully
                scores.append(0.0)
                print(f"Warning: Scoring failed for {self.name}: {e}")

        return scores

    def get_info(self) -> dict[str, Any]:
        """
        Get information about the scorer.

        Returns:
            Dictionary containing scorer metadata
        """
        return {
            "name": self.name,
            "description": self.description,
            "type": self.__class__.__name__,
            "total_scores": self.total_scores,
            "average_score": (
                self.score_sum / self.total_scores if self.total_scores > 0 else 0.0
            ),
            "config": self.kwargs,
        }

    def reset_stats(self) -> None:
        """Reset scoring statistics."""
        self.total_scores = 0
        self.score_sum = 0.0
        self.scores_history = []

    def get_stats(self) -> dict[str, Any]:
        """
        Get scoring statistics.

        Returns:
            Dictionary containing scoring statistics
        """
        if not self.scores_history:
            return {
                "count": 0,
                "mean": 0.0,
                "min": 0.0,
                "max": 0.0,
                "std": 0.0,
            }

        import statistics

        # Convert to numeric values if scores are dicts
        numeric_scores = []
        for score in self.scores_history:
            if isinstance(score, dict):
                # Use the first numeric value or a default key
                if "score" in score:
                    numeric_scores.append(score["score"])
                else:
                    numeric_scores.append(next(iter(score.values())))
            else:
                numeric_scores.append(float(score))

        return {
            "count": len(numeric_scores),
            "mean": statistics.mean(numeric_scores),
            "min": min(numeric_scores),
            "max": max(numeric_scores),
            "std": statistics.stdev(numeric_scores) if len(numeric_scores) > 1 else 0.0,
        }

    def _track_score(self, score: Union[float, dict[str, float]]) -> None:
        """
        Track a score for statistics.

        Args:
            score: Score to track
        """
        self.total_scores += 1
        self.scores_history.append(score)

        # Update sum for numeric scores
        if isinstance(score, (int, float)):
            self.score_sum += float(score)
        elif isinstance(score, dict) and "score" in score:
            self.score_sum += float(score["score"])
        elif isinstance(score, dict):
            # Use first numeric value
            self.score_sum += float(next(iter(score.values())))

    def validate_inputs(
        self,
        prediction: str,
        ground_truth: str,
        context: Optional[dict[str, Any]] = None,
    ) -> bool:
        """
        Validate scorer inputs.

        Args:
            prediction: Model prediction
            ground_truth: Expected output
            context: Additional context

        Returns:
            True if inputs are valid, False otherwise
        """
        if not isinstance(prediction, str) or not isinstance(ground_truth, str):
            return False

        return not (context is not None and not isinstance(context, dict))

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "BaseScorer":
        """
        Create a scorer from configuration.

        Args:
            config: Configuration dictionary

        Returns:
            Configured scorer instance
        """
        return cls(**config)

    def __str__(self) -> str:
        """String representation of the scorer."""
        return f"{self.__class__.__name__}(name='{self.name}')"

    def __repr__(self) -> str:
        """Detailed string representation of the scorer."""
        return (
            f"{self.__class__.__name__}("
            f"name='{self.name}', "
            f"description='{self.description}'"
            f")"
        )
