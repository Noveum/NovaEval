"""
Accuracy-based scorers for NovaEval.

This module provides scorers for exact matching and classification accuracy.
"""

import re
from typing import Any, Optional

from novaeval.scorers.base import BaseScorer


class ExactMatchScorer(BaseScorer):
    """
    Exact string matching scorer.

    Returns 1.0 for exact matches, 0.0 otherwise.
    """

    def __init__(
        self,
        case_sensitive: bool = True,
        strip_whitespace: bool = True,
        normalize_whitespace: bool = False,
        **kwargs: Any,
    ):
        """
        Initialize the exact match scorer.

        Args:
            case_sensitive: Whether to perform case-sensitive matching
            strip_whitespace: Whether to strip leading/trailing whitespace
            normalize_whitespace: Whether to normalize internal whitespace
            **kwargs: Additional parameters
        """
        super().__init__(
            name="exact_match",
            description="Exact string matching scorer",
            case_sensitive=case_sensitive,
            strip_whitespace=strip_whitespace,
            normalize_whitespace=normalize_whitespace,
            **kwargs,
        )

        self.case_sensitive = case_sensitive
        self.strip_whitespace = strip_whitespace
        self.normalize_whitespace = normalize_whitespace

    def score(
        self,
        prediction: str,
        ground_truth: str,
        context: Optional[dict[str, Any]] = None,
    ) -> float:
        """
        Score prediction using exact matching.

        Args:
            prediction: Model's prediction
            ground_truth: Expected output
            context: Additional context (unused)

        Returns:
            1.0 if exact match, 0.0 otherwise
        """
        if not self.validate_inputs(prediction, ground_truth, context):
            return 0.0

        # Preprocess strings
        pred = self._preprocess_string(prediction)
        truth = self._preprocess_string(ground_truth)

        return 1.0 if pred == truth else 0.0

    def _preprocess_string(self, text: str) -> str:
        """
        Preprocess string according to scorer settings.

        Args:
            text: Input text

        Returns:
            Preprocessed text
        """
        if self.strip_whitespace:
            text = text.strip()

        if self.normalize_whitespace:
            text = re.sub(r"\s+", " ", text)

        if self.case_sensitive is False:
            text = text.lower()

        return text


class AccuracyScorer(BaseScorer):
    """
    Classification accuracy scorer.

    Supports multiple choice questions and classification tasks.
    """

    def __init__(
        self,
        extract_answer: bool = True,
        answer_pattern: Optional[str] = None,
        choices: Optional[list[str]] = None,
        **kwargs: Any,
    ):
        """
        Initialize the accuracy scorer.

        Args:
            extract_answer: Whether to extract answer from prediction
            answer_pattern: Regex pattern to extract answer
            choices: List of valid choices for multiple choice
            **kwargs: Additional parameters
        """
        super().__init__(
            name="accuracy",
            description="Classification accuracy scorer",
            extract_answer=extract_answer,
            answer_pattern=answer_pattern,
            choices=choices,
            **kwargs,
        )

        self.extract_answer = extract_answer
        self.answer_pattern = answer_pattern or r"(?:Answer|answer):\s*([A-Za-z0-9]+)"
        self.choices = choices

    def score(
        self,
        prediction: str,
        ground_truth: str,
        context: Optional[dict[str, Any]] = None,
    ) -> float:
        """
        Score prediction using accuracy.

        Args:
            prediction: Model's prediction
            ground_truth: Expected output
            context: Additional context (may contain choices)

        Returns:
            1.0 if correct, 0.0 otherwise
        """
        if not self.validate_inputs(prediction, ground_truth, context):
            return 0.0

        # Extract answer from prediction if needed
        if self.extract_answer:
            extracted_pred = self._extract_answer(prediction, context)
        else:
            extracted_pred = prediction.strip()

        # For MMLU-style questions, convert letter to full choice if available
        if context and "choices" in context and "answer_index" in context:
            # Try to convert extracted letter to choice text
            converted_pred = self._convert_letter_to_choice(extracted_pred, context)
            if converted_pred:
                extracted_pred = converted_pred

        # Normalize answers
        pred_answer = self._normalize_answer(extracted_pred)
        true_answer = self._normalize_answer(ground_truth)

        return 1.0 if pred_answer == true_answer else 0.0

    def _extract_answer(
        self, prediction: str, context: Optional[dict[str, Any]] = None
    ) -> str:
        """
        Extract answer from prediction text.

        Args:
            prediction: Model's prediction
            context: Additional context

        Returns:
            Extracted answer
        """
        # Try multiple patterns in order of preference

        # Pattern 1: "Answer: X" or "answer: X"
        match = re.search(self.answer_pattern, prediction, re.IGNORECASE)
        if match:
            return match.group(1) if match.group(1) else match.group(2)

        # Pattern 2: "The answer is X" or "The correct answer is X"
        match = re.search(
            r"(?:the\s+(?:correct\s+)?answer\s+is\s+)([A-D])", prediction, re.IGNORECASE
        )
        if match:
            return match.group(1)

        # Pattern 3: "**X.**" (bold letter with period)
        match = re.search(r"\*\*([A-D])\.\s*[^*]*\*\*", prediction, re.IGNORECASE)
        if match:
            return match.group(1)

        # Pattern 4: Just a single letter at start of response
        match = re.search(r"^([A-D])\.?\s*$", prediction.strip(), re.IGNORECASE)
        if match:
            return match.group(1)

        # Pattern 5: Letter followed by period or colon at start of line
        match = re.search(r"^([A-D])[\.\:]", prediction, re.MULTILINE | re.IGNORECASE)
        if match:
            return match.group(1)

        # Pattern 6: Stand-alone letter choice (A, B, C, D) near end of text
        lines = prediction.strip().split("\n")
        for line in reversed(lines[-3:]):  # Check last 3 lines
            match = re.search(r"\b([A-D])\b", line, re.IGNORECASE)
            if match:
                return match.group(1)

        # Pattern 7: Letter at the very end of the response
        match = re.search(r"([A-D])\s*$", prediction.strip(), re.IGNORECASE)
        if match:
            return match.group(1)

        # Fallback: find any choice letter in the text (prefer later occurrences)
        choice_matches = list(re.finditer(r"\b([A-D])\b", prediction, re.IGNORECASE))
        if choice_matches:
            return choice_matches[-1].group(1)  # Return last occurrence

        # Final fallback: return first word
        words = prediction.strip().split()
        return words[0] if words else ""

    def _convert_letter_to_choice(
        self, extracted_answer: str, context: dict[str, Any]
    ) -> Optional[str]:
        """
        Convert letter answer (A, B, C, D) to full choice text.

        Args:
            extracted_answer: Extracted letter answer
            context: Context containing choices

        Returns:
            Full choice text or None if conversion fails
        """
        if not extracted_answer or "choices" not in context:
            return None

        # Map letters to indices
        letter_to_index = {"A": 0, "B": 1, "C": 2, "D": 3}
        letter = extracted_answer.strip().upper()

        if letter in letter_to_index:
            choices = context["choices"]
            index = letter_to_index[letter]
            if 0 <= index < len(choices):
                return choices[index]

        return None

    def _normalize_answer(self, answer: str) -> str:
        """
        Normalize answer for comparison.

        Args:
            answer: Answer to normalize

        Returns:
            Normalized answer
        """
        return answer.strip().lower()


class F1Scorer(BaseScorer):
    """
    F1 score for token-level evaluation.

    Useful for tasks like question answering where partial matches matter.
    """

    def __init__(
        self, tokenize: bool = True, case_sensitive: bool = False, **kwargs: Any
    ):
        """
        Initialize the F1 scorer.

        Args:
            tokenize: Whether to tokenize text before comparison
            case_sensitive: Whether to perform case-sensitive comparison
            **kwargs: Additional parameters
        """
        super().__init__(
            name="f1",
            description="Token-level F1 score",
            tokenize=tokenize,
            case_sensitive=case_sensitive,
            **kwargs,
        )

        self.tokenize = tokenize
        self.case_sensitive = case_sensitive

    def score(
        self,
        prediction: str,
        ground_truth: str,
        context: Optional[dict[str, Any]] = None,
    ) -> dict[str, float]:
        """
        Score prediction using F1 score.

        Args:
            prediction: Model's prediction
            ground_truth: Expected output
            context: Additional context (unused)

        Returns:
            Dictionary with precision, recall, and f1 scores
        """
        if not self.validate_inputs(prediction, ground_truth, context):
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

        # Preprocess text
        pred_tokens = self._get_tokens(prediction)
        truth_tokens = self._get_tokens(ground_truth)

        # Calculate overlap
        pred_set = set(pred_tokens)
        truth_set = set(truth_tokens)
        overlap = pred_set & truth_set

        # Calculate metrics
        precision = len(overlap) / len(pred_set) if pred_set else 0.0
        recall = len(overlap) / len(truth_set) if truth_set else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "score": f1,  # Main score for aggregation
        }

    def _get_tokens(self, text: str) -> list[str]:
        """
        Get tokens from text.

        Args:
            text: Input text

        Returns:
            List of tokens
        """
        if not self.case_sensitive:
            text = text.lower()

        # Simple tokenization (split on whitespace and punctuation) or split on whitespace
        tokens = re.findall(r"\b\w+\b", text) if self.tokenize else text.split()

        return tokens
