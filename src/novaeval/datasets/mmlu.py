"""
MMLU dataset implementation for NovaEval.

This module provides a dataset loader for the Massive Multitask Language
Understanding (MMLU) benchmark.
"""

import random
from typing import Any, Optional

from datasets import load_dataset

from novaeval.datasets.base import BaseDataset


class MMLUDataset(BaseDataset):
    """
    MMLU (Massive Multitask Language Understanding) dataset.

    This dataset contains multiple-choice questions across 57 subjects
    including elementary mathematics, US history, computer science, law, and more.
    """

    SUBJECTS = [
        "abstract_algebra",
        "anatomy",
        "astronomy",
        "business_ethics",
        "clinical_knowledge",
        "college_biology",
        "college_chemistry",
        "college_computer_science",
        "college_mathematics",
        "college_medicine",
        "college_physics",
        "computer_security",
        "conceptual_physics",
        "econometrics",
        "electrical_engineering",
        "elementary_mathematics",
        "formal_logic",
        "global_facts",
        "high_school_biology",
        "high_school_chemistry",
        "high_school_computer_science",
        "high_school_european_history",
        "high_school_geography",
        "high_school_government_and_politics",
        "high_school_macroeconomics",
        "high_school_mathematics",
        "high_school_microeconomics",
        "high_school_physics",
        "high_school_psychology",
        "high_school_statistics",
        "high_school_us_history",
        "high_school_world_history",
        "human_aging",
        "human_sexuality",
        "international_law",
        "jurisprudence",
        "logical_fallacies",
        "machine_learning",
        "management",
        "marketing",
        "medical_genetics",
        "miscellaneous",
        "moral_disputes",
        "moral_scenarios",
        "nutrition",
        "philosophy",
        "prehistory",
        "professional_accounting",
        "professional_law",
        "professional_medicine",
        "professional_psychology",
        "public_relations",
        "security_studies",
        "sociology",
        "us_foreign_policy",
        "virology",
        "world_religions",
    ]

    def __init__(
        self,
        subset: Optional[str] = None,
        num_samples: Optional[int] = None,
        split: str = "test",
        seed: int = 42,
        few_shot: int = 0,
        **kwargs: Any,
    ):
        """
        Initialize the MMLU dataset.

        Args:
            subset: Specific MMLU subject to load (None for all subjects)
            num_samples: Maximum number of samples to load
            split: Dataset split to use
            seed: Random seed for reproducibility
            few_shot: Number of few-shot examples to include in prompts
            **kwargs: Additional arguments
        """
        name = f"mmlu_{subset}" if subset else "mmlu_all"
        super().__init__(name, num_samples, split, seed, **kwargs)

        self.subset = subset
        self.few_shot = few_shot

        if subset and subset not in self.SUBJECTS:
            raise ValueError(
                f"Invalid MMLU subset: {subset}. Must be one of {self.SUBJECTS}"
            )

    def load_data(self) -> list[dict[str, Any]]:
        """
        Load MMLU data from HuggingFace datasets.

        Returns:
            List of formatted samples
        """
        random.seed(self.seed)
        samples = []

        if self.subset:
            # Load specific subject
            dataset = load_dataset("cais/mmlu", self.subset, split=self.split)
            samples.extend(self._process_subject_data(dataset, self.subset))
        else:
            # Load all subjects
            for subject in self.SUBJECTS:
                try:
                    dataset = load_dataset("cais/mmlu", subject, split=self.split)
                    samples.extend(self._process_subject_data(dataset, subject))
                except Exception as e:
                    print(f"Warning: Could not load subject {subject}: {e}")

        # Shuffle and limit samples
        random.shuffle(samples)
        if self.num_samples:
            samples = samples[: self.num_samples]

        return samples

    def _process_subject_data(self, dataset: Any, subject: str) -> list[dict[str, Any]]:
        """
        Process data for a specific subject.

        Args:
            dataset: HuggingFace dataset for the subject
            subject: Subject name

        Returns:
            List of processed samples
        """
        samples = []

        for i, item in enumerate(dataset):
            # Format the multiple choice question
            question = item["question"]
            choices = item["choices"]
            answer_idx = item["answer"]

            # Create formatted input
            formatted_input = self._format_question(question, choices, subject)

            # Get the correct answer
            correct_answer = choices[answer_idx]

            sample = {
                "id": f"{subject}_{i}",
                "input": formatted_input,
                "expected": correct_answer,
                "answer_index": answer_idx,
                "choices": choices,
                "subject": subject,
                "question": question,
                "metadata": {
                    "dataset": "mmlu",
                    "subject": subject,
                    "question_type": "multiple_choice",
                },
            }

            samples.append(sample)

        return samples

    def _format_question(self, question: str, choices: list[str], subject: str) -> str:
        """
        Format a question with choices for the model.

        Args:
            question: The question text
            choices: List of answer choices
            subject: Subject name

        Returns:
            Formatted question string
        """
        # Add few-shot examples if requested
        prompt = ""
        if self.few_shot > 0:
            prompt += self._get_few_shot_examples(subject)
            prompt += "\n\n"

        # Add the main question
        prompt += f"Question: {question}\n"

        # Add choices
        choice_labels = ["A", "B", "C", "D"]
        for i, choice in enumerate(choices):
            prompt += f"{choice_labels[i]}. {choice}\n"

        prompt += "Answer:"

        return prompt

    def _get_few_shot_examples(self, subject: str) -> str:
        """
        Get few-shot examples for the given subject.

        Args:
            subject: Subject name

        Returns:
            Formatted few-shot examples
        """
        # This would load examples from the dev/train split
        # For now, return a placeholder
        return (
            f"Here are some example questions from {subject.replace('_', ' ').title()}:"
        )

    def preprocess_sample(self, sample: dict[str, Any]) -> dict[str, Any]:
        """
        Preprocess MMLU sample.

        Args:
            sample: Raw MMLU sample

        Returns:
            Preprocessed sample
        """
        # Add generation kwargs for multiple choice
        sample["generation_kwargs"] = {
            "max_tokens": 10,
            "temperature": 0.0,
            "stop": ["\n", "Question:"],
        }

        return sample

    def get_info(self) -> dict[str, Any]:
        """
        Get information about the MMLU dataset.

        Returns:
            Dictionary containing dataset metadata
        """
        info = super().get_info()
        info.update(
            {
                "subset": self.subset,
                "few_shot": self.few_shot,
                "total_subjects": len(self.SUBJECTS),
                "description": "Massive Multitask Language Understanding benchmark",
            }
        )
        return info
