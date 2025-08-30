"""
Tests for MMLU dataset implementation.

This module tests the MMLU dataset loader and processing functionality.
"""

from unittest.mock import patch

import pytest

from src.novaeval.datasets.mmlu import MMLUDataset

pytestmark = pytest.mark.unit


class TestMMLUDatasetInit:
    """Test MMLU dataset initialization."""

    def test_init_with_valid_subset(self):
        """Test initialization with a valid subset."""
        dataset = MMLUDataset(subset="abstract_algebra", num_samples=10)

        assert dataset.subset == "abstract_algebra"
        assert dataset.name == "mmlu_abstract_algebra"
        assert dataset.num_samples == 10
        assert dataset.few_shot == 0

    def test_init_with_few_shot(self):
        """Test initialization with few-shot examples."""
        dataset = MMLUDataset(subset="anatomy", few_shot=3)

        assert dataset.few_shot == 3
        assert dataset.subset == "anatomy"

    def test_init_without_subset(self):
        """Test initialization without subset (all subjects)."""
        dataset = MMLUDataset()

        assert dataset.subset is None
        assert dataset.name == "mmlu_all"
        assert dataset.few_shot == 0

    def test_init_with_invalid_subset(self):
        """Test initialization with invalid subset raises ValueError."""
        with pytest.raises(ValueError, match="Invalid MMLU subset: invalid_subject"):
            MMLUDataset(subset="invalid_subject")

    def test_init_with_all_parameters(self):
        """Test initialization with all parameters."""
        dataset = MMLUDataset(
            subset="college_biology", num_samples=50, split="test", seed=42, few_shot=2
        )

        assert dataset.subset == "college_biology"
        assert dataset.num_samples == 50
        assert dataset.split == "test"
        assert dataset.seed == 42
        assert dataset.few_shot == 2


class TestMMLUDatasetLoadData:
    """Test MMLU data loading functionality."""

    @patch("src.novaeval.datasets.mmlu.load_dataset")
    def test_load_data_single_subset(self, mock_load_dataset):
        """Test loading data for a single subset."""
        # Mock dataset
        mock_dataset = [
            {"question": "What is 2+2?", "choices": ["2", "3", "4", "5"], "answer": 2},
            {"question": "What is 3+3?", "choices": ["5", "6", "7", "8"], "answer": 1},
        ]
        mock_load_dataset.return_value = mock_dataset

        dataset = MMLUDataset(subset="elementary_mathematics", seed=42)
        samples = dataset.load_data()

        assert len(samples) == 2
        # Check that both samples are present (order may vary due to shuffling)
        questions = [sample["question"] for sample in samples]
        assert "What is 2+2?" in questions
        assert "What is 3+3?" in questions

        # Find the first sample and check its properties
        first_sample = next(s for s in samples if s["question"] == "What is 2+2?")
        assert first_sample["subject"] == "elementary_mathematics"
        assert first_sample["expected"] == "4"
        assert first_sample["answer_index"] == 2
        assert first_sample["choices"] == ["2", "3", "4", "5"]

        mock_load_dataset.assert_called_once_with(
            "cais/mmlu", "elementary_mathematics", split="test"
        )

    @patch("src.novaeval.datasets.mmlu.load_dataset")
    def test_load_data_with_num_samples_limit(self, mock_load_dataset):
        """Test loading data with sample limit."""
        # Create more samples than the limit
        mock_dataset = [
            {"question": f"Question {i}?", "choices": ["A", "B", "C", "D"], "answer": 0}
            for i in range(10)
        ]
        mock_load_dataset.return_value = mock_dataset

        dataset = MMLUDataset(subset="anatomy", num_samples=3, seed=42)
        samples = dataset.load_data()

        assert len(samples) == 3

    @patch("src.novaeval.datasets.mmlu.load_dataset")
    @patch("builtins.print")
    def test_load_data_all_subjects_with_error(self, mock_print, mock_load_dataset):
        """Test loading all subjects with some subjects failing."""

        def side_effect(dataset_name, subject, split):
            if subject == "abstract_algebra":
                return [{"question": "Test", "choices": ["A", "B"], "answer": 0}]
            else:
                raise Exception("Dataset not found")

        mock_load_dataset.side_effect = side_effect

        # Patch SUBJECTS to only test a few
        with patch.object(MMLUDataset, "SUBJECTS", ["abstract_algebra", "anatomy"]):
            dataset = MMLUDataset(seed=42)  # No subset = all subjects
            samples = dataset.load_data()

        # Should have samples from abstract_algebra only
        assert len(samples) == 1
        assert samples[0]["subject"] == "abstract_algebra"

        # Should have printed warning about anatomy
        mock_print.assert_called()
        warning_call = mock_print.call_args_list[-1]
        assert "Could not load subject anatomy" in str(warning_call)

    @patch("src.novaeval.datasets.mmlu.load_dataset")
    def test_load_data_all_subjects_success(self, mock_load_dataset):
        """Test loading all subjects successfully."""
        mock_load_dataset.return_value = [
            {"question": "Test question", "choices": ["A", "B"], "answer": 0}
        ]

        # Patch SUBJECTS to test only a few subjects
        with patch.object(MMLUDataset, "SUBJECTS", ["abstract_algebra", "anatomy"]):
            dataset = MMLUDataset(seed=42)
            samples = dataset.load_data()

        assert len(samples) == 2  # One sample from each subject
        subjects = [sample["subject"] for sample in samples]
        assert "abstract_algebra" in subjects
        assert "anatomy" in subjects


class TestMMLUDatasetProcessing:
    """Test MMLU data processing methods."""

    def test_process_subject_data(self):
        """Test processing subject data."""
        mock_dataset = [
            {
                "question": "What is photosynthesis?",
                "choices": ["Process A", "Process B", "Process C", "Process D"],
                "answer": 2,
            }
        ]

        dataset = MMLUDataset(subset="college_biology")
        samples = dataset._process_subject_data(mock_dataset, "college_biology")

        assert len(samples) == 1
        sample = samples[0]

        assert sample["id"] == "college_biology_0"
        assert sample["question"] == "What is photosynthesis?"
        assert sample["expected"] == "Process C"
        assert sample["answer_index"] == 2
        assert sample["choices"] == ["Process A", "Process B", "Process C", "Process D"]
        assert sample["subject"] == "college_biology"
        assert sample["metadata"]["dataset"] == "mmlu"
        assert sample["metadata"]["subject"] == "college_biology"
        assert sample["metadata"]["question_type"] == "multiple_choice"

    def test_format_question_without_few_shot(self):
        """Test formatting question without few-shot examples."""
        dataset = MMLUDataset(subset="college_physics", few_shot=0)

        formatted = dataset._format_question(
            "What is the speed of light?",
            ["299,792,458 m/s", "300,000,000 m/s", "3x10^8 m/s", "All of the above"],
            "college_physics",
        )

        expected = """Question: What is the speed of light?
A. 299,792,458 m/s
B. 300,000,000 m/s
C. 3x10^8 m/s
D. All of the above
Answer:"""

        assert formatted == expected

    def test_format_question_with_few_shot(self):
        """Test formatting question with few-shot examples."""
        dataset = MMLUDataset(subset="college_chemistry", few_shot=2)

        formatted = dataset._format_question(
            "What is H2O?",
            ["Hydrogen", "Water", "Oxygen", "Carbon"],
            "college_chemistry",
        )

        # Should include few-shot examples
        assert "Here are some example questions from College Chemistry:" in formatted
        assert "Question: What is H2O?" in formatted
        assert "A. Hydrogen" in formatted
        assert "Answer:" in formatted

    def test_get_few_shot_examples(self):
        """Test getting few-shot examples."""
        dataset = MMLUDataset()

        examples = dataset._get_few_shot_examples("computer_science")

        assert "Here are some example questions from Computer Science:" in examples

    def test_get_few_shot_examples_with_underscores(self):
        """Test few-shot examples with subject names containing underscores."""
        dataset = MMLUDataset()

        examples = dataset._get_few_shot_examples("high_school_biology")

        assert "High School Biology" in examples

    def test_preprocess_sample(self):
        """Test sample preprocessing."""
        dataset = MMLUDataset()

        sample = {"id": "test_1", "input": "Question: What is 2+2?", "expected": "4"}

        processed = dataset.preprocess_sample(sample)

        assert "generation_kwargs" in processed
        assert processed["generation_kwargs"]["max_tokens"] == 10
        assert processed["generation_kwargs"]["temperature"] == 0.0
        assert processed["generation_kwargs"]["stop"] == ["\n", "Question:"]


class TestMMLUDatasetInfo:
    """Test MMLU dataset info functionality."""

    @patch("src.novaeval.datasets.mmlu.load_dataset")
    def test_get_info_with_subset(self, mock_load_dataset):
        """Test getting info for dataset with subset."""
        # Mock the dataset to avoid network calls
        mock_load_dataset.return_value = [
            {"question": "Test", "choices": ["A", "B"], "answer": 0}
        ]

        dataset = MMLUDataset(subset="college_mathematics", few_shot=3)

        info = dataset.get_info()

        assert info["subset"] == "college_mathematics"
        assert info["few_shot"] == 3
        assert info["total_subjects"] == len(MMLUDataset.SUBJECTS)
        assert (
            info["description"] == "Massive Multitask Language Understanding benchmark"
        )

    def test_get_info_without_subset(self):
        """Test getting info for dataset without subset."""
        dataset = MMLUDataset(few_shot=1)

        info = dataset.get_info()

        assert info["subset"] is None
        assert info["few_shot"] == 1
        assert info["total_subjects"] == len(MMLUDataset.SUBJECTS)


class TestMMLUDatasetSubjects:
    """Test MMLU subjects list."""

    def test_subjects_list_not_empty(self):
        """Test that subjects list is not empty."""
        assert len(MMLUDataset.SUBJECTS) > 0

    def test_subjects_list_contains_expected_subjects(self):
        """Test that subjects list contains expected subjects."""
        expected_subjects = [
            "abstract_algebra",
            "anatomy",
            "astronomy",
            "college_biology",
            "computer_security",
            "elementary_mathematics",
            "high_school_chemistry",
        ]

        for subject in expected_subjects:
            assert subject in MMLUDataset.SUBJECTS

    def test_subjects_list_length(self):
        """Test that subjects list has expected length."""
        # MMLU has 57 subjects
        assert len(MMLUDataset.SUBJECTS) == 57


class TestMMLUDatasetEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_dataset_handling(self):
        """Test handling of empty dataset."""
        dataset = MMLUDataset(subset="anatomy")

        empty_dataset = []
        samples = dataset._process_subject_data(empty_dataset, "anatomy")

        assert samples == []

    @patch("src.novaeval.datasets.mmlu.load_dataset")
    def test_load_data_with_seed_reproducibility(self, mock_load_dataset):
        """Test that same seed produces same order."""
        mock_dataset = [
            {"question": f"Q{i}", "choices": ["A", "B", "C", "D"], "answer": 0}
            for i in range(10)
        ]
        mock_load_dataset.return_value = mock_dataset

        # Load with same seed twice
        dataset1 = MMLUDataset(subset="anatomy", seed=42)
        samples1 = dataset1.load_data()

        dataset2 = MMLUDataset(subset="anatomy", seed=42)
        samples2 = dataset2.load_data()

        # Should have same order
        assert len(samples1) == len(samples2)
        for s1, s2 in zip(samples1, samples2):
            assert s1["question"] == s2["question"]

    def test_format_question_with_special_characters(self):
        """Test formatting questions with special characters."""
        dataset = MMLUDataset(few_shot=0)

        formatted = dataset._format_question(
            "What is the symbol for alpha?",
            ["alpha", "beta", "gamma", "delta"],
            "greek_letters",
        )

        assert "What is the symbol for alpha?" in formatted
        assert "A. alpha" in formatted

    def test_choices_with_different_lengths(self):
        """Test handling choices with different numbers of options."""
        dataset = MMLUDataset()

        # Test with only 3 choices (less than typical 4)
        mock_dataset = [
            {
                "question": "True or false?",
                "choices": ["True", "False", "Maybe"],
                "answer": 0,
            }
        ]

        samples = dataset._process_subject_data(mock_dataset, "logic")

        assert len(samples) == 1
        assert len(samples[0]["choices"]) == 3
        assert samples[0]["expected"] == "True"
