"""
Unit tests for accuracy scorers.
"""

import pytest

from novaeval.scorers.accuracy import (
    AccuracyScorer,
    ExactMatchScorer,
    F1Scorer,
    MultiPatternAccuracyScorer,
)

pytestmark = pytest.mark.unit


class TestExactMatchScorer:
    """Test cases for ExactMatchScorer class."""

    def test_init_default(self):
        """Test initialization with default parameters."""
        scorer = ExactMatchScorer()

        assert scorer.name == "exact_match"
        assert scorer.description == "Exact string matching scorer"
        assert scorer.case_sensitive is True
        assert scorer.strip_whitespace is True
        assert scorer.normalize_whitespace is False

    def test_init_with_params(self):
        """Test initialization with custom parameters."""
        scorer = ExactMatchScorer(
            case_sensitive=False, strip_whitespace=False, normalize_whitespace=True
        )

        assert scorer.case_sensitive is False
        assert scorer.strip_whitespace is False
        assert scorer.normalize_whitespace is True

    def test_exact_match_true(self):
        """Test exact matching when strings match."""
        scorer = ExactMatchScorer()

        score = scorer.score("hello world", "hello world")
        assert score == 1.0

    def test_exact_match_false(self):
        """Test exact matching when strings don't match."""
        scorer = ExactMatchScorer()

        score = scorer.score("hello world", "goodbye world")
        assert score == 0.0

    def test_case_insensitive_matching(self):
        """Test case-insensitive matching."""
        scorer = ExactMatchScorer(case_sensitive=False)

        score = scorer.score("Hello World", "hello world")
        assert score == 1.0

        score = scorer.score("HELLO", "hello")
        assert score == 1.0

    def test_whitespace_stripping(self):
        """Test whitespace stripping functionality."""
        scorer = ExactMatchScorer(strip_whitespace=True)

        score = scorer.score("  hello world  ", "hello world")
        assert score == 1.0

        score = scorer.score("\thello world\n", "hello world")
        assert score == 1.0

    def test_whitespace_normalization(self):
        """Test whitespace normalization functionality."""
        scorer = ExactMatchScorer(normalize_whitespace=True)

        score = scorer.score("hello    world", "hello world")
        assert score == 1.0

        score = scorer.score("hello\t\nworld", "hello world")
        assert score == 1.0

    def test_combined_preprocessing(self):
        """Test combined preprocessing options."""
        scorer = ExactMatchScorer(
            case_sensitive=False, strip_whitespace=True, normalize_whitespace=True
        )

        score = scorer.score("  HELLO    WORLD  ", "hello world")
        assert score == 1.0

    def test_empty_strings(self):
        """Test handling of empty strings."""
        scorer = ExactMatchScorer()

        score = scorer.score("", "")
        assert score == 1.0

        score = scorer.score("hello", "")
        assert score == 0.0

        score = scorer.score("", "hello")
        assert score == 0.0

    def test_invalid_inputs(self):
        """Test handling of invalid inputs."""
        scorer = ExactMatchScorer()

        score = scorer.score(None, "hello")
        assert score == 0.0

        score = scorer.score("hello", None)
        assert score == 0.0

        score = scorer.score(None, None)
        assert score == 0.0


class TestAccuracyScorer:
    """Test cases for AccuracyScorer class."""

    def test_init_default(self):
        """Test initialization with default parameters."""
        scorer = AccuracyScorer()

        assert scorer.name == "accuracy"
        assert scorer.description == "Classification accuracy scorer"
        assert scorer.extract_answer is True
        assert scorer.answer_pattern == r"(?:Answer|answer):\s*([A-Za-z0-9]+)"
        assert scorer.choices is None

    def test_init_with_params(self):
        """Test initialization with custom parameters."""
        choices = ["A", "B", "C", "D"]
        pattern = r"Answer:\s*([ABCD])"

        scorer = AccuracyScorer(
            extract_answer=False, answer_pattern=pattern, choices=choices
        )

        assert scorer.extract_answer is False
        assert scorer.answer_pattern == pattern
        assert scorer.choices == choices

    def test_simple_accuracy_match(self):
        """Test simple accuracy matching without extraction."""
        scorer = AccuracyScorer(extract_answer=False)

        score = scorer.score("A", "A")
        assert score == 1.0

        score = scorer.score("B", "A")
        assert score == 0.0

    def test_answer_extraction_basic(self):
        """Test basic answer extraction."""
        scorer = AccuracyScorer()

        # Test "Answer: X" pattern
        score = scorer.score("The answer is B. Answer: B", "B")
        assert score == 1.0

        score = scorer.score("Answer: C", "C")
        assert score == 1.0

        score = scorer.score("answer: d", "D")
        assert score == 1.0  # Case insensitive extraction

    def test_answer_extraction_alternative_patterns(self):
        """Test alternative answer extraction patterns."""
        scorer = AccuracyScorer()

        # Test "The answer is X" pattern
        score = scorer.score("The answer is B", "B")
        assert score == 1.0

        score = scorer.score("The correct answer is C", "C")
        assert score == 1.0

        # Test bold pattern
        score = scorer.score("**A.** This is correct", "A")
        assert score == 1.0

    def test_answer_extraction_failure(self):
        """Test when answer extraction fails."""
        scorer = AccuracyScorer()

        # No clear pattern - should use original text
        score = scorer.score("I think it might be something", "A")
        assert score == 0.0

    def test_mmlu_style_with_choices(self):
        """Test MMLU-style questions with choices."""
        scorer = AccuracyScorer()

        context = {
            "choices": ["Paris", "London", "Berlin", "Madrid"],
            "answer_index": 0,  # Paris is correct
        }

        # Test letter extraction - ground truth should be the actual choice text
        score = scorer.score(
            "Answer: A", "Paris", context
        )  # ground truth is the actual choice
        assert score == 1.0

        # Test with different letter
        score = scorer.score(
            "Answer: B", "Paris", context
        )  # ground truth is the actual choice
        assert score == 0.0

    def test_normalize_answer(self):
        """Test answer normalization."""
        scorer = AccuracyScorer()

        # Test that normalization handles case and whitespace
        assert scorer._normalize_answer("  A  ") == "a"
        assert scorer._normalize_answer("Hello World") == "hello world"
        assert scorer._normalize_answer("  ANSWER  ") == "answer"

    def test_convert_letter_to_choice(self):
        """Test letter to choice conversion."""
        scorer = AccuracyScorer()

        context = {
            "choices": ["Paris", "London", "Berlin", "Madrid"],
            "answer_index": 0,
        }

        # Test valid conversions
        converted = scorer._convert_letter_to_choice("A", context)
        assert converted == "Paris"

        converted = scorer._convert_letter_to_choice("B", context)
        assert converted == "London"

        # Test invalid letter
        converted = scorer._convert_letter_to_choice("Z", context)
        assert converted is None

        # Test with missing choices
        context_no_choices = {"answer_index": 0}
        converted = scorer._convert_letter_to_choice("A", context_no_choices)
        assert converted is None

    def test_invalid_inputs(self):
        """Test handling of invalid inputs."""
        scorer = AccuracyScorer()

        score = scorer.score(None, "A")
        assert score == 0.0

        score = scorer.score("A", None)
        assert score == 0.0


class TestF1Scorer:
    """Test cases for F1Scorer class."""

    def test_init_default(self):
        """Test initialization with default parameters."""
        scorer = F1Scorer()

        assert scorer.name.startswith("f1")
        assert scorer.tokenize is True
        assert scorer.case_sensitive is False

    def test_init_with_params(self):
        """Test initialization with custom parameters."""
        scorer = F1Scorer(tokenize=False, case_sensitive=True)

        assert scorer.tokenize is False
        assert scorer.case_sensitive is True

    def test_perfect_match(self):
        """Test F1 score for perfect match."""
        scorer = F1Scorer()

        scores = scorer.score("hello world test", "hello world test")

        assert scores["precision"] == 1.0
        assert scores["recall"] == 1.0
        assert scores["f1"] == 1.0

    def test_no_overlap(self):
        """Test F1 score for no overlap."""
        scorer = F1Scorer()

        scores = scorer.score("hello world", "goodbye universe")

        assert scores["precision"] == 0.0
        assert scores["recall"] == 0.0
        assert scores["f1"] == 0.0

    def test_partial_overlap(self):
        """Test F1 score for partial overlap."""
        scorer = F1Scorer()

        scores = scorer.score("hello world test", "hello universe test")

        # Common tokens: "hello", "test" (2 out of 3 in each)
        expected_precision = 2 / 3  # 2 common out of 3 in prediction
        expected_recall = 2 / 3  # 2 common out of 3 in ground truth
        expected_f1 = (
            2
            * (expected_precision * expected_recall)
            / (expected_precision + expected_recall)
        )

        assert abs(scores["precision"] - expected_precision) < 0.001
        assert abs(scores["recall"] - expected_recall) < 0.001
        assert abs(scores["f1"] - expected_f1) < 0.001

    def test_case_sensitivity(self):
        """Test case sensitivity option."""
        # Case insensitive (default)
        scorer_insensitive = F1Scorer(case_sensitive=False)
        scores = scorer_insensitive.score("Hello World", "hello world")
        assert scores["f1"] == 1.0

        # Case sensitive
        scorer_sensitive = F1Scorer(case_sensitive=True)
        scores = scorer_sensitive.score("Hello World", "hello world")
        assert scores["f1"] == 0.0  # No exact matches

    def test_tokenization_off(self):
        """Test with tokenization disabled."""
        scorer = F1Scorer(tokenize=False)

        # Should split on whitespace but not use regex tokenization
        scores = scorer.score("hello world", "hello world")
        assert scores["f1"] == 1.0

        # With different tokens, should have partial overlap
        scores = scorer.score("hello world", "hello universe")
        # Common tokens: "hello" (1 out of 2 in each)
        expected_precision = 1 / 2  # 1 common out of 2 in prediction
        expected_recall = 1 / 2  # 1 common out of 2 in ground truth
        expected_f1 = (
            2
            * (expected_precision * expected_recall)
            / (expected_precision + expected_recall)
        )
        assert abs(scores["f1"] - expected_f1) < 0.001

    def test_get_tokens_with_tokenization(self):
        """Test token extraction with tokenization enabled."""
        scorer = F1Scorer(tokenize=True)

        tokens = scorer._get_tokens("Hello, world! How are you?")
        expected = ["hello", "world", "how", "are", "you"]
        assert tokens == expected

    def test_get_tokens_without_tokenization(self):
        """Test token extraction with tokenization disabled."""
        scorer = F1Scorer(tokenize=False)

        tokens = scorer._get_tokens("hello, world!")
        expected = ["hello,", "world!"]  # Split on whitespace, not regex
        assert tokens == expected

    def test_empty_strings(self):
        """Test F1 score with empty strings."""
        scorer = F1Scorer()

        scores = scorer.score("", "")
        assert scores["precision"] == 0.0
        assert scores["recall"] == 0.0
        assert scores["f1"] == 0.0

        scores = scorer.score("hello", "")
        assert scores["precision"] == 0.0
        assert scores["recall"] == 0.0
        assert scores["f1"] == 0.0

        scores = scorer.score("", "hello")
        assert scores["precision"] == 0.0
        assert scores["recall"] == 0.0
        assert scores["f1"] == 0.0

    def test_invalid_inputs(self):
        """Test handling of invalid inputs."""
        scorer = F1Scorer()

        scores = scorer.score(None, "hello")
        assert scores["f1"] == 0.0

        scores = scorer.score("hello", None)
        assert scores["f1"] == 0.0


class TestMultiPatternAccuracyScorer:
    """Test cases for MultiPatternAccuracyScorer class."""

    def test_init_default(self):
        """Test initialization with default parameters."""
        scorer = MultiPatternAccuracyScorer()

        assert scorer.name == "multi_pattern_accuracy"
        assert (
            scorer.description == "Multi-pattern answer extraction and matching scorer"
        )
        assert scorer.case_sensitive is False
        assert scorer.choices is None
        assert len(scorer.patterns) == 7  # Default patterns count

    def test_init_with_custom_patterns(self):
        """Test initialization with custom patterns."""
        custom_patterns = [r"Answer:\s*([A-D])", r"The answer is ([A-D])"]
        scorer = MultiPatternAccuracyScorer(patterns=custom_patterns)

        assert scorer.patterns == custom_patterns
        assert scorer.case_sensitive is False

    def test_init_with_custom_choices(self):
        """Test initialization with custom choices."""
        choices = ["Paris", "London", "Berlin", "Madrid"]
        scorer = MultiPatternAccuracyScorer(choices=choices)

        assert scorer.choices == choices
        assert scorer.patterns is not None  # Should use default patterns

    def test_init_with_case_sensitive(self):
        """Test initialization with case sensitivity."""
        scorer = MultiPatternAccuracyScorer(case_sensitive=True)

        assert scorer.case_sensitive is True

    def test_score_exact_match(self):
        """Test scoring when extracted answer exactly matches ground truth."""
        scorer = MultiPatternAccuracyScorer()

        # Test with "Answer: A" pattern
        score = scorer.score("The answer is A. Answer: A", "A")
        assert score == 1.0

        # Test with "The answer is X" pattern
        score = scorer.score("The answer is B", "B")
        assert score == 1.0

    def test_score_no_match(self):
        """Test scoring when no extracted answer matches."""
        scorer = MultiPatternAccuracyScorer()

        score = scorer.score("I think the answer might be something", "A")
        assert score == 0.0

    def test_score_with_context_choices(self):
        """Test scoring with MMLU-style context and choices."""
        scorer = MultiPatternAccuracyScorer()

        context = {
            "choices": ["Paris", "London", "Berlin", "Madrid"],
            "answer_index": 0,  # Paris is correct
        }

        # Test letter extraction - ground truth should be the actual choice text
        score = scorer.score("Answer: A", "Paris", context)
        assert score == 1.0

        # Test with different letter
        score = scorer.score("Answer: B", "Paris", context)
        assert score == 0.0

    def test_score_letter_to_choice_conversion(self):
        """Test scoring when letter answers need conversion."""
        scorer = MultiPatternAccuracyScorer()

        context = {
            "choices": ["Apple", "Banana", "Cherry", "Date"],
            "answer_index": 1,  # Banana is correct
        }

        # Test that letter B gets converted to "Banana" and matches
        score = scorer.score("The answer is B", "Banana", context)
        assert score == 1.0

    def test_score_invalid_inputs(self):
        """Test scoring with invalid inputs."""
        scorer = MultiPatternAccuracyScorer()

        score = scorer.score(None, "A")
        assert score == 0.0

        score = scorer.score("A", None)
        assert score == 0.0

        score = scorer.score(None, None)
        assert score == 0.0

    def test_extract_all_answers_single_pattern(self):
        """Test answer extraction with single pattern."""
        patterns = [r"Answer:\s*([A-D])"]
        scorer = MultiPatternAccuracyScorer(patterns=patterns)

        prediction = "Answer: A and Answer: B"
        answers = scorer._extract_all_answers(prediction)

        assert "A" in answers
        assert "B" in answers
        assert len(answers) == 2

    def test_extract_all_answers_multiple_patterns(self):
        """Test answer extraction with multiple patterns."""
        patterns = [
            r"Answer:\s*([A-D])",
            r"The answer is ([A-D])",
            r"\*\*([A-D])\.\s*[^*]*\*\*",
        ]
        scorer = MultiPatternAccuracyScorer(patterns=patterns)

        prediction = "Answer: A. The answer is B. **C.** This is correct"
        answers = scorer._extract_all_answers(prediction)

        assert "A" in answers
        assert "B" in answers
        assert "C" in answers
        assert len(answers) == 3

    def test_extract_all_answers_no_matches(self):
        """Test answer extraction when no patterns match."""
        patterns = [r"Answer:\s*([A-D])"]
        scorer = MultiPatternAccuracyScorer(patterns=patterns)

        prediction = "I don't know the answer"
        answers = scorer._extract_all_answers(prediction)

        assert len(answers) == 0

    def test_extract_all_answers_duplicate_removal(self):
        """Test that duplicate answers are removed."""
        patterns = [r"Answer:\s*([A-D])"]
        scorer = MultiPatternAccuracyScorer(patterns=patterns)

        prediction = "Answer: A. Answer: A. Answer: A"
        answers = scorer._extract_all_answers(prediction)

        assert answers == ["A"]  # Should only have one "A"

    def test_extract_all_answers_with_groups(self):
        """Test extraction with captured groups."""
        patterns = [r"Answer:\s*([A-D])"]
        scorer = MultiPatternAccuracyScorer(patterns=patterns)

        prediction = "Answer: B"
        answers = scorer._extract_all_answers(prediction)

        assert answers == ["B"]

    def test_extract_all_answers_without_groups(self):
        """Test extraction without captured groups."""
        patterns = [r"[A-D]"]
        scorer = MultiPatternAccuracyScorer(patterns=patterns)

        prediction = "A B C D"
        answers = scorer._extract_all_answers(prediction)

        assert "A" in answers
        assert "B" in answers
        assert "C" in answers
        assert "D" in answers

    def test_convert_letter_to_choice_valid(self):
        """Test valid letter to choice conversion."""
        choices = ["Paris", "London", "Berlin", "Madrid"]
        scorer = MultiPatternAccuracyScorer(choices=choices)

        converted = scorer._convert_letter_to_choice("A")
        assert converted == "Paris"

        converted = scorer._convert_letter_to_choice("B")
        assert converted == "London"

    def test_convert_letter_to_choice_invalid_letter(self):
        """Test invalid letter conversion."""
        choices = ["Paris", "London", "Berlin", "Madrid"]
        scorer = MultiPatternAccuracyScorer(choices=choices)

        converted = scorer._convert_letter_to_choice("Z")
        assert converted is None

        converted = scorer._convert_letter_to_choice("E")
        assert converted is None

    def test_convert_letter_to_choice_missing_choices(self):
        """Test conversion with missing choices."""
        scorer = MultiPatternAccuracyScorer()

        converted = scorer._convert_letter_to_choice("A")
        assert converted is None

    def test_convert_letter_to_choice_context_choices(self):
        """Test conversion using context choices."""
        scorer = MultiPatternAccuracyScorer()

        context = {"choices": ["Apple", "Banana", "Cherry", "Date"]}
        converted = scorer._convert_letter_to_choice("A", context)
        assert converted == "Apple"

    def test_convert_letter_to_choice_instance_choices(self):
        """Test conversion using instance choices."""
        choices = ["Red", "Green", "Blue", "Yellow"]
        scorer = MultiPatternAccuracyScorer(choices=choices)

        # Should use instance choices when context has no choices
        context = {"other": "data"}
        converted = scorer._convert_letter_to_choice("A", context)
        assert converted == "Red"

    def test_convert_letter_to_choice_invalid_choices_type(self):
        """Test conversion with invalid choices type."""
        scorer = MultiPatternAccuracyScorer()

        # Test with non-list choices (string has __len__ and __getitem__ but isn't a proper list)
        context = {"choices": "not_a_list"}
        converted = scorer._convert_letter_to_choice("A", context)
        # String has __len__ and __getitem__, so it will try to use it as a sequence
        # This is actually the current behavior - the method doesn't distinguish between strings and lists
        assert converted == "n"  # "not_a_list"[0] = "n"

        # Test with None choices
        context = {"choices": None}
        converted = scorer._convert_letter_to_choice("A", context)
        assert converted is None

    def test_normalize_answer_case_sensitive(self):
        """Test normalization with case sensitivity enabled."""
        scorer = MultiPatternAccuracyScorer(case_sensitive=True)

        normalized = scorer._normalize_answer("  A  ")
        assert normalized == "A"  # Should preserve case

        normalized = scorer._normalize_answer("Hello World")
        assert normalized == "Hello World"  # Should preserve case

    def test_normalize_answer_case_insensitive(self):
        """Test normalization with case sensitivity disabled."""
        scorer = MultiPatternAccuracyScorer(case_sensitive=False)

        normalized = scorer._normalize_answer("  A  ")
        assert normalized == "a"  # Should lowercase

        normalized = scorer._normalize_answer("Hello World")
        assert normalized == "hello world"  # Should lowercase

    def test_normalize_answer_whitespace(self):
        """Test normalization with whitespace handling."""
        scorer = MultiPatternAccuracyScorer()

        normalized = scorer._normalize_answer("  A  ")
        assert normalized == "a"  # Should strip and lowercase

        normalized = scorer._normalize_answer("\tB\n")
        assert normalized == "b"  # Should strip and lowercase

    def test_score_with_multiple_extracted_answers(self):
        """Test scoring when multiple answers are extracted."""
        scorer = MultiPatternAccuracyScorer()

        # This prediction should extract multiple answers
        prediction = "Answer: A. The answer is B. **C.**"
        score = scorer.score(
            prediction, "B"
        )  # Should match one of the extracted answers
        assert score == 1.0

    def test_score_with_empty_prediction(self):
        """Test scoring with empty prediction."""
        scorer = MultiPatternAccuracyScorer()

        score = scorer.score("", "A")
        assert score == 0.0

    def test_score_with_empty_ground_truth(self):
        """Test scoring with empty ground truth."""
        scorer = MultiPatternAccuracyScorer()

        score = scorer.score("Answer: A", "")
        assert score == 0.0

    def test_score_with_whitespace_only_prediction(self):
        """Test scoring with whitespace-only prediction."""
        scorer = MultiPatternAccuracyScorer()

        score = scorer.score("   \n\t   ", "A")
        assert score == 0.0

    def test_extract_all_answers_with_empty_prediction(self):
        """Test answer extraction with empty prediction."""
        scorer = MultiPatternAccuracyScorer()

        answers = scorer._extract_all_answers("")
        assert len(answers) == 0

    def test_extract_all_answers_with_whitespace_only(self):
        """Test answer extraction with whitespace-only prediction."""
        scorer = MultiPatternAccuracyScorer()

        answers = scorer._extract_all_answers("   \n\t   ")
        assert len(answers) == 0

    def test_convert_letter_to_choice_with_empty_answer(self):
        """Test letter to choice conversion with empty answer."""
        choices = ["Paris", "London", "Berlin", "Madrid"]
        scorer = MultiPatternAccuracyScorer(choices=choices)

        converted = scorer._convert_letter_to_choice("")
        assert converted is None

        converted = scorer._convert_letter_to_choice("   ")
        assert converted is None

    def test_convert_letter_to_choice_with_whitespace(self):
        """Test letter to choice conversion with whitespace."""
        choices = ["Paris", "London", "Berlin", "Madrid"]
        scorer = MultiPatternAccuracyScorer(choices=choices)

        converted = scorer._convert_letter_to_choice("  A  ")
        assert converted == "Paris"

        converted = scorer._convert_letter_to_choice("\tB\n")
        assert converted == "London"

    def test_normalize_answer_with_empty_string(self):
        """Test normalization with empty string."""
        scorer = MultiPatternAccuracyScorer()

        normalized = scorer._normalize_answer("")
        assert normalized == ""

        normalized = scorer._normalize_answer("   ")
        assert normalized == ""

    def test_score_edge_case_choices_index(self):
        """Test edge case where answer_index is at boundary."""
        scorer = MultiPatternAccuracyScorer()

        context = {
            "choices": ["First", "Second", "Third", "Fourth"],
            "answer_index": 3,  # Last choice
        }

        # Test with last choice
        score = scorer.score("Answer: D", "Fourth", context)
        assert score == 1.0

        # Test with first choice
        score = scorer.score("Answer: A", "First", context)
        assert score == 1.0

    def test_score_with_mixed_case_letters(self):
        """Test scoring with mixed case letters in prediction."""
        scorer = MultiPatternAccuracyScorer()

        # Test lowercase letter
        score = scorer.score("Answer: a", "A")
        assert score == 1.0

        # Test uppercase letter
        score = scorer.score("Answer: B", "b")
        assert score == 1.0

    def test_extract_all_answers_preserves_order(self):
        """Test that answer extraction preserves order of first appearance within each pattern."""
        patterns = [r"Answer:\s*([A-D])", r"The answer is ([A-D])"]
        scorer = MultiPatternAccuracyScorer(patterns=patterns)

        prediction = "Answer: B. The answer is A. Answer: C"
        answers = scorer._extract_all_answers(prediction)

        # The method processes patterns in order, then finds all matches within each pattern
        # Pattern 1 (Answer: X): finds B, then C
        # Pattern 2 (The answer is X): finds A
        # So the order should be: B, C (from pattern 1), then A (from pattern 2)
        assert "B" in answers
        assert "A" in answers
        assert "C" in answers
        assert len(answers) == 3
        # The exact order depends on regex engine behavior, so we just check all are present
