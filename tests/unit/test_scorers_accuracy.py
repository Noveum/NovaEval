"""
Unit tests for accuracy scorers.
"""

from novaeval.scorers.accuracy import AccuracyScorer, ExactMatchScorer, F1Scorer


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
