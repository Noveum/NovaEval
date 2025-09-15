"""
Unit tests for RAG prompts module.
"""

import pytest

from novaeval.scorers.rag_prompts import RAGPrompts


class TestRAGPrompts:
    """Test cases for RAGPrompts class."""

    def test_format_prompt_basic(self):
        """Test basic prompt formatting."""
        template = "Question: {question}\nAnswer: {answer}"
        result = RAGPrompts.format_prompt(
            template, question="What is ML?", answer="Machine Learning"
        )

        expected = "Question: What is ML?\nAnswer: Machine Learning"
        assert result == expected

    def test_get_numerical_chunk_relevance_1_5(self):
        """Test 1-5 scale chunk relevance prompt generation."""
        result = RAGPrompts.get_numerical_chunk_relevance_1_5(
            question="What is machine learning?",
            chunk="Machine learning is a subset of artificial intelligence.",
        )

        assert "What is machine learning?" in result
        assert "Machine learning is a subset of artificial intelligence." in result
        assert "1 = Not relevant at all" in result
        assert "5 = Extremely relevant" in result

    def test_get_bias_detection_evaluation(self):
        """Test bias detection evaluation prompt generation."""
        result = RAGPrompts.get_bias_detection_evaluation(
            input_text="What careers are good for young people?",
            output_text="Men should consider engineering, women should consider nursing.",
        )

        assert "What careers are good for young people?" in result
        assert (
            "Men should consider engineering, women should consider nursing." in result
        )
        assert "Gender Bias" in result
        assert "bias_score" in result

    def test_get_factual_accuracy_evaluation(self):
        """Test factual accuracy evaluation prompt generation."""
        result = RAGPrompts.get_factual_accuracy_evaluation(
            context="The Apollo 11 mission landed on the Moon on July 20, 1969.",
            output_text="Apollo 11 landed on the Moon on July 20, 1969.",
        )

        assert "The Apollo 11 mission landed on the Moon on July 20, 1969." in result
        assert "Apollo 11 landed on the Moon on July 20, 1969." in result
        assert "accuracy_score" in result

    def test_prompt_templates_contain_required_elements(self):
        """Test that all prompt templates contain required structural elements."""
        bias_prompt = RAGPrompts.BIAS_DETECTION_EVALUATION
        assert "## Task" in bias_prompt
        assert "## Input" in bias_prompt
        assert "## Evaluation Criteria" in bias_prompt
        assert "## Output Format" in bias_prompt
        assert "```json" in bias_prompt

    def test_prompt_templates_have_examples(self):
        """Test that prompt templates include helpful examples."""
        bias_prompt = RAGPrompts.BIAS_DETECTION_EVALUATION
        assert "**Example 1" in bias_prompt
        assert "**Example 2" in bias_prompt

    def test_prompt_templates_rating_scales(self):
        """Test that rating scales are properly defined in templates."""
        bias_prompt = RAGPrompts.BIAS_DETECTION_EVALUATION
        assert "**1 (No Bias)**" in bias_prompt
        assert "**5 (Major Bias)**" in bias_prompt

    def test_prompt_templates_handle_edge_cases(self):
        """Test that prompt templates handle edge cases properly."""
        result = RAGPrompts.get_bias_detection_evaluation("", "")
        assert result is not None
        assert len(result) > 0

    def test_prompt_templates_consistency(self):
        """Test that similar prompt templates have consistent structure."""
        evaluation_templates = [
            RAGPrompts.BIAS_DETECTION_EVALUATION,
            RAGPrompts.FACTUAL_ACCURACY_EVALUATION,
            RAGPrompts.CLAIM_EXTRACTION_EVALUATION,
            RAGPrompts.CLAIM_VERIFICATION_EVALUATION,
        ]

        for template in evaluation_templates:
            assert "## Task" in template
            assert "## Input" in template
            assert "## Evaluation Criteria" in template
            assert "## Output Format" in template
            assert "```json" in template


if __name__ == "__main__":
    pytest.main([__file__])
