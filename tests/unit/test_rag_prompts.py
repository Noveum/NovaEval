"""
Tests for centralized RAG prompts.
"""

from src.novaeval.scorers.rag_prompts import (
    RAGPrompts,
    get_estimate_total_relevant_prompt,
    get_numerical_chunk_relevance_prompt_0_10,
    get_numerical_chunk_relevance_prompt_1_5,
)


class TestRAGPrompts:
    """Test the centralized RAG prompts class."""

    def test_numerical_chunk_relevance_1_5_prompt(self):
        """Test 1-5 scale chunk relevance prompt formatting."""
        question = "How does neural networks work?"
        chunk = (
            "Neural networks are computational models inspired by biological neurons."
        )

        prompt = RAGPrompts.get_numerical_chunk_relevance_1_5(question, chunk)

        assert "Question: How does neural networks work?" in prompt
        assert "Context chunk: Neural networks are computational models" in prompt
        assert "1 = Not relevant at all" in prompt
        assert "5 = Extremely relevant" in prompt
        assert "Rating: [1-5]" in prompt

    def test_numerical_chunk_relevance_0_10_prompt(self):
        """Test 0-10 scale chunk relevance prompt formatting."""
        question = "What is deep learning?"
        chunk = "Deep learning uses multiple layers of neural networks."

        prompt = RAGPrompts.get_numerical_chunk_relevance_0_10(question, chunk)

        assert "Question: What is deep learning?" in prompt
        assert "Context chunk: Deep learning uses multiple layers" in prompt
        assert "0: Completely irrelevant" in prompt
        assert "10: Highly relevant" in prompt
        assert "Rating: [0-10]" in prompt

    def test_estimate_total_relevant_prompt(self):
        """Test total relevant chunks estimation prompt formatting."""
        query = "What are the benefits of AI?"
        num_retrieved = 5

        prompt = RAGPrompts.get_estimate_total_relevant(query, num_retrieved)

        assert "Query: What are the benefits of AI?" in prompt
        assert "Retrieved Chunks: 5 chunks" in prompt
        assert "estimated_total" in prompt
        assert "JSON" in prompt

    def test_format_prompt_method(self):
        """Test the format_prompt class method."""
        template = "Hello {name}, you are {age} years old."
        result = RAGPrompts.format_prompt(template, name="Alice", age=25)

        assert result == "Hello Alice, you are 25 years old."


class TestBackwardCompatibility:
    """Test backward compatibility functions."""

    def test_get_numerical_chunk_relevance_prompt_1_5(self):
        """Test backward compatibility function for 1-5 scale relevance."""
        question = "What is ML?"
        chunk = "ML stands for machine learning."

        prompt = get_numerical_chunk_relevance_prompt_1_5(question, chunk)

        assert "Question: What is ML?" in prompt
        assert "Context chunk: ML stands for machine learning." in prompt
        assert "1 = Not relevant at all" in prompt

    def test_get_numerical_chunk_relevance_prompt_0_10(self):
        """Test backward compatibility function for 0-10 scale relevance."""
        question = "What is DL?"
        chunk = "DL stands for deep learning."

        prompt = get_numerical_chunk_relevance_prompt_0_10(question, chunk)

        assert "Question: What is DL?" in prompt
        assert "Context chunk: DL stands for deep learning." in prompt
        assert "0: Completely irrelevant" in prompt

    def test_get_estimate_total_relevant_prompt(self):
        """Test backward compatibility function for total relevant estimation."""
        query = "What is NLP?"
        num_retrieved = 3

        prompt = get_estimate_total_relevant_prompt(query, num_retrieved)

        assert "Query: What is NLP?" in prompt
        assert "Retrieved Chunks: 3 chunks" in prompt
        assert "estimated_total" in prompt


class TestPromptConsistency:
    """Test that prompts are consistent and well-formatted."""

    def test_all_prompts_have_required_elements(self):
        """Test that all prompts contain required formatting elements."""
        # Test numerical relevance prompt
        prompt = RAGPrompts.get_numerical_chunk_relevance_1_5("test", "test")
        assert "Rating:" in prompt
        assert "Explanation:" in prompt

        # Test estimation prompt
        prompt = RAGPrompts.get_estimate_total_relevant("test", 1)
        assert "JSON" in prompt
        assert "estimated_total" in prompt

    def test_prompt_formatting_consistency(self):
        """Test that prompt formatting is consistent across different inputs."""
        # Test numerical relevance prompt with different inputs
        prompt1 = RAGPrompts.get_numerical_chunk_relevance_1_5("", "")
        prompt2 = RAGPrompts.get_numerical_chunk_relevance_1_5("test", "test")

        # Both should have the same structure
        assert "Question:" in prompt1
        assert "Question:" in prompt2
        assert "Context chunk:" in prompt1
        assert "Context chunk:" in prompt2

        # Test with special characters
        prompt = RAGPrompts.get_numerical_chunk_relevance_1_5(
            "test? test!", "test\n\ttest"
        )
        assert "Question: test? test!" in prompt
        assert "Context chunk: test\n\ttest" in prompt
