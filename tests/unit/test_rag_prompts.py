"""
Tests for centralized RAG prompts.
"""

from src.novaeval.scorers.rag_prompts import RAGPrompts


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

    def test_bias_detection_evaluation_prompt(self):
        """Test bias detection evaluation prompt formatting."""
        input_text = "What are good careers?"
        output_text = "Men should consider engineering, women should consider nursing."

        prompt = RAGPrompts.get_bias_detection_evaluation(
            input_text=input_text, output_text=output_text
        )

        assert "**Question:** What are good careers?" in prompt
        assert (
            "**Answer:** Men should consider engineering, women should consider nursing."
            in prompt
        )
        assert "Bias Detection Evaluation" in prompt
        assert "Gender Bias" in prompt

    def test_factual_accuracy_evaluation_prompt(self):
        """Test factual accuracy evaluation prompt formatting."""
        context = "The Apollo 11 mission landed on the Moon on July 20, 1969."
        output_text = "Apollo 11 landed on Mars in 1970."

        prompt = RAGPrompts.get_factual_accuracy_evaluation(
            context=context, output_text=output_text
        )

        assert (
            "**Context:** The Apollo 11 mission landed on the Moon on July 20, 1969."
            in prompt
        )
        assert "**Answer:** Apollo 11 landed on Mars in 1970." in prompt
        assert "Factual Accuracy Evaluation" in prompt

    def test_hallucination_detection_evaluation_prompt(self):
        """Test hallucination detection evaluation prompt formatting."""
        context = "The Earth is round."
        output_text = "The Earth is flat and the Moon is made of cheese."

        prompt = RAGPrompts.get_hallucination_detection_evaluation(
            context=context, output_text=output_text
        )

        assert "**Context:** The Earth is round." in prompt
        assert "**Answer:** The Earth is flat and the Moon is made of cheese." in prompt
        assert "Hallucination Detection Evaluation" in prompt

    def test_context_faithfulness_evaluation_prompt(self):
        """Test context faithfulness evaluation prompt formatting."""
        context = "Python is a programming language."
        output_text = "Python is a snake species."

        prompt = RAGPrompts.get_context_faithfulness_evaluation(
            context=context, output_text=output_text
        )

        assert "**Context:** Python is a programming language." in prompt
        assert "**Answer:** Python is a snake species." in prompt
        assert "Context Faithfulness Evaluation" in prompt

    def test_question_answer_alignment_evaluation_prompt(self):
        """Test question-answer alignment evaluation prompt formatting."""
        input_text = "What is the capital of France?"
        output_text = "The capital of France is Paris."

        prompt = RAGPrompts.get_question_answer_alignment_evaluation(
            input_text=input_text, output_text=output_text
        )

        assert "**Question:** What is the capital of France?" in prompt
        assert "**Answer:** The capital of France is Paris." in prompt
        assert "Question-Answer Alignment Evaluation" in prompt
