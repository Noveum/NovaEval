"""
Test utilities for NovaEval unit tests.

This module provides shared test fixtures and mock objects used across
multiple test files to ensure consistency and reduce duplication.
"""

import asyncio

import pytest


class MockLLM:
    """
    Mock LLM class that simulates async model.generate() calls.

    This mock provides realistic responses for RAG scorer prompts by matching
    substrings from actual prompts used in the scorer implementations.

    Usage:
        mock_llm = MockLLM()
        response = await mock_llm.generate("some prompt")

    The mock uses substring matching to identify the type of prompt and return
    appropriate responses. This allows tests to verify that scorers handle
    different types of LLM responses correctly.
    """

    def __init__(self):
        # Comprehensive response mapping based on actual prompts from RAG scorers
        self.responses = {
            # Answer Relevancy Scorer
            "generate 3-5 questions": "1. What is machine learning?\n2. How does ML work?\n3. What are ML applications?",
            "Given the following answer, generate": "1. What is machine learning?\n2. How does ML work?\n3. What are ML applications?",
            # Faithfulness Scorer
            "Extract all factual claims": "1. Machine learning is a subset of AI\n2. ML involves training algorithms\n3. ML makes predictions from data",
            "Extract all factual claims from this answer": "1. Machine learning is a subset of AI\n2. ML involves training algorithms\n3. ML makes predictions from data",
            "Can this claim be verified": "Verification: SUPPORTED\nExplanation: The claim is directly supported by the context.",
            'Respond with either "SUPPORTED"': "Verification: SUPPORTED\nExplanation: The claim is directly supported by the context.",
            # Contextual Precision Scorer
            "Is this context chunk relevant": "Rating: 4\nExplanation: Highly relevant for answering the question.",
            "Rate the relevance on a scale of 1-5": "Rating: 4\nExplanation: Highly relevant for answering the question.",
            # Contextual Recall Scorer
            "Extract the key pieces of information": "1. Machine learning definition\n2. AI subset relationship\n3. Algorithm training process",
            "Extract the key pieces of information from the following expected answer": "1. Machine learning definition\n2. AI subset relationship\n3. Algorithm training process",
            "Is this information present in the context": "Status: PRESENT\nExplanation: The information is clearly stated in the context.",
            'Respond with either "PRESENT"': "Status: PRESENT\nExplanation: The information is clearly stated in the context.",
            # Advanced Generation Scorers - Bias Detection
            "Detect any bias in this answer": "Rating: 2\nExplanation: Minor bias detected in terminology.",
            "Look for:": "Rating: 2\nExplanation: Minor bias detected in terminology.",
            # Advanced Generation Scorers - Factual Accuracy
            "Verify the factual accuracy": "Rating: 4\nExplanation: Most claims are factually accurate.",
            "Check for:": "Rating: 4\nExplanation: Most claims are factually accurate.",
            # Advanced Generation Scorers - Claim Verification
            "Extract all specific claims": "1. Specific claim about ML\n2. Specific claim about AI\n3. Specific claim about algorithms",
            "Extract all specific claims from this answer": "1. Specific claim about ML\n2. Specific claim about AI\n3. Specific claim about algorithms",
            "Can this specific claim be verified": "Rating: 4\nExplanation: The specific claim is verified.",
            # Advanced Generation Scorers - Information Density
            "Evaluate the information density": "Rating: 4\nExplanation: High information density with relevant details.",
            "Rate the information density": "Rating: 4\nExplanation: High information density with relevant details.",
            # Advanced Generation Scorers - Clarity and Coherence
            "Evaluate the clarity and coherence": "Rating: 4\nExplanation: Clear and well-structured response.",
            "Rate the clarity": "Rating: 4\nExplanation: Clear and well-structured response.",
            # Advanced Generation Scorers - Conflict Resolution
            "Evaluate how well this answer handles potential conflicts": "Rating: 4\nExplanation: Effectively resolves conflicts.",
            "handles potential conflicts": "Rating: 4\nExplanation: Effectively resolves conflicts.",
            # Advanced Generation Scorers - Context Prioritization
            "Evaluate how well this answer prioritizes": "Rating: 4\nExplanation: Good prioritization of information.",
            "prioritizes": "Rating: 4\nExplanation: Good prioritization of information.",
            # Advanced Generation Scorers - Citation Quality
            "Evaluate the quality of citations": "Rating: 4\nExplanation: Good citation quality.",
            "quality of citations": "Rating: 4\nExplanation: Good citation quality.",
            # Advanced Generation Scorers - Technical Accuracy
            "Evaluate the technical accuracy": "Rating: 4\nExplanation: Technically accurate information.",
            "technical accuracy": "Rating: 4\nExplanation: Technically accurate information.",
            # Advanced Generation Scorers - Tone Consistency
            "Evaluate the appropriateness and consistency of tone": "Rating: 4\nExplanation: Consistent and appropriate tone.",
            "consistency of tone": "Rating: 4\nExplanation: Consistent and appropriate tone.",
            # Advanced Generation Scorers - Terminology Consistency
            "Evaluate the consistency of terminology": "Rating: 4\nExplanation: Consistent terminology usage.",
            "consistency of terminology": "Rating: 4\nExplanation: Consistent terminology usage.",
            # Advanced Generation Scorers - Context Faithfulness
            "Evaluate if this answer is consistent with this specific context chunk": "Rating: 4\nExplanation: Consistent with context.",
            "consistent with this specific context chunk": "Rating: 4\nExplanation: Consistent with context.",
            # Advanced Generation Scorers - Context Groundedness
            "Evaluate how well this answer is grounded": "Rating: 4\nExplanation: Well grounded in context.",
            "how well this answer is grounded": "Rating: 4\nExplanation: Well grounded in context.",
            # Advanced Generation Scorers - Context Completeness
            "Evaluate if the provided context is complete": "Rating: 4\nExplanation: Context is complete for the question.",
            "if the provided context is complete": "Rating: 4\nExplanation: Context is complete for the question.",
            # Advanced Generation Scorers - Context Consistency
            "Evaluate the overall quality of this RAG-generated answer": "Rating: 4\nExplanation: High quality RAG answer.",
            "overall quality of this RAG-generated answer": "Rating: 4\nExplanation: High quality RAG answer.",
            # Advanced Generation Scorers - Hallucination Detection
            "Detect any hallucinations": "Rating: 2\nExplanation: Minor hallucinations detected.",
            "hallucinations": "Rating: 2\nExplanation: Minor hallucinations detected.",
            # Advanced Generation Scorers - Source Attribution
            "Evaluate the quality of source attribution": "Rating: 4\nExplanation: Good source attribution.",
            "quality of source attribution": "Rating: 4\nExplanation: Good source attribution.",
            # Advanced Generation Scorers - Answer Completeness
            "Evaluate the completeness of this answer": "Rating: 4\nExplanation: Complete answer provided.",
            "completeness of this answer": "Rating: 4\nExplanation: Complete answer provided.",
            # Advanced Generation Scorers - Question Answer Alignment
            "Evaluate how well this answer directly addresses": "Rating: 4\nExplanation: Directly addresses the question.",
            "directly addresses": "Rating: 4\nExplanation: Directly addresses the question.",
            # Advanced Generation Scorers - Cross Context Synthesis
            "Evaluate how well this answer synthesizes information": "Rating: 4\nExplanation: Good synthesis of information.",
            "synthesizes information": "Rating: 4\nExplanation: Good synthesis of information.",
            # G-Eval Scorers
            "Evaluate the helpfulness": "Rating: 4\nExplanation: Very helpful response.",
            "Evaluate the correctness": "Rating: 4\nExplanation: Correct information provided.",
            "Evaluate the coherence": "Rating: 4\nExplanation: Coherent response.",
            "Evaluate the relevance": "Rating: 4\nExplanation: Highly relevant response.",
            # Basic RAG Scorers
            "Evaluate the faithfulness": "Rating: 4\nExplanation: Faithful to context.",
            "Evaluate the answer relevancy": "Rating: 4\nExplanation: Highly relevant answer.",
            "Evaluate the contextual precision": "Rating: 4\nExplanation: High contextual precision.",
            "Evaluate the contextual recall": "Rating: 4\nExplanation: Good contextual recall.",
            # Default fallback for unmatched prompts
            "default": "Rating: 3\nExplanation: Moderate quality response.",
        }

    async def generate(self, prompt: str) -> str:
        """
        Async method to generate responses based on prompt content.

        Args:
            prompt: The input prompt string

        Returns:
            A string response that matches the expected format for the prompt type
        """
        # Find the best matching response based on prompt content
        for key, response in self.responses.items():
            if key.lower() in prompt.lower():
                return response

        # Return default response if no match found
        return self.responses["default"]

    def __call__(self, prompt: str) -> str:
        """
        Synchronous fallback for compatibility with older test patterns.

        This method is deprecated and should be replaced with await model.generate()
        in all test code.
        """
        # Run the async method in a new event loop
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If we're already in an event loop, create a new one
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                result = loop.run_until_complete(self.generate(prompt))
                loop.close()
                return result
            else:
                return loop.run_until_complete(self.generate(prompt))
        except RuntimeError:
            # No event loop available, create one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(self.generate(prompt))
            loop.close()
            return result


# Shared fixtures that can be imported by multiple test files
@pytest.fixture
def mock_llm():
    """Shared fixture providing a MockLLM instance."""
    return MockLLM()


@pytest.fixture
def sample_agent_data():
    """Shared fixture providing sample AgentData for testing."""
    from novaeval.scorers.rag_assessment import AgentData

    return AgentData(
        ground_truth="What is machine learning?",
        agent_response="Machine learning is a subset of artificial intelligence that enables computers to learn from data.",
        retrieved_context="Machine learning is a subset of artificial intelligence. It involves training algorithms on data to make predictions or decisions.",
    )


@pytest.fixture
def sample_agent_data_list():
    """Shared fixture providing a list of sample AgentData for batch testing."""
    from novaeval.scorers.rag_assessment import AgentData

    return [
        AgentData(
            ground_truth="What is machine learning?",
            agent_response="Machine learning is a subset of artificial intelligence that enables computers to learn from data.",
            retrieved_context="Machine learning is a subset of artificial intelligence. It involves training algorithms on data to make predictions or decisions.",
        ),
        AgentData(
            ground_truth="What is deep learning?",
            agent_response="Deep learning is a subset of machine learning that uses neural networks with multiple layers.",
            retrieved_context="Deep learning is a subset of machine learning. It uses artificial neural networks with multiple layers to process data.",
        ),
    ]
