import pytest
import asyncio
from unittest.mock import Mock, AsyncMock
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from novaeval.scorers.rag import (
    AnswerRelevancyScorer,
    FaithfulnessScorer,
    ContextualPrecisionScorer,
    ContextualRecallScorer,
    RAGASScorer
)
from novaeval.scorers.base import ScoreResult

# Mock LLM for testing
class MockLLM:
    def __init__(self):
        self.total_requests = 0
    
    async def generate(self, prompt: str) -> str:
        self.total_requests += 1
        if "generate questions" in prompt.lower():
            return "1. What is the capital of France?\n2. Where is France located?\n3. What is Paris known for?"
        elif "extract claims" in prompt.lower():
            return "1. The capital is Paris\n2. France is a European country\n3. Paris is known for culture"
        elif "extract key facts" in prompt.lower():
            return "1. Paris is the capital of France\n2. France is in Europe\n3. Paris has the Eiffel Tower"
        elif "rate relevance" in prompt.lower():
            return "Rating: 4\nExplanation: This context is highly relevant."
        elif "verify" in prompt.lower():
            return "Verification: SUPPORTED\nExplanation: This claim is supported by the context."
        else:
            return "Mock response for testing."

@pytest.mark.asyncio
async def test_answer_relevancy_scorer():
    """Test AnswerRelevancyScorer."""
    mock_llm = MockLLM()
    scorer = AnswerRelevancyScorer(model=mock_llm, threshold=0.7)
    
    result = await scorer.evaluate(
        input_text="What is the capital of France?",
        output_text="Paris is the capital of France.",
        context="Paris is the capital of France."
    )
    
    assert isinstance(result, ScoreResult)
    assert 0 <= result.score <= 1
    assert isinstance(result.passed, bool)
    assert isinstance(result.reasoning, str)
    assert isinstance(result.metadata, dict)
    print(f"Answer Relevancy: {result.score:.3f} (Passed: {result.passed})")

@pytest.mark.asyncio
async def test_faithfulness_scorer():
    """Test FaithfulnessScorer."""
    mock_llm = MockLLM()
    scorer = FaithfulnessScorer(model=mock_llm, threshold=0.8)
    
    result = await scorer.evaluate(
        input_text="What is the capital of France?",
        output_text="Paris is the capital of France.",
        context="Paris is the capital of France. France is in Europe."
    )
    
    assert isinstance(result, ScoreResult)
    assert 0 <= result.score <= 1
    assert isinstance(result.passed, bool)
    assert isinstance(result.reasoning, str)
    assert isinstance(result.metadata, dict)
    print(f"Faithfulness: {result.score:.3f} (Passed: {result.passed})")

@pytest.mark.asyncio
async def test_contextual_precision_scorer():
    """Test ContextualPrecisionScorer."""
    mock_llm = MockLLM()
    scorer = ContextualPrecisionScorer(model=mock_llm, threshold=0.7)
    
    result = await scorer.evaluate(
        input_text="What is the capital of France?",
        output_text="Paris is the capital of France.",
        context="Paris is the capital of France.\n\nFrance is in Europe.\n\nParis has the Eiffel Tower."
    )
    
    assert isinstance(result, ScoreResult)
    assert 0 <= result.score <= 1
    assert isinstance(result.passed, bool)
    assert isinstance(result.reasoning, str)
    assert isinstance(result.metadata, dict)
    print(f"Contextual Precision: {result.score:.3f} (Passed: {result.passed})")

@pytest.mark.asyncio
async def test_contextual_recall_scorer():
    """Test ContextualRecallScorer."""
    mock_llm = MockLLM()
    scorer = ContextualRecallScorer(model=mock_llm, threshold=0.7)
    
    result = await scorer.evaluate(
        input_text="What is the capital of France?",
        output_text="Paris is the capital of France.",
        expected_output="Paris is the capital of France and is known for the Eiffel Tower.",
        context="Paris is the capital of France.\n\nFrance is in Europe.\n\nParis has the Eiffel Tower."
    )
    
    assert isinstance(result, ScoreResult)
    assert 0 <= result.score <= 1
    assert isinstance(result.passed, bool)
    assert isinstance(result.reasoning, str)
    assert isinstance(result.metadata, dict)
    print(f"Contextual Recall: {result.score:.3f} (Passed: {result.passed})")

@pytest.mark.asyncio
async def test_ragas_scorer():
    """Test RAGASScorer (composite scorer)."""
    mock_llm = MockLLM()
    scorer = RAGASScorer(model=mock_llm, threshold=0.7)
    
    result = await scorer.evaluate(
        input_text="What is the capital of France?",
        output_text="Paris is the capital of France.",
        expected_output="Paris is the capital of France and is known for the Eiffel Tower.",
        context="Paris is the capital of France.\n\nFrance is in Europe.\n\nParis has the Eiffel Tower."
    )
    
    assert isinstance(result, ScoreResult)
    assert 0 <= result.score <= 1
    assert isinstance(result.passed, bool)
    assert isinstance(result.reasoning, str)
    assert isinstance(result.metadata, dict)
    print(f"RAGAS Score: {result.score:.3f} (Passed: {result.passed})")

# @pytest.mark.asyncio
# async def test_sync_score_methods():
#     """Test synchronous score methods."""
#     mock_llm = MockLLM()
#     
#     # Test AnswerRelevancyScorer sync method
#     scorer = AnswerRelevancyScorer(model=mock_llm, threshold=0.7)
#     score = scorer.score(
#         prediction="Paris is the capital of France.",
#         ground_truth="What is the capital of France?",
#         context={"context": "Paris is the capital of France."}
#     )
#     assert isinstance(score, (float, dict))
#     print(f"Sync Answer Relevancy: {score}")

@pytest.mark.asyncio
async def test_error_handling():
    """Test error handling with invalid inputs."""
    mock_llm = MockLLM()
    scorer = AnswerRelevancyScorer(model=mock_llm, threshold=0.7)
    
    # Test with missing context
    result = await scorer.evaluate(
        input_text="What is the capital of France?",
        output_text="Paris is the capital of France.",
        context=None
    )
    
    # Should handle gracefully
    assert isinstance(result, ScoreResult)
    print(f"Error handling test passed: {result.score}")

if __name__ == "__main__":
    # Run all tests
    pytest.main([__file__, "-v", "-s"]) 