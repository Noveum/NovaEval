"""
Integration tests for enhanced RAG evaluation system.

These tests verify the end-to-end functionality of the RAG evaluation system
with real model interactions and complex scenarios.
"""

import asyncio
import json
import os

# Import the RAG evaluation system
import sys
from unittest.mock import MagicMock, patch

import pytest

from novaeval.models.openai import OpenAIModel
from novaeval.scorers.base import ScoreResult

sys.path.append(os.path.join(os.path.dirname(__file__), "../../src"))

from novaeval.scorers.rag_comprehensive import (
    RAGEvaluationConfig,
    RAGEvaluationSuite,
    get_optimized_rag_config,
)


class TestRAGEvaluationWorkflow:
    """Test complete RAG evaluation workflows."""

    @pytest.fixture
    def mock_openai_model(self):
        """Create a mock OpenAI model for testing."""
        with patch("openai.AsyncOpenAI") as mock_client:
            # Configure mock responses for different types of prompts
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = self._get_mock_response

            mock_client.return_value.chat.completions.create.return_value = (
                mock_response
            )

            model = OpenAIModel(model_name="gpt-3.5-turbo", api_key="test-key")
            return model

    def _get_mock_response(self, *args, **kwargs):
        """Generate appropriate mock responses based on prompt content."""
        # This would be called with the actual prompt, but for testing
        # we'll return structured responses
        return json.dumps(
            {
                "relevance_score": 4,
                "reasoning": "Content is highly relevant to the question",
                "questions": ["What is the main concept?", "How does it work?"],
                "structural_similarity": 0.8,
                "classification": "TRUE_POSITIVE",
                "confidence": 0.9,
                "presence_status": "FULLY_PRESENT",
                "supporting_evidence": "Information found in context",
            }
        )

    @pytest.mark.asyncio
    async def test_end_to_end_rag_evaluation(self, mock_openai_model):
        """Test complete end-to-end RAG evaluation workflow."""
        # Create evaluation suite
        config = RAGEvaluationConfig(
            similarity_threshold=0.7, faithfulness_threshold=0.8
        )
        suite = RAGEvaluationSuite(mock_openai_model, config)

        # Realistic RAG scenario: Question about climate change
        question = "What are the main causes of climate change and what can be done to address it?"

        generated_answer = """
        Climate change is primarily caused by human activities that increase greenhouse gas
        concentrations in the atmosphere. The main causes include:

        1. Burning fossil fuels (coal, oil, gas) for electricity, heat, and transportation
        2. Deforestation and land use changes
        3. Industrial processes and manufacturing
        4. Agriculture, particularly livestock farming

        To address climate change, we can:
        - Transition to renewable energy sources like solar and wind
        - Improve energy efficiency in buildings and transportation
        - Protect and restore forests
        - Develop carbon capture technologies
        - Implement policy measures like carbon pricing
        """

        expected_answer = """
        The primary drivers of climate change are greenhouse gas emissions from human activities.
        Key sources include fossil fuel combustion, deforestation, and industrial processes.
        Solutions involve transitioning to clean energy, enhancing energy efficiency,
        protecting natural ecosystems, and implementing supportive policies.
        """

        context = [
            "Climate change refers to long-term shifts in global temperatures and weather patterns. While climate variations are natural, scientific evidence shows that human activities have been the main driver since the 1800s.",
            "The primary cause is the emission of greenhouse gases, particularly carbon dioxide from burning fossil fuels like coal, oil, and gas. These gases trap heat in Earth's atmosphere.",
            "Deforestation contributes to climate change by reducing the Earth's capacity to absorb CO2. Forests act as carbon sinks, storing carbon that would otherwise remain in the atmosphere.",
            "Solutions include transitioning to renewable energy sources, improving energy efficiency, electrifying transportation, and implementing carbon pricing policies.",
            "International cooperation through agreements like the Paris Climate Accord is essential for coordinated global action on climate change.",
        ]

        # Run comprehensive evaluation
        results = await suite.evaluate_comprehensive(
            question, generated_answer, expected_answer, context
        )

        # Verify all expected metrics are present
        expected_metrics = [
            "context_precision",
            "context_relevancy",
            "context_recall",
            "context_entity_recall",
            "answer_relevancy",
            "answer_similarity",
            "answer_correctness",
            "faithfulness",
            "ragas",
            "rag_triad",
        ]

        for metric in expected_metrics:
            assert metric in results
            assert isinstance(results[metric], ScoreResult)
            assert 0.0 <= results[metric].score <= 1.0

        # Verify composite scores have expected structure
        ragas_result = results["ragas"]
        assert "component_scores" in ragas_result.metadata
        assert "retrieval_score" in ragas_result.metadata
        assert "generation_score" in ragas_result.metadata

        triad_result = results["rag_triad"]
        assert "context_relevance" in triad_result.metadata
        assert "groundedness" in triad_result.metadata
        assert "answer_relevance" in triad_result.metadata

    @pytest.mark.asyncio
    async def test_retrieval_pipeline_evaluation(self, mock_openai_model):
        """Test focused evaluation of retrieval pipeline."""
        suite = RAGEvaluationSuite(mock_openai_model)

        # Scenario: Technical documentation retrieval
        question = "How do you implement authentication in a REST API?"

        answer = """
        To implement authentication in a REST API, you can use several approaches:
        1. JWT (JSON Web Tokens) for stateless authentication
        2. OAuth 2.0 for third-party authentication
        3. API keys for simple service-to-service authentication
        4. Basic authentication for simple use cases
        """

        expected_answer = """
        REST API authentication can be implemented using JWT tokens, OAuth 2.0,
        API keys, or basic authentication depending on security requirements.
        """

        # Mixed quality context - some relevant, some irrelevant
        context = [
            "JWT (JSON Web Tokens) are a compact way to securely transmit information between parties as a JSON object. They are commonly used for authentication in web applications.",
            "OAuth 2.0 is an authorization framework that enables applications to obtain limited access to user accounts on an HTTP service.",
            "API keys are simple authentication tokens that identify the calling program to the API.",
            "The weather today is sunny with a high of 75 degrees.",  # Irrelevant
            "Basic authentication is a simple authentication scheme built into the HTTP protocol.",
            "Machine learning algorithms can be used to predict user behavior.",  # Irrelevant
        ]

        # Evaluate retrieval pipeline
        results = await suite.evaluate_retrieval_pipeline(
            question, answer, expected_answer, context
        )

        # Verify retrieval metrics
        retrieval_metrics = [
            "context_precision",
            "context_relevancy",
            "context_recall",
            "context_entity_recall",
        ]

        for metric in retrieval_metrics:
            assert metric in results
            result = results[metric]
            assert isinstance(result, ScoreResult)
            assert 0.0 <= result.score <= 1.0

            # Context precision should detect irrelevant content
            if metric == "context_precision":
                assert "relevance_evaluations" in result.metadata

            # Context relevancy should measure overall relevance
            if metric == "context_relevancy":
                assert "relevant_percentage" in result.metadata

    @pytest.mark.asyncio
    async def test_generation_pipeline_evaluation(self, mock_openai_model):
        """Test focused evaluation of generation pipeline."""
        suite = RAGEvaluationSuite(mock_openai_model)

        # Scenario: Medical information generation
        question = "What are the symptoms of diabetes?"

        generated_answer = """
        Common symptoms of diabetes include:
        - Increased thirst and frequent urination
        - Extreme fatigue and weakness
        - Blurred vision
        - Slow-healing cuts and bruises
        - Unexplained weight loss (Type 1)
        - Tingling or numbness in hands and feet
        """

        expected_answer = """
        Diabetes symptoms include excessive thirst, frequent urination, fatigue,
        blurred vision, slow wound healing, and unexplained weight changes.
        """

        context = """
        Diabetes is a group of metabolic disorders characterized by high blood sugar levels.
        Common symptoms include polydipsia (excessive thirst), polyuria (frequent urination),
        fatigue, blurred vision, and slow wound healing. Type 1 diabetes may also cause
        rapid weight loss, while Type 2 diabetes may cause gradual weight gain.
        """

        # Evaluate generation pipeline
        results = await suite.evaluate_generation_pipeline(
            question, generated_answer, expected_answer, context
        )

        # Verify generation metrics
        generation_metrics = [
            "answer_relevancy",
            "answer_similarity",
            "answer_correctness",
            "faithfulness",
        ]

        for metric in generation_metrics:
            assert metric in results
            result = results[metric]
            assert isinstance(result, ScoreResult)
            assert 0.0 <= result.score <= 1.0

            # Answer relevancy should use multi-approach evaluation
            if metric == "answer_relevancy":
                assert "approach" in result.metadata
                assert result.metadata["approach"] == "multi_method"

            # Answer similarity should have multiple similarity measures
            if metric == "answer_similarity":
                assert "semantic_similarity" in result.metadata
                assert "lexical_similarity" in result.metadata

            # Faithfulness should verify claims against context
            if metric == "faithfulness":
                assert "total_claims" in result.metadata


class TestRAGEvaluationScenarios:
    """Test various RAG evaluation scenarios."""

    @pytest.fixture
    def evaluation_suite(self):
        """Create evaluation suite with mock model."""

        class MockModel:
            async def generate(self, prompt, **kwargs):
                # Return appropriate JSON responses based on prompt type
                if "relevance_score" in prompt:
                    return '{"relevance_score": 4, "reasoning": "Highly relevant"}'
                elif "questions" in prompt:
                    return '{"questions": ["Q1?", "Q2?", "Q3?"]}'
                elif "similarity" in prompt:
                    return '{"structural_similarity": 0.85, "reasoning": "Similar structure"}'
                elif "classification" in prompt:
                    return '{"classification": "TRUE_POSITIVE", "confidence": 0.9}'
                elif "presence" in prompt:
                    return '{"presence_status": "FULLY_PRESENT", "confidence": 0.9}'
                elif "supported" in prompt:
                    return '{"status": "SUPPORTED", "confidence": 0.9}'
                elif "claims" in prompt:
                    return '{"factual_claims": ["Claim 1", "Claim 2"]}'
                elif "statements" in prompt:
                    return '{"statements": ["Statement 1", "Statement 2"]}'
                elif "entities" in prompt:
                    return '{"persons": ["John"], "organizations": ["ACME"]}'
                else:
                    return "Default response"

        return RAGEvaluationSuite(MockModel())

    @pytest.mark.asyncio
    async def test_high_quality_rag_scenario(self, evaluation_suite):
        """Test scenario with high-quality RAG components."""
        question = "What is photosynthesis?"

        # High-quality generated answer
        answer = """
        Photosynthesis is the process by which plants convert light energy into chemical energy.
        It occurs in chloroplasts and involves two main stages: light-dependent reactions
        and the Calvin cycle. The overall equation is: 6CO2 + 6H2O + light energy → C6H12O6 + 6O2.
        """

        # Accurate expected answer
        expected = """
        Photosynthesis is the biological process where plants use sunlight to convert
        carbon dioxide and water into glucose and oxygen, occurring primarily in chloroplasts.
        """

        # Highly relevant context
        context = """
        Photosynthesis is a fundamental biological process that occurs in plants, algae,
        and some bacteria. It takes place in specialized organelles called chloroplasts,
        which contain chlorophyll. The process consists of light-dependent reactions
        (occurring in thylakoids) and light-independent reactions (the Calvin cycle,
        occurring in the stroma). The chemical equation for photosynthesis is
        6CO2 + 6H2O + light energy → C6H12O6 + 6O2.
        """

        results = await evaluation_suite.evaluate_comprehensive(
            question, answer, expected, context
        )

        # High-quality scenario should have good scores
        ragas_score = results["ragas"].score
        triad_score = results["rag_triad"].score

        # Scores should be reasonably high for this high-quality scenario
        assert ragas_score > 0.5  # Should be above average
        assert triad_score > 0.5

        # Most individual metrics should pass
        passed_count = sum(1 for result in results.values() if result.passed)
        assert passed_count >= len(results) * 0.6  # At least 60% should pass

    @pytest.mark.asyncio
    async def test_poor_quality_rag_scenario(self, evaluation_suite):
        """Test scenario with poor-quality RAG components."""
        question = "What causes earthquakes?"

        # Poor quality answer (off-topic and inaccurate)
        answer = """
        Earthquakes are caused by angry underground spirits that shake the earth
        when they are disturbed. This happens mostly during full moons and can
        be prevented by offering sacrifices to the earth gods.
        """

        # Accurate expected answer
        expected = """
        Earthquakes are caused by the sudden release of energy stored in rocks
        due to tectonic plate movements, fault line activity, and stress
        accumulation in the Earth's crust.
        """

        # Irrelevant context
        context = """
        The stock market had a volatile day yesterday with major indices
        fluctuating significantly. Technology stocks led the decline while
        energy sector showed some resilience. Weather patterns across the
        country are expected to remain stable this week.
        """

        results = await evaluation_suite.evaluate_comprehensive(
            question, answer, expected, context
        )

        # Poor quality scenario should have low scores
        ragas_score = results["ragas"].score
        triad_score = results["rag_triad"].score

        # Scores should be low for this poor-quality scenario
        assert ragas_score < 0.5  # Should be below average
        assert triad_score < 0.5

        # Most individual metrics should fail
        failed_count = sum(1 for result in results.values() if not result.passed)
        assert failed_count >= len(results) * 0.6  # At least 60% should fail

    @pytest.mark.asyncio
    async def test_mixed_quality_rag_scenario(self, evaluation_suite):
        """Test scenario with mixed quality RAG components."""
        question = "How does machine learning work?"

        # Partially correct answer with some inaccuracies
        answer = """
        Machine learning is a type of artificial intelligence where computers
        learn from data without being explicitly programmed. It uses algorithms
        to find patterns in data and make predictions. There are three main types:
        supervised learning, unsupervised learning, and reinforcement learning.
        However, machine learning requires quantum computers to work effectively.
        """

        expected = """
        Machine learning is a subset of AI that enables computers to learn and
        improve from experience without explicit programming. It uses algorithms
        to analyze data, identify patterns, and make predictions or decisions.
        """

        # Mixed quality context (some relevant, some irrelevant)
        context = [
            "Machine learning is a method of data analysis that automates analytical model building.",
            "It is based on the idea that systems can learn from data and make decisions with minimal human intervention.",
            "The weather forecast shows rain is expected tomorrow.",  # Irrelevant
            "Common ML algorithms include linear regression, decision trees, and neural networks.",
            "Quantum computing is a separate field from machine learning.",  # Contradicts error in answer
        ]

        results = await evaluation_suite.evaluate_comprehensive(
            question, answer, expected, context
        )

        # Mixed quality should have moderate scores
        ragas_score = results["ragas"].score
        assert 0.3 <= ragas_score <= 0.7  # Should be in middle range

        # Some metrics should pass, others fail
        passed_count = sum(1 for result in results.values() if result.passed)
        failed_count = len(results) - passed_count

        # Should have a mix of passing and failing metrics
        assert passed_count > 0
        assert failed_count > 0


class TestRAGEvaluationConfiguration:
    """Test different RAG evaluation configurations."""

    @pytest.mark.asyncio
    async def test_precision_optimized_config(self):
        """Test precision-optimized configuration."""

        class MockModel:
            async def generate(self, prompt, **kwargs):
                return '{"relevance_score": 4, "reasoning": "Test response"}'

        # Use precision-optimized config
        config = get_optimized_rag_config("precision")
        suite = RAGEvaluationSuite(MockModel(), config)

        # Test with scenario that should emphasize precision
        question = "What is the exact melting point of gold?"
        answer = "The melting point of gold is 1064.18°C (1947.52°F)."
        expected = "Gold melts at 1064.18 degrees Celsius."
        context = "Gold has a melting point of 1064.18°C and a boiling point of 2970°C."

        results = await suite.evaluate_comprehensive(
            question, answer, expected, context
        )

        # Verify precision-focused weights are applied
        ragas_metadata = results["ragas"].metadata
        weights = ragas_metadata["weights"]

        # Precision config should emphasize faithfulness and correctness
        assert weights["faithfulness"] >= 0.25
        assert weights["answer_correctness"] >= 0.2

    @pytest.mark.asyncio
    async def test_recall_optimized_config(self):
        """Test recall-optimized configuration."""

        class MockModel:
            async def generate(self, prompt, **kwargs):
                return '{"relevance_score": 4, "reasoning": "Test response"}'

        # Use recall-optimized config
        config = get_optimized_rag_config("recall")
        suite = RAGEvaluationSuite(MockModel(), config)

        question = "What are all the benefits of renewable energy?"
        answer = "Renewable energy reduces pollution and costs."
        expected = "Benefits include environmental protection, cost savings, job creation, and energy independence."
        context = "Renewable energy sources provide environmental benefits, economic advantages, create jobs, and enhance energy security."

        results = await suite.evaluate_comprehensive(
            question, answer, expected, context
        )

        # Verify recall-focused weights are applied
        ragas_metadata = results["ragas"].metadata
        weights = ragas_metadata["weights"]

        # Recall config should emphasize context recall and relevancy
        assert weights["context_recall"] >= 0.2
        assert weights["context_relevancy"] >= 0.15

    @pytest.mark.asyncio
    async def test_speed_optimized_config(self):
        """Test speed-optimized configuration."""

        class MockModel:
            async def generate(self, prompt, **kwargs):
                return '{"relevance_score": 4, "reasoning": "Test response"}'

        # Use speed-optimized config
        config = get_optimized_rag_config("speed")
        suite = RAGEvaluationSuite(MockModel(), config)

        question = "What is AI?"
        answer = "AI is artificial intelligence."
        expected = "Artificial intelligence is machine intelligence."
        context = "AI refers to machine intelligence and automated decision making."

        results = await suite.evaluate_comprehensive(
            question, answer, expected, context
        )

        # Verify speed-focused weights skip slower metrics
        ragas_metadata = results["ragas"].metadata
        weights = ragas_metadata["weights"]

        # Speed config should skip context recall and entity recall
        assert weights["context_recall"] == 0.0
        assert weights["context_entity_recall"] == 0.0


class TestRAGEvaluationRobustness:
    """Test robustness of RAG evaluation system."""

    @pytest.mark.asyncio
    async def test_empty_inputs_handling(self):
        """Test handling of empty inputs."""

        class MockModel:
            async def generate(self, prompt, **kwargs):
                return '{"relevance_score": 1, "reasoning": "Empty content"}'

        suite = RAGEvaluationSuite(MockModel())

        # Test with empty inputs
        results = await suite.evaluate_comprehensive("", "", "", "")

        # Should handle empty inputs gracefully
        for _metric_name, result in results.items():
            assert isinstance(result, ScoreResult)
            assert 0.0 <= result.score <= 1.0

    @pytest.mark.asyncio
    async def test_very_long_inputs_handling(self):
        """Test handling of very long inputs."""

        class MockModel:
            async def generate(self, prompt, **kwargs):
                return '{"relevance_score": 3, "reasoning": "Long content processed"}'

        suite = RAGEvaluationSuite(MockModel())

        # Create very long inputs
        long_text = "This is a very long text. " * 1000  # ~25,000 characters

        results = await suite.evaluate_comprehensive(
            long_text, long_text, long_text, long_text
        )

        # Should handle long inputs without errors
        for _metric_name, result in results.items():
            assert isinstance(result, ScoreResult)
            assert 0.0 <= result.score <= 1.0

    @pytest.mark.asyncio
    async def test_special_characters_handling(self):
        """Test handling of special characters and unicode."""

        class MockModel:
            async def generate(self, prompt, **kwargs):
                return '{"relevance_score": 3, "reasoning": "Special chars handled"}'

        suite = RAGEvaluationSuite(MockModel())

        # Test with special characters and unicode
        special_text = "Test with émojis 🚀, symbols ∑∆, and quotes 'smart quotes'"

        results = await suite.evaluate_comprehensive(
            special_text, special_text, special_text, special_text
        )

        # Should handle special characters gracefully
        for _metric_name, result in results.items():
            assert isinstance(result, ScoreResult)
            assert 0.0 <= result.score <= 1.0


class TestRAGEvaluationPerformance:
    """Test performance characteristics of RAG evaluation."""

    @pytest.mark.asyncio
    async def test_parallel_evaluation_performance(self):
        """Test that parallel evaluation improves performance."""
        import time

        class SlowMockModel:
            async def generate(self, prompt, **kwargs):
                await asyncio.sleep(0.1)  # Simulate slow model
                return '{"relevance_score": 3, "reasoning": "Slow response"}'

        suite = RAGEvaluationSuite(SlowMockModel())

        question = "Test question"
        answer = "Test answer"
        expected = "Expected answer"
        context = "Test context"

        # Measure time for comprehensive evaluation
        start_time = time.time()
        results = await suite.evaluate_comprehensive(
            question, answer, expected, context
        )
        end_time = time.time()

        # Should complete in reasonable time despite slow model
        # With parallel execution, should be much faster than sequential
        execution_time = end_time - start_time

        # With 10 metrics and 0.1s delay each, sequential would take ~1s
        # Parallel should be much faster (closer to 0.1s + overhead)
        assert execution_time < 0.5  # Should be significantly faster than sequential

        # Verify all metrics were evaluated
        assert len(results) == 10

    @pytest.mark.asyncio
    async def test_batch_evaluation_capability(self):
        """Test capability to evaluate multiple RAG examples."""

        class MockModel:
            async def generate(self, prompt, **kwargs):
                return '{"relevance_score": 4, "reasoning": "Batch test"}'

        suite = RAGEvaluationSuite(MockModel())

        # Create multiple test cases
        test_cases = [
            {
                "question": f"Question {i}",
                "answer": f"Answer {i}",
                "expected": f"Expected {i}",
                "context": f"Context {i}",
            }
            for i in range(5)
        ]

        # Evaluate all test cases
        all_results = []
        for case in test_cases:
            results = await suite.evaluate_comprehensive(
                case["question"], case["answer"], case["expected"], case["context"]
            )
            all_results.append(results)

        # Verify all evaluations completed successfully
        assert len(all_results) == 5
        for results in all_results:
            assert len(results) == 10  # All metrics evaluated
            for result in results.values():
                assert isinstance(result, ScoreResult)


if __name__ == "__main__":
    # Run the integration tests
    pytest.main([__file__, "-v", "-s"])
