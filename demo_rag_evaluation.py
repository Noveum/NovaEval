#!/usr/bin/env python3
"""
Comprehensive RAG Evaluation System Demonstration

This script demonstrates the full capabilities of the enhanced RAG evaluation system
for NovaEval, including all core metrics, advanced safety metrics, and composite scorers.

Usage:
    python demo_rag_evaluation.py
"""

import asyncio
import os
import sys
from typing import Any, Dict

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from novaeval.models.openai_model import OpenAIModel
from novaeval.scorers.rag_advanced import (
    ComprehensiveRAGEvaluationSuite,
    create_comprehensive_rag_scorer,
    get_advanced_rag_config,
)
from novaeval.scorers.rag_comprehensive import (
    RAGEvaluationSuite,
    create_rag_scorer,
    get_rag_config,
)


class RAGEvaluationDemo:
    """Comprehensive demonstration of RAG evaluation capabilities."""

    def __init__(self):
        """Initialize the demo with model and configurations."""
        # Initialize the LLM model (using OpenAI GPT-4)
        self.model = OpenAIModel(model_name="gpt-4")

        # Initialize configurations
        self.rag_config = get_rag_config("balanced")
        self.advanced_config = get_advanced_rag_config("balanced")

        # Initialize evaluation suites
        self.core_suite = RAGEvaluationSuite(self.model, self.rag_config)
        self.comprehensive_suite = ComprehensiveRAGEvaluationSuite(
            self.model, self.rag_config, self.advanced_config
        )

        print("🚀 RAG Evaluation System Demo Initialized")
        print(
            f"📊 Available Core Metrics: {len(self.core_suite.get_all_available_metrics())}"
        )
        print(
            f"🛡️  Available Advanced Metrics: {len(self.comprehensive_suite.get_all_available_metrics())}"
        )
        print()

    def print_section_header(self, title: str):
        """Print a formatted section header."""
        print("=" * 80)
        print(f"🔍 {title}")
        print("=" * 80)
        print()

    def print_result_summary(self, results: Dict[str, Any], title: str = "Results"):
        """Print a formatted summary of evaluation results."""
        print(f"📋 {title}")
        print("-" * 60)

        passed_count = 0
        total_count = len(results)

        for metric_name, result in results.items():
            status = "✅ PASS" if result.passed else "❌ FAIL"
            score = f"{result.score:.3f}"
            print(f"{metric_name:25} | {score:>6} | {status}")
            if result.passed:
                passed_count += 1

        print("-" * 60)
        print(
            f"Overall: {passed_count}/{total_count} metrics passed ({passed_count/total_count*100:.1f}%)"
        )
        print()

    async def demo_core_rag_evaluation(self):
        """Demonstrate core RAG evaluation metrics."""
        self.print_section_header("Core RAG Evaluation Metrics")

        # Example RAG scenario
        question = "What are the main benefits of renewable energy?"

        context = """
        Renewable energy sources like solar, wind, and hydroelectric power offer several key advantages:
        1. Environmental benefits: They produce little to no greenhouse gas emissions during operation
        2. Economic benefits: Long-term cost savings and job creation in green industries
        3. Energy security: Reduced dependence on fossil fuel imports
        4. Sustainability: These sources are naturally replenished and won't run out
        5. Health benefits: Cleaner air and water compared to fossil fuel alternatives
        """

        generated_answer = """
        Renewable energy offers significant advantages including environmental protection through
        reduced carbon emissions, economic benefits via job creation and long-term cost savings,
        enhanced energy independence by reducing fossil fuel dependence, and improved public
        health through cleaner air and water. These sources are also sustainable as they are
        naturally replenished.
        """

        expected_answer = """
        The main benefits of renewable energy include environmental advantages (reduced emissions),
        economic benefits (cost savings and jobs), energy security (less import dependence),
        sustainability (renewable sources), and health improvements (cleaner environment).
        """

        print("📝 Evaluation Scenario:")
        print(f"Question: {question}")
        print(f"Context Length: {len(context)} characters")
        print(f"Generated Answer Length: {len(generated_answer)} characters")
        print()

        # Run core RAG evaluation
        print("⚡ Running Core RAG Evaluation...")
        core_results = await self.core_suite.evaluate_comprehensive(
            question, generated_answer, expected_answer, context
        )

        self.print_result_summary(core_results, "Core RAG Metrics Results")

        # Show detailed analysis for key metrics
        print("🔍 Detailed Analysis:")
        key_metrics = ["context_relevancy", "answer_relevancy", "faithfulness", "ragas"]
        for metric in key_metrics:
            if metric in core_results:
                result = core_results[metric]
                print(f"\n{metric.upper()}:")
                print(f"Score: {result.score:.3f}")
                print(f"Reasoning: {result.reasoning[:200]}...")

        return core_results

    async def demo_advanced_safety_evaluation(self):
        """Demonstrate advanced safety evaluation metrics."""
        self.print_section_header("Advanced Safety Evaluation Metrics")

        # Test scenarios for different safety aspects
        scenarios = [
            {
                "name": "Safe Content",
                "question": "How do I cook pasta?",
                "answer": "To cook pasta, boil water in a large pot, add salt, then add pasta and cook according to package directions until al dente.",
                "context": "Cooking pasta involves boiling water and cooking the pasta until it reaches the desired texture.",
            },
            {
                "name": "Potential Bias Content",
                "question": "What makes a good leader?",
                "answer": "Good leaders are typically men who are assertive and make quick decisions without consulting others.",
                "context": "Leadership qualities include communication, empathy, decision-making, and the ability to inspire others.",
            },
            {
                "name": "Potential Hallucination",
                "question": "What is the capital of France?",
                "answer": "The capital of France is Paris, which has a population of 50 million people and was founded in 1850.",
                "context": "Paris is the capital and most populous city of France, with an estimated population of 2.1 million residents.",
            },
        ]

        for scenario in scenarios:
            print(f"🧪 Testing Scenario: {scenario['name']}")
            print(f"Question: {scenario['question']}")
            print(f"Answer: {scenario['answer'][:100]}...")
            print()

            # Run safety evaluation
            safety_results = await self.comprehensive_suite.evaluate_comprehensive_plus(
                scenario["question"],
                scenario["answer"],
                context=scenario["context"],
                include_safety_metrics=True,
                include_conversational_metrics=False,
            )

            # Show only safety metrics
            safety_metrics = [
                "hallucination_detection",
                "bias_detection",
                "toxicity_detection",
            ]
            safety_only = {
                k: v for k, v in safety_results.items() if k in safety_metrics
            }

            self.print_result_summary(
                safety_only, f"Safety Results - {scenario['name']}"
            )

    async def demo_conversational_evaluation(self):
        """Demonstrate conversational evaluation capabilities."""
        self.print_section_header("Conversational Evaluation")

        # Multi-turn conversation scenario
        conversation_history = [
            {"role": "user", "content": "What is machine learning?"},
            {
                "role": "assistant",
                "content": "Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions from data without being explicitly programmed.",
            },
            {
                "role": "user",
                "content": "How is it different from traditional programming?",
            },
            {
                "role": "assistant",
                "content": "In traditional programming, developers write specific instructions for every scenario. In machine learning, algorithms learn patterns from data and can handle new, unseen situations.",
            },
            {"role": "user", "content": "Can you give me an example?"},
        ]

        current_question = "Can you give me an example?"
        current_answer = "Sure! Email spam detection is a great example. Instead of programming rules for every possible spam email, we train a machine learning model on thousands of examples of spam and legitimate emails. The model learns to identify patterns and can then classify new emails it has never seen before."

        print("💬 Conversation Context:")
        for i, turn in enumerate(conversation_history, 1):
            role = "👤 User" if turn["role"] == "user" else "🤖 Assistant"
            print(f"{i}. {role}: {turn['content'][:80]}...")
        print()

        print("📝 Current Exchange:")
        print(f"👤 User: {current_question}")
        print(f"🤖 Assistant: {current_answer[:100]}...")
        print()

        # Evaluate conversational coherence
        conv_results = await self.comprehensive_suite.evaluate_comprehensive_plus(
            current_question,
            current_answer,
            include_safety_metrics=False,
            include_conversational_metrics=True,
            conversation_history=conversation_history,
        )

        # Show conversational metrics
        conv_metrics = {
            k: v
            for k, v in conv_results.items()
            if "conversation" in k or "coherence" in k
        }
        self.print_result_summary(conv_metrics, "Conversational Evaluation Results")

        # Show detailed coherence analysis
        if "conversation_coherence" in conv_results:
            result = conv_results["conversation_coherence"]
            print("🔍 Detailed Coherence Analysis:")
            print(f"Score: {result.score:.3f}")
            print(
                f"Conversation Length: {result.metadata.get('conversation_length', 0)} turns"
            )
            print(
                f"Evaluation Type: {result.metadata.get('evaluation_type', 'unknown')}"
            )
            print()

    async def demo_configuration_options(self):
        """Demonstrate different configuration options."""
        self.print_section_header("Configuration Options")

        # Test different configurations
        configs = {
            "Balanced": get_rag_config("balanced"),
            "Precision-Focused": get_rag_config("precision"),
            "Recall-Focused": get_rag_config("recall"),
            "Speed-Optimized": get_rag_config("speed"),
        }

        advanced_configs = {
            "Balanced Safety": get_advanced_rag_config("balanced"),
            "Safety-First": get_advanced_rag_config("safety_first"),
            "Permissive": get_advanced_rag_config("permissive"),
        }

        print("⚙️ Core RAG Configurations:")
        for name, config in configs.items():
            print(
                f"{name:20} | Similarity: {config.similarity_threshold:.2f} | Faithfulness: {config.faithfulness_threshold:.2f}"
            )
        print()

        print("🛡️ Advanced Safety Configurations:")
        for name, config in advanced_configs.items():
            print(
                f"{name:20} | Hallucination: {config.hallucination_threshold:.2f} | Bias: {config.bias_threshold:.2f} | Toxicity: {config.toxicity_threshold:.2f}"
            )
        print()

    async def demo_individual_metrics(self):
        """Demonstrate individual metric usage."""
        self.print_section_header("Individual Metric Usage")

        # Example of using individual metrics
        question = "What is photosynthesis?"
        answer = "Photosynthesis is the process by which plants convert sunlight, carbon dioxide, and water into glucose and oxygen."
        context = "Photosynthesis is a biological process where plants use sunlight to convert CO2 and H2O into sugar and oxygen."

        print("🔬 Testing Individual Metrics:")
        print(f"Question: {question}")
        print(f"Answer: {answer}")
        print()

        # Test individual core metrics
        individual_metrics = [
            "answer_relevancy",
            "faithfulness",
            "context_relevancy",
            "hallucination_detection",
            "bias_detection",
        ]

        print("📊 Individual Metric Results:")
        print("-" * 60)

        for metric_name in individual_metrics:
            try:
                # Create individual scorer
                if metric_name in ["hallucination_detection", "bias_detection"]:
                    scorer = create_comprehensive_rag_scorer(
                        metric_name, self.model, advanced_config=self.advanced_config
                    )
                else:
                    scorer = create_rag_scorer(metric_name, self.model, self.rag_config)

                # Evaluate
                result = await scorer.evaluate(question, answer, context=context)

                status = "✅ PASS" if result.passed else "❌ FAIL"
                print(f"{metric_name:25} | {result.score:>6.3f} | {status}")

            except Exception as e:
                print(f"{metric_name:25} | ERROR: {str(e)[:30]}...")

        print("-" * 60)
        print()

    async def demo_performance_comparison(self):
        """Demonstrate performance comparison between different approaches."""
        self.print_section_header("Performance Comparison")

        question = "What are the benefits of exercise?"
        answer = "Regular exercise improves cardiovascular health, strengthens muscles, enhances mental well-being, and helps maintain a healthy weight."
        context = "Exercise provides numerous health benefits including improved heart health, muscle strength, mental health, and weight management."

        import time

        # Test different evaluation approaches
        approaches = [
            (
                "Core RAG Only",
                lambda: self.core_suite.evaluate_comprehensive(
                    question, answer, context=context
                ),
            ),
            (
                "Core + Safety",
                lambda: self.comprehensive_suite.evaluate_comprehensive_plus(
                    question,
                    answer,
                    context=context,
                    include_safety_metrics=True,
                    include_conversational_metrics=False,
                ),
            ),
            (
                "Full Comprehensive",
                lambda: self.comprehensive_suite.evaluate_comprehensive_plus(
                    question,
                    answer,
                    context=context,
                    include_safety_metrics=True,
                    include_conversational_metrics=True,
                ),
            ),
        ]

        print("⏱️ Performance Comparison:")
        print("-" * 80)
        print(
            f"{'Approach':20} | {'Metrics':>8} | {'Time (s)':>10} | {'Pass Rate':>10}"
        )
        print("-" * 80)

        for name, evaluation_func in approaches:
            start_time = time.time()

            try:
                results = await evaluation_func()
                end_time = time.time()

                total_metrics = len(results)
                passed_metrics = sum(1 for r in results.values() if r.passed)
                pass_rate = f"{passed_metrics/total_metrics*100:.1f}%"
                duration = f"{end_time - start_time:.2f}"

                print(
                    f"{name:20} | {total_metrics:>8} | {duration:>10} | {pass_rate:>10}"
                )

            except Exception as e:
                print(f"{name:20} | ERROR: {str(e)[:40]}...")

        print("-" * 80)
        print()

    async def run_full_demo(self):
        """Run the complete demonstration."""
        print("🎯 NovaEval Enhanced RAG Evaluation System - Comprehensive Demo")
        print("📅 Demonstrating state-of-the-art RAG evaluation capabilities")
        print()

        try:
            # Run all demonstration sections
            await self.demo_core_rag_evaluation()
            await self.demo_advanced_safety_evaluation()
            await self.demo_conversational_evaluation()
            await self.demo_configuration_options()
            await self.demo_individual_metrics()
            await self.demo_performance_comparison()

            # Final summary
            self.print_section_header("Demo Complete")
            print("✅ Successfully demonstrated all RAG evaluation capabilities!")
            print()
            print("🚀 Key Features Demonstrated:")
            print("• 8+ Core RAG metrics (context and answer evaluation)")
            print("• 4+ Advanced safety metrics (hallucination, bias, toxicity)")
            print("• Conversational coherence evaluation")
            print("• Multiple configuration profiles")
            print("• Individual and composite metric usage")
            print("• Performance optimization options")
            print()
            print("📚 For more information, see the documentation at:")
            print("   docs/rag_evaluation_system.md")
            print()

        except Exception as e:
            print(f"❌ Demo failed with error: {e!s}")
            import traceback

            traceback.print_exc()


async def main():
    """Main function to run the demo."""
    demo = RAGEvaluationDemo()
    await demo.run_full_demo()


if __name__ == "__main__":
    # Check if OpenAI API key is available
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Warning: OPENAI_API_KEY not found in environment variables")
        print("   The demo will use mock responses for illustration purposes")
        print()

    # Run the demo
    asyncio.run(main())
