#!/usr/bin/env python3
"""
Basic RAG Evaluation Example

This example demonstrates how to use the core RAG evaluation metrics
to assess the quality of a simple RAG system response.
"""

import asyncio
import os
import sys

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), "../../src"))

from novaeval.models.openai_model import OpenAIModel
from novaeval.scorers.rag_comprehensive import (
    RAGEvaluationSuite,
    create_rag_scorer,
    get_rag_config,
)


async def basic_rag_evaluation_example():
    """Demonstrate basic RAG evaluation with core metrics."""

    print("🔍 Basic RAG Evaluation Example")
    print("=" * 50)

    # Initialize model and configuration
    model = OpenAIModel(model_name="gpt-4")
    config = get_rag_config("balanced")

    # Create evaluation suite
    suite = RAGEvaluationSuite(model, config)

    # Example RAG scenario: Question about renewable energy
    question = "What are the main environmental benefits of renewable energy sources?"

    context = """
    Renewable energy sources such as solar, wind, hydroelectric, and geothermal power
    offer significant environmental advantages over fossil fuels. These sources produce
    little to no greenhouse gas emissions during operation, helping to mitigate climate
    change. They also reduce air pollution, as they don't release harmful pollutants
    like sulfur dioxide, nitrogen oxides, or particulate matter. Additionally, renewable
    energy sources have minimal water usage compared to traditional power plants,
    helping preserve water resources. The extraction and transportation of renewable
    energy sources also have lower environmental impacts compared to fossil fuel
    extraction and transport.
    """

    generated_answer = """
    Renewable energy sources provide several key environmental benefits. First, they
    significantly reduce greenhouse gas emissions, which helps combat climate change.
    Unlike fossil fuels, renewable sources like solar and wind produce clean energy
    without releasing carbon dioxide during operation. Second, they improve air quality
    by eliminating the release of harmful pollutants such as sulfur compounds and
    particulate matter. Third, renewable energy requires much less water than traditional
    power generation, helping conserve this precious resource. Finally, the environmental
    impact of harvesting renewable energy is minimal compared to the extraction and
    transportation of fossil fuels.
    """

    expected_answer = """
    The main environmental benefits of renewable energy include: reduced greenhouse gas
    emissions that help fight climate change, improved air quality through elimination
    of harmful pollutants, conservation of water resources due to lower water usage,
    and minimal environmental impact from energy harvesting compared to fossil fuel
    extraction.
    """

    print("📝 Evaluation Scenario:")
    print(f"Question: {question}")
    print(f"Context Length: {len(context)} characters")
    print(f"Generated Answer Length: {len(generated_answer)} characters")
    print()

    # Run comprehensive RAG evaluation
    print("⚡ Running Core RAG Evaluation...")
    results = await suite.evaluate_comprehensive(
        question, generated_answer, expected_answer, context
    )

    # Display results
    print("\n📊 Evaluation Results:")
    print("-" * 70)
    print(f"{'Metric':<25} | {'Score':<8} | {'Status':<8} | {'Threshold'}")
    print("-" * 70)

    for metric_name, result in results.items():
        status = "✅ PASS" if result.passed else "❌ FAIL"
        threshold = result.metadata.get("threshold_used", "N/A")
        print(f"{metric_name:<25} | {result.score:<8.3f} | {status:<8} | {threshold}")

    print("-" * 70)

    # Calculate overall performance
    total_metrics = len(results)
    passed_metrics = sum(1 for r in results.values() if r.passed)
    overall_score = sum(r.score for r in results.values()) / total_metrics

    print(
        f"Overall Performance: {passed_metrics}/{total_metrics} metrics passed ({passed_metrics/total_metrics*100:.1f}%)"
    )
    print(f"Average Score: {overall_score:.3f}")

    # Show detailed analysis for key metrics
    print("\n🔍 Detailed Analysis:")
    key_metrics = [
        "context_relevancy",
        "answer_relevancy",
        "faithfulness",
        "answer_correctness",
    ]

    for metric in key_metrics:
        if metric in results:
            result = results[metric]
            print(f"\n{metric.upper().replace('_', ' ')}:")
            print(f"Score: {result.score:.3f}")
            print(f"Status: {'✅ PASSED' if result.passed else '❌ FAILED'}")
            print(f"Reasoning: {result.reasoning[:200]}...")

            # Show metadata insights
            if "confidence" in result.metadata:
                print(f"Confidence: {result.metadata['confidence']:.3f}")

    return results


async def individual_metric_example():
    """Demonstrate using individual RAG metrics."""

    print("\n🎯 Individual Metric Usage Example")
    print("=" * 50)

    model = OpenAIModel(model_name="gpt-4")

    # Example data
    question = "How does photosynthesis work?"
    answer = "Photosynthesis is the process where plants convert sunlight, carbon dioxide, and water into glucose and oxygen using chlorophyll."
    context = "Photosynthesis is a biological process in which plants use sunlight to convert CO2 and H2O into sugar and oxygen."

    # Test individual metrics
    individual_metrics = [
        "answer_relevancy",
        "faithfulness",
        "context_relevancy",
        "answer_correctness",
    ]

    print("📊 Individual Metric Results:")
    print("-" * 60)

    for metric_name in individual_metrics:
        try:
            # Create individual scorer
            scorer = create_rag_scorer(metric_name, model)

            # Evaluate
            result = await scorer.evaluate(
                question, answer, expected_output=answer, context=context
            )

            status = "✅ PASS" if result.passed else "❌ FAIL"
            print(f"{metric_name:<25} | {result.score:>6.3f} | {status}")

        except Exception as e:
            print(f"{metric_name:<25} | ERROR: {str(e)[:30]}...")

    print("-" * 60)


async def configuration_example():
    """Demonstrate different configuration options."""

    print("\n⚙️ Configuration Options Example")
    print("=" * 50)

    model = OpenAIModel(model_name="gpt-4")

    # Test different configurations
    configs = {
        "Balanced": get_rag_config("balanced"),
        "Precision-Focused": get_rag_config("precision"),
        "Recall-Focused": get_rag_config("recall"),
        "Speed-Optimized": get_rag_config("speed"),
    }

    print("Configuration Comparison:")
    print("-" * 80)
    print(
        f"{'Config':<20} | {'Similarity':<10} | {'Faithfulness':<12} | {'Relevancy':<10}"
    )
    print("-" * 80)

    for name, config in configs.items():
        print(
            f"{name:<20} | {config.similarity_threshold:<10.2f} | {config.faithfulness_threshold:<12.2f} | {config.relevancy_threshold:<10.2f}"
        )

    print("-" * 80)

    # Example evaluation with different configs
    question = "What is machine learning?"
    answer = (
        "Machine learning is a subset of AI that enables computers to learn from data."
    )
    context = "Machine learning is a field of artificial intelligence that uses algorithms to learn patterns from data."

    print("\nConfiguration Impact on Evaluation:")
    print("-" * 60)

    for config_name, config in configs.items():
        suite = RAGEvaluationSuite(model, config)

        # Run a quick evaluation with just answer relevancy
        scorer = create_rag_scorer("answer_relevancy", model, config)
        result = await scorer.evaluate(question, answer, context=context)

        status = "✅ PASS" if result.passed else "❌ FAIL"
        print(f"{config_name:<20} | {result.score:>6.3f} | {status}")

    print("-" * 60)


async def main():
    """Run all examples."""

    # Check if OpenAI API key is available
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Warning: OPENAI_API_KEY not found in environment variables")
        print("   This example requires an OpenAI API key to run properly.")
        print("   Set your API key: export OPENAI_API_KEY='your-key-here'")
        return

    try:
        # Run all examples
        await basic_rag_evaluation_example()
        await individual_metric_example()
        await configuration_example()

        print("\n✅ Basic RAG Evaluation Example completed successfully!")
        print("\n📚 Next Steps:")
        print(
            "• Try the advanced safety evaluation example: python advanced_safety_example.py"
        )
        print("• Explore conversational evaluation: python conversational_example.py")
        print("• Run the comprehensive demo: python ../demo_rag_evaluation.py")

    except Exception as e:
        print(f"\n❌ Example failed with error: {e!s}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
