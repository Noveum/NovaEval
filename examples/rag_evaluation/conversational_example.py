#!/usr/bin/env python3
"""
Conversational RAG Evaluation Example

This example demonstrates how to evaluate RAG systems in conversational contexts,
including multi-turn dialogue coherence and context maintenance.
"""

import asyncio
import os
import sys

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), "../../src"))

from novaeval.models.openai_model import OpenAIModel
from novaeval.scorers.rag_advanced import ComprehensiveRAGEvaluationSuite


async def conversational_coherence_example():
    """Demonstrate conversational coherence evaluation."""

    print("💬 Conversational RAG Evaluation Example")
    print("=" * 50)

    # Initialize model and suite
    model = OpenAIModel(model_name="gpt-4")
    suite = ComprehensiveRAGEvaluationSuite(model)

    # Example multi-turn conversation about machine learning
    conversation_scenarios = [
        {
            "name": "Technical Learning Conversation",
            "description": "A conversation where user learns about machine learning concepts",
            "conversation_history": [
                {"role": "user", "content": "What is machine learning?"},
                {
                    "role": "assistant",
                    "content": "Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions from data without being explicitly programmed for every scenario.",
                },
                {
                    "role": "user",
                    "content": "How is it different from traditional programming?",
                },
                {
                    "role": "assistant",
                    "content": "In traditional programming, developers write specific instructions for every possible situation. In machine learning, algorithms learn patterns from data and can handle new, unseen situations based on what they've learned.",
                },
                {"role": "user", "content": "Can you give me a practical example?"},
            ],
            "current_question": "Can you give me a practical example?",
            "current_answer": "Sure! Email spam detection is a great example. Instead of programming rules for every possible spam email, we train a machine learning model on thousands of examples of spam and legitimate emails. The model learns to identify patterns and can then classify new emails it has never seen before.",
            "context": "Machine learning applications include email filtering, recommendation systems, image recognition, and natural language processing. These systems learn from training data to make predictions on new data.",
        },
        {
            "name": "Inconsistent Conversation",
            "description": "A conversation with topic drift and inconsistencies",
            "conversation_history": [
                {"role": "user", "content": "Tell me about renewable energy."},
                {
                    "role": "assistant",
                    "content": "Renewable energy comes from natural sources like solar, wind, and hydroelectric power that are constantly replenished.",
                },
                {"role": "user", "content": "What are the benefits?"},
                {
                    "role": "assistant",
                    "content": "The main benefits include environmental protection, energy independence, and long-term cost savings.",
                },
                {"role": "user", "content": "How efficient are solar panels?"},
            ],
            "current_question": "How efficient are solar panels?",
            "current_answer": "Cooking pasta is simple - just boil water, add salt, and cook the pasta until al dente. The efficiency depends on the type of pasta you choose.",
            "context": "Solar panel efficiency typically ranges from 15-22% for residential panels, with newer technologies achieving higher rates. Efficiency is measured by how much sunlight is converted to electricity.",
        },
    ]

    print("🧪 Testing Conversational Scenarios")
    print("=" * 60)

    for i, scenario in enumerate(conversation_scenarios, 1):
        print(f"\n{i}. {scenario['name']}")
        print(f"Description: {scenario['description']}")
        print(f"Conversation Length: {len(scenario['conversation_history'])} turns")
        print(f"Current Question: {scenario['current_question']}")
        print(f"Current Answer: {scenario['current_answer'][:100]}...")
        print()

        # Evaluate conversational coherence
        results = await suite.evaluate_comprehensive_plus(
            scenario["current_question"],
            scenario["current_answer"],
            context=scenario["context"],
            include_safety_metrics=False,
            include_conversational_metrics=True,
            conversation_history=scenario["conversation_history"],
        )

        # Show conversational metrics
        conv_metrics = {
            k: v for k, v in results.items() if "conversation" in k or "coherence" in k
        }

        print("Conversational Evaluation Results:")
        print("-" * 50)

        for metric_name, result in conv_metrics.items():
            status = "✅ COHERENT" if result.passed else "❌ INCOHERENT"
            confidence = result.metadata.get("confidence", 0.0)
            print(
                f"{metric_name:<25} | {result.score:>6.3f} | {status} | Conf: {confidence:.2f}"
            )

        print("-" * 50)

        # Show detailed analysis
        if "conversation_coherence" in results:
            result = results["conversation_coherence"]
            print("\n🔍 Detailed Coherence Analysis:")
            print(f"Score: {result.score:.3f}")
            print(f"Reasoning: {result.reasoning[:250]}...")

            # Show conversation metadata
            conv_length = result.metadata.get("conversation_length", 0)
            topic_consistency = result.metadata.get("topic_consistency", 0.0)
            context_maintenance = result.metadata.get("context_maintenance", 0.0)

            print(f"Conversation Length: {conv_length} turns")
            print(f"Topic Consistency: {topic_consistency:.3f}")
            print(f"Context Maintenance: {context_maintenance:.3f}")

        print("\n" + "=" * 60)


async def multi_turn_evaluation_example():
    """Demonstrate evaluation across multiple conversation turns."""

    print("\n🔄 Multi-Turn Evaluation Example")
    print("=" * 50)

    model = OpenAIModel(model_name="gpt-4")
    suite = ComprehensiveRAGEvaluationSuite(model)

    # Simulate a complete conversation with evaluation at each turn
    conversation = []
    context = "Python is a high-level programming language known for its simplicity and readability. It's widely used in web development, data science, artificial intelligence, and automation."

    # Conversation turns
    turns = [
        {
            "user": "What is Python programming language?",
            "assistant": "Python is a high-level, interpreted programming language known for its simple and readable syntax. It's widely used for web development, data analysis, artificial intelligence, and automation tasks.",
            "expected": "Python is a versatile programming language with clear syntax, used in many domains including web development and data science.",
        },
        {
            "user": "What makes Python different from other languages?",
            "assistant": "Python's main differences include its emphasis on code readability with significant whitespace, extensive standard library, and 'batteries included' philosophy. It's also dynamically typed and interpreted rather than compiled.",
            "expected": "Python differs through its readable syntax, comprehensive standard library, dynamic typing, and interpreted nature.",
        },
        {
            "user": "Is Python good for beginners?",
            "assistant": "Yes, Python is excellent for beginners because of its simple, English-like syntax and gentle learning curve. The language emphasizes readability and has a supportive community with extensive documentation and tutorials.",
            "expected": "Python is beginner-friendly due to its simple syntax, readability, and strong community support.",
        },
    ]

    print("Evaluating Multi-Turn Conversation:")
    print("-" * 60)

    cumulative_scores = {
        "conversation_coherence": [],
        "answer_relevancy": [],
        "faithfulness": [],
    }

    for i, turn in enumerate(turns):
        print(f"\nTurn {i+1}:")
        print(f"User: {turn['user']}")
        print(f"Assistant: {turn['assistant'][:80]}...")

        # Add current turn to conversation history
        conversation.extend(
            [
                {"role": "user", "content": turn["user"]},
                {"role": "assistant", "content": turn["assistant"]},
            ]
        )

        # Evaluate current response in context of full conversation
        results = await suite.evaluate_comprehensive_plus(
            turn["user"],
            turn["assistant"],
            expected_output=turn["expected"],
            context=context,
            include_safety_metrics=False,
            include_conversational_metrics=True,
            conversation_history=conversation[:-2],  # Exclude current turn from history
        )

        # Track key metrics
        key_metrics = ["conversation_coherence", "answer_relevancy", "faithfulness"]

        print("Turn Results:")
        for metric in key_metrics:
            if metric in results:
                score = results[metric].score
                cumulative_scores[metric].append(score)
                status = "✅" if results[metric].passed else "❌"
                print(f"  {metric}: {score:.3f} {status}")

    # Show conversation progression
    print("\n📈 Conversation Quality Progression:")
    print("-" * 60)

    for metric, scores in cumulative_scores.items():
        if scores:
            print(f"{metric}:")
            for i, score in enumerate(scores, 1):
                trend = ""
                if i > 1:
                    if score > scores[i - 2]:
                        trend = "↗️"
                    elif score < scores[i - 2]:
                        trend = "↘️"
                    else:
                        trend = "→"
                print(f"  Turn {i}: {score:.3f} {trend}")

            avg_score = sum(scores) / len(scores)
            print(f"  Average: {avg_score:.3f}")
            print()


async def context_switching_example():
    """Demonstrate evaluation when conversation context switches."""

    print("\n🔀 Context Switching Evaluation Example")
    print("=" * 50)

    model = OpenAIModel(model_name="gpt-4")
    suite = ComprehensiveRAGEvaluationSuite(model)

    # Conversation with topic switches
    context_switch_scenario = {
        "conversation_history": [
            {"role": "user", "content": "Tell me about machine learning algorithms."},
            {
                "role": "assistant",
                "content": "Machine learning algorithms are mathematical models that learn patterns from data. Common types include supervised learning (like linear regression), unsupervised learning (like clustering), and reinforcement learning.",
            },
            {"role": "user", "content": "What about neural networks?"},
            {
                "role": "assistant",
                "content": "Neural networks are inspired by biological neurons and consist of interconnected nodes that process information. They're particularly effective for complex pattern recognition tasks like image classification and natural language processing.",
            },
            {
                "role": "user",
                "content": "Actually, let's talk about cooking instead. How do I make a good pasta sauce?",
            },
        ],
        "current_question": "Actually, let's talk about cooking instead. How do I make a good pasta sauce?",
        "current_answer": "To make a good pasta sauce, start with quality ingredients. For a basic tomato sauce, sauté garlic and onions, add crushed tomatoes, season with herbs like basil and oregano, and let it simmer to develop flavors.",
        "context": "Pasta sauce preparation involves combining ingredients like tomatoes, herbs, garlic, and onions. The key is using quality ingredients and allowing time for flavors to develop through cooking.",
    }

    print("Testing Context Switch Scenario:")
    print("Previous Topic: Machine Learning")
    print("New Topic: Cooking")
    print(f"Question: {context_switch_scenario['current_question']}")
    print(f"Answer: {context_switch_scenario['current_answer'][:100]}...")
    print()

    # Evaluate the context switch
    results = await suite.evaluate_comprehensive_plus(
        context_switch_scenario["current_question"],
        context_switch_scenario["current_answer"],
        context=context_switch_scenario["context"],
        include_conversational_metrics=True,
        conversation_history=context_switch_scenario["conversation_history"],
    )

    # Analyze context switch handling
    if "conversation_coherence" in results:
        result = results["conversation_coherence"]

        print("Context Switch Analysis:")
        print("-" * 40)
        print(f"Coherence Score: {result.score:.3f}")
        print(f"Status: {'✅ HANDLED WELL' if result.passed else '❌ POOR HANDLING'}")
        print(f"Reasoning: {result.reasoning[:200]}...")

        # Check for context switch detection
        if "context_switch_detected" in result.metadata:
            switch_detected = result.metadata["context_switch_detected"]
            print(f"Context Switch Detected: {'Yes' if switch_detected else 'No'}")

        if "topic_transition_quality" in result.metadata:
            transition_quality = result.metadata["topic_transition_quality"]
            print(f"Transition Quality: {transition_quality:.3f}")


async def main():
    """Run all conversational evaluation examples."""

    # Check if OpenAI API key is available
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Warning: OPENAI_API_KEY not found in environment variables")
        print("   This example requires an OpenAI API key to run properly.")
        print("   Set your API key: export OPENAI_API_KEY='your-key-here'")
        return

    try:
        # Run all examples
        await conversational_coherence_example()
        await multi_turn_evaluation_example()
        await context_switching_example()

        print("\n✅ Conversational RAG Evaluation Example completed successfully!")
        print("\n📚 Next Steps:")
        print("• Try the batch evaluation example: python batch_evaluation_example.py")
        print(
            "• Explore production integration: python production_integration_example.py"
        )
        print("• Run the comprehensive demo: python ../demo_rag_evaluation.py")

    except Exception as e:
        print(f"\n❌ Example failed with error: {e!s}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
