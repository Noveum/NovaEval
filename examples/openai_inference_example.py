#!/usr/bin/env python3
"""
OpenAI Model Inference Example

This example demonstrates how to use the OpenAI model from novaeval.models.openai
to perform inference over multiple queries.
"""

import os
import sys

# Add the src directory to the path so we can import novaeval
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from novaeval.models.openai import OpenAIModel


def run_openai_inference_example():
    """Run the OpenAI inference example with 5 different queries."""

    # Initialize the OpenAI model
    # You can specify a different model if needed
    model = OpenAIModel(
        model_name="gpt-3.5-turbo",  # Using GPT-3.5 for cost efficiency
        max_retries=3,
        timeout=30.0,
    )

    # Define 5 different queries to test
    queries = [
        "What is the capital of France?",
        "Explain quantum computing in simple terms.",
        "Write a haiku about artificial intelligence.",
        "What are the main benefits of renewable energy?",
        "How does machine learning differ from traditional programming?",
    ]

    print("🚀 Starting OpenAI Model Inference Example")
    print("=" * 50)
    print(f"Model: {model.name}")
    print(f"Provider: {model.get_provider()}")
    print(f"Model Info: {model.get_info()}")
    print("=" * 50)
    print()

    # Test single query generation
    print("📝 Single Query Generation:")
    print("-" * 30)

    try:
        response = model.generate(prompt=queries[0], max_tokens=100, temperature=0.7)
        print(f"Query: {queries[0]}")
        print(f"Response: {response}")
        print()
    except Exception as e:
        print(f"❌ Error in single generation: {e}")
        print()

    # Test batch generation
    print("🔄 Batch Query Generation:")
    print("-" * 30)

    try:
        responses = model.generate_batch(
            prompts=queries, max_tokens=150, temperature=0.5
        )

        for i, (query, response) in enumerate(zip(queries, responses), 1):
            print(f"Query {i}: {query}")
            print(f"Response {i}: {response}")
            print("-" * 20)

    except Exception as e:
        print(f"❌ Error in batch generation: {e}")

    # Test different parameters
    print("⚙️  Testing Different Parameters:")
    print("-" * 30)

    try:
        # High temperature for more creative responses
        creative_response = model.generate(
            prompt="Write a short story about a robot learning to paint",
            max_tokens=200,
            temperature=0.9,
        )
        print("Creative Response (high temp):")
        print(creative_response)
        print()

        # Low temperature for more focused responses
        focused_response = model.generate(
            prompt="What is the mathematical formula for the area of a circle?",
            max_tokens=50,
            temperature=0.1,
        )
        print("Focused Response (low temp):")
        print(focused_response)
        print()

    except Exception as e:
        print(f"❌ Error in parameter testing: {e}")

    # Test cost estimation
    print("💰 Cost Estimation:")
    print("-" * 30)

    try:
        sample_prompt = "Explain the concept of neural networks."
        sample_response = "Neural networks are computational models inspired by biological neural networks."

        estimated_cost = model.estimate_cost(sample_prompt, sample_response)
        input_tokens = model.count_tokens(sample_prompt)
        output_tokens = model.count_tokens(sample_response)

        print(f"Sample prompt tokens: {input_tokens}")
        print(f"Sample response tokens: {output_tokens}")
        print(f"Estimated cost: ${estimated_cost:.6f}")
        print()

    except Exception as e:
        print(f"❌ Error in cost estimation: {e}")

    # Test connection validation
    print("🔗 Connection Validation:")
    print("-" * 30)

    try:
        is_valid = model.validate_connection()
        print(f"Connection valid: {is_valid}")
    except Exception as e:
        print(f"❌ Error in connection validation: {e}")

    print()
    print("✅ OpenAI Model Inference Example Complete!")


if __name__ == "__main__":
    # Check if OpenAI API key is available
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Warning: OPENAI_API_KEY environment variable not set.")
        print("   Please set your OpenAI API key before running this example.")
        print("   Example: export OPENAI_API_KEY='your-api-key-here'")
        print()

    run_openai_inference_example()
