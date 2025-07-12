"""
Basic evaluation example for NovaEval.

This example demonstrates how to run a simple evaluation using
the NovaEval framework.
"""

from novaeval import Evaluator
from novaeval.datasets import MMLUDataset
from novaeval.models import OpenAIModel
from novaeval.scorers import AccuracyScorer


def main():
    """
    Run a basic evaluation example.

    Note: This example uses max_tokens=500 to allow detailed reasoning,
    but this significantly increases API costs. Adjust MAX_TOKENS based
    on your use case and budget constraints.
    """

    # Configure max_tokens based on your use case and budget
    # For MMLU multiple choice: 5-10 tokens (just the letter answer)
    # For reasoning evaluation: 100 tokens (explanation + answer)
    # Note: Higher values increase API costs and latency significantly
    MAX_TOKENS = 100  # Configurable - adjust based on your needs and budget

    # Initialize dataset - use easier subset for higher accuracy
    print("Loading MMLU dataset...")
    dataset = MMLUDataset(
        subset="elementary_mathematics",
        num_samples=10,
        split="test",  # Easier questions for demo
    )

    # Initialize model with appropriate generation settings for MMLU
    print("Initializing OpenAI model...")
    print(
        f"⚠️  Using max_tokens={MAX_TOKENS} - this may increase API costs significantly"
    )
    print("   For budget-conscious evaluation, consider reducing MAX_TOKENS to 5-10")
    model = OpenAIModel(
        model_name="gpt-4o-mini",
        temperature=0.0,
        max_tokens=MAX_TOKENS,  # Configurable - see MAX_TOKENS comment above
        # Let the model complete its reasoning naturally
    )

    # Initialize scorer - use built-in robust answer extraction
    print("Setting up accuracy scorer...")
    scorer = AccuracyScorer(
        extract_answer=True,
        # Use the built-in patterns which are more robust
    )

    # Create evaluator
    print("Creating evaluator...")
    evaluator = Evaluator(
        dataset=dataset,
        models=[model],
        scorers=[scorer],
        output_dir="./results/basic_example",
    )

    # Run evaluation
    print("Running evaluation...")
    results = evaluator.run()

    # Display results
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)

    for model_name, model_results in results["model_results"].items():
        print(f"\nModel: {model_name}")
        print("-" * 30)

        for scorer_name, score_info in model_results["scores"].items():
            if isinstance(score_info, dict):
                mean_score = score_info.get("mean", 0)
                count = score_info.get("count", 0)
                print(f"{scorer_name}: {mean_score:.4f} ({count} samples)")
            else:
                print(f"{scorer_name}: {score_info}")

        if model_results["errors"]:
            print(f"Errors: {len(model_results['errors'])}")

    print(f"\nResults saved to: {evaluator.output_dir}")
    print("Check the generated reports for detailed analysis!")


if __name__ == "__main__":
    main()
