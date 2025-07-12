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
    """Run a basic evaluation example."""
    
    # Initialize dataset
    print("Loading MMLU dataset...")
    dataset = MMLUDataset(
        subset="abstract_algebra",
        num_samples=10,  # Small sample for demo
        split="test"
    )
    
    # Initialize model
    print("Initializing OpenAI model...")
    model = OpenAIModel(
        model_name="gpt-3.5-turbo",
        temperature=0.0
    )
    
    # Initialize scorer
    print("Setting up accuracy scorer...")
    scorer = AccuracyScorer(extract_answer=True)
    
    # Create evaluator
    print("Creating evaluator...")
    evaluator = Evaluator(
        dataset=dataset,
        models=[model],
        scorers=[scorer],
        output_dir="./results/basic_example"
    )
    
    # Run evaluation
    print("Running evaluation...")
    results = evaluator.run()
    
    # Display results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    
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

