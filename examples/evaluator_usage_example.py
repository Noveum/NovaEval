"""
Practical Example: Using NovaEval Evaluators

This example demonstrates how to use both the standard evaluator 
and create custom evaluators for your specific needs.
"""

from pathlib import Path
from typing import Any, Dict, List

# NovaEval imports
from novaeval.datasets.custom import CustomDataset
from novaeval.models.openai import OpenAIModel
from novaeval.scorers.accuracy import ExactMatchScorer
from novaeval.evaluators.standard import Evaluator as StandardEvaluator

# Import our custom evaluator
from simple_custom_evaluator import SimpleCustomEvaluator


def create_sample_dataset():
    """Create a simple dataset for testing."""
    # Create sample data
    sample_data = [
        {
            "id": "sample_1",
            "input": "What is the capital of France?",
            "expected": "Paris"
        },
        {
            "id": "sample_2", 
            "input": "What is 2 + 2?",
            "expected": "4"
        },
        {
            "id": "sample_3",
            "input": "What color is the sky?",
            "expected": "blue"
        },
        {
            "id": "sample_4",
            "input": "Who wrote Romeo and Juliet?",
            "expected": "William Shakespeare"
        },
        {
            "id": "sample_5",
            "input": "What is the largest planet in our solar system?",
            "expected": "Jupiter"
        }
    ]
    
    # Save to a temporary file
    import json
    import tempfile
    
    temp_file = Path(tempfile.mktemp(suffix='.jsonl'))
    with open(temp_file, 'w') as f:
        for item in sample_data:
            f.write(json.dumps(item) + '\n')
    
    return temp_file


def example_standard_evaluator():
    """Example using the standard NovaEval evaluator."""
    print("=" * 50)
    print("Running Standard Evaluator Example")
    print("=" * 50)
    
    # Create dataset
    dataset_file = create_sample_dataset()
    dataset = CustomDataset(str(dataset_file))
    
    # Create models (you'd need actual API keys for this to work)
    models = [
        # Note: You need actual API keys for these to work
        # OpenAIModel("gpt-3.5-turbo", api_key="your-openai-key"),
        # For demonstration, we'll create a mock model
        MockModel("mock-model-1"),
        MockModel("mock-model-2"), 
    ]
    
    # Create scorers
    scorers = [
        ExactMatchScorer(case_sensitive=False),
    ]
    
    # Create and run evaluator
    evaluator = StandardEvaluator(
        dataset=dataset,
        models=models,
        scorers=scorers,
        output_dir="./results/standard_evaluation",
        max_workers=2,
        batch_size=1
    )
    
    try:
        results = evaluator.run()
        print(f"✅ Standard evaluation completed!")
        print(f"📊 Models evaluated: {results['summary']['total_models']}")
        print(f"📝 Samples processed: {results['summary']['total_samples']}")
        print(f"⚠️  Errors encountered: {results['summary']['total_errors']}")
        
        # Show best models
        best_models = results['summary'].get('best_model', {})
        if best_models:
            print("\n🏆 Best models by scorer:")
            for scorer_name, info in best_models.items():
                print(f"  {scorer_name}: {info['model']} (score: {info['score']:.3f})")
        
    except Exception as e:
        print(f"❌ Error running standard evaluator: {e}")
    
    # Cleanup
    dataset_file.unlink()


def example_custom_evaluator():
    """Example using our custom evaluator."""
    print("\n" + "=" * 50)
    print("Running Custom Evaluator Example")
    print("=" * 50)
    
    # Create dataset
    dataset_file = create_sample_dataset()
    dataset = CustomDataset(str(dataset_file))
    
    # Create models
    models = [
        MockModel("custom-model-1"),
        MockModel("custom-model-2"),
    ]
    
    # Create scorers
    scorers = [
        ExactMatchScorer(case_sensitive=False),
    ]
    
    # Create and run custom evaluator
    evaluator = SimpleCustomEvaluator(
        dataset=dataset,
        models=models,
        scorers=scorers,
        output_dir="./results/custom_evaluation",
        verbose=True
    )
    
    try:
        results = evaluator.run()
        print(f"✅ Custom evaluation completed!")
        print(f"📊 Models evaluated: {results['summary']['total_models']}")
        print(f"📝 Samples processed: {results['summary']['total_samples']}")  
        print(f"⚠️  Errors encountered: {results['summary']['total_errors']}")
        print(f"📈 Average error rate: {results['summary']['average_error_rate']:.2%}")
        
        # Show model comparison
        model_comparison = results['summary'].get('model_comparison', {})
        if model_comparison:
            print("\n📋 Model Performance Comparison:")
            for model_name, stats in model_comparison.items():
                print(f"  {model_name}:")
                print(f"    Samples: {stats['samples_processed']}")
                print(f"    Errors: {stats['errors']}")  
                print(f"    Error Rate: {stats['error_rate']:.2%}")
        
    except Exception as e:
        print(f"❌ Error running custom evaluator: {e}")
    
    # Cleanup
    dataset_file.unlink()


class MockModel:
    """A mock model for demonstration purposes."""
    
    def __init__(self, name: str):
        self.name = name
        self.model_name = name
        
        # Mock responses for our sample questions
        self.responses = {
            "What is the capital of France?": "Paris",
            "What is 2 + 2?": "4", 
            "What color is the sky?": "blue",
            "Who wrote Romeo and Juliet?": "Shakespeare",  # Slightly different to test scoring
            "What is the largest planet in our solar system?": "Jupiter"
        }
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate a mock response."""
        # Return predefined responses or a default
        return self.responses.get(prompt, f"Mock response to: {prompt}")
    
    def get_info(self) -> Dict[str, Any]:
        """Return model information."""
        return {
            "name": self.name,
            "model_name": self.model_name,
            "type": "mock",
        }


def compare_evaluators():
    """Compare the results from both evaluators."""
    print("\n" + "=" * 50)
    print("Evaluator Comparison")
    print("=" * 50)
    
    print("📊 Standard Evaluator:")
    print("  ✅ Built-in parallel processing")
    print("  ✅ Comprehensive score aggregation") 
    print("  ✅ CSV export for analysis")
    print("  ✅ Production-ready error handling")
    
    print("\n📊 Custom Evaluator:")
    print("  ✅ Detailed progress logging")
    print("  ✅ Custom summary statistics")
    print("  ✅ Enhanced metadata tracking")
    print("  ✅ Custom report generation")
    print("  ✅ Flexible for specific needs")
    
    print("\n💡 When to use which:")
    print("  • Standard Evaluator: Production evaluations, large datasets")
    print("  • Custom Evaluator: Research, specific requirements, custom metrics")


if __name__ == "__main__":
    print("🚀 NovaEval Evaluator Examples")
    print("This example shows how to use both standard and custom evaluators.")
    print("Note: Using mock models for demonstration - replace with real models in practice.\n")
    
    # Run examples
    example_standard_evaluator()
    example_custom_evaluator()
    compare_evaluators()
    
    print("\n✨ Examples completed! Check the ./results/ directory for output files.")
    print("🔍 Look at custom_report.txt for the enhanced custom evaluator output.") 