---
layout: default
title: Examples
nav_order: 6
---

# Examples

This section provides practical examples and real-world use cases for NovaEval across different domains and scenarios.

## Table of Contents

1. [Basic Examples](#basic-examples)
2. [Model Comparison](#model-comparison)
3. [Custom Datasets](#custom-datasets)
4. [Advanced Scoring](#advanced-scoring)
5. [Production Use Cases](#production-use-cases)
6. [CI/CD Integration](#cicd-integration)

## Basic Examples

### Simple MMLU Evaluation

**Use Case**: Quick evaluation of a model on a standard benchmark

```python
from novaeval import Evaluator
from novaeval.datasets import MMLUDataset
from novaeval.models import OpenAIModel
from novaeval.scorers import AccuracyScorer

# Setup
dataset = MMLUDataset(subset="elementary_mathematics", num_samples=50)
model = OpenAIModel(model_name="gpt-4o-mini", temperature=0.0)
scorer = AccuracyScorer(extract_answer=True)

# Run evaluation
evaluator = Evaluator(dataset=dataset, models=[model], scorers=[scorer])
results = evaluator.run()

# Print results
accuracy = results["model_results"]["gpt-4o-mini"]["scores"]["accuracy"]["mean"]
print(f"Accuracy: {accuracy:.4f}")
```

### Multi-Subject Evaluation

**Use Case**: Evaluate performance across multiple MMLU subjects

```python
from novaeval import Evaluator
from novaeval.datasets import MMLUDataset
from novaeval.models import OpenAIModel
from novaeval.scorers import AccuracyScorer

# Define subjects to evaluate
subjects = [
    "elementary_mathematics",
    "high_school_physics",
    "college_biology",
    "computer_security"
]

results_by_subject = {}

for subject in subjects:
    print(f"Evaluating {subject}...")

    dataset = MMLUDataset(subset=subject, num_samples=20)
    model = OpenAIModel(model_name="gpt-4o-mini", temperature=0.0)
    scorer = AccuracyScorer(extract_answer=True)

    evaluator = Evaluator(
        dataset=dataset,
        models=[model],
        scorers=[scorer],
        output_dir=f"./results/{subject}"
    )

    result = evaluator.run()
    accuracy = result["model_results"]["gpt-4o-mini"]["scores"]["accuracy"]["mean"]
    results_by_subject[subject] = accuracy

    print(f"  Accuracy: {accuracy:.4f}")

# Summary
print("\nSummary:")
for subject, accuracy in results_by_subject.items():
    print(f"  {subject}: {accuracy:.4f}")

average_accuracy = sum(results_by_subject.values()) / len(results_by_subject)
print(f"  Average: {average_accuracy:.4f}")
```

## Model Comparison

### OpenAI vs Anthropic Comparison

**Use Case**: Compare performance between different AI providers

```python
from novaeval import Evaluator
from novaeval.datasets import MMLUDataset
from novaeval.models import OpenAIModel, AnthropicModel
from novaeval.scorers import AccuracyScorer

# Setup dataset and scorer
dataset = MMLUDataset(subset="elementary_mathematics", num_samples=100)
scorer = AccuracyScorer(extract_answer=True)

# Define models to compare
models = [
    OpenAIModel(model_name="gpt-4o-mini", temperature=0.0),
    OpenAIModel(model_name="gpt-4-turbo", temperature=0.0),
    AnthropicModel(model_name="claude-3-haiku-20240307", temperature=0.0),
    AnthropicModel(model_name="claude-3-sonnet-20240229", temperature=0.0)
]

# Run comparison
evaluator = Evaluator(
    dataset=dataset,
    models=models,
    scorers=[scorer],
    output_dir="./model_comparison"
)

results = evaluator.run()

# Display comparison
print("Model Comparison Results:")
print("=" * 50)

for model_name, model_results in results["model_results"].items():
    accuracy = model_results["scores"]["accuracy"]["mean"]
    print(f"{model_name:30} | Accuracy: {accuracy:.4f}")

# Find best model
best_model = max(
    results["model_results"].items(),
    key=lambda x: x[1]["scores"]["accuracy"]["mean"]
)
print(f"\nBest performing model: {best_model[0]} ({best_model[1]['scores']['accuracy']['mean']:.4f})")
```

### Cost-Performance Analysis

**Use Case**: Analyze cost vs performance tradeoffs

```python
from novaeval import Evaluator
from novaeval.datasets import MMLUDataset
from novaeval.models import OpenAIModel
from novaeval.scorers import AccuracyScorer

# Models with different cost/performance profiles
models = [
    OpenAIModel(model_name="gpt-4o-mini", temperature=0.0, max_tokens=50),
    OpenAIModel(model_name="gpt-3.5-turbo", temperature=0.0, max_tokens=50),
    OpenAIModel(model_name="gpt-4-turbo", temperature=0.0, max_tokens=50),
    OpenAIModel(model_name="gpt-4", temperature=0.0, max_tokens=50),
]

dataset = MMLUDataset(subset="elementary_mathematics", num_samples=50)
scorer = AccuracyScorer(extract_answer=True)

evaluator = Evaluator(dataset=dataset, models=models, scorers=[scorer])
results = evaluator.run()

# Extract data for analysis
print("Cost-Performance Analysis:")
print("=" * 50)

for model in evaluator.models:
    model_name = model.model_name
    accuracy = results["model_results"][model.name]["scores"]["accuracy"]["mean"]
    estimated_cost = model.estimate_cost("Sample prompt", "Sample response")

    print(f"{model_name:20} | Accuracy: {accuracy:.4f} | Est. Cost: ${estimated_cost:.4f}")
```

## Custom Datasets

### Loading Custom JSON Data

**Use Case**: Evaluate on proprietary or custom datasets

```python
from novaeval import Evaluator
from novaeval.datasets import CustomDataset
from novaeval.models import OpenAIModel
from novaeval.scorers import AccuracyScorer

# Create custom dataset file (custom_qa.json)
import json

custom_data = [
    {
        "id": "q1",
        "input": "What is the capital of France?",
        "expected": "Paris",
        "metadata": {"category": "geography", "difficulty": "easy"}
    },
    {
        "id": "q2",
        "input": "What is 2 + 2?",
        "expected": "4",
        "metadata": {"category": "math", "difficulty": "easy"}
    },
    {
        "id": "q3",
        "input": "Who wrote Romeo and Juliet?",
        "expected": "Shakespeare",
        "metadata": {"category": "literature", "difficulty": "medium"}
    }
]

with open("custom_qa.json", "w") as f:
    json.dump(custom_data, f, indent=2)

# Load and evaluate
dataset = CustomDataset(path="custom_qa.json", format="json")
model = OpenAIModel(model_name="gpt-4o-mini", temperature=0.0)
scorer = AccuracyScorer()

evaluator = Evaluator(
    dataset=dataset,
    models=[model],
    scorers=[scorer],
    output_dir="./custom_results"
)

results = evaluator.run()
print(f"Custom dataset accuracy: {results['model_results']['gpt-4o-mini']['scores']['accuracy']['mean']:.4f}")
```

### HuggingFace Dataset Integration

**Use Case**: Evaluate on popular HuggingFace datasets

```python
from novaeval import Evaluator
from novaeval.datasets import HuggingFaceDataset
from novaeval.models import OpenAIModel
from novaeval.scorers import AccuracyScorer

# Load SQuAD dataset
dataset = HuggingFaceDataset(
    dataset_name="squad",
    subset="v1.1",
    split="validation",
    num_samples=50
)

model = OpenAIModel(model_name="gpt-4o-mini", temperature=0.0, max_tokens=100)
scorer = AccuracyScorer()

evaluator = Evaluator(
    dataset=dataset,
    models=[model],
    scorers=[scorer],
    output_dir="./squad_results"
)

results = evaluator.run()
print(f"SQuAD accuracy: {results['model_results']['gpt-4o-mini']['scores']['accuracy']['mean']:.4f}")
```

### Programmatic Dataset Creation

**Use Case**: Create datasets from databases or APIs

```python
from novaeval.datasets import BaseDataset
from novaeval import Evaluator
from novaeval.models import OpenAIModel
from novaeval.scorers import AccuracyScorer

class DatabaseDataset(BaseDataset):
    def __init__(self, **kwargs):
        super().__init__(name="database_dataset", **kwargs)

    def load_data(self):
        # Simulate loading from database
        return [
            {
                "id": "db_1",
                "input": "What is machine learning?",
                "expected": "A subset of artificial intelligence",
                "metadata": {"source": "database", "category": "AI"}
            },
            {
                "id": "db_2",
                "input": "What is Python?",
                "expected": "A programming language",
                "metadata": {"source": "database", "category": "programming"}
            }
        ]

# Use custom dataset
dataset = DatabaseDataset(num_samples=10)
model = OpenAIModel(model_name="gpt-4o-mini", temperature=0.0)
scorer = AccuracyScorer()

evaluator = Evaluator(dataset=dataset, models=[model], scorers=[scorer])
results = evaluator.run()
```

## Advanced Scoring

### Multiple Scoring Metrics

**Use Case**: Evaluate with multiple scoring approaches

```python
from novaeval import Evaluator
from novaeval.datasets import MMLUDataset
from novaeval.models import OpenAIModel
from novaeval.scorers import AccuracyScorer, ExactMatchScorer, F1Scorer

# Setup components
dataset = MMLUDataset(subset="elementary_mathematics", num_samples=50)
model = OpenAIModel(model_name="gpt-4o-mini", temperature=0.0)

# Multiple scorers
scorers = [
    AccuracyScorer(extract_answer=True),
    ExactMatchScorer(ignore_case=True),
    F1Scorer()
]

# Run evaluation
evaluator = Evaluator(
    dataset=dataset,
    models=[model],
    scorers=scorers,
    output_dir="./multi_score_results"
)

results = evaluator.run()

# Display all scores
print("Multiple Scoring Results:")
print("=" * 30)

for scorer_name, score_info in results["model_results"]["gpt-4o-mini"]["scores"].items():
    if isinstance(score_info, dict):
        mean_score = score_info.get("mean", 0)
        print(f"{scorer_name:15} | {mean_score:.4f}")
```

### Custom Scoring Logic

**Use Case**: Implement domain-specific scoring

```python
from novaeval.scorers import BaseScorer, ScoreResult
from novaeval import Evaluator
from novaeval.datasets import MMLUDataset
from novaeval.models import OpenAIModel

class MathScorer(BaseScorer):
    def __init__(self, **kwargs):
        super().__init__(name="math_scorer", **kwargs)

    def score(self, prediction, ground_truth, context=None):
        # Custom math scoring logic
        pred_clean = prediction.strip().lower()
        truth_clean = ground_truth.strip().lower()

        # Extract numeric values
        import re
        pred_nums = re.findall(r'\d+\.?\d*', pred_clean)
        truth_nums = re.findall(r'\d+\.?\d*', truth_clean)

        if pred_nums and truth_nums:
            try:
                pred_val = float(pred_nums[0])
                truth_val = float(truth_nums[0])

                # Score based on numeric closeness
                if pred_val == truth_val:
                    score = 1.0
                elif abs(pred_val - truth_val) / max(abs(truth_val), 1) < 0.1:
                    score = 0.8  # Close enough
                else:
                    score = 0.0

                return ScoreResult(
                    score=score,
                    details={
                        "predicted_value": pred_val,
                        "true_value": truth_val,
                        "difference": abs(pred_val - truth_val)
                    }
                )
            except ValueError:
                pass

        # Fallback to string comparison
        return ScoreResult(
            score=1.0 if pred_clean == truth_clean else 0.0,
            details={"method": "string_comparison"}
        )

# Use custom scorer
dataset = MMLUDataset(subset="elementary_mathematics", num_samples=20)
model = OpenAIModel(model_name="gpt-4o-mini", temperature=0.0)
scorer = MathScorer()

evaluator = Evaluator(dataset=dataset, models=[model], scorers=[scorer])
results = evaluator.run()
```

## Production Use Cases

### Batch Evaluation System

**Use Case**: Process large datasets efficiently

```python
from novaeval import Evaluator
from novaeval.datasets import MMLUDataset
from novaeval.models import OpenAIModel
from novaeval.scorers import AccuracyScorer
import time

def batch_evaluate(dataset_name, subjects, batch_size=50):
    """Run evaluation across multiple subjects in batches"""

    model = OpenAIModel(model_name="gpt-4o-mini", temperature=0.0)
    scorer = AccuracyScorer(extract_answer=True)

    all_results = {}

    for subject in subjects:
        print(f"Processing {subject}...")

        # Load dataset
        dataset = MMLUDataset(subset=subject, num_samples=batch_size)

        # Run evaluation
        evaluator = Evaluator(
            dataset=dataset,
            models=[model],
            scorers=[scorer],
            output_dir=f"./batch_results/{subject}",
            max_workers=2  # Control concurrency
        )

        start_time = time.time()
        results = evaluator.run()
        end_time = time.time()

        accuracy = results["model_results"]["gpt-4o-mini"]["scores"]["accuracy"]["mean"]
        all_results[subject] = {
            "accuracy": accuracy,
            "time": end_time - start_time,
            "samples": batch_size
        }

        print(f"  Accuracy: {accuracy:.4f} (took {end_time - start_time:.2f}s)")

    return all_results

# Run batch evaluation
subjects = ["elementary_mathematics", "high_school_physics", "college_biology"]
results = batch_evaluate("mmlu", subjects, batch_size=25)

# Summary report
print("\nBatch Evaluation Summary:")
print("=" * 50)
total_time = sum(r["time"] for r in results.values())
avg_accuracy = sum(r["accuracy"] for r in results.values()) / len(results)
print(f"Average accuracy: {avg_accuracy:.4f}")
print(f"Total time: {total_time:.2f}s")
```

### A/B Testing Framework

**Use Case**: Compare model versions or configurations

```python
from novaeval import Evaluator
from novaeval.datasets import MMLUDataset
from novaeval.models import OpenAIModel
from novaeval.scorers import AccuracyScorer

def ab_test_models(model_configs, dataset_config, test_name):
    """Run A/B test between different model configurations"""

    # Setup dataset
    dataset = MMLUDataset(**dataset_config)
    scorer = AccuracyScorer(extract_answer=True)

    # Create models
    models = []
    for config in model_configs:
        model = OpenAIModel(**config)
        models.append(model)

    # Run evaluation
    evaluator = Evaluator(
        dataset=dataset,
        models=models,
        scorers=[scorer],
        output_dir=f"./ab_test_{test_name}"
    )

    results = evaluator.run()

    # Analyze results
    print(f"\nA/B Test Results: {test_name}")
    print("=" * 50)

    for model_name, model_results in results["model_results"].items():
        accuracy = model_results["scores"]["accuracy"]["mean"]
        sample_count = model_results["scores"]["accuracy"]["count"]
        print(f"{model_name:20} | Accuracy: {accuracy:.4f} ({sample_count} samples)")

    return results

# Define A/B test configurations
model_configs = [
    {
        "model_name": "gpt-4o-mini",
        "temperature": 0.0,
        "max_tokens": 50
    },
    {
        "model_name": "gpt-4o-mini",
        "temperature": 0.3,
        "max_tokens": 50
    }
]

dataset_config = {
    "subset": "elementary_mathematics",
    "num_samples": 100
}

# Run A/B test
ab_results = ab_test_models(model_configs, dataset_config, "temperature_comparison")
```

## CI/CD Integration

### GitHub Actions Workflow

**Use Case**: Automated model evaluation in CI/CD

```yaml
# .github/workflows/model_evaluation.yml
name: Model Evaluation

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  evaluate:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v4

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'

    - name: Install dependencies
      run: |
        pip install novaeval

    - name: Run quick evaluation
      run: |
        python -c "
        from novaeval import Evaluator
        from novaeval.datasets import MMLUDataset
        from novaeval.models import OpenAIModel
        from novaeval.scorers import AccuracyScorer

        dataset = MMLUDataset(subset='elementary_mathematics', num_samples=5)
        model = OpenAIModel(model_name='gpt-4o-mini', temperature=0.0)
        scorer = AccuracyScorer(extract_answer=True)

        evaluator = Evaluator(dataset=dataset, models=[model], scorers=[scorer])
        results = evaluator.run()

        accuracy = results['model_results']['gpt-4o-mini']['scores']['accuracy']['mean']
        print(f'Accuracy: {accuracy:.4f}')

        # Fail if accuracy is too low
        if accuracy < 0.5:
            raise ValueError(f'Accuracy {accuracy:.4f} is below threshold 0.5')
        "
      env:
        OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}

    - name: Upload results
      uses: actions/upload-artifact@v4
      with:
        name: evaluation-results
        path: ./results/
```

### Configuration-Based CI Pipeline

**Use Case**: Flexible CI evaluation with YAML config

```python
# ci_evaluation.py
import os
import sys
from novaeval import Evaluator

def main():
    # Load configuration
    config_path = os.getenv("EVAL_CONFIG", "ci_config.yaml")

    try:
        evaluator = Evaluator.from_config(config_path)
        results = evaluator.run()

        # Extract key metrics
        for model_name, model_results in results["model_results"].items():
            accuracy = model_results["scores"]["accuracy"]["mean"]
            print(f"{model_name} accuracy: {accuracy:.4f}")

            # Set threshold check
            threshold = float(os.getenv("ACCURACY_THRESHOLD", "0.7"))
            if accuracy < threshold:
                print(f"❌ {model_name} failed threshold check ({accuracy:.4f} < {threshold})")
                sys.exit(1)
            else:
                print(f"✅ {model_name} passed threshold check ({accuracy:.4f} >= {threshold})")

        print("All models passed evaluation!")

    except Exception as e:
        print(f"Evaluation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
```

```yaml
# ci_config.yaml
name: "CI Model Evaluation"
description: "Automated evaluation for CI/CD pipeline"
version: "1.0"

models:
  - provider: "openai"
    model_name: "gpt-4o-mini"
    temperature: 0.0
    max_tokens: 50

datasets:
  - type: "mmlu"
    subset: "elementary_mathematics"
    num_samples: 20

scorers:
  - type: "accuracy"
    extract_answer: true

output:
  directory: "./ci_results"
  formats: ["json"]
```

### Docker-Based Evaluation

**Use Case**: Containerized evaluation for reproducibility

```dockerfile
# Dockerfile.evaluation
FROM python:3.9-slim

WORKDIR /app

# Install NovaEval
RUN pip install novaeval

# Copy evaluation script and config
COPY evaluation_script.py .
COPY evaluation_config.yaml .

# Run evaluation
CMD ["python", "evaluation_script.py"]
```

```python
# evaluation_script.py
from novaeval import Evaluator
import os

def main():
    # Load config from environment or file
    config_path = os.getenv("CONFIG_PATH", "evaluation_config.yaml")

    evaluator = Evaluator.from_config(config_path)
    results = evaluator.run()

    # Save results
    import json
    with open("/app/results/evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("Evaluation completed successfully!")

if __name__ == "__main__":
    main()
```

```bash
# Run containerized evaluation
docker build -f Dockerfile.evaluation -t novaeval-runner .
docker run -e OPENAI_API_KEY=$OPENAI_API_KEY -v $(pwd)/results:/app/results novaeval-runner
```

## Advanced Configuration Examples

### Multi-Model Benchmark Suite

**Use Case**: Comprehensive benchmarking across models and datasets

```python
from novaeval import Evaluator
from novaeval.datasets import MMLUDataset, HuggingFaceDataset
from novaeval.models import OpenAIModel, AnthropicModel
from novaeval.scorers import AccuracyScorer, ExactMatchScorer

def run_benchmark_suite():
    # Define test matrix
    models = [
        OpenAIModel(model_name="gpt-4o-mini", temperature=0.0),
        AnthropicModel(model_name="claude-3-haiku-20240307", temperature=0.0)
    ]

    datasets = [
        MMLUDataset(subset="elementary_mathematics", num_samples=25),
        MMLUDataset(subset="high_school_physics", num_samples=25),
        HuggingFaceDataset(dataset_name="squad", subset="v1.1", num_samples=25)
    ]

    scorers = [
        AccuracyScorer(extract_answer=True),
        ExactMatchScorer(ignore_case=True)
    ]

    # Run comprehensive evaluation
    all_results = {}

    for dataset in datasets:
        print(f"\n--- Evaluating {dataset.name} ---")

        evaluator = Evaluator(
            dataset=dataset,
            models=models,
            scorers=scorers,
            output_dir=f"./benchmark/{dataset.name}",
            max_workers=2
        )

        results = evaluator.run()
        all_results[dataset.name] = results

        # Print summary
        for model_name, model_results in results["model_results"].items():
            for scorer_name, score_info in model_results["scores"].items():
                if isinstance(score_info, dict):
                    mean_score = score_info.get("mean", 0)
                    print(f"  {model_name} - {scorer_name}: {mean_score:.4f}")

    return all_results

# Run benchmark
benchmark_results = run_benchmark_suite()
```

---

**Next Steps**:
- Explore the [API Reference](api-reference.md) for detailed documentation
- Check out the [User Guide](user-guide.md) for best practices
- Visit the [Getting Started](getting-started.md) guide for setup instructions
