---
layout: default
title: User Guide
nav_order: 3
---

# User Guide

This comprehensive guide covers advanced features, best practices, and real-world use cases for NovaEval.

## Table of Contents

1. [Core Concepts](#core-concepts)
2. [Datasets](#datasets)
3. [Models](#models)
4. [Scorers](#scorers)
5. [Configuration](#configuration)
6. [Best Practices](#best-practices)
7. [Advanced Features](#advanced-features)

## Core Concepts

### The Evaluation Pipeline

NovaEval follows a simple but powerful evaluation pipeline:

```
Dataset → Model → Scorer → Results
```

1. **Dataset**: Provides the evaluation data (questions, tasks, etc.)
2. **Model**: The AI model being evaluated
3. **Scorer**: Evaluates the model's output against expected results
4. **Results**: Aggregated metrics and detailed output

### Key Components

```python
from novaeval import Evaluator
from novaeval.datasets import BaseDataset
from novaeval.models import BaseModel
from novaeval.scorers import BaseScorer

# The main orchestrator
evaluator = Evaluator(
    dataset=dataset,    # What to evaluate on
    models=[model],     # What to evaluate
    scorers=[scorer],   # How to evaluate
    output_dir="./results"  # Where to save results
)
```

## Datasets

### Built-in Datasets

#### MMLU (Massive Multitask Language Understanding)

```python
from novaeval.datasets import MMLUDataset

# Load specific subject
dataset = MMLUDataset(
    subset="abstract_algebra",
    num_samples=100,
    split="test"
)

# Load all subjects
dataset = MMLUDataset(
    subset="all",
    num_samples=1000,
    split="test"
)

# Available subjects
subjects = [
    "abstract_algebra", "anatomy", "astronomy", "business_ethics",
    "clinical_knowledge", "college_biology", "college_chemistry",
    "college_computer_science", "college_mathematics", "college_medicine",
    "college_physics", "computer_security", "conceptual_physics",
    "econometrics", "electrical_engineering", "elementary_mathematics",
    "formal_logic", "global_facts", "high_school_biology",
    "high_school_chemistry", "high_school_computer_science",
    "high_school_european_history", "high_school_geography",
    "high_school_government_and_politics", "high_school_macroeconomics",
    "high_school_mathematics", "high_school_microeconomics",
    "high_school_physics", "high_school_psychology", "high_school_statistics",
    "high_school_us_history", "high_school_world_history", "human_aging",
    "human_sexuality", "international_law", "jurisprudence", "logical_fallacies",
    "machine_learning", "management", "marketing", "medical_genetics",
    "miscellaneous", "moral_disputes", "moral_scenarios", "nutrition",
    "philosophy", "prehistory", "professional_accounting", "professional_law",
    "professional_medicine", "professional_psychology", "public_relations",
    "security_studies", "sociology", "us_foreign_policy", "virology", "world_religions"
]
```

#### HuggingFace Datasets

```python
from novaeval.datasets import HuggingFaceDataset

# Load any HuggingFace dataset
dataset = HuggingFaceDataset(
    dataset_name="squad",
    subset="v1.1",
    split="validation",
    num_samples=100
)

# Common datasets
datasets = [
    "squad",           # Reading comprehension
    "hellaswag",       # Commonsense reasoning
    "arc",             # Science questions
    "winogrande",      # Commonsense reasoning
    "truthful_qa",     # Truthfulness
    "gsm8k",           # Math word problems
]
```

### Custom Datasets

#### From JSON/JSONL Files

```python
from novaeval.datasets import CustomDataset

# JSONL format
dataset = CustomDataset(
    path="./my_dataset.jsonl",
    format="jsonl"
)

# JSON format
dataset = CustomDataset(
    path="./my_dataset.json",
    format="json"
)
```

Required format for custom datasets:
```json
{
  "id": "sample_1",
  "input": "What is the capital of France?",
  "expected": "Paris",
  "metadata": {
    "difficulty": "easy",
    "category": "geography"
  }
}
```

#### Programmatic Dataset Creation

```python
from novaeval.datasets import BaseDataset

class MyCustomDataset(BaseDataset):
    def __init__(self, **kwargs):
        super().__init__(name="my_custom_dataset", **kwargs)

    def load_data(self):
        # Load from database, API, etc.
        return [
            {
                "id": "1",
                "input": "Question 1",
                "expected": "Answer 1",
                "metadata": {"category": "math"}
            },
            # ... more samples
        ]
```

## Models

### OpenAI Models

```python
from novaeval.models import OpenAIModel

# GPT-4 models
model = OpenAIModel(
    model_name="gpt-4-turbo",
    temperature=0.0,
    max_tokens=1000,
    timeout=30
)

# GPT-3.5 models
model = OpenAIModel(
    model_name="gpt-3.5-turbo",
    temperature=0.7,
    max_tokens=500
)

# GPT-4o models (cost-effective)
model = OpenAIModel(
    model_name="gpt-4o-mini",
    temperature=0.0,
    max_tokens=100
)
```

### Anthropic Models

```python
from novaeval.models import AnthropicModel

# Claude 3 models
model = AnthropicModel(
    model_name="claude-3-sonnet-20240229",
    temperature=0.0,
    max_tokens=1000
)

# Claude 3 Haiku (cost-effective)
model = AnthropicModel(
    model_name="claude-3-haiku-20240307",
    temperature=0.0,
    max_tokens=500
)
```

### Custom Models

```python
from novaeval.models import BaseModel

class MyCustomModel(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(name="my_custom_model", **kwargs)

    def generate(self, prompt, **kwargs):
        # Implement your model inference logic
        response = self.my_model_api_call(prompt)
        return response

    def estimate_cost(self, prompt, response=""):
        # Implement cost estimation
        return 0.001  # Example cost
```

## Scorers

### Built-in Scorers

#### Accuracy Scorer

```python
from novaeval.scorers import AccuracyScorer

# Basic accuracy scoring
scorer = AccuracyScorer()

# With answer extraction for multiple choice
scorer = AccuracyScorer(extract_answer=True)

# With custom patterns
scorer = AccuracyScorer(
    extract_answer=True,
    answer_patterns=[r'\b([A-D])\b', r'answer is ([A-D])']
)
```

#### Exact Match Scorer

```python
from novaeval.scorers import ExactMatchScorer

# Exact string matching
scorer = ExactMatchScorer()

# Case-insensitive matching
scorer = ExactMatchScorer(ignore_case=True)

# With normalization
scorer = ExactMatchScorer(
    ignore_case=True,
    strip_whitespace=True,
    normalize_punct=True
)
```

#### F1 Scorer

```python
from novaeval.scorers import F1Scorer

# Token-level F1 scoring
scorer = F1Scorer()

# With custom tokenizer
scorer = F1Scorer(tokenizer=my_tokenizer)
```

### Custom Scorers

```python
from novaeval.scorers import BaseScorer, ScoreResult

class MyCustomScorer(BaseScorer):
    def __init__(self, **kwargs):
        super().__init__(name="my_custom_scorer", **kwargs)

    def score(self, prediction, ground_truth, context=None):
        # Implement your scoring logic
        score = self.calculate_score(prediction, ground_truth)

        return ScoreResult(
            score=score,
            details={
                "prediction": prediction,
                "ground_truth": ground_truth,
                "explanation": "Custom scoring explanation"
            }
        )
```

## Configuration

### YAML Configuration

```yaml
# evaluation_config.yaml
name: "Production Evaluation"
description: "Evaluate multiple models on MMLU"
version: "1.0"

models:
  - provider: "openai"
    model_name: "gpt-4o-mini"
    temperature: 0.0
    max_tokens: 100
    timeout: 30

  - provider: "anthropic"
    model_name: "claude-3-haiku-20240307"
    temperature: 0.0
    max_tokens: 100
    timeout: 30

datasets:
  - type: "mmlu"
    subset: "elementary_mathematics"
    split: "test"
    num_samples: 100

  - type: "huggingface"
    dataset_name: "squad"
    subset: "v1.1"
    split: "validation"
    num_samples: 50

scorers:
  - type: "accuracy"
    extract_answer: true

  - type: "exact_match"
    ignore_case: true

  - type: "f1"

output:
  directory: "./results"
  formats: ["json", "csv", "html"]

evaluation:
  max_workers: 4
  batch_size: 1
  timeout: 300
```

### Programmatic Configuration

```python
from novaeval import Evaluator
from novaeval.config import EvaluationConfig

# Create configuration object
config = EvaluationConfig(
    name="My Evaluation",
    models=[
        {"provider": "openai", "model_name": "gpt-4o-mini"},
        {"provider": "anthropic", "model_name": "claude-3-haiku-20240307"}
    ],
    datasets=[
        {"type": "mmlu", "subset": "elementary_mathematics", "num_samples": 50}
    ],
    scorers=[
        {"type": "accuracy", "extract_answer": True}
    ],
    output={"directory": "./results", "formats": ["json", "csv"]}
)

# Create evaluator from config
evaluator = Evaluator.from_config(config)
results = evaluator.run()
```

## Best Practices

### Cost Management

```python
# Use cost-effective models for development
model = OpenAIModel(
    model_name="gpt-4o-mini",  # More cost-effective than gpt-4
    max_tokens=50,             # Limit output length
    temperature=0.0            # Deterministic results
)

# Use small samples for testing
dataset = MMLUDataset(
    subset="elementary_mathematics",
    num_samples=10,  # Small sample for quick testing
    split="test"
)

# Monitor costs
evaluator = Evaluator(dataset=dataset, models=[model], scorers=[scorer])
results = evaluator.run()

# Check estimated costs
total_cost = sum(model.estimated_cost for model in evaluator.models)
print(f"Total estimated cost: ${total_cost:.4f}")
```

### Performance Optimization

```python
# Parallel processing
evaluator = Evaluator(
    dataset=dataset,
    models=[model],
    scorers=[scorer],
    max_workers=4,      # Parallel model calls
    batch_size=10       # Process in batches
)

# Timeout configuration
model = OpenAIModel(
    model_name="gpt-4o-mini",
    timeout=30  # 30 second timeout
)
```

### Error Handling

```python
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

# Run evaluation with error handling
try:
    evaluator = Evaluator(dataset=dataset, models=[model], scorers=[scorer])
    results = evaluator.run()

    # Check for errors
    for model_name, model_results in results["model_results"].items():
        if model_results["errors"]:
            print(f"Errors in {model_name}: {len(model_results['errors'])}")
            for error in model_results["errors"]:
                print(f"  - {error}")

except Exception as e:
    print(f"Evaluation failed: {e}")
```

## Advanced Features

### Multiple Dataset Evaluation

```python
# Evaluate across multiple datasets
datasets = [
    MMLUDataset(subset="elementary_mathematics", num_samples=50),
    MMLUDataset(subset="high_school_physics", num_samples=50),
    HuggingFaceDataset(dataset_name="squad", num_samples=50)
]

all_results = {}
for dataset in datasets:
    evaluator = Evaluator(
        dataset=dataset,
        models=[model],
        scorers=[scorer],
        output_dir=f"./results/{dataset.name}"
    )

    results = evaluator.run()
    all_results[dataset.name] = results
```

### Model Comparison

```python
# Compare multiple models
models = [
    OpenAIModel(model_name="gpt-4o-mini", temperature=0.0),
    OpenAIModel(model_name="gpt-3.5-turbo", temperature=0.0),
    AnthropicModel(model_name="claude-3-haiku-20240307", temperature=0.0)
]

evaluator = Evaluator(
    dataset=dataset,
    models=models,
    scorers=[scorer],
    output_dir="./model_comparison"
)

results = evaluator.run()

# Compare results
for model_name, model_results in results["model_results"].items():
    accuracy = model_results["scores"]["accuracy"]["mean"]
    print(f"{model_name}: {accuracy:.4f}")
```

### Custom Evaluation Pipeline

```python
from novaeval.evaluators import BaseEvaluator

class MyCustomEvaluator(BaseEvaluator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def evaluate_sample(self, sample, model, scorers):
        # Custom evaluation logic
        prediction = model.generate(sample["input"])

        # Custom processing
        processed_prediction = self.custom_processing(prediction)

        # Score with custom logic
        scores = {}
        for scorer in scorers:
            score = scorer.score(processed_prediction, sample["expected"])
            scores[scorer.name] = score

        return scores

    def custom_processing(self, prediction):
        # Implement custom processing
        return prediction.strip().lower()
```

### Batch Processing

```python
# Process large datasets efficiently
dataset = MMLUDataset(subset="all", num_samples=1000)

# Split into batches
batch_size = 100
num_batches = len(dataset) // batch_size

for i in range(num_batches):
    start_idx = i * batch_size
    end_idx = min((i + 1) * batch_size, len(dataset))

    batch_dataset = dataset.slice(start_idx, end_idx)

    evaluator = Evaluator(
        dataset=batch_dataset,
        models=[model],
        scorers=[scorer],
        output_dir=f"./results/batch_{i}"
    )

    results = evaluator.run()
    print(f"Batch {i+1}/{num_batches} completed")
```

## Troubleshooting

### Common Issues

1. **Rate Limiting**: Reduce `max_workers` or add delays
2. **Memory Issues**: Reduce `batch_size` or `num_samples`
3. **Timeout Errors**: Increase model timeout or reduce complexity
4. **API Errors**: Check API keys and quotas

### Debug Mode

```python
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

# Run with detailed logging
evaluator = Evaluator(
    dataset=dataset,
    models=[model],
    scorers=[scorer],
    output_dir="./debug_results"
)

results = evaluator.run()
```

---

**Next Steps**: Explore the [Examples](examples.md) section for real-world use cases and advanced scenarios.
