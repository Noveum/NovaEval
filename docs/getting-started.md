---
layout: default
title: Getting Started
nav_order: 2
---

# Getting Started with NovaEval

Welcome to NovaEval! This guide will help you get up and running with your first AI model evaluation in just a few minutes.

## 📋 Prerequisites

Before starting, ensure you have:

- **Python 3.8+** installed on your system
- **pip** package manager
- **API keys** for the AI models you want to evaluate (OpenAI, Anthropic)
- **Git** (optional, for development installation)

## 🚀 Installation

### Option 1: Install from PyPI (Recommended)

```bash
pip install novaeval
```

### Option 2: Install from Source

```bash
git clone https://github.com/Noveum/NovaEval.git
cd NovaEval
pip install -e .
```

### Option 3: Docker Installation

```bash
# Pull the latest image
docker pull noveum/novaeval:latest

# Or from GitHub Container Registry
docker pull ghcr.io/noveum/novaeval:latest
```

## 🔑 Setting Up API Keys

NovaEval supports OpenAI and Anthropic models. Set up your API keys as environment variables:

```bash
# OpenAI
export OPENAI_API_KEY="sk-your-openai-key-here"

# Anthropic
export ANTHROPIC_API_KEY="sk-ant-your-anthropic-key-here"
```

### Using .env Files

Create a `.env` file in your project directory:

```bash
# .env file
OPENAI_API_KEY=sk-your-openai-key-here
ANTHROPIC_API_KEY=sk-ant-your-anthropic-key-here
```

## 🏃‍♂️ Your First Evaluation

Let's start with a simple evaluation using the MMLU dataset and OpenAI's GPT model:

### Step 1: Create Your First Script

Create a file called `first_evaluation.py`:

```python
from novaeval import Evaluator
from novaeval.datasets import MMLUDataset
from novaeval.models import OpenAIModel
from novaeval.scorers import AccuracyScorer

# Configuration for cost-conscious evaluation
MAX_TOKENS = 50  # Limit tokens to control costs
NUM_SAMPLES = 5  # Small sample for testing

def main():
    print("🚀 Starting your first NovaEval evaluation!")

    # Step 1: Setup the dataset
    print("📚 Loading dataset...")
    dataset = MMLUDataset(
        subset="elementary_mathematics",  # Easier subset for demo
        num_samples=NUM_SAMPLES,
        split="test"
    )

    # Step 2: Setup the model
    print("🤖 Initializing model...")
    model = OpenAIModel(
        model_name="gpt-4o-mini",  # Cost-effective model
        temperature=0.0,           # Deterministic results
        max_tokens=MAX_TOKENS
    )

    # Step 3: Setup the scorer
    print("🎯 Setting up scorer...")
    scorer = AccuracyScorer(extract_answer=True)

    # Step 4: Create and run evaluator
    print("⚡ Running evaluation...")
    evaluator = Evaluator(
        dataset=dataset,
        models=[model],
        scorers=[scorer],
        output_dir="./results"
    )

    # Run the evaluation
    results = evaluator.run()

    # Step 5: Display results
    print("\n🎉 Evaluation completed!")
    print("=" * 50)

    for model_name, model_results in results["model_results"].items():
        print(f"\n📊 Results for {model_name}:")
        for scorer_name, score_info in model_results["scores"].items():
            if isinstance(score_info, dict):
                mean_score = score_info.get("mean", 0)
                count = score_info.get("count", 0)
                print(f"  {scorer_name}: {mean_score:.4f} ({count} samples)")

    print(f"\n📁 Results saved to: {evaluator.output_dir}")

if __name__ == "__main__":
    main()
```

### Step 2: Run Your Evaluation

```bash
python first_evaluation.py
```

Expected output:
```
🚀 Starting your first NovaEval evaluation!
📚 Loading dataset...
🤖 Initializing model...
🎯 Setting up scorer...
⚡ Running evaluation...

🎉 Evaluation completed!
==================================================

📊 Results for gpt-4o-mini:
  accuracy: 0.8000 (5 samples)

📁 Results saved to: ./results
```

## 🔧 Configuration-Based Evaluation

For more complex evaluations, use YAML configuration files:

### Step 1: Create Configuration File

Create `evaluation_config.yaml`:

```yaml
# evaluation_config.yaml
name: "My First Evaluation"
description: "Testing NovaEval with multiple models and scorers"
version: "1.0"

models:
  - provider: "openai"
    model_name: "gpt-4o-mini"
    temperature: 0.0
    max_tokens: 100
  - provider: "anthropic"
    model_name: "claude-3-haiku-20240307"
    temperature: 0.0
    max_tokens: 100

datasets:
  - type: "mmlu"
    subset: "elementary_mathematics"
    split: "test"
    limit: 10

scorers:
  - type: "accuracy"
    extract_answer: true

output:
  formats: ["json", "csv"]
  directory: "./config_results"
```

### Step 2: Run Configuration-Based Evaluation

```python
from novaeval import Evaluator

# Load and run from configuration
evaluator = Evaluator.from_config("evaluation_config.yaml")
results = evaluator.run()

print("Configuration-based evaluation completed!")
```

## 🛠️ Development Setup

If you're planning to contribute or modify NovaEval:

### Step 1: Clone and Setup

```bash
git clone https://github.com/Noveum/NovaEval.git
cd NovaEval

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -e ".[dev]"
```

### Step 2: Install Pre-commit Hooks

```bash
pre-commit install
```

### Step 3: Run Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/novaeval --cov-report=html

# Run only unit tests
pytest tests/unit/

# Run only integration tests
pytest tests/integration/
```

## 📊 Understanding Results

NovaEval generates comprehensive evaluation results:

### Result Structure

```python
{
    "evaluation_id": "eval_20240101_123456",
    "timestamp": "2024-01-01T12:34:56Z",
    "model_results": {
        "gpt-4o-mini": {
            "scores": {
                "accuracy": {
                    "mean": 0.8500,
                    "std": 0.0577,
                    "count": 20,
                    "values": [0.8, 0.9, 0.85, ...]
                }
            },
            "errors": [],
            "metadata": {
                "model_name": "gpt-4o-mini",
                "total_samples": 20,
                "successful_samples": 20
            }
        }
    },
    "dataset_info": {
        "name": "mmlu",
        "subset": "elementary_mathematics",
        "split": "test",
        "num_samples": 20
    }
}
```

### Output Files

NovaEval creates several output files in your results directory:

```
results/
├── evaluation_results.json    # Complete results
├── summary.json              # High-level summary
├── detailed_results.csv      # Per-sample results
├── model_outputs/            # Raw model responses
└── artifacts/               # Additional files
```

## 🔍 Troubleshooting

### Common Issues

#### API Key Not Found

```bash
Error: OpenAI API key not found. Please set OPENAI_API_KEY environment variable.
```

**Solution**: Set your API key:
```bash
export OPENAI_API_KEY="sk-your-key-here"
```

#### Rate Limiting

```bash
Error: Rate limit exceeded. Please try again later.
```

**Solution**: Add delays between requests or reduce `max_workers`:
```python
evaluator = Evaluator(
    dataset=dataset,
    models=[model],
    scorers=[scorer],
    max_workers=1  # Reduce concurrent requests
)
```

#### Out of Memory

```bash
Error: Out of memory during evaluation.
```

**Solution**: Reduce `num_samples` or `batch_size`:
```python
dataset = MMLUDataset(num_samples=10)  # Smaller dataset
evaluator = Evaluator(batch_size=1)    # Smaller batches
```

### Getting Help

- **Documentation**: [https://noveum.github.io/NovaEval](https://noveum.github.io/NovaEval)
- **GitHub Issues**: [Report bugs](https://github.com/Noveum/NovaEval/issues)
- **GitHub Discussions**: [Ask questions](https://github.com/Noveum/NovaEval/discussions)
- **Email**: [team@noveum.ai](mailto:team@noveum.ai)

## 🎯 Next Steps

Now that you've completed your first evaluation, explore:

1. **[User Guide](user-guide.md)**: Learn about advanced features
2. **[Examples](examples.md)**: See real-world use cases
3. **[API Reference](api-reference.md)**: Detailed API documentation
4. **[Contributing](contributing.md)**: Contribute to the project

---

**Ready to dive deeper?** Check out the [User Guide](user-guide.md) for advanced features and best practices.
