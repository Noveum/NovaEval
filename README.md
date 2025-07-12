# NovaEval by Noveum.ai

[![CI](https://github.com/Noveum/NovaEval/actions/workflows/ci.yml/badge.svg)](https://github.com/Noveum/NovaEval/actions/workflows/ci.yml)
[![Release](https://github.com/Noveum/NovaEval/actions/workflows/release.yml/badge.svg)](https://github.com/Noveum/NovaEval/actions/workflows/release.yml)
[![codecov](https://codecov.io/gh/Noveum/NovaEval/branch/main/graph/badge.svg)](https://codecov.io/gh/Noveum/NovaEval)
[![PyPI version](https://badge.fury.io/py/novaeval.svg)](https://badge.fury.io/py/novaeval)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

A comprehensive, extensible AI model evaluation framework designed for production use. NovaEval provides a unified interface for evaluating language models across various datasets, metrics, and deployment scenarios.

## 🚀 Features

- **Multi-Model Support**: Evaluate models from OpenAI, Anthropic, AWS Bedrock, and custom providers
- **Extensible Scoring**: Built-in scorers for accuracy, semantic similarity, code evaluation, and custom metrics
- **Dataset Integration**: Support for MMLU, HuggingFace datasets, custom datasets, and more
- **Production Ready**: Docker support, Kubernetes deployment, and cloud integrations
- **Comprehensive Reporting**: Detailed evaluation reports, artifacts, and visualizations
- **Secure**: Built-in credential management and secret store integration
- **Scalable**: Designed for both local testing and large-scale production evaluations
- **Cross-Platform**: Tested on macOS, Linux, and Windows with comprehensive CI/CD

## 📦 Installation

### From PyPI (Recommended)

```bash
pip install novaeval
```

### From Source

```bash
git clone https://github.com/Noveum/NovaEval.git
cd NovaEval
pip install -e .
```

### Docker

```bash
docker pull noveum/novaeval:latest
```

## 🏃‍♂️ Quick Start

### Basic Evaluation

```python
from novaeval import Evaluator
from novaeval.datasets import MMLUDataset
from novaeval.models import OpenAIModel
from novaeval.scorers import AccuracyScorer

# Configure for cost-conscious evaluation
MAX_TOKENS = 100  # Adjust based on budget: 5-10 for answers, 100+ for reasoning

# Initialize components
dataset = MMLUDataset(
    subset="elementary_mathematics",  # Easier subset for demo
    num_samples=10,
    split="test"
)

model = OpenAIModel(
    model_name="gpt-4o-mini",  # Cost-effective model
    temperature=0.0,
    max_tokens=MAX_TOKENS
)

scorer = AccuracyScorer(extract_answer=True)

# Create and run evaluation
evaluator = Evaluator(
    dataset=dataset,
    models=[model],
    scorers=[scorer],
    output_dir="./results"
)

results = evaluator.run()

# Display detailed results
for model_name, model_results in results["model_results"].items():
    for scorer_name, score_info in model_results["scores"].items():
        if isinstance(score_info, dict):
            mean_score = score_info.get("mean", 0)
            count = score_info.get("count", 0)
            print(f"{scorer_name}: {mean_score:.4f} ({count} samples)")
```

### Configuration-Based Evaluation

```python
from novaeval import Evaluator

# Load configuration from YAML/JSON
evaluator = Evaluator.from_config("evaluation_config.yaml")
results = evaluator.run()
```

### Example Configuration

```yaml
# evaluation_config.yaml
dataset:
  type: "mmlu"
  subset: "abstract_algebra"
  num_samples: 500

models:
  - type: "openai"
    model_name: "gpt-4"
    temperature: 0.0
  - type: "anthropic"
    model_name: "claude-3-opus"
    temperature: 0.0

scorers:
  - type: "accuracy"
  - type: "semantic_similarity"
    threshold: 0.8

output:
  directory: "./results"
  formats: ["json", "csv", "html"]
  upload_to_s3: true
  s3_bucket: "my-eval-results"
```

## 🏗️ Architecture

NovaEval is built with extensibility and modularity in mind:

```
src/novaeval/
├── datasets/          # Dataset loaders and processors
├── evaluators/        # Core evaluation logic
├── integrations/      # External service integrations
├── models/           # Model interfaces and adapters
├── reporting/        # Report generation and visualization
├── scorers/          # Scoring mechanisms and metrics
└── utils/            # Utility functions and helpers
```

### Core Components

- **Datasets**: Standardized interface for loading evaluation datasets
- **Models**: Unified API for different AI model providers
- **Scorers**: Pluggable scoring mechanisms for various evaluation metrics
- **Evaluators**: Orchestrates the evaluation process
- **Reporting**: Generates comprehensive reports and artifacts
- **Integrations**: Handles external services (S3, credential stores, etc.)

## 📊 Supported Datasets

- **MMLU**: Massive Multitask Language Understanding
- **HuggingFace**: Any dataset from the HuggingFace Hub
- **Custom**: JSON, CSV, or programmatic dataset definitions
- **Code Evaluation**: Programming benchmarks and code generation tasks
- **Agent Traces**: Multi-turn conversation and agent evaluation

## 🤖 Supported Models

- **OpenAI**: GPT-3.5, GPT-4, and newer models
- **Anthropic**: Claude family models
- **AWS Bedrock**: Amazon's managed AI services
- **Noveum AI Gateway**: Integration with Noveum's model gateway
- **Custom**: Extensible interface for any API-based model

## 📏 Built-in Scorers

### Accuracy-Based
- **ExactMatch**: Exact string matching
- **Accuracy**: Classification accuracy
- **F1Score**: F1 score for classification tasks

### Semantic-Based
- **SemanticSimilarity**: Embedding-based similarity scoring
- **BERTScore**: BERT-based semantic evaluation
- **RougeScore**: ROUGE metrics for text generation

### Code-Specific
- **CodeExecution**: Execute and validate code outputs
- **SyntaxChecker**: Validate code syntax
- **TestCoverage**: Code coverage analysis

### Custom
- **LLMJudge**: Use another LLM as a judge
- **HumanEval**: Integration with human evaluation workflows

## 🚀 Deployment

### Local Development

```bash
# Install dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run example evaluation
python examples/basic_evaluation.py
```

### Docker

```bash
# Build image
docker build -t nova-eval .

# Run evaluation
docker run -v $(pwd)/config:/config -v $(pwd)/results:/results nova-eval --config /config/eval.yaml
```

### Kubernetes

```bash
# Deploy to Kubernetes
kubectl apply -f kubernetes/

# Check status
kubectl get pods -l app=nova-eval
```

## 🔧 Configuration

NovaEval supports configuration through:

- **YAML/JSON files**: Declarative configuration
- **Environment variables**: Runtime configuration
- **Python code**: Programmatic configuration
- **CLI arguments**: Command-line overrides

### Environment Variables

```bash
export NOVA_EVAL_OUTPUT_DIR="./results"
export NOVA_EVAL_LOG_LEVEL="INFO"
export OPENAI_API_KEY="your-api-key"
export AWS_ACCESS_KEY_ID="your-aws-key"
```

### CI/CD Integration

NovaEval includes optimized GitHub Actions workflows:
- **Unit tests** run on all PRs and pushes for quick feedback
- **Integration tests** run on main branch only to minimize API costs
- **Cross-platform testing** on macOS, Linux, and Windows

## 📈 Reporting and Artifacts

NovaEval generates comprehensive evaluation reports:

- **Summary Reports**: High-level metrics and insights
- **Detailed Results**: Per-sample predictions and scores
- **Visualizations**: Charts and graphs for result analysis
- **Artifacts**: Model outputs, intermediate results, and debug information
- **Export Formats**: JSON, CSV, HTML, PDF

### Example Report Structure

```
results/
├── summary.json              # High-level metrics
├── detailed_results.csv      # Per-sample results
├── artifacts/
│   ├── model_outputs/        # Raw model responses
│   ├── intermediate/         # Processing artifacts
│   └── debug/               # Debug information
├── visualizations/
│   ├── accuracy_by_category.png
│   ├── score_distribution.png
│   └── confusion_matrix.png
└── report.html              # Interactive HTML report
```

## 🔌 Extending NovaEval

### Custom Datasets

```python
from novaeval.datasets import BaseDataset

class MyCustomDataset(BaseDataset):
    def load_data(self):
        # Implement data loading logic
        return samples

    def get_sample(self, index):
        # Return individual sample
        return sample
```

### Custom Scorers

```python
from novaeval.scorers import BaseScorer

class MyCustomScorer(BaseScorer):
    def score(self, prediction, ground_truth, context=None):
        # Implement scoring logic
        return score
```

### Custom Models

```python
from novaeval.models import BaseModel

class MyCustomModel(BaseModel):
    def generate(self, prompt, **kwargs):
        # Implement model inference
        return response
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Clone repository
git clone https://github.com/Noveum/NovaEval.git
cd NovaEval

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run tests
pytest

# Run with coverage (23% overall, 90%+ for core modules)
pytest --cov=src/novaeval --cov-report=html
```

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Inspired by evaluation frameworks like DeepEval, Confident AI, and Braintrust
- Built with modern Python best practices and industry standards
- Designed for the AI evaluation community

## 📞 Support

- **Documentation**: [https://noveum.github.io/NovaEval](https://noveum.github.io/NovaEval)
- **Issues**: [GitHub Issues](https://github.com/Noveum/NovaEval/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Noveum/NovaEval/discussions)
- **Email**: support@noveum.ai

---

Made with ❤️ by the Noveum.ai team
