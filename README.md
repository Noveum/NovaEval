# NovaEval by Noveum.ai

[![CI](https://github.com/Noveum/NovaEval/actions/workflows/ci.yml/badge.svg)](https://github.com/Noveum/NovaEval/actions/workflows/ci.yml)
[![Release](https://github.com/Noveum/NovaEval/actions/workflows/release.yml/badge.svg)](https://github.com/Noveum/NovaEval/actions/workflows/release.yml)
[![codecov](https://codecov.io/gh/Noveum/NovaEval/branch/main/graph/badge.svg)](https://codecov.io/gh/Noveum/NovaEval)
[![PyPI version](https://badge.fury.io/py/novaeval.svg)](https://badge.fury.io/py/novaeval)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

A comprehensive, extensible AI model evaluation framework designed for production use. NovaEval provides a unified interface for evaluating language models across various datasets, metrics, and deployment scenarios.

## 🚧 Development Status

> **⚠️ ACTIVE DEVELOPMENT - NOT PRODUCTION READY**
>
> NovaEval is currently in active development and **not recommended for production use**. We are actively working on improving stability, adding features, and expanding test coverage. APIs may change without notice.
>
> **We're looking for contributors!** See the [Contributing](#-contributing) section below for ways to help.

## 🤝 We Need Your Help!

NovaEval is an open-source project that thrives on community contributions. Whether you're a seasoned developer or just getting started, there are many ways to contribute:

### 🎯 High-Priority Contribution Areas

We're actively looking for contributors in these key areas:

- **🧪 Unit Tests**: Help us improve our test coverage (currently 23% overall, 90%+ for core modules)
- **📚 Examples**: Create real-world evaluation examples and use cases
- **📝 Guides & Notebooks**: Write evaluation guides and interactive Jupyter notebooks
- **📖 Documentation**: Improve API documentation and user guides
- **🔍 RAG Metrics**: Add more metrics specifically for Retrieval-Augmented Generation evaluation
- **🤖 Agent Evaluation**: Build frameworks for evaluating AI agents and multi-turn conversations

### 🚀 Getting Started as a Contributor

1. **Start Small**: Pick up issues labeled `good first issue` or `help wanted`
2. **Join Discussions**: Share your ideas in [GitHub Discussions](https://github.com/Noveum/NovaEval/discussions)
3. **Review Code**: Help review pull requests and provide feedback
4. **Report Issues**: Found a bug? Report it in [GitHub Issues](https://github.com/Noveum/NovaEval/issues)
5. **Spread the Word**: Star the repository and share with your network

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

### Command Line Interface

NovaEval provides a comprehensive CLI for running evaluations:

```bash
# Run evaluation from configuration file
novaeval run config.yaml

# Quick evaluation with minimal setup
novaeval quick -d mmlu -m gpt-4 -s accuracy

# List available datasets, models, and scorers
novaeval list-datasets
novaeval list-models
novaeval list-scorers

# Generate sample configuration
novaeval generate-config sample-config.yaml
```

📖 **[Complete CLI Reference](docs/cli-reference.md)** - Detailed documentation for all CLI commands and options

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

We welcome contributions! NovaEval is actively seeking contributors to help build a robust AI evaluation framework. Please see our [Contributing Guide](CONTRIBUTING.md) for detailed guidelines.

### 🎯 Priority Contribution Areas

As mentioned in the [We Need Your Help](#-we-need-your-help) section, we're particularly looking for help with:

1. **Unit Tests** - Expand test coverage beyond the current 23%
2. **Examples** - Real-world evaluation scenarios and use cases
3. **Guides & Notebooks** - Interactive evaluation tutorials
4. **Documentation** - API docs, user guides, and tutorials
5. **RAG Metrics** - Specialized metrics for retrieval-augmented generation
6. **Agent Evaluation** - Frameworks for multi-turn and agent-based evaluations

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

# Run with coverage
pytest --cov=src/novaeval --cov-report=html
```

### 🏗️ Contribution Workflow

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Make** your changes following our coding standards
4. **Add** tests for your changes
5. **Commit** your changes (`git commit -m 'Add amazing feature'`)
6. **Push** to the branch (`git push origin feature/amazing-feature`)
7. **Open** a Pull Request

### 📋 Contribution Guidelines

- **Code Quality**: Follow PEP 8 and use the provided pre-commit hooks
- **Testing**: Add unit tests for new features and bug fixes
- **Documentation**: Update documentation for API changes
- **Commit Messages**: Use conventional commit format
- **Issues**: Reference relevant issues in your PR description

### 🎉 Recognition

Contributors will be:
- Listed in our contributors page
- Mentioned in release notes for significant contributions
- Invited to join our contributor Discord community

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
