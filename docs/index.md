---
layout: default
title: Home
nav_order: 1
---

# NovaEval Documentation

[![CI](https://github.com/Noveum/NovaEval/actions/workflows/ci.yml/badge.svg)](https://github.com/Noveum/NovaEval/actions/workflows/ci.yml)
[![Release](https://github.com/Noveum/NovaEval/actions/workflows/release.yml/badge.svg)](https://github.com/Noveum/NovaEval/actions/workflows/release.yml)
[![codecov](https://codecov.io/gh/Noveum/NovaEval/branch/main/graph/badge.svg)](https://codecov.io/gh/Noveum/NovaEval)
[![PyPI version](https://badge.fury.io/py/novaeval.svg)](https://badge.fury.io/py/novaeval)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

<div class="hero-banner">
  <div class="hero-content">
    <div class="hero-text">
      <h2>Comprehensive AI Model Evaluation Framework</h2>
      <p>A unified interface for evaluating language models across various datasets, metrics, and deployment scenarios.</p>
    </div>
    <div class="hero-brand">
      <img src="/NovaEval/assets/images/noveum-logo.png" alt="Noveum AI" class="noveum-logo">
      <p class="brand-text">Built by <strong>Noveum AI</strong></p>
    </div>
  </div>
</div>

## ✨ What is NovaEval?

NovaEval is an open-source framework that simplifies the complex process of evaluating AI models. Whether you're a researcher comparing model performance, a developer ensuring quality in production, or a data scientist benchmarking custom models, NovaEval provides the tools you need.

### 🎯 Key Features

- **🤖 Multi-Model Support**: Evaluate models from OpenAI and Anthropic
- **📊 Extensible Scoring**: Built-in scorers for accuracy, exact match, and F1 score
- **📚 Dataset Integration**: Support for MMLU, HuggingFace datasets, and custom datasets
- **🏭 Production Ready**: Docker support, Kubernetes deployment, and comprehensive CI/CD
- **📈 Comprehensive Reporting**: Detailed evaluation reports with JSON, CSV, and HTML outputs
- **⚡ Scalable**: Designed for both local testing and large-scale production evaluations
- **🔧 Cross-Platform**: Tested on macOS, Linux, and Windows with comprehensive CI/CD

## 🚀 Quick Start

### Installation

```bash
# Install from PyPI
pip install novaeval

# Or install from source
git clone https://github.com/Noveum/NovaEval.git
cd NovaEval
pip install -e .
```

### Your First Evaluation

```python
from novaeval import Evaluator
from novaeval.datasets import MMLUDataset
from novaeval.models import OpenAIModel
from novaeval.scorers import AccuracyScorer

# Setup components
dataset = MMLUDataset(subset="elementary_mathematics", num_samples=10)
model = OpenAIModel(model_name="gpt-4o-mini", temperature=0.0)
scorer = AccuracyScorer()

# Run evaluation
evaluator = Evaluator(dataset=dataset, models=[model], scorers=[scorer])
results = evaluator.run()

print(f"Accuracy: {results['model_results']['gpt-4o-mini']['scores']['accuracy']['mean']:.4f}")
```

## 🏗️ Architecture Overview

NovaEval is built with modularity and extensibility in mind:

```
NovaEval Framework
├── 📁 Datasets        # Load and process evaluation data
├── 🤖 Models          # AI model integrations
├── 🎯 Scorers         # Evaluation metrics and scoring
├── 🔄 Evaluators      # Orchestrate evaluation workflows
├── 📊 Reporting       # Generate comprehensive reports
└── 🔗 Integrations    # External service connections
```

## 🌟 Why Choose NovaEval?

### For Researchers
- **Standardized Benchmarks**: Access to popular datasets like MMLU, HuggingFace Hub
- **Reproducible Results**: Consistent evaluation protocols and seeding
- **Comprehensive Metrics**: Multiple scoring mechanisms for different use cases

### For Developers
- **CI/CD Integration**: YAML-based configuration for automated testing
- **Production Monitoring**: Track model performance over time
- **Cost Optimization**: Built-in cost tracking and optimization features

### For Data Scientists
- **Custom Datasets**: Easy integration of proprietary evaluation data
- **Flexible Scoring**: Create custom metrics for specific use cases
- **Rich Visualizations**: Detailed reports and performance insights

## 📚 Documentation Structure

<div class="doc-grid">
  <div class="doc-card">
    <h3><a href="/getting-started/">🚀 Getting Started</a></h3>
    <p>Installation, setup, and your first evaluation</p>
  </div>

  <div class="doc-card">
    <h3><a href="/user-guide/">📖 User Guide</a></h3>
    <p>Comprehensive guide to using NovaEval</p>
  </div>

  <div class="doc-card">
    <h3><a href="/api-reference/">🔧 API Reference</a></h3>
    <p>Detailed API documentation</p>
  </div>

  <div class="doc-card">
    <h3><a href="/examples/">💡 Examples</a></h3>
    <p>Real-world use cases and examples</p>
  </div>
</div>

## 🤝 Community & Support

- **📖 Documentation**: [Full documentation](https://noveum.github.io/NovaEval)
- **💬 GitHub Issues**: [Report bugs or request features](https://github.com/Noveum/NovaEval/issues)
- **🗣️ Discussions**: [Join the community](https://github.com/Noveum/NovaEval/discussions)
- **📧 Support**: [team@noveum.ai](mailto:team@noveum.ai)
- **🌐 Noveum.ai**: [Visit our main site](https://noveum.ai)

## 📊 Project Status

- **🧪 Tests**: 203 tests (177 unit + 26 integration)
- **📈 Coverage**: 23% overall, 90%+ for core modules
- **🔄 CI/CD**: Automated testing and deployment
- **📦 Releases**: Available on PyPI and Docker Hub

## 🔗 Related Projects

Explore more from the [Noveum AI](https://noveum.ai) ecosystem:

- **[Noveum Platform](https://noveum.ai/en/docs)**: Enterprise AI evaluation and monitoring
- **[Noveum Gateway](https://noveum.ai)**: Unified API for multiple AI providers
- **[Noveum Analytics](https://noveum.ai)**: Advanced AI model analytics and insights

---

<div class="footer-cta">
  <p><strong>Ready to get started?</strong> <a href="/getting-started/">Install NovaEval</a> and run your first evaluation in minutes.</p>
  <p>Built with ❤️ by the <a href="https://noveum.ai" target="_blank">Noveum.ai</a> team</p>
</div>
