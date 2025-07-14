---
layout: default
title: Home
nav_order: 1
---

# NovaEval Documentation

[![GitHub Repo](https://img.shields.io/badge/GitHub-NovaEval-blue?logo=github)](https://github.com/Noveum/NovaEval)
[![GitHub Stars](https://img.shields.io/github/stars/Noveum/NovaEval?style=social)](https://github.com/Noveum/NovaEval/stargazers)
[![GitHub Forks](https://img.shields.io/github/forks/Noveum/NovaEval?style=social)](https://github.com/Noveum/NovaEval/network/members)
[![GitHub Issues](https://img.shields.io/github/issues/Noveum/NovaEval)](https://github.com/Noveum/NovaEval/issues)
[![GitHub Watchers](https://img.shields.io/github/watchers/Noveum/NovaEval?style=social)](https://github.com/Noveum/NovaEval/watchers)

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
      <div class="hero-buttons">
        <a href="https://github.com/Noveum/NovaEval" class="github-button" target="_blank" rel="noopener">
          <svg class="github-icon" viewBox="0 0 16 16" width="16" height="16">
            <path fill="currentColor" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"/>
          </svg>
          View on GitHub
        </a>
        <a href="/getting-started/" class="primary-button">Get Started →</a>
      </div>
    </div>
    <div class="hero-brand">
      <img src="/assets/images/noveum-logo.png" alt="Noveum AI" class="noveum-logo">
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
