# NovaEval by Noveum.ai

[![CI](https://github.com/Noveum/NovaEval/actions/workflows/ci.yml/badge.svg)](https://github.com/Noveum/NovaEval/actions/workflows/ci.yml)
[![Release](https://github.com/Noveum/NovaEval/actions/workflows/release.yml/badge.svg)](https://github.com/Noveum/NovaEval/actions/workflows/release.yml)
[![codecov](https://codecov.io/gh/Noveum/NovaEval/branch/main/graph/badge.svg)](https://codecov.io/gh/Noveum/NovaEval)
[![PyPI version](https://badge.fury.io/py/novaeval.svg)](https://badge.fury.io/py/novaeval)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

A comprehensive, extensible AI model evaluation framework designed for production use. NovaEval provides a unified interface for evaluating language models across various datasets, metrics, and deployment scenarios.

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

## 🌐 HTTP API

NovaEval provides a FastAPI-based HTTP API for programmatic access to evaluation capabilities. This enables easy integration with web applications, microservices, and CI/CD pipelines.

### Quick API Start

```bash
# Install API dependencies
pip install -e ".[api]"

# Run the API server
uvicorn app.main:app --host 0.0.0.0 --port 8000

# Access interactive documentation
open http://localhost:8000/docs
```

### Core API Endpoints

- **Health Check**: `GET /health` - Service health status
- **Component Discovery**: `GET /api/v1/components/` - List available models, datasets, scorers
- **Model Operations**: `POST /api/v1/models/{model}/predict` - Generate predictions
- **Dataset Operations**: `POST /api/v1/datasets/{dataset}/load` - Load and query datasets
- **Scorer Operations**: `POST /api/v1/scorers/{scorer}/score` - Score predictions
- **Evaluation Jobs**: `POST /api/v1/evaluations/submit` - Submit async evaluation jobs

### Example API Usage

```python
import requests

# Submit evaluation via API
evaluation_config = {
    "name": "api_evaluation",
    "models": [{"provider": "openai", "identifier": "gpt-3.5-turbo"}],
    "datasets": [{"name": "mmlu", "split": "test", "limit": 10}],
    "scorers": [{"name": "accuracy"}]
}

response = requests.post(
    "http://localhost:8000/api/v1/evaluations/submit",
    json=evaluation_config
)

task_id = response.json()["task_id"]
print(f"Evaluation started: {task_id}")
```

### Deployment Options

- **Docker**: `docker run -p 8000:8000 novaeval-api:latest`
- **Kubernetes**: Full manifests provided in `kubernetes/`
- **Cloud Platforms**: Supports AWS, GCP, Azure with environment variable configuration

📖 **[Complete API Documentation](app/README.md)** - Detailed API reference, examples, and deployment guide

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

## 💡 Advanced Evaluation Examples

NovaEval supports sophisticated evaluation scenarios across different AI system types. Here are comprehensive examples showing how to evaluate LLMs, RAG systems, and AI agents with detailed traces and tool calls.

### 🧠 LLM Evaluation with Conversation Traces

```python
from novaeval import Evaluator
from novaeval.models import OpenAIModel
from novaeval.scorers.conversational import (
    ConversationalMetricsScorer,
    KnowledgeRetentionScorer,
    Conversation,
    ConversationTurn
)
from novaeval.datasets import CustomDataset

# Initialize LLM model for evaluation
model = OpenAIModel(
    model_name="gpt-4o-mini",
    temperature=0.0,
    max_tokens=150
)

# Create conversation trace for evaluation
conversation_trace = Conversation(
    turns=[
        ConversationTurn(speaker="user", message="Hi, my name is Sarah and I'm a software engineer from Seattle."),
        ConversationTurn(speaker="assistant", message="Hello Sarah! It's nice to meet you. How can I help you today?"),
        ConversationTurn(speaker="user", message="I'm working on a Python project and need help with error handling."),
        ConversationTurn(speaker="assistant", message="I'd be happy to help with Python error handling! What specific aspect would you like to know about?"),
        ConversationTurn(speaker="user", message="What's my name again?"),
        ConversationTurn(speaker="assistant", message="Your name is Sarah. You mentioned you're a software engineer from Seattle working on a Python project with error handling. How can I help you with that?")
    ],
    context="You are a helpful programming assistant",
    topic="Python programming help"
)

# Prepare evaluation data with conversation context
evaluation_data = [
    {
        "input": "What's my name again?",
        "expected_output": "Your name is Sarah.",
        "context": {
            "conversation": conversation_trace,
            "evaluation_type": "knowledge_retention"
        }
    }
]

# Create custom dataset
dataset = CustomDataset(data=evaluation_data)

# Initialize conversational metrics scorer
conv_scorer = ConversationalMetricsScorer(
    model=model,
    include_knowledge_retention=True,
    include_relevancy=True,
    include_completeness=True,
    include_role_adherence=True,
    window_size=5
)

# Run evaluation
evaluator = Evaluator(
    dataset=dataset,
    models=[model],
    scorers=[conv_scorer],
    output_dir="./conversation_results"
)

results = evaluator.run()

# Display detailed conversation analysis
for model_name, model_results in results["model_results"].items():
    print(f"\n=== Conversational Evaluation Results for {model_name} ===")
    for scorer_name, score_info in model_results["scores"].items():
        if isinstance(score_info, dict):
            individual_scores = score_info.get("individual_scores", {})
            overall_score = score_info.get("mean", 0)
            print(f"\nOverall Score: {overall_score:.3f}")
            print("Individual Metrics:")
            for metric, score in individual_scores.items():
                print(f"  • {metric.replace('_', ' ').title()}: {score:.3f}")
```

### 🔍 RAG System Evaluation with Context

```python
from novaeval import Evaluator
from novaeval.models import OpenAIModel
from novaeval.scorers.rag import (
    RAGASScorer,
    AnswerRelevancyScorer,
    FaithfulnessScorer,
    ContextualPrecisionScorer
)
from novaeval.datasets import CustomDataset

# Initialize model for RAG evaluation
model = OpenAIModel(
    model_name="gpt-4o-mini",
    temperature=0.0,
    max_tokens=200
)

# RAG evaluation dataset with retrieved context
rag_evaluation_data = [
    {
        "input": "What are the key benefits of renewable energy?",
        "prediction": "Renewable energy offers several key benefits: it reduces greenhouse gas emissions, provides energy independence, creates jobs in green technology sectors, and offers long-term cost savings. Solar and wind power are becoming increasingly cost-competitive with fossil fuels.",
        "expected_output": "Renewable energy reduces emissions, provides energy security, and creates economic opportunities.",
        "context": {
            "context": """
            Renewable energy sources like solar, wind, and hydroelectric power offer numerous advantages:

            Environmental Benefits:
            - Significantly lower greenhouse gas emissions compared to fossil fuels
            - Reduced air and water pollution
            - Minimal environmental impact during operation

            Economic Benefits:
            - Job creation in manufacturing, installation, and maintenance
            - Long-term cost savings due to free fuel sources
            - Energy price stability and independence from volatile fossil fuel markets

            Technical Benefits:
            - Distributed generation capabilities
            - Scalable from residential to utility scale
            - Increasingly competitive costs with traditional energy sources
            """,
            "retrieved_documents": [
                {"id": "doc1", "title": "Renewable Energy Economics", "relevance": 0.95},
                {"id": "doc2", "title": "Environmental Impact of Clean Energy", "relevance": 0.88}
            ]
        }
    },
    {
        "input": "How does solar panel efficiency work?",
        "prediction": "Solar panel efficiency measures how much sunlight is converted to electricity. Modern panels achieve 15-22% efficiency. Higher efficiency means more power from smaller installations. Factors affecting efficiency include temperature, shading, and panel orientation.",
        "expected_output": "Solar efficiency measures sunlight-to-electricity conversion, typically 15-22% for modern panels.",
        "context": {
            "context": """
            Solar Panel Efficiency Overview:

            Definition: Solar panel efficiency refers to the percentage of sunlight that gets converted into usable electricity.

            Current Technology:
            - Standard silicon panels: 15-17% efficiency
            - High-efficiency panels: 18-22% efficiency
            - Laboratory records: Over 26% efficiency

            Factors Affecting Efficiency:
            - Panel temperature (lower is better)
            - Shading and obstructions
            - Panel orientation and tilt angle
            - Age and degradation of panels
            - Weather conditions and sunlight intensity
            """,
            "retrieved_documents": [
                {"id": "doc3", "title": "Solar Technology Guide", "relevance": 0.92}
            ]
        }
    }
]

# Create dataset
rag_dataset = CustomDataset(data=rag_evaluation_data)

# Initialize RAG scorers
ragas_scorer = RAGASScorer(
    model=model,
    threshold=0.7,
    weights={
        "answer_relevancy": 0.25,
        "faithfulness": 0.35,
        "contextual_precision": 0.20,
        "contextual_recall": 0.20
    }
)

answer_relevancy = AnswerRelevancyScorer(model=model, threshold=0.8)
faithfulness = FaithfulnessScorer(model=model, threshold=0.8)

# Run comprehensive RAG evaluation
evaluator = Evaluator(
    dataset=rag_dataset,
    models=[model],
    scorers=[ragas_scorer, answer_relevancy, faithfulness],
    output_dir="./rag_evaluation_results"
)

results = evaluator.run()

# Display detailed RAG analysis
for model_name, model_results in results["model_results"].items():
    print(f"\n=== RAG Evaluation Results for {model_name} ===")
    for scorer_name, score_info in model_results["scores"].items():
        if isinstance(score_info, dict):
            mean_score = score_info.get("mean", 0)
            print(f"\n{scorer_name}: {mean_score:.3f}")

            # Show individual RAGAS component scores if available
            if "individual_scores" in score_info:
                individual = score_info["individual_scores"]
                if isinstance(individual, dict):
                    print("  Component Scores:")
                    for component, score in individual.items():
                        if component != "overall":
                            print(f"    • {component.replace('_', ' ').title()}: {score:.3f}")
```

### 🤖 Agent Evaluation with Tool Calls

```python
from novaeval.agents.agent_data import AgentData, ToolCall, ToolCallResult
from novaeval.scorers.agent_scorers import AgentScorers
from novaeval.models import OpenAIModel
import json

# Initialize model for agent evaluation
model = OpenAIModel(
    model_name="gpt-4o-mini",
    temperature=0.0,
    max_tokens=300
)

# Create comprehensive agent data with tool calls and trace
agent_evaluation_data = AgentData(
    agent_role="Research Assistant",
    agent_task="Find and summarize information about renewable energy adoption rates in Europe",
    system_prompt="You are a research assistant specialized in gathering and analyzing data about energy markets. Use available tools to find accurate, up-to-date information.",

    # Available tools for the agent
    tools_available=[
        {
            "name": "web_search",
            "description": "Search the web for current information",
            "parameters": {"query": "string", "max_results": "integer"}
        },
        {
            "name": "data_analysis",
            "description": "Analyze numerical data and create summaries",
            "parameters": {"data": "array", "analysis_type": "string"}
        },
        {
            "name": "report_generator",
            "description": "Generate formatted reports from research data",
            "parameters": {"content": "string", "format": "string"}
        }
    ],

    # Actual tool calls made by the agent
    tool_calls=[
        ToolCall(
            call_id="call_1",
            tool_name="web_search",
            parameters={"query": "Europe renewable energy adoption rates 2024", "max_results": 5}
        ),
        ToolCall(
            call_id="call_2",
            tool_name="data_analysis",
            parameters={"data": [45, 52, 38, 67, 41], "analysis_type": "summary_statistics"}
        ),
        ToolCall(
            call_id="call_3",
            tool_name="report_generator",
            parameters={"content": "European renewable energy summary", "format": "markdown"}
        )
    ],

    # Expected optimal tool call for comparison
    expected_tool_call=ToolCall(
        call_id="expected_1",
        tool_name="web_search",
        parameters={"query": "European renewable energy statistics 2024", "max_results": 10}
    ),

    # Parameters that were passed to tools
    parameters_passed={
        "call_1": {"query": "Europe renewable energy adoption rates 2024", "max_results": 5},
        "call_2": {"data": [45, 52, 38, 67, 41], "analysis_type": "summary_statistics"},
        "call_3": {"content": "European renewable energy summary", "format": "markdown"}
    },

    # Results from tool calls
    tool_call_results=[
        ToolCallResult(
            call_id="call_1",
            result={"status": "success", "results": ["EU renewable energy reached 48% in 2024", "Wind power leads adoption", "Solar growth accelerating"]},
            success=True
        ),
        ToolCallResult(
            call_id="call_2",
            result={"mean": 48.6, "median": 45, "std_dev": 11.2},
            success=True
        ),
        ToolCallResult(
            call_id="call_3",
            result={"report_url": "https://example.com/report", "format": "markdown", "length": 1250},
            success=True
        )
    ],

    # Agent's final response
    agent_response="Based on my research, European renewable energy adoption has reached significant milestones in 2024. The data shows an average adoption rate of 48.6% across major EU countries, with some nations exceeding 67%. Wind power continues to lead the renewable sector, while solar energy is experiencing accelerated growth. I've compiled this information into a comprehensive markdown report for your review.",

    # Complete interaction trace
    trace=[
        {"timestamp": "2024-01-15T10:00:00Z", "type": "task_received", "content": "Find and summarize information about renewable energy adoption rates in Europe"},
        {"timestamp": "2024-01-15T10:00:30Z", "type": "tool_call", "tool": "web_search", "parameters": {"query": "Europe renewable energy adoption rates 2024", "max_results": 5}},
        {"timestamp": "2024-01-15T10:01:15Z", "type": "tool_result", "call_id": "call_1", "status": "success", "data": "Retrieved 5 relevant articles"},
        {"timestamp": "2024-01-15T10:01:45Z", "type": "tool_call", "tool": "data_analysis", "parameters": {"data": [45, 52, 38, 67, 41], "analysis_type": "summary_statistics"}},
        {"timestamp": "2024-01-15T10:02:10Z", "type": "tool_result", "call_id": "call_2", "status": "success", "data": "Statistical analysis completed"},
        {"timestamp": "2024-01-15T10:02:30Z", "type": "tool_call", "tool": "report_generator", "parameters": {"content": "European renewable energy summary", "format": "markdown"}},
        {"timestamp": "2024-01-15T10:03:00Z", "type": "tool_result", "call_id": "call_3", "status": "success", "data": "Report generated successfully"},
        {"timestamp": "2024-01-15T10:03:30Z", "type": "response", "content": "Research completed and report generated"},
        {"timestamp": "2024-01-15T10:03:35Z", "type": "task_completed", "success": True}
    ],

    # Agent has completed the task
    agent_exit=True
)

# Initialize agent scorers
agent_scorers = AgentScorers(model=model)

# Run comprehensive agent evaluation
print("=== Comprehensive Agent Evaluation ===\n")

# Individual scoring functions
print("1. Tool Relevancy Assessment:")
tool_relevancy_results = agent_scorers.score_tool_relevancy(agent_evaluation_data)
if isinstance(tool_relevancy_results, list):
    for i, result in enumerate(tool_relevancy_results):
        print(f"   Tool Call {i+1}: Score {result.score:.2f} - {result.reasoning[:100]}...")

print("\n2. Tool Correctness Evaluation:")
tool_correctness_results = agent_scorers.score_tool_correctness(agent_evaluation_data)
if isinstance(tool_correctness_results, list):
    for i, result in enumerate(tool_correctness_results):
        print(f"   Tool Call {i+1}: Score {result.score:.2f} - {result.reasoning[:100]}...")

print("\n3. Parameter Correctness Analysis:")
param_correctness_results = agent_scorers.score_parameter_correctness(agent_evaluation_data)
if isinstance(param_correctness_results, list):
    for i, result in enumerate(param_correctness_results):
        print(f"   Tool Call {i+1}: Score {result.score:.2f} - {result.reasoning[:100]}...")

print("\n4. Task Progression Evaluation:")
task_progression_result = agent_scorers.score_task_progression(agent_evaluation_data)
if hasattr(task_progression_result, 'score'):
    print(f"   Task: {task_progression_result.original_task}")
    print(f"   Score: {task_progression_result.score:.2f}")
    print(f"   Reasoning: {task_progression_result.reasoning[:150]}...")

print("\n5. Goal Achievement Assessment:")
goal_achievement_result = agent_scorers.score_goal_achievement(agent_evaluation_data)
if hasattr(goal_achievement_result, 'score'):
    print(f"   Original Task: {goal_achievement_result.original_task}")
    print(f"   Achievement Score: {goal_achievement_result.score:.2f}")
    print(f"   Reasoning: {goal_achievement_result.reasoning[:150]}...")

print("\n6. Conversation Coherence Analysis:")
coherence_result = agent_scorers.score_conversation_coherence(agent_evaluation_data)
if hasattr(coherence_result, 'score'):
    print(f"   Coherence Score: {coherence_result.score:.2f}")
    print(f"   Reasoning: {coherence_result.reasoning[:150]}...")

print("\n7. Complete Agent Assessment:")
all_results = agent_scorers.score_all(agent_evaluation_data)

# Summary of all evaluations
print("\n=== Agent Evaluation Summary ===")
for metric, result in all_results.items():
    if hasattr(result, 'score'):
        print(f"{metric.replace('_', ' ').title()}: {result.score:.2f}")
    elif isinstance(result, list) and result and hasattr(result[0], 'score'):
        avg_score = sum(r.score for r in result) / len(result)
        print(f"{metric.replace('_', ' ').title()}: {avg_score:.2f} (avg of {len(result)} tool calls)")
    elif isinstance(result, dict) and 'error' in result:
        print(f"{metric.replace('_', ' ').title()}: Error - {result['error']}")
```

### 🔄 Batch Evaluation Across Multiple Systems

```python
from novaeval import Evaluator
from novaeval.datasets import CustomDataset
from novaeval.models import OpenAIModel, AnthropicModel
from novaeval.scorers import AccuracyScorer
from novaeval.scorers.conversational import ConversationalMetricsScorer
from novaeval.scorers.rag import RAGASScorer

# Initialize multiple models for comparison
models = [
    OpenAIModel(model_name="gpt-4o-mini", temperature=0.0),
    OpenAIModel(model_name="gpt-3.5-turbo", temperature=0.0),
    AnthropicModel(model_name="claude-3-haiku", temperature=0.0)
]

# Multi-domain evaluation dataset
mixed_evaluation_data = [
    # Classification task
    {
        "input": "Classify this sentiment: 'I love this product, it works perfectly!'",
        "expected_output": "positive",
        "task_type": "classification"
    },
    # RAG task
    {
        "input": "What is machine learning?",
        "expected_output": "Machine learning is a subset of AI that enables computers to learn without explicit programming.",
        "context": {
            "context": "Machine learning is a method of data analysis that automates analytical model building. It is a branch of artificial intelligence based on the idea that systems can learn from data, identify patterns and make decisions with minimal human intervention."
        },
        "task_type": "rag"
    },
    # Conversational task
    {
        "input": "What's the weather like?",
        "expected_output": "I don't have access to current weather data, but I can help you find weather information.",
        "context": {
            "conversation": Conversation(
                turns=[
                    ConversationTurn(speaker="user", message="Hello, I'm planning a trip"),
                    ConversationTurn(speaker="assistant", message="That sounds exciting! How can I help with your trip planning?"),
                    ConversationTurn(speaker="user", message="What's the weather like?")
                ]
            )
        },
        "task_type": "conversation"
    }
]

# Create dataset
multi_dataset = CustomDataset(data=mixed_evaluation_data)

# Initialize appropriate scorers
accuracy_scorer = AccuracyScorer(extract_answer=True)
conv_scorer = ConversationalMetricsScorer(model=models[0], include_relevancy=True)
ragas_scorer = RAGASScorer(model=models[0])

# Run comprehensive evaluation
evaluator = Evaluator(
    dataset=multi_dataset,
    models=models,
    scorers=[accuracy_scorer, conv_scorer, ragas_scorer],
    output_dir="./comprehensive_evaluation"
)

results = evaluator.run()

# Model comparison analysis
print("=== Multi-Model Evaluation Results ===\n")
for model_name, model_results in results["model_results"].items():
    print(f"Model: {model_name}")
    print("-" * 40)
    for scorer_name, score_info in model_results["scores"].items():
        if isinstance(score_info, dict):
            mean_score = score_info.get("mean", 0)
            count = score_info.get("count", 0)
            print(f"  {scorer_name}: {mean_score:.3f} ({count} samples)")
    print()
```

## 📏 Built-in Scorers & Metrics

NovaEval provides a comprehensive suite of scorers organized by evaluation domain. All scorers implement the `BaseScorer` interface and support both synchronous and asynchronous evaluation.

### 🎯 Accuracy & Classification Metrics

#### **ExactMatchScorer**
- **Purpose**: Performs exact string matching between prediction and ground truth
- **Features**:
  - Case-sensitive/insensitive matching options
  - Whitespace normalization and stripping
  - Perfect for classification tasks with exact expected outputs
- **Use Cases**: Multiple choice questions, command validation, exact answer matching
- **Configuration**: `case_sensitive`, `strip_whitespace`, `normalize_whitespace`

#### **AccuracyScorer**
- **Purpose**: Advanced classification accuracy with answer extraction capabilities
- **Features**:
  - Intelligent answer extraction from model responses using multiple regex patterns
  - Support for MMLU-style multiple choice questions (A, B, C, D)
  - Letter-to-choice text conversion
  - Robust parsing of various answer formats
- **Use Cases**: MMLU evaluations, multiple choice tests, classification benchmarks
- **Configuration**: `extract_answer`, `answer_pattern`, `choices`

#### **F1Scorer**
- **Purpose**: Token-level F1 score for partial matching scenarios
- **Features**:
  - Calculates precision, recall, and F1 score
  - Configurable tokenization (word-level or character-level)
  - Case-sensitive/insensitive options
- **Use Cases**: Question answering, text summarization, partial credit evaluation
- **Returns**: Dictionary with `precision`, `recall`, `f1`, and `score` values

### 💬 Conversational AI Metrics

#### **KnowledgeRetentionScorer**
- **Purpose**: Evaluates if the LLM retains information provided by users throughout conversations
- **Features**:
  - Sophisticated knowledge extraction from conversation history
  - Sliding window approach for relevant context (configurable window size)
  - Detects when LLM asks for previously provided information
  - Tracks knowledge items with confidence scores
- **Use Cases**: Chatbots, virtual assistants, multi-turn conversations
- **Requirements**: LLM model for knowledge extraction, conversation context

#### **ConversationRelevancyScorer**
- **Purpose**: Measures response relevance to recent conversation context
- **Features**:
  - Sliding window context analysis
  - LLM-based relevance assessment (1-5 scale)
  - Context coherence evaluation
  - Conversation flow maintenance tracking
- **Use Cases**: Dialogue systems, context-aware assistants
- **Configuration**: `window_size` for context scope

#### **ConversationCompletenessScorer**
- **Purpose**: Assesses whether user intentions and requests are fully addressed
- **Features**:
  - Extracts user intentions from conversation history
  - Evaluates fulfillment level of each intention
  - Comprehensive coverage analysis
  - Outcome-based evaluation
- **Use Cases**: Customer service bots, task-oriented dialogue systems

#### **RoleAdherenceScorer**
- **Purpose**: Evaluates consistency with assigned persona or role
- **Features**:
  - Role consistency tracking throughout conversations
  - Character maintenance assessment
  - Persona adherence evaluation
  - Customizable role expectations
- **Use Cases**: Character-based chatbots, role-playing AI, specialized assistants
- **Configuration**: `expected_role` parameter

#### **ConversationalMetricsScorer**
- **Purpose**: Comprehensive conversational evaluation combining multiple metrics
- **Features**:
  - Combines knowledge retention, relevancy, completeness, and role adherence
  - Configurable metric inclusion/exclusion
  - Weighted aggregation of individual scores
  - Detailed per-metric breakdown
- **Use Cases**: Holistic conversation quality assessment
- **Configuration**: Enable/disable individual metrics, window sizes, role expectations

### 🔍 RAG (Retrieval-Augmented Generation) Metrics

#### **AnswerRelevancyScorer**
- **Purpose**: Evaluates how relevant answers are to given questions
- **Features**:
  - Generates questions from answers using LLM
  - Semantic similarity comparison using embeddings (SentenceTransformers)
  - Multiple question generation for robust evaluation
  - Cosine similarity scoring
- **Use Cases**: RAG systems, Q&A applications, knowledge bases
- **Configuration**: `threshold`, `embedding_model`

#### **FaithfulnessScorer**
- **Purpose**: Measures if responses are faithful to provided context without hallucinations
- **Features**:
  - Extracts factual claims from responses
  - Verifies each claim against source context
  - Three-tier verification: SUPPORTED/PARTIALLY_SUPPORTED/NOT_SUPPORTED
  - Detailed claim-by-claim analysis
- **Use Cases**: RAG faithfulness, fact-checking, source attribution
- **Configuration**: `threshold` for pass/fail determination

#### **ContextualPrecisionScorer**
- **Purpose**: Evaluates precision of retrieved context relevance
- **Features**:
  - Splits context into chunks for granular analysis
  - Relevance scoring per chunk (1-5 scale)
  - Intelligent context segmentation
  - Average relevance calculation
- **Use Cases**: Retrieval system evaluation, context quality assessment
- **Requirements**: Context must be provided for evaluation

#### **ContextualRecallScorer**
- **Purpose**: Measures if all necessary information for answering is present in context
- **Features**:
  - Extracts key information from expected outputs
  - Checks presence of each key fact in provided context
  - Three-tier presence detection: PRESENT/PARTIALLY_PRESENT/NOT_PRESENT
  - Comprehensive information coverage analysis
- **Use Cases**: Retrieval completeness, context sufficiency evaluation
- **Requirements**: Both context and expected output required

#### **RAGASScorer**
- **Purpose**: Composite RAGAS methodology combining multiple RAG metrics
- **Features**:
  - Integrates Answer Relevancy, Faithfulness, Contextual Precision, and Contextual Recall
  - Configurable weighted aggregation
  - Parallel execution of individual metrics
  - Comprehensive RAG pipeline evaluation
- **Use Cases**: Complete RAG system assessment, benchmark evaluation
- **Configuration**: Custom weights for each metric component

### 🤖 LLM-as-Judge Metrics

#### **GEvalScorer**
- **Purpose**: Uses LLMs with chain-of-thought reasoning for custom evaluation criteria
- **Features**:
  - Based on G-Eval research paper methodology
  - Configurable evaluation criteria and steps
  - Chain-of-thought reasoning support
  - Multiple evaluation iterations for consistency
  - Custom score ranges and thresholds
- **Use Cases**: Custom evaluation criteria, human-aligned assessment, complex judgments
- **Configuration**: `criteria`, `use_cot`, `num_iterations`, `threshold`

#### **CommonGEvalCriteria** (Predefined Criteria)
- **Correctness**: Factual accuracy and completeness assessment
- **Relevance**: Topic adherence and query alignment evaluation
- **Coherence**: Logical flow and structural consistency analysis
- **Helpfulness**: Practical value and actionability assessment

#### **PanelOfJudgesScorer**
- **Purpose**: Multi-LLM evaluation with diverse perspectives and aggregation
- **Features**:
  - Multiple LLM judges with individual weights and specialties
  - Configurable aggregation methods (mean, median, weighted, consensus, etc.)
  - Consensus requirement and threshold controls
  - Parallel judge evaluation for efficiency
  - Detailed individual and aggregate reasoning
- **Use Cases**: High-stakes evaluation, bias reduction, robust assessment
- **Configuration**: Judge models, weights, specialties, aggregation method

#### **SpecializedPanelScorer** (Panel Configurations)
- **Diverse Panel**: Different models with varied specialties (accuracy, clarity, completeness)
- **Consensus Panel**: High-consensus requirement for agreement-based decisions
- **Weighted Expert Panel**: Domain experts with expertise-based weighting

### 🎭 Agent Evaluation Metrics

#### **Tool Relevancy Scoring**
- **Purpose**: Evaluates appropriateness of tool calls given available tools
- **Features**: Compares selected tools against available tool catalog
- **Use Cases**: Agent tool selection assessment, action planning evaluation

#### **Tool Correctness Scoring**
- **Purpose**: Compares actual tool calls against expected tool calls
- **Features**: Detailed tool call comparison and correctness assessment
- **Use Cases**: Agent behavior validation, expected action verification

#### **Parameter Correctness Scoring**
- **Purpose**: Evaluates correctness of parameters passed to tool calls
- **Features**: Parameter validation against tool call results and expectations
- **Use Cases**: Tool usage quality, parameter selection accuracy

#### **Task Progression Scoring**
- **Purpose**: Measures agent progress toward assigned tasks
- **Features**: Analyzes task completion status and advancement quality
- **Use Cases**: Agent effectiveness measurement, task completion tracking

#### **Context Relevancy Scoring**
- **Purpose**: Assesses response appropriateness given agent's role and task
- **Features**: Role-task-response alignment evaluation
- **Use Cases**: Agent behavior consistency, contextual appropriateness

#### **Role Adherence Scoring**
- **Purpose**: Evaluates consistency with assigned agent role across actions
- **Features**: Comprehensive role consistency across tool calls and responses
- **Use Cases**: Agent persona maintenance, role-based behavior validation

#### **Goal Achievement Scoring**
- **Purpose**: Measures overall goal accomplishment using complete interaction traces
- **Features**: End-to-end goal evaluation with G-Eval methodology
- **Use Cases**: Agent effectiveness assessment, outcome-based evaluation

#### **Conversation Coherence Scoring**
- **Purpose**: Evaluates logical flow and context maintenance in agent conversations
- **Features**: Conversational coherence and context tracking analysis
- **Use Cases**: Agent dialogue quality, conversation flow assessment

#### **AgentScorers** (Convenience Class)
- **Purpose**: Unified interface for all agent evaluation metrics
- **Features**: Single class providing access to all agent scorers with consistent LLM model
- **Methods**: Individual scoring methods plus `score_all()` for comprehensive evaluation

### 🔧 Advanced Features

#### **BaseScorer Interface**
All scorers inherit from `BaseScorer` providing:
- **Statistics Tracking**: Automatic score history and statistics
- **Batch Processing**: Efficient batch scoring capabilities
- **Input Validation**: Robust input validation and error handling
- **Configuration Support**: Flexible configuration from dictionaries
- **Metadata Reporting**: Detailed scoring metadata and information

#### **ScoreResult Model**
Comprehensive scoring results include:
- **Numerical Score**: Primary evaluation score
- **Pass/Fail Status**: Threshold-based binary result
- **Detailed Reasoning**: Human-readable evaluation explanation
- **Rich Metadata**: Additional context and scoring details

### 📊 Usage Examples

```python
# Basic accuracy scoring
scorer = AccuracyScorer(extract_answer=True)
score = scorer.score("The answer is B", "B")

# Advanced conversational evaluation
conv_scorer = ConversationalMetricsScorer(
    model=your_llm_model,
    include_knowledge_retention=True,
    include_relevancy=True,
    window_size=10
)
result = await conv_scorer.evaluate(input_text, output_text, context=conv_context)

# RAG system evaluation
ragas = RAGASScorer(
    model=your_llm_model,
    weights={"faithfulness": 0.4, "answer_relevancy": 0.3, "contextual_precision": 0.3}
)
result = await ragas.evaluate(question, answer, context=retrieved_context)

# Panel-based evaluation
panel = SpecializedPanelScorer.create_diverse_panel(
    models=[model1, model2, model3],
    evaluation_criteria="overall quality and helpfulness"
)
result = await panel.evaluate(input_text, output_text)

# Agent evaluation
agent_scorers = AgentScorers(model=your_llm_model)
all_scores = agent_scorers.score_all(agent_data)
```

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
