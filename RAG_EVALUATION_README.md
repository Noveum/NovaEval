# Enhanced RAG Evaluation System for NovaEval

A comprehensive, state-of-the-art evaluation framework for Retrieval-Augmented Generation (RAG) systems, featuring 12+ evaluation metrics, advanced safety detection, and production-ready performance optimization.

## 🚀 Features

### Core RAG Metrics (8 metrics)
- **Context Evaluation**: Precision, Relevancy, Recall, Entity Recall
- **Answer Evaluation**: Relevancy, Similarity, Correctness, Faithfulness

### Advanced Safety Metrics (4+ metrics)
- **Hallucination Detection**: Multi-category claim verification
- **Bias Detection**: 9-category bias analysis (gender, race, political, etc.)
- **Toxicity Detection**: 7-category harmful content detection
- **Conversation Coherence**: Multi-turn conversation evaluation

### Composite Scorers
- **Enhanced RAGAS**: Comprehensive RAG assessment with configurable weights
- **RAG Triad**: Simplified 3-metric evaluation for rapid assessment

### Production Features
- ⚡ **Parallel Execution**: Concurrent metric evaluation for optimal performance
- 🔧 **Configurable Profiles**: Optimized settings for precision, recall, or speed
- 📊 **Detailed Analytics**: Comprehensive scoring with explanations and metadata
- 🛡️ **Safety-First Design**: Built-in detection for harmful, biased, or hallucinated content
- 🔄 **Async/Await Support**: Modern Python async patterns for scalability

## 📦 Installation

The Enhanced RAG Evaluation System is part of NovaEval. Install NovaEval with all dependencies:

```bash
pip install novaeval[rag]
# or for development
pip install -e .[dev]
```

## 🎯 Quick Start

### Basic RAG Evaluation

```python
import asyncio
from novaeval.models.openai_model import OpenAIModel
from novaeval.scorers.rag_comprehensive import RAGEvaluationSuite, get_rag_config

async def evaluate_rag():
    # Setup
    model = OpenAIModel(model_name="gpt-4")
    config = get_rag_config("balanced")
    suite = RAGEvaluationSuite(model, config)

    # Evaluate
    results = await suite.evaluate_comprehensive(
        input_text="What are the benefits of renewable energy?",
        output_text="Renewable energy reduces emissions and provides sustainable power.",
        expected_output="Benefits include environmental protection and sustainability.",
        context="Renewable sources like solar and wind provide clean energy..."
    )

    # Results
    for metric, result in results.items():
        print(f"{metric}: {result.score:.3f} ({'✅ PASS' if result.passed else '❌ FAIL'})")

asyncio.run(evaluate_rag())
```

### Advanced Safety Evaluation

```python
from novaeval.scorers.rag_advanced import ComprehensiveRAGEvaluationSuite

async def safety_evaluation():
    model = OpenAIModel(model_name="gpt-4")
    suite = ComprehensiveRAGEvaluationSuite(model)

    results = await suite.evaluate_comprehensive_plus(
        input_text="How do I stay healthy?",
        output_text="Eat well, exercise regularly, and get enough sleep.",
        context="Health involves proper nutrition, exercise, and rest.",
        include_safety_metrics=True  # Enables hallucination, bias, toxicity detection
    )

    # Check safety specifically
    safety_metrics = ["hallucination_detection", "bias_detection", "toxicity_detection"]
    for metric in safety_metrics:
        result = results[metric]
        print(f"{metric}: {result.score:.3f} - {'✅ SAFE' if result.passed else '⚠️ FLAGGED'}")

asyncio.run(safety_evaluation())
```

## 📊 Evaluation Metrics

### Context Evaluation Metrics

| Metric | Purpose | Key Features |
|--------|---------|--------------|
| **Context Precision** | Evaluates retrieval ranking quality | LLM-based relevance scoring, chunk-level analysis |
| **Context Relevancy** | Measures overall context relevance | Multi-dimensional assessment, optimization insights |
| **Context Recall** | Assesses information completeness | Key information extraction, gap identification |
| **Context Entity Recall** | Evaluates entity-level coverage | Multi-category NER, fuzzy matching |

### Answer Evaluation Metrics

| Metric | Purpose | Key Features |
|--------|---------|--------------|
| **Answer Relevancy** | Measures response relevance | Multi-method assessment, question generation |
| **Answer Similarity** | Compares generated vs expected | Semantic + lexical + structural similarity |
| **Answer Correctness** | Evaluates factual accuracy | Statement-level verification, precision/recall/F1 |
| **Enhanced Faithfulness** | Assesses context grounding | Multi-category claim verification, confidence scoring |

### Advanced Safety Metrics

| Metric | Purpose | Detection Categories |
|--------|---------|---------------------|
| **Hallucination Detection** | Identifies unsupported claims | Factual, numerical, temporal, entity, relational |
| **Bias Detection** | Detects unfair bias | Gender, race, religion, political, age, disability, etc. |
| **Toxicity Detection** | Identifies harmful content | Offensive language, hate speech, threats, harassment |
| **Conversation Coherence** | Evaluates dialogue quality | Context consistency, logical flow, thread maintenance |

## ⚙️ Configuration

### Pre-built Configurations

```python
from novaeval.scorers.rag_comprehensive import get_rag_config
from novaeval.scorers.rag_advanced import get_advanced_rag_config

# Core RAG configurations
balanced_config = get_rag_config("balanced")      # Equal emphasis on all metrics
precision_config = get_rag_config("precision")   # High precision, strict thresholds
recall_config = get_rag_config("recall")         # Comprehensive coverage
speed_config = get_rag_config("speed")           # Fast evaluation, essential metrics

# Advanced safety configurations
safety_config = get_advanced_rag_config("safety_first")  # Strict safety thresholds
permissive_config = get_advanced_rag_config("permissive")  # Relaxed thresholds
```

### Custom Configuration

```python
from novaeval.scorers.rag_comprehensive import RAGEvaluationConfig
from novaeval.scorers.rag_advanced import AdvancedRAGConfig

# Custom core configuration
custom_config = RAGEvaluationConfig(
    similarity_threshold=0.8,
    faithfulness_threshold=0.9,
    relevancy_threshold=0.75,
    ragas_weights={
        "context_precision": 0.3,
        "context_recall": 0.2,
        "answer_relevancy": 0.25,
        "faithfulness": 0.25
    }
)

# Custom safety configuration
safety_config = AdvancedRAGConfig(
    hallucination_threshold=0.95,
    bias_threshold=0.1,
    toxicity_threshold=0.05
)
```

## 🔧 Individual Metric Usage

```python
from novaeval.scorers.rag_comprehensive import create_rag_scorer
from novaeval.scorers.rag_advanced import create_comprehensive_rag_scorer

# Use individual core metrics
faithfulness_scorer = create_rag_scorer("faithfulness", model)
result = await faithfulness_scorer.evaluate(input_text, output_text, context=context)

# Use individual advanced metrics
bias_scorer = create_comprehensive_rag_scorer("bias_detection", model)
result = await bias_scorer.evaluate(input_text, output_text)
```

## 🎭 Conversational Evaluation

```python
# Multi-turn conversation evaluation
conversation_history = [
    {"role": "user", "content": "What is machine learning?"},
    {"role": "assistant", "content": "ML is a subset of AI..."},
    {"role": "user", "content": "How does it work?"}
]

results = await suite.evaluate_comprehensive_plus(
    input_text="How does it work?",
    output_text="ML algorithms identify patterns in data...",
    include_conversational_metrics=True,
    conversation_history=conversation_history
)

coherence = results["conversation_coherence"]
print(f"Conversation Coherence: {coherence.score:.3f}")
```

## 📈 Performance Optimization

### Parallel Execution
```python
# Automatic parallel execution of compatible metrics
results = await suite.evaluate_comprehensive(...)  # Runs metrics concurrently
```

### Selective Evaluation
```python
# Evaluate only retrieval quality
retrieval_results = await suite.evaluate_retrieval_pipeline(question, answer, context)

# Evaluate only generation quality
generation_results = await suite.evaluate_generation_pipeline(question, answer, expected)

# Evaluate only safety metrics
safety_results = await suite.evaluate_comprehensive_plus(
    question, answer, context,
    include_safety_metrics=True,
    include_conversational_metrics=False
)
```

### Batch Processing
```python
async def batch_evaluate(examples):
    tasks = [
        suite.evaluate_comprehensive(ex['question'], ex['answer'], ex['expected'], ex['context'])
        for ex in examples
    ]
    return await asyncio.gather(*tasks)
```

## 🏭 Production Integration

### CI/CD Pipeline Integration
```python
# Add to your test suite
async def test_rag_quality():
    results = await suite.evaluate_comprehensive(question, answer, expected, context)

    # Assert quality thresholds
    assert results["faithfulness"].passed, "Faithfulness check failed"
    assert results["bias_detection"].passed, "Bias detected"
    assert results["toxicity_detection"].passed, "Toxic content detected"
```

### Real-time Monitoring
```python
# Production monitoring
async def monitor_rag_response(question, answer, context):
    results = await suite.evaluate_comprehensive_plus(
        question, answer, context=context,
        include_safety_metrics=True
    )

    # Alert on quality issues
    if not results["faithfulness"].passed:
        alert_system.send_alert("Faithfulness violation detected")

    # Log metrics
    logger.info({
        "faithfulness": results["faithfulness"].score,
        "bias_score": results["bias_detection"].score,
        "toxicity_score": results["toxicity_detection"].score
    })
```

## 🧪 Testing and Validation

Run the comprehensive test suite:

```bash
# Run all RAG evaluation tests
pytest tests/unit/test_rag_comprehensive.py -v
pytest tests/unit/test_rag_advanced.py -v
pytest tests/integration/test_rag_integration.py -v

# Run specific test categories
pytest tests/unit/test_rag_comprehensive.py::TestContextEvaluationScorers -v
pytest tests/unit/test_rag_advanced.py::TestHallucinationDetectionScorer -v
```

## 📚 Documentation

- **[Complete Documentation](docs/rag_evaluation_system.md)** - Comprehensive guide with architecture details
- **[API Reference](docs/rag_evaluation_system.md#api-reference)** - Detailed API documentation
- **[Performance Guide](docs/rag_evaluation_system.md#performance-considerations)** - Optimization strategies
- **[Integration Guide](docs/rag_evaluation_system.md#integration-guide)** - Framework integration patterns

## 🎯 Demo Script

Run the comprehensive demonstration:

```bash
python demo_rag_evaluation.py
```

This will showcase:
- Core RAG evaluation metrics
- Advanced safety evaluation
- Conversational evaluation
- Configuration options
- Performance comparisons

## 🔍 Comparison with Other Frameworks

| Feature | NovaEval Enhanced RAG | Other Frameworks |
|---------|----------------------|------------------|
| **Core RAG Metrics** | 8 comprehensive metrics | 3-5 basic metrics |
| **Safety Detection** | 4 advanced safety metrics | Limited or none |
| **Conversational Support** | Multi-turn coherence evaluation | Basic or none |
| **Configuration Profiles** | 4 optimized profiles | Fixed configuration |
| **Parallel Execution** | Automatic concurrent evaluation | Sequential only |
| **Production Ready** | Built for scale with monitoring | Research-focused |
| **Bias Detection** | 9-category comprehensive analysis | Basic or none |
| **Hallucination Detection** | Multi-strategy claim verification | Simple faithfulness |

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

```bash
git clone https://github.com/Noveum/NovaEval.git
cd NovaEval
pip install -e .[dev]
pre-commit install
```

### Running Tests

```bash
# Run all tests
pytest

# Run RAG-specific tests
pytest tests/unit/test_rag_*.py -v

# Run with coverage
pytest --cov=novaeval.scorers.rag_comprehensive --cov=novaeval.scorers.rag_advanced
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

This enhanced RAG evaluation system incorporates insights and methodologies from leading evaluation frameworks while providing significant improvements in comprehensiveness, safety, and production readiness.

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/Noveum/NovaEval/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Noveum/NovaEval/discussions)
- **Documentation**: [docs/rag_evaluation_system.md](docs/rag_evaluation_system.md)

---

**Built with ❤️ by the NovaEval team**
