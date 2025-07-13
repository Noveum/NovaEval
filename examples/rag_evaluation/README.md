# RAG Evaluation Examples

This directory contains comprehensive examples demonstrating how to use the enhanced RAG evaluation system in NovaEval. These examples cover everything from basic usage to advanced production scenarios.

## 📋 Examples Overview

### 1. Basic RAG Evaluation (`basic_rag_example.py`)
**Purpose**: Introduction to core RAG evaluation metrics
**What it demonstrates**:
- Setting up RAG evaluation suite
- Running comprehensive evaluation with core metrics
- Understanding evaluation results and thresholds
- Using individual metrics
- Comparing different configuration profiles

**Key Features Shown**:
- Context evaluation (precision, relevancy, recall)
- Answer evaluation (relevancy, similarity, correctness, faithfulness)
- Configuration options (balanced, precision, recall, speed)
- Detailed result analysis and interpretation

### 2. Advanced Safety Evaluation (`advanced_safety_example.py`)
**Purpose**: Comprehensive safety and quality assessment
**What it demonstrates**:
- Hallucination detection across multiple categories
- Bias detection with 9-category analysis
- Toxicity detection and harmful content identification
- Safety configuration profiles
- Production monitoring patterns

**Key Features Shown**:
- Multi-scenario safety testing
- Individual safety metric usage
- Configuration impact on safety thresholds
- Alert generation and monitoring workflows

### 3. Conversational Evaluation (`conversational_example.py`)
**Purpose**: Multi-turn dialogue quality assessment
**What it demonstrates**:
- Conversational coherence evaluation
- Multi-turn conversation tracking
- Context switching detection
- Topic consistency analysis
- Dialogue quality progression

**Key Features Shown**:
- Conversation history management
- Turn-by-turn evaluation
- Context switch handling
- Coherence scoring and analysis

### 4. Batch Evaluation (`batch_evaluation_example.py`)
**Purpose**: Large-scale evaluation and performance optimization
**What it demonstrates**:
- Parallel batch processing
- Performance comparison between approaches
- Domain-level analysis
- Results export and reporting
- Production-scale evaluation patterns

**Key Features Shown**:
- Sequential vs parallel evaluation
- Comprehensive batch safety evaluation
- Performance benchmarking
- Results export (JSON, summary reports)

## 🚀 Getting Started

### Prerequisites

1. **Install NovaEval** with RAG evaluation dependencies:
   ```bash
   pip install novaeval[rag]
   # or for development
   pip install -e .[dev]
   ```

2. **Set up OpenAI API Key** (required for examples):
   ```bash
   export OPENAI_API_KEY="your-openai-api-key-here"
   ```

3. **Navigate to examples directory**:
   ```bash
   cd examples/rag_evaluation/
   ```

### Running Examples

#### Start with Basic Example
```bash
python basic_rag_example.py
```
This will show you:
- Core RAG evaluation metrics in action
- How to interpret evaluation results
- Different configuration options
- Individual metric usage patterns

#### Explore Safety Features
```bash
python advanced_safety_example.py
```
This demonstrates:
- Hallucination detection capabilities
- Bias and toxicity detection
- Safety monitoring workflows
- Production alert patterns

#### Try Conversational Evaluation
```bash
python conversational_example.py
```
This covers:
- Multi-turn dialogue evaluation
- Conversation coherence analysis
- Context switching scenarios
- Turn-by-turn quality tracking

#### Scale with Batch Processing
```bash
python batch_evaluation_example.py
```
This shows:
- Large-scale evaluation techniques
- Performance optimization strategies
- Results analysis and export
- Production integration patterns

## 📊 Example Outputs

### Basic Evaluation Results
```
📊 Evaluation Results:
----------------------------------------------------------------------
Metric                    | Score    | Status   | Threshold
----------------------------------------------------------------------
context_relevancy         | 0.850    | ✅ PASS  | 0.7
answer_relevancy          | 0.920    | ✅ PASS  | 0.7
faithfulness             | 0.780    | ❌ FAIL  | 0.8
answer_correctness       | 0.890    | ✅ PASS  | 0.7
----------------------------------------------------------------------
Overall Performance: 3/4 metrics passed (75.0%)
Average Score: 0.860
```

### Safety Evaluation Results
```
Safety Results:
--------------------------------------------------
hallucination_detection   |  0.950 | ✅ SAFE | Conf: 0.92
bias_detection           |  0.250 | ⚠️ FLAGGED | Conf: 0.85
toxicity_detection       |  0.980 | ✅ SAFE | Conf: 0.95
--------------------------------------------------

🚨 Detailed Analysis of Flagged Content:
BIAS_DETECTION:
Score: 0.250
Detected Biases:
  • gender: "men are typically better at math"
  • role: "women are more suited for caregiving"
```

### Batch Evaluation Summary
```
📈 Batch Results Analysis:
----------------------------------------------------------------------
ID       | Domain       | Avg Score  | Pass Rate  | Status
----------------------------------------------------------------------
qa_001   | Science      | 0.875      | 87.5%      | ✅ GOOD
qa_002   | Technology   | 0.820      | 75.0%      | ⚠️ REVIEW
qa_003   | Health       | 0.910      | 100.0%     | ✅ GOOD
----------------------------------------------------------------------
```

## 🔧 Customization Guide

### Modifying Examples for Your Use Case

#### 1. **Custom Evaluation Data**
Replace the example datasets with your own data:

```python
# In basic_rag_example.py
your_evaluation_data = {
    "question": "Your question here",
    "context": "Your context here",
    "generated_answer": "Your RAG system's answer",
    "expected_answer": "Expected/reference answer"
}
```

#### 2. **Custom Configuration**
Adjust evaluation thresholds for your requirements:

```python
from novaeval.scorers.rag_comprehensive import RAGEvaluationConfig

custom_config = RAGEvaluationConfig(
    similarity_threshold=0.8,      # Adjust based on your quality requirements
    faithfulness_threshold=0.9,    # Higher for critical applications
    relevancy_threshold=0.75       # Balance precision vs recall
)
```

#### 3. **Custom Safety Thresholds**
Modify safety detection sensitivity:

```python
from novaeval.scorers.rag_advanced import AdvancedRAGConfig

safety_config = AdvancedRAGConfig(
    hallucination_threshold=0.95,  # Very strict hallucination detection
    bias_threshold=0.1,            # Low tolerance for bias
    toxicity_threshold=0.05        # Very strict toxicity detection
)
```

#### 4. **Custom Metrics Selection**
Choose specific metrics for your evaluation:

```python
# Evaluate only core quality metrics
results = await suite.evaluate_retrieval_pipeline(question, answer, context)

# Evaluate only safety metrics
results = await suite.evaluate_comprehensive_plus(
    question, answer, context,
    include_safety_metrics=True,
    include_conversational_metrics=False
)
```

## 🏭 Production Integration

### CI/CD Pipeline Integration

Add RAG evaluation to your continuous integration:

```python
# In your test suite
async def test_rag_quality():
    suite = RAGEvaluationSuite(model, get_rag_config("precision"))

    for test_case in test_cases:
        results = await suite.evaluate_comprehensive(
            test_case.question,
            test_case.generated_answer,
            test_case.expected_answer,
            test_case.context
        )

        # Assert quality requirements
        assert results["faithfulness"].passed, "Faithfulness check failed"
        assert results["answer_relevancy"].passed, "Relevancy check failed"
```

### Real-time Monitoring

Monitor production RAG responses:

```python
async def monitor_rag_response(question, answer, context):
    # Quick safety check for production
    results = await suite.evaluate_comprehensive_plus(
        question, answer, context,
        include_safety_metrics=True
    )

    # Alert on safety issues
    if not results["toxicity_detection"].passed:
        alert_system.send_urgent_alert("Toxic content detected")

    # Log quality metrics
    logger.info({
        "faithfulness": results["faithfulness"].score,
        "bias_score": results["bias_detection"].score
    })
```

### Batch Analysis

Regular batch evaluation of production data:

```python
# Weekly quality assessment
async def weekly_quality_assessment():
    production_samples = get_weekly_rag_samples()

    batch_results = await evaluate_batch(production_samples)

    # Generate quality report
    generate_quality_report(batch_results)

    # Alert on quality degradation
    if average_quality < threshold:
        alert_system.send_alert("RAG quality degradation detected")
```

## 📚 Learning Path

### Beginner Path
1. **Start with `basic_rag_example.py`** - Learn core concepts
2. **Understand evaluation metrics** - Study the detailed analysis sections
3. **Experiment with configurations** - Try different threshold settings
4. **Practice with your own data** - Replace example data with your use cases

### Intermediate Path
1. **Explore `advanced_safety_example.py`** - Understand safety evaluation
2. **Try `conversational_example.py`** - Learn dialogue evaluation
3. **Customize configurations** - Create domain-specific settings
4. **Integrate individual metrics** - Use specific metrics in your pipeline

### Advanced Path
1. **Master `batch_evaluation_example.py`** - Scale evaluation processes
2. **Implement production monitoring** - Set up real-time quality checks
3. **Create custom metrics** - Extend the evaluation framework
4. **Optimize performance** - Fine-tune for your specific requirements

## 🔍 Troubleshooting

### Common Issues

#### 1. **API Key Not Set**
```
⚠️ Warning: OPENAI_API_KEY not found in environment variables
```
**Solution**: Set your OpenAI API key:
```bash
export OPENAI_API_KEY="your-key-here"
```

#### 2. **Import Errors**
```
ModuleNotFoundError: No module named 'novaeval'
```
**Solution**: Install NovaEval properly:
```bash
pip install -e .[dev]
```

#### 3. **Slow Performance**
If evaluation is slow, try:
- Use `get_rag_config("speed")` for faster evaluation
- Reduce batch size for batch evaluation
- Use individual metrics instead of comprehensive evaluation

#### 4. **Memory Issues**
For large-scale evaluation:
- Process data in smaller batches
- Use `gc.collect()` between batches
- Consider using lighter configuration profiles

### Getting Help

- **Documentation**: See `../docs/rag_evaluation_system.md` for detailed API reference
- **Issues**: Report problems on GitHub Issues
- **Discussions**: Join conversations on GitHub Discussions

## 🎯 Next Steps

After running these examples:

1. **Integrate with your RAG system** - Adapt the patterns to your specific implementation
2. **Set up monitoring** - Implement production quality monitoring
3. **Create custom configurations** - Tune thresholds for your domain
4. **Scale evaluation** - Use batch processing for large-scale assessment
5. **Contribute back** - Share improvements and new examples with the community

---

**Happy Evaluating! 🚀**

These examples provide a comprehensive foundation for implementing robust RAG evaluation in your projects. Start with the basic example and progressively explore more advanced features as your needs grow.
