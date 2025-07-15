# Enhanced RAG Evaluation System for NovaEval

## Overview

This document provides comprehensive documentation for the Enhanced RAG (Retrieval-Augmented Generation) Evaluation System implemented for NovaEval. This system represents a significant advancement in RAG evaluation capabilities, incorporating state-of-the-art metrics and methodologies inspired by leading evaluation frameworks like DeepEval, Braintrust, and AutoEvals.

## Table of Contents

1. [Introduction](#introduction)
2. [Architecture Overview](#architecture-overview)
3. [Core Components](#core-components)
4. [Evaluation Metrics](#evaluation-metrics)
5. [Configuration System](#configuration-system)
6. [Usage Examples](#usage-examples)
7. [API Reference](#api-reference)
8. [Performance Considerations](#performance-considerations)
9. [Best Practices](#best-practices)
10. [Integration Guide](#integration-guide)

## Introduction

Retrieval-Augmented Generation (RAG) systems combine the power of information retrieval with large language model generation to provide accurate, contextually relevant responses. However, evaluating RAG systems presents unique challenges as it requires assessing both the quality of retrieved information and the generated responses.

The Enhanced RAG Evaluation System addresses these challenges by providing a comprehensive suite of evaluation metrics that cover every aspect of the RAG pipeline. The system is designed to be modular, configurable, and scalable, making it suitable for both research and production environments.

### Key Features

- **Comprehensive Metric Coverage**: 8+ individual metrics covering retrieval and generation quality
- **Composite Evaluation Methods**: Enhanced RAGAS and RAG Triad scorers for holistic assessment
- **Configurable Evaluation**: Optimized configurations for different use cases (precision, recall, speed)
- **Parallel Processing**: Efficient evaluation through concurrent metric computation
- **Extensible Architecture**: Easy to add new metrics and evaluation methods
- **Production Ready**: Robust error handling and performance optimization

## Architecture Overview

The Enhanced RAG Evaluation System follows a modular architecture with clear separation of concerns:

```
RAGEvaluationSuite
├── Context Evaluation Metrics
│   ├── ContextPrecisionScorer
│   ├── ContextRelevancyScorer
│   ├── ContextRecallScorer
│   └── ContextEntityRecallScorer
├── Answer Evaluation Metrics
│   ├── AnswerRelevancyScorer
│   ├── AnswerSimilarityScorer
│   ├── AnswerCorrectnessScorer
│   └── EnhancedFaithfulnessScorer
├── Composite Scorers
│   ├── EnhancedRAGASScorer
│   └── RAGTriadScorer
└── Configuration Management
    └── RAGEvaluationConfig
```

### Design Principles

1. **Modularity**: Each metric is implemented as an independent scorer that can be used individually or as part of composite evaluations.

2. **Consistency**: All scorers follow the same interface pattern, returning standardized `ScoreResult` objects with score, pass/fail status, reasoning, and metadata.

3. **Configurability**: The system supports multiple configuration profiles optimized for different evaluation priorities (precision, recall, speed).

4. **Extensibility**: New metrics can be easily added by implementing the base scorer interface.

5. **Performance**: Parallel execution of metrics ensures efficient evaluation even with multiple complex assessments.




## Core Components

### RAGEvaluationConfig

The `RAGEvaluationConfig` class serves as the central configuration hub for all RAG evaluation parameters. It provides fine-grained control over thresholds, model settings, and metric weights.

```python
class RAGEvaluationConfig:
    embedding_model: str = "all-MiniLM-L6-v2"
    similarity_threshold: float = 0.7
    faithfulness_threshold: float = 0.8
    relevancy_threshold: float = 0.7
    precision_threshold: float = 0.7
    recall_threshold: float = 0.7
    answer_correctness_threshold: float = 0.8
    ragas_weights: Dict[str, float] = {...}
```

The configuration system supports multiple optimization profiles:

- **Balanced**: Default configuration with equal emphasis on all metrics
- **Precision**: Optimized for high-precision scenarios with emphasis on faithfulness and correctness
- **Recall**: Optimized for comprehensive coverage with emphasis on context recall
- **Speed**: Optimized for fast evaluation by skipping computationally expensive metrics

### RAGEvaluationSuite

The `RAGEvaluationSuite` class provides a unified interface for accessing all RAG evaluation capabilities. It manages scorer instances, coordinates parallel evaluation, and provides convenient methods for different evaluation scenarios.

Key methods include:
- `evaluate_single_metric()`: Evaluate using a specific metric
- `evaluate_retrieval_pipeline()`: Focus on retrieval quality metrics
- `evaluate_generation_pipeline()`: Focus on generation quality metrics
- `evaluate_comprehensive()`: Run all metrics for complete assessment

## Evaluation Metrics

The Enhanced RAG Evaluation System provides comprehensive coverage of RAG system quality through eight core metrics organized into two categories: Context Evaluation and Answer Evaluation.

### Context Evaluation Metrics

Context evaluation metrics assess the quality of the retrieval component in RAG systems, ensuring that the retrieved information is relevant, comprehensive, and well-organized.

#### Context Precision

**Purpose**: Evaluates whether the reranker effectively ranks more relevant content higher than irrelevant content in the retrieved context.

**Methodology**:
- Analyzes each chunk of retrieved context for relevance to the input question
- Uses LLM-based relevance scoring with detailed reasoning
- Calculates precision as the ratio of relevant chunks to total chunks
- Provides chunk-level analysis for debugging retrieval quality

**Key Features**:
- Supports both string and list context inputs
- Provides detailed relevance analysis for each context chunk
- Includes confidence scoring for relevance assessments
- Offers actionable insights for improving retrieval ranking

**Use Cases**:
- Optimizing reranker performance in multi-stage retrieval systems
- Identifying irrelevant content in retrieved context
- Tuning retrieval parameters for better precision

#### Context Relevancy

**Purpose**: Measures whether the text chunk size and top-K retrieval parameters effectively capture relevant information without excessive irrelevancy.

**Methodology**:
- Evaluates overall relevance of the combined retrieved context
- Assesses information density and relevance distribution
- Analyzes the balance between comprehensive coverage and focused relevance
- Provides recommendations for chunk size and retrieval parameter optimization

**Key Features**:
- Multi-dimensional relevance assessment (topical, semantic, contextual)
- Chunk size optimization recommendations
- Top-K parameter tuning insights
- Irrelevancy detection and quantification

**Use Cases**:
- Optimizing retrieval parameters for specific domains
- Balancing retrieval breadth vs. precision
- Improving context window utilization in generation

#### Context Recall

**Purpose**: Evaluates whether the embedding model can accurately capture and retrieve all relevant information needed to answer the question comprehensively.

**Methodology**:
- Extracts key information elements from the expected answer
- Verifies presence of each element in the retrieved context
- Calculates recall as the percentage of key information successfully retrieved
- Provides detailed analysis of missing information categories

**Key Features**:
- Multi-category information extraction (facts, concepts, details)
- Granular recall analysis by information type
- Missing information identification and categorization
- Embedding model performance insights

**Use Cases**:
- Evaluating embedding model effectiveness for specific domains
- Identifying gaps in knowledge base coverage
- Optimizing retrieval strategies for comprehensive information capture

#### Context Entity Recall

**Purpose**: Assesses entity-level recall in retrieved context, ensuring that important entities mentioned in the expected answer are present in the retrieved information.

**Methodology**:
- Extracts entities from expected answers using advanced NER techniques
- Categorizes entities by type (persons, organizations, locations, dates, concepts)
- Verifies entity presence in retrieved context with fuzzy matching
- Calculates category-specific and overall entity recall rates

**Key Features**:
- Multi-category entity extraction and analysis
- Fuzzy matching for entity variations and synonyms
- Entity importance weighting based on context
- Detailed entity coverage reporting

**Use Cases**:
- Ensuring factual completeness in knowledge-intensive domains
- Optimizing retrieval for entity-rich content
- Validating information coverage in specialized domains

### Answer Evaluation Metrics

Answer evaluation metrics assess the quality of the generation component in RAG systems, ensuring that generated responses are relevant, accurate, and faithful to the retrieved context.

#### Answer Relevancy

**Purpose**: Measures how relevant the generated answer is to the original question using multiple complementary approaches.

**Methodology**:
The system employs a multi-method approach for robust relevance assessment:

1. **Question Generation Approach**: Generates questions from the answer and measures similarity to the original question
2. **Direct Relevance Assessment**: Uses LLM-based direct evaluation of answer-question relevance
3. **Semantic Similarity**: Computes embedding-based semantic similarity between question and answer

**Key Features**:
- Multi-method relevance assessment for robust evaluation
- Confidence scoring and uncertainty quantification
- Detailed reasoning for relevance judgments
- Adaptive method selection based on content characteristics

**Use Cases**:
- Ensuring generated responses directly address user questions
- Optimizing prompt engineering for better relevance
- Detecting and preventing off-topic responses

#### Answer Similarity

**Purpose**: Measures similarity between generated and expected answers using multiple similarity dimensions.

**Methodology**:
Comprehensive similarity assessment through three complementary measures:

1. **Semantic Similarity**: Embedding-based semantic comparison using sentence transformers
2. **Lexical Similarity**: Token-level overlap analysis with TF-IDF weighting
3. **Structural Similarity**: LLM-based assessment of organizational and structural alignment

**Key Features**:
- Multi-dimensional similarity analysis
- Weighted combination of similarity measures
- Detailed similarity breakdown by dimension
- Threshold-based pass/fail determination

**Use Cases**:
- Comparing generated responses against gold standard answers
- Evaluating consistency across multiple generation attempts
- Benchmarking different generation models or prompts

#### Answer Correctness

**Purpose**: Evaluates generated answers against ground truth using statement-level verification with precision, recall, and F1 analysis.

**Methodology**:
- Decomposes both generated and expected answers into atomic statements
- Classifies each generated statement as True Positive, False Positive, or False Negative
- Calculates precision, recall, and F1 scores for comprehensive correctness assessment
- Provides detailed statement-level analysis for error identification

**Key Features**:
- Statement-level factual verification
- Comprehensive precision/recall/F1 analysis
- Detailed error categorization and analysis
- Actionable feedback for improving answer quality

**Use Cases**:
- Factual accuracy assessment in knowledge-intensive domains
- Identifying and correcting systematic generation errors
- Benchmarking factual reliability across different models

#### Enhanced Faithfulness

**Purpose**: Evaluates whether the generated response is faithful to the provided context through categorized claim verification.

**Methodology**:
Advanced faithfulness assessment through multi-category claim analysis:

1. **Claim Extraction**: Identifies and categorizes claims by type (factual, numerical, temporal, relational, opinion)
2. **Claim Verification**: Verifies each claim against the provided context with confidence scoring
3. **Category Analysis**: Provides detailed analysis by claim category
4. **Confidence Assessment**: Includes uncertainty quantification for verification decisions

**Key Features**:
- Multi-category claim extraction and verification
- Confidence-weighted faithfulness scoring
- Detailed claim-level analysis and reasoning
- Category-specific faithfulness insights

**Use Cases**:
- Preventing hallucination in generated responses
- Ensuring factual grounding in context-dependent generation
- Identifying and addressing systematic faithfulness issues

### Composite Scorers

The system provides two sophisticated composite scorers that combine multiple individual metrics for holistic RAG system assessment.

#### Enhanced RAGAS Scorer

**Purpose**: Provides comprehensive RAG assessment by combining all eight individual metrics with configurable weights.

**Methodology**:
- Executes all individual metrics in parallel for efficiency
- Applies configurable weights based on evaluation priorities
- Calculates component group scores (retrieval vs. generation)
- Provides detailed component analysis and overall assessment

**Key Features**:
- Configurable metric weighting for different use cases
- Parallel execution for optimal performance
- Detailed component and group-level analysis
- Comprehensive pass/fail determination logic

**Use Cases**:
- Holistic RAG system evaluation and benchmarking
- Identifying strengths and weaknesses across RAG components
- Tracking system performance over time

#### RAG Triad Scorer

**Purpose**: Simplified but comprehensive RAG assessment focusing on three core aspects: Context Relevance, Groundedness (Faithfulness), and Answer Relevance.

**Methodology**:
- Focuses on the three most critical aspects of RAG quality
- Provides equal weighting to each triad component
- Requires all three components to pass for overall success
- Offers streamlined evaluation for rapid assessment

**Key Features**:
- Simplified three-metric approach for rapid evaluation
- Equal emphasis on retrieval quality, faithfulness, and relevance
- Clear pass/fail logic requiring success in all three areas
- Detailed component-level analysis and reasoning

**Use Cases**:
- Rapid RAG system quality assessment
- Production monitoring and alerting
- Simplified evaluation for non-technical stakeholders



## Usage Examples

### Basic RAG Evaluation

```python
import asyncio
from novaeval.models.openai_model import OpenAIModel
from novaeval.scorers.rag_comprehensive import RAGEvaluationSuite, get_rag_config

async def basic_rag_evaluation():
    # Initialize model and configuration
    model = OpenAIModel(model_name="gpt-4")
    config = get_rag_config("balanced")

    # Create evaluation suite
    suite = RAGEvaluationSuite(model, config)

    # Define evaluation inputs
    question = "What are the benefits of renewable energy?"
    generated_answer = "Renewable energy reduces carbon emissions and provides sustainable power."
    expected_answer = "Benefits include environmental protection and energy sustainability."
    context = "Renewable energy sources like solar and wind provide clean, sustainable power..."

    # Run comprehensive evaluation
    results = await suite.evaluate_comprehensive(
        question, generated_answer, expected_answer, context
    )

    # Process results
    for metric_name, result in results.items():
        print(f"{metric_name}: {result.score:.3f} ({'PASS' if result.passed else 'FAIL'})")

# Run the evaluation
asyncio.run(basic_rag_evaluation())
```

### Advanced Safety Evaluation

```python
from novaeval.scorers.rag_advanced import ComprehensiveRAGEvaluationSuite, get_advanced_rag_config

async def safety_evaluation():
    # Initialize with safety-focused configuration
    model = OpenAIModel(model_name="gpt-4")
    rag_config = get_rag_config("balanced")
    safety_config = get_advanced_rag_config("safety_first")

    # Create comprehensive suite
    suite = ComprehensiveRAGEvaluationSuite(model, rag_config, safety_config)

    # Evaluate with safety metrics
    results = await suite.evaluate_comprehensive_plus(
        question="How do I stay healthy?",
        output_text="Eat well, exercise regularly, and get enough sleep.",
        context="Health maintenance involves proper nutrition, physical activity, and rest.",
        include_safety_metrics=True,
        include_conversational_metrics=False
    )

    # Check safety metrics specifically
    safety_metrics = ["hallucination_detection", "bias_detection", "toxicity_detection"]
    for metric in safety_metrics:
        result = results[metric]
        print(f"{metric}: {result.score:.3f} - {result.reasoning[:100]}...")

asyncio.run(safety_evaluation())
```

### Individual Metric Usage

```python
from novaeval.scorers.rag_comprehensive import create_rag_scorer
from novaeval.scorers.rag_advanced import create_comprehensive_rag_scorer

async def individual_metrics():
    model = OpenAIModel(model_name="gpt-4")

    # Use individual core metrics
    faithfulness_scorer = create_rag_scorer("faithfulness", model)
    faithfulness_result = await faithfulness_scorer.evaluate(
        input_text="What is AI?",
        output_text="AI is artificial intelligence used in computers.",
        context="Artificial intelligence refers to machine intelligence and automation."
    )

    # Use individual advanced metrics
    bias_scorer = create_comprehensive_rag_scorer("bias_detection", model)
    bias_result = await bias_scorer.evaluate(
        input_text="Describe leadership qualities.",
        output_text="Good leaders are empathetic, decisive, and communicate effectively."
    )

    print(f"Faithfulness: {faithfulness_result.score:.3f}")
    print(f"Bias Detection: {bias_result.score:.3f}")

asyncio.run(individual_metrics())
```

### Conversational Evaluation

```python
async def conversational_evaluation():
    model = OpenAIModel(model_name="gpt-4")
    suite = ComprehensiveRAGEvaluationSuite(model)

    # Define conversation history
    conversation_history = [
        {"role": "user", "content": "What is machine learning?"},
        {"role": "assistant", "content": "ML is a subset of AI that learns from data."},
        {"role": "user", "content": "How does it work?"}
    ]

    # Evaluate current response in context
    results = await suite.evaluate_comprehensive_plus(
        input_text="How does it work?",
        output_text="ML algorithms identify patterns in data to make predictions.",
        include_conversational_metrics=True,
        conversation_history=conversation_history
    )

    coherence_result = results["conversation_coherence"]
    print(f"Conversation Coherence: {coherence_result.score:.3f}")
    print(f"Analysis: {coherence_result.reasoning}")

asyncio.run(conversational_evaluation())
```

### Configuration Customization

```python
from novaeval.scorers.rag_comprehensive import RAGEvaluationConfig
from novaeval.scorers.rag_advanced import AdvancedRAGConfig

# Custom core configuration
custom_rag_config = RAGEvaluationConfig(
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

# Custom advanced configuration
custom_advanced_config = AdvancedRAGConfig(
    hallucination_threshold=0.95,
    bias_threshold=0.1,
    toxicity_threshold=0.05,
    bias_categories=["gender", "race", "religion", "political"]
)

# Use custom configurations
suite = ComprehensiveRAGEvaluationSuite(
    model, custom_rag_config, custom_advanced_config
)
```

## API Reference

### Core Classes

#### RAGEvaluationSuite

Main class for core RAG evaluation functionality.

**Constructor:**
```python
RAGEvaluationSuite(model: LLMModel, config: RAGEvaluationConfig = None)
```

**Key Methods:**
- `evaluate_single_metric(metric_name, input_text, output_text, **kwargs)` - Evaluate using a specific metric
- `evaluate_retrieval_pipeline(input_text, output_text, context, **kwargs)` - Focus on retrieval metrics
- `evaluate_generation_pipeline(input_text, output_text, expected_output, **kwargs)` - Focus on generation metrics
- `evaluate_comprehensive(input_text, output_text, expected_output, context, **kwargs)` - Run all core metrics

#### ComprehensiveRAGEvaluationSuite

Extended class that includes advanced safety and conversational metrics.

**Constructor:**
```python
ComprehensiveRAGEvaluationSuite(
    model: LLMModel,
    rag_config: RAGEvaluationConfig = None,
    advanced_config: AdvancedRAGConfig = None
)
```

**Key Methods:**
- `evaluate_comprehensive_plus(input_text, output_text, include_safety_metrics=True, include_conversational_metrics=False, **kwargs)` - Run comprehensive evaluation
- `get_all_available_metrics()` - Get list of all available metrics
- `get_safety_metrics()` - Get list of safety-focused metrics
- `get_conversational_metrics()` - Get list of conversational metrics

### Individual Scorers

#### Context Evaluation Scorers

- **ContextPrecisionScorer** - Evaluates retrieval ranking quality
- **ContextRelevancyScorer** - Measures context relevance to question
- **ContextRecallScorer** - Assesses information completeness in context
- **ContextEntityRecallScorer** - Evaluates entity-level recall

#### Answer Evaluation Scorers

- **AnswerRelevancyScorer** - Measures answer relevance to question
- **AnswerSimilarityScorer** - Compares generated vs expected answers
- **AnswerCorrectnessScorer** - Evaluates factual correctness
- **EnhancedFaithfulnessScorer** - Assesses faithfulness to context

#### Advanced Safety Scorers

- **HallucinationDetectionScorer** - Detects hallucinated content
- **BiasDetectionScorer** - Identifies various forms of bias
- **ToxicityDetectionScorer** - Detects harmful or toxic content
- **ConversationCoherenceScorer** - Evaluates conversational coherence

#### Composite Scorers

- **EnhancedRAGASScorer** - Comprehensive RAGAS-style evaluation
- **RAGTriadScorer** - Simplified three-metric evaluation

### Configuration Classes

#### RAGEvaluationConfig

Configuration for core RAG evaluation parameters.

**Key Parameters:**
- `embedding_model: str` - Model for semantic similarity (default: "all-MiniLM-L6-v2")
- `similarity_threshold: float` - Threshold for similarity metrics (default: 0.7)
- `faithfulness_threshold: float` - Threshold for faithfulness evaluation (default: 0.8)
- `relevancy_threshold: float` - Threshold for relevancy metrics (default: 0.7)
- `ragas_weights: Dict[str, float]` - Weights for RAGAS composite scoring

#### AdvancedRAGConfig

Configuration for advanced safety and quality metrics.

**Key Parameters:**
- `hallucination_threshold: float` - Threshold for hallucination detection (default: 0.8)
- `bias_threshold: float` - Threshold for bias detection (default: 0.3)
- `toxicity_threshold: float` - Threshold for toxicity detection (default: 0.2)
- `bias_categories: List[str]` - Categories of bias to evaluate
- `toxicity_severity_levels: List[str]` - Severity levels for toxicity classification

### Utility Functions

#### Configuration Helpers

```python
# Get optimized configurations
get_rag_config(focus: str) -> RAGEvaluationConfig
# Options: "balanced", "precision", "recall", "speed"

get_advanced_rag_config(focus: str) -> AdvancedRAGConfig
# Options: "balanced", "safety_first", "permissive"
```

#### Scorer Factory Functions

```python
# Create core RAG scorers
create_rag_scorer(scorer_type: str, model: LLMModel, config: RAGEvaluationConfig = None) -> BaseScorer

# Create comprehensive scorers (including advanced)
create_comprehensive_rag_scorer(scorer_type: str, model: LLMModel, rag_config: RAGEvaluationConfig = None, advanced_config: AdvancedRAGConfig = None) -> BaseScorer
```

### ScoreResult Object

All evaluation methods return a `ScoreResult` object with the following structure:

```python
@dataclass
class ScoreResult:
    score: float                    # Numerical score (0.0 to 1.0)
    passed: bool                   # Whether the evaluation passed the threshold
    reasoning: str                 # Detailed explanation of the evaluation
    metadata: Dict[str, Any]       # Additional evaluation metadata
```

## Performance Considerations

### Optimization Strategies

1. **Parallel Execution**: The system automatically runs compatible metrics in parallel to reduce evaluation time.

2. **Configuration Profiles**: Use optimized configurations based on your priorities:
   - `"speed"` - Fastest evaluation with essential metrics only
   - `"precision"` - High-precision evaluation with stricter thresholds
   - `"recall"` - Comprehensive evaluation prioritizing information completeness

3. **Selective Evaluation**: Use individual metrics or metric subsets when full evaluation isn't needed:
   ```python
   # Evaluate only retrieval quality
   retrieval_results = await suite.evaluate_retrieval_pipeline(question, answer, context)

   # Evaluate only safety metrics
   safety_results = await suite.evaluate_comprehensive_plus(
       question, answer, context,
       include_safety_metrics=True,
       include_conversational_metrics=False
   )
   ```

4. **Batch Processing**: For large-scale evaluation, process multiple examples in batches:
   ```python
   async def batch_evaluate(examples):
       tasks = []
       for example in examples:
           task = suite.evaluate_comprehensive(
               example['question'], example['answer'],
               example['expected'], example['context']
           )
           tasks.append(task)

       results = await asyncio.gather(*tasks)
       return results
   ```

### Memory Management

- The system automatically manages embedding model loading and caching
- Large context windows are processed efficiently through chunking strategies
- Metadata is selectively stored to balance detail with memory usage

### Scalability

- Designed for both single-example evaluation and large-scale batch processing
- Supports async/await patterns for efficient concurrent evaluation
- Configurable timeout and retry mechanisms for robust operation

## Best Practices

### Evaluation Strategy

1. **Start with Core Metrics**: Begin with basic RAG evaluation using core metrics to establish baseline performance.

2. **Add Safety Gradually**: Introduce safety metrics (hallucination, bias, toxicity) based on your application's risk profile.

3. **Use Appropriate Configurations**: Select configuration profiles that match your evaluation priorities and performance requirements.

4. **Validate with Human Judgment**: Regularly compare automated evaluation results with human assessments to ensure alignment.

### Threshold Setting

1. **Domain-Specific Tuning**: Adjust thresholds based on your specific domain and quality requirements.

2. **A/B Testing**: Use the evaluation system to compare different RAG implementations or configurations.

3. **Progressive Enhancement**: Start with permissive thresholds and gradually tighten them as your system improves.

### Integration Patterns

1. **CI/CD Integration**: Incorporate evaluation into your continuous integration pipeline for automated quality assurance.

2. **Production Monitoring**: Use lightweight metric subsets for real-time production monitoring.

3. **Development Feedback**: Provide detailed evaluation results to developers for iterative improvement.

## Integration Guide

### Adding to Existing Projects

1. **Installation**: Install the enhanced RAG evaluation system as part of NovaEval.

2. **Basic Integration**:
   ```python
   from novaeval.scorers.rag_comprehensive import RAGEvaluationSuite

   # Add to your existing evaluation pipeline
   rag_evaluator = RAGEvaluationSuite(your_model, your_config)
   results = await rag_evaluator.evaluate_comprehensive(question, answer, expected, context)
   ```

3. **Custom Metrics**: Extend the system with domain-specific metrics by inheriting from `BaseScorer`.

### Framework Integration

The system integrates seamlessly with popular ML frameworks:

- **LangChain**: Use with LangChain RAG implementations
- **LlamaIndex**: Compatible with LlamaIndex evaluation patterns
- **Haystack**: Integrate with Haystack pipeline evaluation
- **Custom Frameworks**: Adapt to any RAG implementation through the standardized interface

### Monitoring and Alerting

```python
# Example monitoring integration
async def monitor_rag_quality(question, answer, context):
    results = await suite.evaluate_comprehensive_plus(
        question, answer, context=context,
        include_safety_metrics=True
    )

    # Alert on quality issues
    if not results["faithfulness"].passed:
        alert_system.send_alert("Faithfulness threshold violated")

    if not results["toxicity_detection"].passed:
        alert_system.send_urgent_alert("Toxic content detected")

    # Log metrics for monitoring
    metrics_logger.log({
        "faithfulness_score": results["faithfulness"].score,
        "bias_score": results["bias_detection"].score,
        "overall_quality": sum(r.score for r in results.values()) / len(results)
    })
```

This comprehensive RAG evaluation system represents a significant advancement in automated RAG quality assessment, providing the tools necessary for building, monitoring, and improving RAG applications at scale.
