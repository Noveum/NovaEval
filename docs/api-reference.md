---
layout: default
title: API Reference
nav_order: 4
---

# API Reference

Complete reference for all NovaEval classes, methods, and functions.

## Table of Contents

1. [Core Classes](#core-classes)
2. [Datasets](#datasets)
3. [Models](#models)
4. [Scorers](#scorers)
5. [Configuration](#configuration)
6. [Utilities](#utilities)

## Core Classes

### Evaluator

The main orchestrator class for running evaluations.

```python
class Evaluator:
    def __init__(
        self,
        dataset: BaseDataset,
        models: list[BaseModel],
        scorers: list[BaseScorer],
        output_dir: Optional[Union[str, Path]] = None,
        config: Optional[dict[str, Any]] = None,
        max_workers: int = 4,
        batch_size: int = 1,
    )
```

#### Parameters

- **dataset** (`BaseDataset`): The dataset to evaluate on
- **models** (`list[BaseModel]`): List of models to evaluate
- **scorers** (`list[BaseScorer]`): List of scorers to use for evaluation
- **output_dir** (`Optional[Union[str, Path]]`): Directory to save results (default: "./results")
- **config** (`Optional[dict[str, Any]]`): Additional configuration options
- **max_workers** (`int`): Maximum number of worker threads (default: 4)
- **batch_size** (`int`): Batch size for processing samples (default: 1)

#### Methods

##### `run() -> dict[str, Any]`

Run the complete evaluation process.

**Returns**: Dictionary containing aggregated evaluation results

**Example**:
```python
evaluator = Evaluator(dataset, models, scorers)
results = evaluator.run()
```

##### `validate_inputs() -> None`

Validate that all inputs are properly configured.

**Raises**: `ValueError` if any inputs are invalid

##### `from_config(config_path: Union[str, Path]) -> Evaluator`

Create an evaluator from a configuration file.

**Parameters**:
- **config_path** (`Union[str, Path]`): Path to the configuration file

**Returns**: Configured evaluator instance

**Example**:
```python
evaluator = Evaluator.from_config("config.yaml")
```

## Datasets

### BaseDataset

Abstract base class for all datasets.

```python
class BaseDataset(ABC):
    def __init__(
        self,
        name: str,
        num_samples: Optional[int] = None,
        split: str = "test",
        seed: int = 42,
        **kwargs: Any,
    )
```

#### Parameters

- **name** (`str`): Name of the dataset
- **num_samples** (`Optional[int]`): Maximum number of samples to load (None for all)
- **split** (`str`): Dataset split to use (default: "test")
- **seed** (`int`): Random seed for reproducibility (default: 42)

#### Methods

##### `load_data() -> list[dict[str, Any]]`

Load the dataset from its source. **Must be implemented by subclasses.**

**Returns**: List of dataset samples

##### `get_sample(index: int) -> dict[str, Any]`

Get a specific sample by index.

**Parameters**:
- **index** (`int`): Index of the sample

**Returns**: Sample dictionary

##### `get_info() -> dict[str, Any]`

Get dataset information and metadata.

**Returns**: Dictionary with dataset information

### MMLUDataset

Dataset loader for MMLU (Massive Multitask Language Understanding).

```python
class MMLUDataset(BaseDataset):
    def __init__(
        self,
        subset: str = "all",
        num_samples: Optional[int] = None,
        split: str = "test",
        seed: int = 42,
        **kwargs: Any,
    )
```

#### Parameters

- **subset** (`str`): MMLU subset to load (default: "all")
- **num_samples** (`Optional[int]`): Maximum number of samples
- **split** (`str`): Dataset split ("test", "validation", "dev")
- **seed** (`int`): Random seed for reproducibility

#### Available Subsets

```python
MMLU_SUBJECTS = [
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

### HuggingFaceDataset

Dataset loader for HuggingFace datasets.

```python
class HuggingFaceDataset(BaseDataset):
    def __init__(
        self,
        dataset_name: str,
        subset: Optional[str] = None,
        split: str = "test",
        num_samples: Optional[int] = None,
        seed: int = 42,
        **kwargs: Any,
    )
```

#### Parameters

- **dataset_name** (`str`): Name of the HuggingFace dataset
- **subset** (`Optional[str]`): Dataset subset/configuration
- **split** (`str`): Dataset split to use
- **num_samples** (`Optional[int]`): Maximum number of samples
- **seed** (`int`): Random seed for reproducibility

### CustomDataset

Dataset loader for custom data files.

```python
class CustomDataset(BaseDataset):
    def __init__(
        self,
        path: Union[str, Path],
        format: str = "json",
        num_samples: Optional[int] = None,
        seed: int = 42,
        **kwargs: Any,
    )
```

#### Parameters

- **path** (`Union[str, Path]`): Path to the dataset file
- **format** (`str`): File format ("json", "jsonl", "csv")
- **num_samples** (`Optional[int]`): Maximum number of samples
- **seed** (`int`): Random seed for reproducibility

## Models

### BaseModel

Abstract base class for all models.

```python
class BaseModel(ABC):
    def __init__(
        self,
        name: str,
        model_name: str,
        temperature: float = 0.0,
        max_tokens: int = 1000,
        timeout: int = 30,
        **kwargs: Any,
    )
```

#### Parameters

- **name** (`str`): Display name for the model
- **model_name** (`str`): API model name
- **temperature** (`float`): Sampling temperature (default: 0.0)
- **max_tokens** (`int`): Maximum tokens to generate (default: 1000)
- **timeout** (`int`): Request timeout in seconds (default: 30)

#### Methods

##### `generate(prompt: str, **kwargs) -> str`

Generate text from a prompt. **Must be implemented by subclasses.**

**Parameters**:
- **prompt** (`str`): Input prompt for the model
- **kwargs**: Additional generation parameters

**Returns**: Generated text response

##### `estimate_cost(prompt: str, response: str = "") -> float`

Estimate the cost of a generation request.

**Parameters**:
- **prompt** (`str`): Input prompt for the model
- **response** (`str`): Generated response

**Returns**: Estimated cost in USD

### OpenAIModel

Model implementation for OpenAI API.

```python
class OpenAIModel(BaseModel):
    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.0,
        max_tokens: int = 1000,
        timeout: int = 30,
        api_key: Optional[str] = None,
        **kwargs: Any,
    )
```

#### Parameters

- **model_name** (`str`): OpenAI model name (default: "gpt-4o-mini")
- **temperature** (`float`): Sampling temperature
- **max_tokens** (`int`): Maximum tokens to generate
- **timeout** (`int`): Request timeout in seconds
- **api_key** (`Optional[str]`): OpenAI API key (uses OPENAI_API_KEY env var if None)

#### Supported Models

- `gpt-4o-mini`
- `gpt-4o`
- `gpt-4-turbo`
- `gpt-4`
- `gpt-3.5-turbo`

### AnthropicModel

Model implementation for Anthropic API.

```python
class AnthropicModel(BaseModel):
    def __init__(
        self,
        model_name: str = "claude-3-haiku-20240307",
        temperature: float = 0.0,
        max_tokens: int = 1000,
        timeout: int = 30,
        api_key: Optional[str] = None,
        **kwargs: Any,
    )
```

#### Parameters

- **model_name** (`str`): Anthropic model name (default: "claude-3-haiku-20240307")
- **temperature** (`float`): Sampling temperature
- **max_tokens** (`int`): Maximum tokens to generate
- **timeout** (`int`): Request timeout in seconds
- **api_key** (`Optional[str]`): Anthropic API key (uses ANTHROPIC_API_KEY env var if None)

#### Supported Models

- `claude-3-haiku-20240307`
- `claude-3-sonnet-20240229`
- `claude-3-opus-20240229`

## Scorers

### BaseScorer

Abstract base class for all scorers.

```python
class BaseScorer(ABC):
    def __init__(
        self,
        name: str,
        **kwargs: Any,
    )
```

#### Parameters

- **name** (`str`): Name of the scorer

#### Methods

##### `score(prediction: str, ground_truth: str, context: Optional[dict] = None) -> ScoreResult`

Score a prediction against ground truth. **Must be implemented by subclasses.**

**Parameters**:
- **prediction** (`str`): Model prediction
- **ground_truth** (`str`): Expected answer
- **context** (`Optional[dict]`): Additional context

**Returns**: ScoreResult object

### ScoreResult

Result object returned by scorers.

```python
class ScoreResult:
    def __init__(
        self,
        score: float,
        details: Optional[dict[str, Any]] = None,
        metadata: Optional[dict[str, Any]] = None,
    )
```

#### Parameters

- **score** (`float`): Numeric score (0.0 to 1.0)
- **details** (`Optional[dict[str, Any]]`): Additional scoring details
- **metadata** (`Optional[dict[str, Any]]`): Metadata about the scoring process

### AccuracyScorer

Scorer for accuracy evaluation.

```python
class AccuracyScorer(BaseScorer):
    def __init__(
        self,
        extract_answer: bool = False,
        answer_patterns: Optional[list[str]] = None,
        ignore_case: bool = True,
        **kwargs: Any,
    )
```

#### Parameters

- **extract_answer** (`bool`): Whether to extract answer from response (default: False)
- **answer_patterns** (`Optional[list[str]]`): Regex patterns for answer extraction
- **ignore_case** (`bool`): Whether to ignore case in comparison (default: True)

### ExactMatchScorer

Scorer for exact string matching.

```python
class ExactMatchScorer(BaseScorer):
    def __init__(
        self,
        ignore_case: bool = True,
        strip_whitespace: bool = True,
        normalize_punct: bool = False,
        **kwargs: Any,
    )
```

#### Parameters

- **ignore_case** (`bool`): Whether to ignore case (default: True)
- **strip_whitespace** (`bool`): Whether to strip whitespace (default: True)
- **normalize_punct** (`bool`): Whether to normalize punctuation (default: False)

### F1Scorer

Scorer for F1 score calculation.

```python
class F1Scorer(BaseScorer):
    def __init__(
        self,
        tokenizer: Optional[callable] = None,
        **kwargs: Any,
    )
```

#### Parameters

- **tokenizer** (`Optional[callable]`): Custom tokenizer function (default: split on whitespace)

## Configuration

### EvaluationConfig

Configuration class for evaluations.

```python
class EvaluationConfig:
    def __init__(
        self,
        name: str,
        description: Optional[str] = None,
        version: str = "1.0",
        models: list[dict[str, Any]] = None,
        datasets: list[dict[str, Any]] = None,
        scorers: list[dict[str, Any]] = None,
        output: dict[str, Any] = None,
        **kwargs: Any,
    )
```

#### Parameters

- **name** (`str`): Evaluation name
- **description** (`Optional[str]`): Evaluation description
- **version** (`str`): Configuration version
- **models** (`list[dict[str, Any]]`): Model configurations
- **datasets** (`list[dict[str, Any]]`): Dataset configurations
- **scorers** (`list[dict[str, Any]]`): Scorer configurations
- **output** (`dict[str, Any]`): Output configuration

## Utilities

### Logging

```python
from novaeval.utils.logging import setup_logging

setup_logging(
    level: str = "INFO",
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers: Optional[list] = None,
)
```

#### Parameters

- **level** (`str`): Logging level ("DEBUG", "INFO", "WARNING", "ERROR")
- **format** (`str`): Log message format
- **handlers** (`Optional[list]`): Custom log handlers

### Configuration Utilities

```python
from novaeval.utils.config import load_config, save_config

# Load configuration from file
config = load_config("config.yaml")

# Save configuration to file
save_config(config, "config.yaml")
```

## Example Usage

### Basic Evaluation

```python
from novaeval import Evaluator
from novaeval.datasets import MMLUDataset
from novaeval.models import OpenAIModel
from novaeval.scorers import AccuracyScorer

# Setup components
dataset = MMLUDataset(subset="elementary_mathematics", num_samples=10)
model = OpenAIModel(model_name="gpt-4o-mini", temperature=0.0)
scorer = AccuracyScorer(extract_answer=True)

# Create evaluator
evaluator = Evaluator(
    dataset=dataset,
    models=[model],
    scorers=[scorer],
    output_dir="./results"
)

# Run evaluation
results = evaluator.run()

# Access results
accuracy = results["model_results"]["gpt-4o-mini"]["scores"]["accuracy"]["mean"]
print(f"Accuracy: {accuracy:.4f}")
```

### Custom Scorer Example

```python
from novaeval.scorers import BaseScorer, ScoreResult

class CustomScorer(BaseScorer):
    def __init__(self, **kwargs):
        super().__init__(name="custom_scorer", **kwargs)

    def score(self, prediction, ground_truth, context=None):
        # Custom scoring logic
        score = len(prediction) / len(ground_truth)  # Example metric

        return ScoreResult(
            score=min(score, 1.0),
            details={
                "prediction_length": len(prediction),
                "ground_truth_length": len(ground_truth)
            }
        )

# Use custom scorer
scorer = CustomScorer()
evaluator = Evaluator(dataset=dataset, models=[model], scorers=[scorer])
results = evaluator.run()
```

### Configuration File Example

```yaml
# config.yaml
name: "API Reference Example"
description: "Example evaluation configuration"
version: "1.0"

models:
  - provider: "openai"
    model_name: "gpt-4o-mini"
    temperature: 0.0
    max_tokens: 100

datasets:
  - type: "mmlu"
    subset: "elementary_mathematics"
    num_samples: 50

scorers:
  - type: "accuracy"
    extract_answer: true

output:
  directory: "./results"
  formats: ["json", "csv"]
```

```python
# Load and run from config
evaluator = Evaluator.from_config("config.yaml")
results = evaluator.run()
```

---

**Next Steps**: See the [Examples](examples.md) section for more detailed use cases and real-world scenarios.
