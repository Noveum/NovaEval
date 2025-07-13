# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Testing
```bash
# Run all tests
pytest

# Run unit tests only
pytest -m unit

# Run integration tests only
pytest -m integration

# Run tests with coverage
pytest --cov=src/novaeval --cov-report=html

# Run specific test file
pytest tests/unit/test_models_anthropic.py

# Run tests in parallel
pytest -n auto
```

### Code Quality
```bash
# Format code
black src/ tests/

# Sort imports
isort src/ tests/

# Lint with ruff
ruff check src/ tests/

# Type checking
mypy src/

# Security check
bandit -r src/

# All quality checks
black src/ tests/ && isort src/ tests/ && ruff check src/ tests/ && mypy src/
```

### Building and Installation
```bash
# Install in development mode
pip install -e ".[dev]"

# Install all optional dependencies
pip install -e ".[all]"

# Build distribution
python -m build

# Install pre-commit hooks
pre-commit install
```

### Running Examples
```bash
# Basic evaluation example
python examples/basic_evaluation.py

# CLI usage
novaeval --config examples/config.yaml

# Run with specific dataset
novaeval --dataset mmlu --subset abstract_algebra --num-samples 10
```

## Architecture Overview

NovaEval is a modular AI evaluation framework with plugin-based architecture:

### Core Components
- **`evaluators/`** - Orchestrates evaluation workflow (BaseEvaluator → Evaluator)
- **`models/`** - AI model interfaces (OpenAI, Anthropic, Noveum providers)
- **`datasets/`** - Data loading (MMLU, HuggingFace, Custom datasets)
- **`scorers/`** - Evaluation metrics (Accuracy, G-Eval, RAG, Conversational)
- **`config/`** - YAML/JSON configuration with Pydantic validation
- **`integrations/`** - External platform integrations (Noveum.ai)
- **`reporting/`** - Results output in multiple formats

### Key Patterns
1. **Abstract Base Classes** - All components inherit from Base* classes using strategy pattern
2. **Factory Pattern** - Configuration-driven instantiation via ModelFactory, DatasetFactory, ScorerFactory
3. **Plugin Architecture** - Extensible via setuptools entry points in pyproject.toml
4. **Configuration-Driven** - YAML/JSON configs with hierarchical overrides and environment variables

### Entry Points
- **CLI**: `novaeval.cli:main`
- **Plugin Registration**: Entry points for datasets, models, scorers in pyproject.toml under `[project.entry-points.*]`

### Evaluation Flow
```
Config → Factories → Components (Models/Datasets/Scorers) → Evaluator → Results
```

## Configuration

### Environment Variables
- `NOVAEVAL_*` - Framework configuration overrides
- `OPENAI_API_KEY` - OpenAI API authentication
- `ANTHROPIC_API_KEY` - Anthropic API authentication
- `AWS_*` - AWS credentials for Bedrock

### Config File Structure
YAML/JSON configs follow hierarchical pattern:
```yaml
dataset:
  type: "mmlu"
  subset: "abstract_algebra"
models:
  - type: "openai"
    model_name: "gpt-4"
scorers:
  - type: "accuracy"
```

## Extension Points

### Adding Custom Components
1. Inherit from appropriate Base* class (BaseModel, BaseDataset, BaseScorer)
2. Register via entry points in pyproject.toml
3. Components auto-discovered through plugin system

### Key Base Classes
- `BaseEvaluator` - Custom evaluation orchestration
- `BaseModel` - New AI model providers
- `BaseDataset` - Custom data sources
- `BaseScorer` - New evaluation metrics

## Important Files
- `src/novaeval/cli.py` - Command-line interface entry point
- `src/novaeval/config/schema.py` - Pydantic configuration schemas
- `src/novaeval/evaluators/standard.py` - Main evaluation logic
- `pyproject.toml` - Plugin registration and build configuration

## Testing Strategy
- Unit tests focus on individual components with mocking
- Integration tests verify component interactions
- Markers: `@pytest.mark.unit`, `@pytest.mark.integration`, `@pytest.mark.slow`
- API tests run only on main branch to minimize costs
