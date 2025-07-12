---
layout: default
title: Contributing
nav_order: 6
---

# Contributing to NovaEval

Thank you for your interest in contributing to NovaEval! This guide will help you get started with contributing to our open-source AI evaluation framework.

## Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [Getting Started](#getting-started)
3. [Development Setup](#development-setup)
4. [Contribution Types](#contribution-types)
5. [Development Workflow](#development-workflow)
6. [Code Standards](#code-standards)
7. [Testing](#testing)
8. [Documentation](#documentation)
9. [Submitting Changes](#submitting-changes)
10. [Community](#community)

## Code of Conduct

We are committed to fostering a welcoming and inclusive community. Please read our [Code of Conduct](https://github.com/Noveum/NovaEval/blob/main/CODE_OF_CONDUCT.md) before participating.

### Our Pledge

- **Be respectful**: Treat all community members with respect and kindness
- **Be inclusive**: Welcome contributors from all backgrounds and experience levels
- **Be collaborative**: Work together to improve NovaEval for everyone
- **Be constructive**: Provide helpful feedback and suggestions

## Getting Started

### Prerequisites

Before contributing, ensure you have:

- **Python 3.9+** installed
- **Git** for version control
- **GitHub account** for submitting contributions
- **API keys** for testing (OpenAI, Anthropic, etc.)

### Quick Start

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/NovaEval.git
   cd NovaEval
   ```
3. **Set up development environment** (see detailed instructions below)
4. **Make your changes**
5. **Submit a pull request**

## Development Setup

### 1. Clone and Fork

```bash
# Fork the repository on GitHub first, then clone your fork
git clone https://github.com/YOUR_USERNAME/NovaEval.git
cd NovaEval

# Add the original repository as upstream
git remote add upstream https://github.com/Noveum/NovaEval.git
```

### 2. Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
# Install development dependencies
pip install -e ".[dev]"

# Install hooks
pre-commit install --hook-type pre-commit --hook-type commit-msg
```

### 4. Verify Installation

```bash
# Run tests to ensure everything is working
pytest

# Run linting
pre-commit run --all-files
```

### 5. Environment Setup

Create a `.env` file for development:

```bash
# .env file for development
OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here
AWS_ACCESS_KEY_ID=your_aws_key_here
AWS_SECRET_ACCESS_KEY=your_aws_secret_here
```

## Contribution Types

We welcome various types of contributions:

### 🐛 Bug Reports

Help us identify and fix bugs by:
- **Searching existing issues** before creating new ones
- **Providing detailed reproduction steps**
- **Including error messages and stack traces**
- **Specifying your environment** (OS, Python version, etc.)

**Bug Report Template**:
```markdown
**Bug Description**
A clear description of the bug.

**Reproduction Steps**
1. Step one
2. Step two
3. Step three

**Expected Behavior**
What you expected to happen.

**Actual Behavior**
What actually happened.

**Environment**
- OS: [e.g., macOS 12.0]
- Python: [e.g., 3.11.5]
- NovaEval: [e.g., 0.2.2]

**Additional Context**
Any other relevant information.
```

### ✨ Feature Requests

Suggest new features by:
- **Describing the problem** you're trying to solve
- **Explaining the proposed solution**
- **Providing use cases** and examples
- **Considering backwards compatibility**

### 📝 Documentation

Improve documentation by:
- **Fixing typos** and grammatical errors
- **Adding examples** and tutorials
- **Improving API documentation**
- **Translating content** to other languages

### 🔧 Code Contributions

Contribute code by:
- **Implementing new features**
- **Fixing bugs**
- **Improving performance**
- **Adding tests**

## Development Workflow

### 1. Create a Branch

```bash
# Create and switch to a new branch
git checkout -b feature/your-feature-name

# Or for bug fixes:
git checkout -b fix/bug-description
```

### 2. Make Changes

Follow these guidelines:
- **Write clear commit messages**
- **Keep changes focused** and atomic
- **Add tests** for new functionality
- **Update documentation** as needed

### 3. Test Your Changes

```bash
# Run all tests
pytest

# Run specific test categories
pytest tests/unit/          # Unit tests only
pytest tests/integration/   # Integration tests only

# Run tests with coverage
pytest --cov=src/novaeval --cov-report=html

# Run linting
pre-commit run --all-files
```

### 4. Keep Your Branch Updated

```bash
# Fetch upstream changes
git fetch upstream

# Rebase your branch on latest main
git rebase upstream/main
```

## Code Standards

### Python Code Style

We use several tools to maintain code quality:

- **Black**: Code formatting
- **isort**: Import sorting
- **Ruff**: Linting and additional checks
- **MyPy**: Type checking
- **pytest**: Testing

### Style Guidelines

```python
# Good example
from typing import Optional
from novaeval.models import BaseModel

class MyModel(BaseModel):
    """Example model implementation."""

    def __init__(
        self,
        model_name: str,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
    ) -> None:
        super().__init__(
            name="my_model",
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from the model."""
        # Implementation here
        return "Generated text"
```

### Naming Conventions

- **Classes**: `PascalCase` (e.g., `MyModelClass`)
- **Functions/Methods**: `snake_case` (e.g., `my_function`)
- **Variables**: `snake_case` (e.g., `my_variable`)
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `MAX_TOKENS`)
- **Private methods**: `_snake_case` (e.g., `_private_method`)

### Type Hints

Always use type hints:

```python
from typing import Dict, List, Optional, Union

def process_data(
    data: List[Dict[str, str]],
    max_items: Optional[int] = None,
) -> Dict[str, Union[int, float]]:
    """Process data with type hints."""
    # Implementation
    return {"processed": len(data)}
```

## Testing

### Test Structure

```
tests/
├── unit/                    # Unit tests
│   ├── test_models.py
│   ├── test_scorers.py
│   └── test_datasets.py
├── integration/             # Integration tests
│   ├── test_full_workflow.py
│   └── test_cli.py
└── fixtures/               # Test fixtures
    └── sample_data.json
```

### Writing Tests

#### Unit Tests

```python
import pytest
from unittest.mock import Mock, patch
from novaeval.models import OpenAIModel

class TestOpenAIModel:
    def test_initialization(self):
        """Test model initialization."""
        model = OpenAIModel(model_name="gpt-4")
        assert model.model_name == "gpt-4"
        assert model.temperature == 0.0

    @patch("novaeval.models.openai.OpenAI")
    def test_generation(self, mock_openai):
        """Test text generation."""
        # Mock the OpenAI client
        mock_client = Mock()
        mock_openai.return_value = mock_client

        model = OpenAIModel(model_name="gpt-4")
        result = model.generate("Test prompt")

        assert isinstance(result, str)
        mock_client.chat.completions.create.assert_called_once()
```

#### Integration Tests

```python
import pytest
from novaeval import Evaluator
from novaeval.datasets import MMLUDataset
from novaeval.models import OpenAIModel
from novaeval.scorers import AccuracyScorer

@pytest.mark.integration
def test_full_evaluation_workflow():
    """Test complete evaluation workflow."""
    # Setup
    dataset = MMLUDataset(subset="elementary_mathematics", num_samples=5)
    model = OpenAIModel(model_name="gpt-4o-mini")
    scorer = AccuracyScorer()

    # Run evaluation
    evaluator = Evaluator(dataset=dataset, models=[model], scorers=[scorer])
    results = evaluator.run()

    # Assertions
    assert "model_results" in results
    assert "gpt-4o-mini" in results["model_results"]
    assert "accuracy" in results["model_results"]["gpt-4o-mini"]["scores"]
```

### Test Categories

Mark tests with appropriate categories:

```python
@pytest.mark.unit
def test_unit_functionality():
    """Unit test example."""
    pass

@pytest.mark.integration
def test_integration_workflow():
    """Integration test example."""
    pass

@pytest.mark.slow
def test_expensive_operation():
    """Slow test example."""
    pass
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific categories
pytest -m unit
pytest -m integration
pytest -m "not slow"

# Run with coverage
pytest --cov=src/novaeval --cov-report=html

# Run specific test files
pytest tests/unit/test_models.py
pytest tests/integration/test_cli.py::test_specific_function
```

## Documentation

### Docstring Style

Use Google-style docstrings:

```python
def evaluate_model(
    model: BaseModel,
    dataset: BaseDataset,
    scorer: BaseScorer,
    max_samples: Optional[int] = None,
) -> Dict[str, Any]:
    """Evaluate a model on a dataset.

    Args:
        model: The model to evaluate.
        dataset: The dataset to evaluate on.
        scorer: The scorer to use for evaluation.
        max_samples: Maximum number of samples to evaluate.

    Returns:
        Dictionary containing evaluation results.

    Raises:
        ValueError: If inputs are invalid.

    Example:
        >>> model = OpenAIModel("gpt-4")
        >>> dataset = MMLUDataset("math")
        >>> scorer = AccuracyScorer()
        >>> results = evaluate_model(model, dataset, scorer)
    """
    # Implementation
    pass
```

### README Updates

When adding new features:

1. **Update the main README.md**
2. **Add examples** to the examples section
3. **Update the feature list**
4. **Add API documentation**

### Documentation Site

We use GitHub Pages with Jekyll. To work on documentation:

```bash
# Install Jekyll (macOS)
brew install ruby
gem install jekyll bundler

# Serve documentation locally
cd docs
bundle install
bundle exec jekyll serve

# View at http://localhost:4000
```

## Submitting Changes

### Before Submitting

1. **Run all tests**: `pytest`
2. **Run linting**: `pre-commit run --all-files`
3. **Update documentation** if needed
4. **Add tests** for new functionality
5. **Write clear commit messages**

### Commit Messages

Follow conventional commit format:

```
type(scope): short description

Longer description if needed.

- List any breaking changes
- Reference relevant issues (#123)
```

**Types**:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes
- `refactor`: Code refactoring
- `test`: Test additions/modifications
- `chore`: Maintenance tasks

**Examples**:
```
feat(models): add support for Claude 3 models

Add support for the new Claude 3 model family from Anthropic.

- Add claude-3-opus-20240229 model
- Add claude-3-sonnet-20240229 model
- Add claude-3-haiku-20240307 model
- Update model pricing information

Closes #123
```

### Pull Request Process

1. **Push your branch** to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

2. **Create a pull request** on GitHub

3. **Fill out the PR template** with:
   - **Description** of changes
   - **Testing** performed
   - **Breaking changes** (if any)
   - **Related issues**

4. **Wait for review** and address feedback

5. **Merge** when approved

### Pull Request Template

```markdown
## Description
Brief description of the changes.

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Manual testing performed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Comments added for complex code
- [ ] Documentation updated
- [ ] Tests added/updated
- [ ] No breaking changes (or documented)
```

## Community

### Getting Help

- **GitHub Issues**: For bugs and feature requests
- **GitHub Discussions**: For questions and general discussion
- **Email**: [team@noveum.ai](mailto:team@noveum.ai)
- **Documentation**: [noveum.github.io/NovaEval](https://noveum.github.io/NovaEval)

### Communication Channels

- **GitHub Issues**: Technical discussions
- **GitHub Discussions**: General questions and ideas
- **Pull Requests**: Code review and collaboration
- **Email**: Direct communication with maintainers

### Recognition

We recognize contributors in several ways:

- **Contributors file**: Listed in CONTRIBUTORS.md
- **Release notes**: Major contributions highlighted
- **Social media**: Contributions shared on our channels
- **Noveum community**: Invitation to join our developer community

### Maintainers

Current maintainers:

- **Noveum Team** ([@NoveumTeam](https://github.com/NoveumTeam))
- **Core Contributors** (see CONTRIBUTORS.md)

## Development Tips

### Debugging

Use the following for debugging:

```python
import logging
from novaeval.utils.logging import setup_logging

# Enable debug logging
setup_logging(level="DEBUG")

# Add breakpoints
import pdb; pdb.set_trace()
```

### Performance Profiling

```python
import cProfile
import pstats

# Profile your code
profiler = cProfile.Profile()
profiler.enable()
# Your code here
profiler.disable()

# Analyze results
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(20)
```

### Common Issues

**1. Import Errors**
```bash
# Make sure you're in the right environment
source venv/bin/activate

# Install in development mode
pip install -e ".[dev]"
```

**2. Test Failures**
```bash
# Run specific failing test
pytest tests/unit/test_models.py::test_specific_function -v

# Run with output
pytest -s tests/unit/test_models.py
```

**3. Linting Issues**
```bash
# Auto-fix most issues
black src/ tests/
isort src/ tests/
```

## License

By contributing to NovaEval, you agree that your contributions will be licensed under the [Apache License 2.0](https://github.com/Noveum/NovaEval/blob/main/LICENSE).

---

<div class="contributing-footer">
  <h3>🚀 Ready to Contribute?</h3>
  <p>
    Thank you for considering contributing to NovaEval! Every contribution, no matter how small, helps make AI evaluation better for everyone.
  </p>
  <p>
    <strong>Questions?</strong> Don't hesitate to reach out:
  </p>
  <ul>
    <li><a href="https://github.com/Noveum/NovaEval/discussions">GitHub Discussions</a></li>
    <li><a href="mailto:team@noveum.ai">Email us</a></li>
    <li><a href="https://noveum.ai">Visit Noveum.ai</a></li>
  </ul>
</div>

<style>
.contributing-footer {
  background: #e8f5e8;
  padding: 2rem;
  border-radius: 8px;
  border-left: 4px solid #28a745;
  margin-top: 3rem;
}

.contributing-footer h3 {
  margin-top: 0;
  color: #155724;
}

.contributing-footer ul {
  margin-bottom: 0;
}
</style>
