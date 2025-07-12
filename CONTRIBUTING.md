# Contributing to NovaEval

Thank you for your interest in contributing to NovaEval! This document provides guidelines and information for contributors.

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- Git
- Docker (optional, for containerized development)

### Development Setup

1. **Fork and clone the repository**
   ```bash
   git clone https://github.com/your-username/NovaEval.git
   cd NovaEval
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install development dependencies**
   ```bash
   pip install -e ".[dev]"
   ```

4. **Install pre-commit hooks**
   ```bash
   pre-commit install
   ```

5. **Run tests to verify setup**
   ```bash
   pytest tests/
   ```

## 🛠️ Development Workflow

### Code Style

We use several tools to maintain code quality:

- **Black** for code formatting
- **isort** for import sorting
- **flake8** for linting
- **mypy** for type checking

Run all checks:
```bash
# Format code
black src/novaeval tests/
isort src/novaeval tests/

# Lint code
flake8 src/novaeval tests/

# Type check
mypy src/novaeval
```

### Testing

We maintain high test coverage with different types of tests:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=novaeval --cov-report=html

# Run specific test types
pytest -m unit          # Unit tests only
pytest -m integration   # Integration tests only
pytest -m "not slow"    # Skip slow tests
```

### Adding New Features

1. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Implement your feature**
   - Add code in appropriate modules
   - Follow existing patterns and conventions
   - Add comprehensive tests
   - Update documentation

3. **Test your changes**
   ```bash
   pytest tests/
   ```

4. **Commit your changes**
   ```bash
   git add .
   git commit -m "feat: add your feature description"
   ```

5. **Push and create a pull request**
   ```bash
   git push origin feature/your-feature-name
   ```

## 📝 Contribution Guidelines

### Types of Contributions

We welcome various types of contributions:

- **Bug fixes** - Fix issues and improve stability
- **New features** - Add new evaluation capabilities
- **Documentation** - Improve docs, examples, and tutorials
- **Performance** - Optimize code and reduce resource usage
- **Tests** - Increase test coverage and reliability

### Commit Message Convention

We follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

Examples:
```
feat(datasets): add support for custom dataset formats
fix(models): handle API timeout errors gracefully
docs(readme): update installation instructions
test(scorers): add tests for accuracy scorer
```

### Pull Request Process

1. **Ensure your PR addresses an issue**
   - Reference the issue number in your PR description
   - If no issue exists, create one first

2. **Provide a clear description**
   - Explain what changes you made
   - Include motivation and context
   - List any breaking changes

3. **Ensure all checks pass**
   - All tests must pass
   - Code coverage should not decrease
   - Linting and type checks must pass

4. **Request review**
   - Tag relevant maintainers
   - Be responsive to feedback

### Code Review Guidelines

When reviewing code:

- **Be constructive** - Provide helpful feedback
- **Be specific** - Point to exact lines and suggest improvements
- **Be respectful** - Remember there's a person behind the code
- **Test the changes** - Verify functionality works as expected

## 🏗️ Architecture Guidelines

### Project Structure

```
src/novaeval/
├── datasets/          # Dataset loaders and processors
├── evaluators/        # Core evaluation logic
├── integrations/      # External service integrations
├── models/           # Model interfaces and implementations
├── reporting/        # Report generation and visualization
├── scorers/          # Scoring mechanisms
└── utils/            # Utility functions and helpers
```

### Design Principles

1. **Modularity** - Components should be loosely coupled
2. **Extensibility** - Easy to add new datasets, models, and scorers
3. **Configurability** - Support configuration-driven workflows
4. **Performance** - Optimize for speed and resource efficiency
5. **Reliability** - Handle errors gracefully and provide good logging

### Adding New Components

#### New Dataset

1. Create a new file in `src/novaeval/datasets/`
2. Inherit from `BaseDataset`
3. Implement required methods
4. Add tests in `tests/unit/datasets/`
5. Update documentation

#### New Model

1. Create a new file in `src/novaeval/models/`
2. Inherit from `BaseModel`
3. Implement required methods
4. Add tests in `tests/unit/models/`
5. Update documentation

#### New Scorer

1. Create a new file in `src/novaeval/scorers/`
2. Inherit from `BaseScorer`
3. Implement required methods
4. Add tests in `tests/unit/scorers/`
5. Update documentation

## 📚 Documentation

### Writing Documentation

- Use clear, concise language
- Provide examples for complex concepts
- Keep documentation up-to-date with code changes
- Follow the existing documentation style

### Building Documentation

```bash
cd docs/
make html
```

## 🐛 Reporting Issues

### Bug Reports

When reporting bugs, please include:

- **Environment details** (Python version, OS, etc.)
- **Steps to reproduce** the issue
- **Expected behavior** vs actual behavior
- **Error messages** and stack traces
- **Minimal code example** that reproduces the issue

### Feature Requests

When requesting features:

- **Describe the problem** you're trying to solve
- **Explain the proposed solution**
- **Consider alternatives** you've thought about
- **Provide use cases** and examples

## 🤝 Community

### Getting Help

- **GitHub Issues** - For bugs and feature requests
- **GitHub Discussions** - For questions and general discussion
- **Documentation** - Check the docs first

### Code of Conduct

We are committed to providing a welcoming and inclusive environment. Please read and follow our [Code of Conduct](CODE_OF_CONDUCT.md).

## 📄 License

By contributing to NovaEval, you agree that your contributions will be licensed under the Apache License 2.0.

## 🙏 Recognition

Contributors will be recognized in:

- The project's README
- Release notes for significant contributions
- The project's contributors page

Thank you for contributing to NovaEval! 🎉
