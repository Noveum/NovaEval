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

### Pre-commit Hooks

We use pre-commit hooks to ensure code quality before commits:

```bash
# Install hooks (run once after cloning)
pre-commit install

# Run hooks on all files manually
pre-commit run --all-files

# Run specific hooks
pre-commit run black --all-files
pre-commit run ruff --all-files
pre-commit run mypy --all-files

# Update hook versions
pre-commit autoupdate
```

**Available hooks:**
- **black** - Code formatting
- **isort** - Import sorting
- **ruff** - Fast Python linter
- **mypy** - Type checking
- **check-yaml** - YAML validation
- **bandit** - Security linting
- **safety** - Dependency vulnerability scanning

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

## 📦 Release Process

### For Maintainers: Publishing to PyPI

**Prerequisites:**
- PyPI account with API token
- Push access to the main repository
- All tests passing on main branch

**Step 1: Prepare Release**
```bash
# Ensure you're on main branch and up-to-date
git checkout main
git pull origin main

# Update version in pyproject.toml (semantic versioning)
# Update CHANGELOG.md with new version and changes
# Commit version bump
git add pyproject.toml CHANGELOG.md
git commit -m "chore: bump version to X.Y.Z"
git push origin main
```

**Step 2: Create Release Tag**
```bash
# Create and push git tag
git tag -a v0.1.0 -m "Release v0.1.0"
git push origin v0.1.0
```

**Step 3: Build Package**
```bash
# Clean previous builds
rm -rf dist/ build/ src/*.egg-info

# Build wheel and source distribution
python -m build

# Verify package quality
twine check dist/*
```

**Step 4: Upload to PyPI**
```bash
# Upload to PyPI (production)
twine upload dist/*

# When prompted:
# Username: __token__
# Password: pypi-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx (your API token)
```

**Step 5: Verify Release**
```bash
# Test installation from PyPI
pip install novaeval==X.Y.Z

# Test CLI works
novaeval --version
```

### PyPI Configuration

**Set up API Token:**
1. Go to https://pypi.org/manage/account/token/
2. Create new token with "Entire account" scope
3. Store securely (use in upload command)

**For TestPyPI (optional testing):**
```bash
# Upload to TestPyPI first
twine upload --repository testpypi dist/*

# Test install from TestPyPI
pip install --index-url https://test.pypi.org/simple/ novaeval
```

### Version Management

We follow [Semantic Versioning](https://semver.org/):

- **MAJOR** (X.0.0) - Incompatible API changes
- **MINOR** (X.Y.0) - New features (backwards compatible)
- **PATCH** (X.Y.Z) - Bug fixes (backwards compatible)

**Pre-release versions:**
- **Alpha** (X.Y.Z-alpha.N) - Early development
- **Beta** (X.Y.Z-beta.N) - Feature complete, testing
- **RC** (X.Y.Z-rc.N) - Release candidate

### Release Checklist

- [ ] All CI checks passing
- [ ] Version bumped in `pyproject.toml`
- [ ] `CHANGELOG.md` updated
- [ ] Git tag created
- [ ] Package builds successfully
- [ ] Package passes `twine check`
- [ ] Uploaded to PyPI
- [ ] Installation verified
- [ ] Release notes published
- [ ] Documentation updated

### Automated Releases (Future)

Consider setting up automated releases with:
- GitHub Actions for CI/CD
- Automatic PyPI publishing on tag creation
- Changelog generation from commit messages
- Release notes from GitHub releases

## 🔧 Administrative Tasks

### Updating Dependencies

```bash
# Update all dependencies
pip-compile requirements.in
pip-compile requirements-dev.in

# Update pre-commit hooks
pre-commit autoupdate

# Test with updated dependencies
pip install -r requirements-dev.txt
pytest tests/
```

### Security Scanning

```bash
# Check for security vulnerabilities
safety check

# Run security linting
bandit -r src/novaeval/

# Check for outdated packages
pip list --outdated
```

### Performance Monitoring

```bash
# Profile code performance
python -m cProfile -o profile.stats examples/basic_evaluation.py

# Memory usage analysis
python -m memory_profiler examples/basic_evaluation.py
```

## 🙏 Recognition

Contributors will be recognized in:

- The project's README
- Release notes for significant contributions
- The project's contributors page

Thank you for contributing to NovaEval! 🎉
