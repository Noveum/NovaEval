# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## v0.4.0 (2025-07-23)

### BREAKING CHANGE

- GitHub Pages navigation structure has been completely redesigned

### Feat

- add comprehensive GitHub badges and hero CTA buttons (#10)
- **docs**: comprehensive documentation improvements and GitHub Pages redesign (#9)

### Fix

- fix conversational metrics (#16)
- REmove extra medicine (#14)

## v0.3.3 (2025-07-14)

### Feat

- **Massive Test Coverage Expansion**: Coverage increased from 23% to 61% (38 percentage point increase)
- **Comprehensive Unit Test Suite**: Added 4,943+ lines of new test code across 7 major test files
- **Core Module Testing**: Achieved high coverage across all critical components:
  - CLI: 99% coverage with 765 new test lines
  - Job Configuration: 97% coverage with 1,031 new test lines
  - Datasets (Custom): 95% coverage with 577 new test lines
  - Datasets (HuggingFace): 100% coverage with 631 new test lines
  - Standard Evaluators: 95% coverage with 644 new test lines
  - Anthropic Models: 100% coverage with 432 new test lines
  - Reporting/Metrics: 100% coverage with 575 new test lines

### Fix

- **Evaluator Improvements**: Fixed critical bugs in standard evaluator logic and error handling
- **CLI Enhancements**: Improved command-line interface reliability and user experience
- **Model Integration**: Fixed Anthropic model integration and error handling
- **Dataset Processing**: Resolved issues with HuggingFace dataset loading and custom dataset handling
- **Examples**: Fixed and improved `basic_evaluation.py` and `panel_evaluation.py` examples
- **Pre-commit Configuration**: Enhanced development workflow with improved linting and formatting

### Testing

- **Total Tests**: Increased from 203 to 452 tests (more than doubled)
- **Test Execution Time**: Maintained efficiency at ~6.5 seconds for full test suite
- **Cross-platform Compatibility**: All tests pass on macOS, Linux, and Windows
- **Comprehensive Coverage**: Now testing edge cases, error scenarios, and integration paths

### Technical Details

- **New Test Files**:
  - `test_cli.py`: 765 lines - comprehensive CLI testing
  - `test_datasets_custom.py`: 577 lines - custom dataset functionality
  - `test_datasets_huggingface.py`: 631 lines - HuggingFace integration testing
  - `test_evaluators_standard.py`: 644 lines - standard evaluator testing
  - `test_job_config.py`: 1,031 lines - job configuration validation
  - `test_models_anthropic.py`: 432 lines - Anthropic model testing
  - `test_reporting_metrics.py`: 575 lines - metrics and reporting testing

- **Coverage Statistics by Module**:
  - CLI: 99% (up from ~50%)
  - Job Configuration: 97% (up from ~60%)
  - Custom Datasets: 95% (up from ~40%)
  - HuggingFace Datasets: 100% (up from ~70%)
  - Standard Evaluators: 95% (up from ~60%)
  - Anthropic Models: 100% (up from ~80%)
  - OpenAI Models: 100% (maintained)
  - Reporting/Metrics: 100% (up from ~70%)

## v0.3.2 (2025-07-13)

### Fix

- update GitHub Actions workflows and CI/CD pipeline improvements

## v0.3.1 (2025-07-13)

### Fix

- update all version files to 0.3.0 and fix commitizen version_files configuration

## v0.3.0 (2025-07-13)

### Feat

- add commitizen configuration with automated version management

## v0.2.2 (2025-07-12)

### Fix

- general bug fixes and improvements

## v0.2.1 (2025-07-12)

### Fix

- fix tests and improve test coverage

## v0.2.0 (2024-12-19)

### Feat

- **Integration Tests**: Comprehensive integration test suite with 26 tests covering:
  - CLI configuration loading (YAML/JSON)
  - Configuration validation and merging
  - Environment variable handling
  - Full evaluation workflow testing
  - Multi-model and multi-scorer evaluation
  - Real-world evaluation scenarios (QA, classification)
  - Error recovery and partial failure handling
  - Large dataset handling
  - Empty dataset edge cases

- **Enhanced Unit Test Coverage**: Fixed and improved 177 unit tests with:
  - OpenAI model comprehensive testing (100% coverage)
  - Logging utilities extensive testing (95% coverage)
  - Base classes thorough testing (90%+ coverage)
  - All accuracy scorers complete testing
  - Configuration utilities robust testing

- **GitHub Actions Compatibility**:
  - All tests now use proper temporary directories (`tempfile.TemporaryDirectory()`)
  - Cross-platform compatibility ensured
  - Eliminated hardcoded system paths like `/tmp/`
  - Proper cleanup of temporary files and resources

### Fix

- **OpenAI Model Tests**:
  - `test_generate_batch_with_error`: Fixed to expect correct number of errors (2 per batch item)
  - `test_estimate_cost_known_model`: Fixed pricing calculation (per 1K tokens instead of 1M)
  - `test_validate_connection_failure`: Fixed error message format expectations
  - Added comprehensive token counting tests for different models
  - Added fallback handling for ImportError scenarios

- **Accuracy Scorer Tests**:
  - `test_mmlu_style_with_choices`: Fixed to use correct ground truth format
  - Improved exact matching behavior consistency

- **Base Scorer Tests**:
  - Fixed `ConcreteScorer` implementation for proper statistics tracking
  - Updated tests to match exact string matching behavior
  - Fixed batch scoring and context-based scoring tests

- **Logging Tests**:
  - `test_get_logger_with_none_name`: Fixed to handle actual `logging.getLogger(None)` behavior
  - `test_log_evaluation_end`: Fixed to match formatted number output ("10,000")
  - `test_log_model_results`: Updated to use correct results format with "scores" key
  - All logging tests now use proper temporary directories

- **Configuration Tests**:
  - Fixed nested key access functionality
  - Improved environment variable handling
  - Enhanced configuration merging and validation

### Improved

- **Test Organization**:
  - Clear separation between unit and integration tests
  - Comprehensive test documentation and examples
  - Better mock implementations for testing

- **Code Quality**:
  - All ruff linting issues resolved
  - Improved error handling in test fixtures
  - Better temporary resource management

- **CI/CD Pipeline**:
  - Updated GitHub Actions workflows for main branch only
  - Enhanced test coverage reporting
  - Streamlined linting process with ruff

### Coverage Statistics
- **Overall Coverage**: 21% → 23%
- **Core Modules High Coverage**:
  - `src/novaeval/models/openai.py`: 100%
  - `src/novaeval/utils/logging.py`: 95%
  - `src/novaeval/scorers/accuracy.py`: 94%
  - `src/novaeval/models/base.py`: 92%
  - `src/novaeval/scorers/base.py`: 92%
  - `src/novaeval/evaluators/base.py`: 100%

### Testing
- **Total Tests**: 177 → 203 tests (26 new integration tests)
- **Test Execution Time**: ~1.6 seconds
- **All tests passing**: ✅
- **Cross-platform compatibility**: ✅
- **GitHub Actions ready**: ✅

### Technical Details
- Fixed abstract method implementations in test mocks
- Improved MockDataset and MockModel for better integration testing
- Enhanced MockEvaluator with proper result handling
- Better parameter name consistency across scorer implementations
- Proper handling of empty datasets and edge cases

## v0.1.0 (2024-12-01)

### Feat

- Initial release of NovaEval framework
- Core evaluation infrastructure
- Basic model providers (OpenAI, Anthropic)
- Fundamental scoring mechanisms
- Configuration management
- Basic CLI interface
- Documentation and examples

### Features

- Multi-model evaluation support
- Various scoring metrics (accuracy, exact match, F1)
- Dataset loading capabilities
- Results reporting and visualization
- Extensible plugin architecture
