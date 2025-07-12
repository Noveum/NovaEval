# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2024-12-19

### Added
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

### Fixed
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

---

## [0.1.0] - 2024-12-01

### Added
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

---

## Legend
- 🎉 **Added**: New features
- 🐛 **Fixed**: Bug fixes
- 🔧 **Improved**: Enhancements to existing functionality
- 📊 **Coverage**: Test coverage improvements
- 🧪 **Testing**: Test-related changes
- 📝 **Documentation**: Documentation updates
- 🚀 **Performance**: Performance improvements
- ⚠️ **Breaking**: Breaking changes

## v0.3.0 (2025-07-13)

### Feat

- add commitizen configuration with automated version management

## v0.2.2 (2025-07-12)

## v0.2.1 (2025-07-12)

## v0.2.0 (2025-07-12)
