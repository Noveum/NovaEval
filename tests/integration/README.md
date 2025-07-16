# Gemini Integration Tests

This directory contains comprehensive integration tests for the Gemini model implementation in NovaEval.

## Overview

The integration tests validate the `GeminiModel` class against real Google Gemini API endpoints, ensuring:

- ✅ **Authentication & Initialization**: API key validation, model initialization
- ✅ **Text Generation**: Single and batch text generation with various parameters
- ✅ **Cost Tracking**: Token counting and cost estimation accuracy
- ✅ **Error Handling**: Network issues, rate limiting, malformed responses
- ✅ **Framework Integration**: Configuration-based initialization and evaluation workflows
- ✅ **Connection Validation**: API connectivity and model metadata

## Test Categories

### Core Integration Tests (`TestGeminiModelIntegration`)
- Model initialization with real API keys
- Authentication failure scenarios
- Different model variant initialization
- Single and batch text generation
- Response validation and statistics tracking

### Cost Tracking Tests (`TestGeminiModelCostTracking`)
- Token counting accuracy against known samples
- Cost estimation validation
- Edge cases and special characters
- Consistency across model variants

### Framework Integration Tests (`TestGeminiModelEvaluationIntegration`)
- Configuration-based initialization
- Evaluation workflow integration
- Default parameter handling

### Error Handling Tests (`TestGeminiModelErrorHandling`)
- Invalid API key scenarios
- Network connectivity issues
- Rate limiting and quota exceeded
- Malformed response handling
- Timeout scenarios

### Connection Validation Tests (`TestGeminiConnectionValidation`)
- `validate_connection()` method testing
- `get_info()` method accuracy
- Model metadata validation

### Token Counting Tests (`TestGeminiModelTokenCounting`)
- Basic accuracy against known samples
- Edge cases (empty strings, long text, special characters)
- Consistency and performance testing
- API validation when available

### Stress Tests (`TestGeminiModelStressTests`)
- Rapid request stress testing
- Large batch processing
- Concurrent model instances

### Smoke Tests (`TestGeminiModelSmokeTests`)
- Basic functionality verification
- Quick validation of core features

## Test Markers

Tests are categorized using pytest markers:

- `@integration_test`: Full integration tests with real API calls
- `@smoke_test`: Quick smoke tests for basic functionality
- `@slow_test`: Tests that may take longer to complete
- `@stress_test`: Stress tests for performance validation
- `@requires_api_key`: Tests that require a valid API key
- `@gemini`: Gemini-specific tests

## Setup

### Prerequisites

1. **API Key**: You need a valid Google Gemini API key
   ```bash
   export GEMINI_API_KEY="your_api_key_here"
   # or
   export GOOGLE_API_KEY="your_api_key_here"
   ```

2. **Dependencies**: Ensure all required packages are installed
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   ```

3. **Virtual Environment**: Activate your virtual environment
   ```bash
   source .venv/bin/activate  # or your preferred venv
   ```

## Running Tests

### Run All Integration Tests
```bash
pytest tests/integration/
```

### Run Only Gemini Tests
```bash
pytest tests/integration/test_models_gemini_integration.py
```

### Run Specific Test Categories

**Smoke Tests (Quick validation):**
```bash
pytest tests/integration/test_models_gemini_integration.py -m smoke
```

**Core Integration Tests:**
```bash
pytest tests/integration/test_models_gemini_integration.py::TestGeminiModelIntegration
```

**Cost Tracking Tests:**
```bash
pytest tests/integration/test_models_gemini_integration.py::TestGeminiModelCostTracking
```

**Error Handling Tests:**
```bash
pytest tests/integration/test_models_gemini_integration.py::TestGeminiModelErrorHandling
```

**Stress Tests (Performance):**
```bash
pytest tests/integration/test_models_gemini_integration.py::TestGeminiModelStressTests
```

### Run Tests Without API Key
Tests marked with `@requires_api_key` will be skipped if no API key is available:
```bash
pytest tests/integration/test_models_gemini_integration.py -v
```

### Run Tests with Verbose Output
```bash
pytest tests/integration/test_models_gemini_integration.py -v -s
```

### Run Tests with Coverage
```bash
pytest tests/integration/test_models_gemini_integration.py --cov=novaeval.models.gemini
```

## CI/CD Integration

### GitHub Actions
The tests are configured to run in CI/CD pipelines:

```yaml
- name: Run Gemini Integration Tests
  run: |
    pytest tests/integration/test_models_gemini_integration.py -v
  env:
    GEMINI_API_KEY: ${{ secrets.GEMINI_API_KEY }}
```

### Conditional Execution
Tests automatically adapt based on API key availability:
- Tests requiring API keys are skipped if no key is provided
- Non-API tests run regardless of key availability
- Clear error messages indicate why tests are skipped

## Test Data and Fixtures

### Test Prompts
The tests use various prompt sets:
- Basic Q&A prompts
- Mathematical calculations
- Science questions
- Edge cases (empty, long, special characters)

### Expected Patterns
Tests validate responses against expected patterns:
- Factual accuracy
- Response format
- Content relevance

### Performance Benchmarks
Tests include performance expectations:
- Response time limits
- Token counting accuracy
- Cost estimation validation

## Troubleshooting

### Common Issues

1. **API Key Not Found**
   ```
   pytest.skip("No API key available")
   ```
   **Solution**: Set `GEMINI_API_KEY` or `GOOGLE_API_KEY` environment variable

2. **Connection Validation Fails**
   ```
   assert False is True  # validate_connection() returned False
   ```
   **Solution**: Check API key validity and network connectivity

3. **Rate Limiting**
   ```
   Rate limit exceeded
   ```
   **Solution**: Wait and retry, or use a different API key

4. **Import Errors**
   ```
   ModuleNotFoundError: No module named 'google.genai'
   ```
   **Solution**: Install required dependencies: `pip install google-generativeai`

### Debug Mode
Run tests with debug output:
```bash
pytest tests/integration/test_models_gemini_integration.py -v -s --tb=long
```

### Test Isolation
Run individual tests to isolate issues:
```bash
pytest tests/integration/test_models_gemini_integration.py::TestGeminiModelIntegration::test_model_initialization_with_real_api -v
```

## Test Results

### Expected Output
```
============================= test session starts ==============================
platform linux -- Python 3.10.12, pytest-8.4.1, pluggy-1.6.0
Integration Tests Configuration:
  Gemini API Key: Available
  Test Environment: development
collected 45 items

test_models_gemini_integration.py::TestGeminiModelIntegration::test_model_initialization_with_real_api PASSED [  2%]
test_models_gemini_integration.py::TestGeminiModelIntegration::test_single_text_generation PASSED [  4%]
...
============================== 45 passed in 120.34s ==============================
```

### Test Coverage
The integration tests cover:
- ✅ Model initialization and authentication
- ✅ Single and batch text generation
- ✅ Token counting and cost estimation
- ✅ Error handling and edge cases
- ✅ Framework integration
- ✅ Connection validation
- ✅ Performance and stress testing

## Contributing

When adding new tests:

1. **Follow the existing structure** and naming conventions
2. **Use appropriate markers** (`@integration_test`, `@smoke_test`, etc.)
3. **Handle API key requirements** with `@requires_api_key` marker
4. **Add comprehensive docstrings** explaining test purpose
5. **Include edge cases** and error scenarios
6. **Validate test data** and expected patterns

## Performance Considerations

- **API Rate Limits**: Tests are designed to respect API rate limits
- **Cost Management**: Tests use minimal tokens to reduce costs
- **Timeout Handling**: Tests include reasonable timeout expectations
- **Batch Processing**: Batch tests optimize API usage

## Security Notes

- API keys are handled securely through environment variables
- No hardcoded credentials in test files
- Tests validate proper error handling for invalid credentials
- Sensitive data is not logged or exposed in test output 