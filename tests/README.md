# LLM-OCR Test Suite

Comprehensive test suite for the llm-ocr package.

## Setup

Install test dependencies:

```bash
pip install -r requirements-dev.txt
```

## Running Tests

### Run all tests

```bash
pytest
```

### Run with coverage report

```bash
pytest --cov=llm_ocr --cov-report=html
```

### Run specific test file

```bash
pytest tests/test_config.py
```

### Run specific test class or function

```bash
pytest tests/test_openai.py::TestOpenAIProvider::test_init_with_api_key
```

### Run tests by marker

```bash
# Run only unit tests
pytest -m unit

# Run only integration tests
pytest -m integration

# Skip slow tests
pytest -m "not slow"

# Skip tests requiring API keys
pytest -m "not requires_api"
```

### Run with verbose output

```bash
pytest -v
```

### Run with print statements visible

```bash
pytest -s
```

## Test Structure

```
tests/
├── __init__.py              # Test package initialization
├── conftest.py              # Shared fixtures and test configuration
├── test_config.py           # Tests for OCRConfig
├── test_base.py             # Tests for LLMProvider abstract class
├── test_openai.py           # Tests for OpenAI provider
├── test_gemini.py           # Tests for Gemini provider
├── test_prompts.py          # Tests for markdown prompts
├── test_processor.py        # Tests for LLMOCR processor
└── test_integration.py      # Integration tests
```

## Test Categories

### Unit Tests

- `test_config.py`: Configuration validation
- `test_base.py`: Abstract base class behavior
- `test_openai.py`: OpenAI provider functionality
- `test_gemini.py`: Gemini provider functionality
- `test_prompts.py`: Prompt template tests
- `test_processor.py`: Core processing logic

### Integration Tests

- `test_integration.py`: End-to-end workflows, provider switching, package imports

## Mocking Strategy

- **API Calls**: All external API calls are mocked by default
- **File I/O**: Uses pytest's `tmp_path` fixture for temporary files
- **Environment Variables**: Uses `monkeypatch` to set/unset env vars

## Real API Tests

Some tests are marked with `@pytest.mark.requires_api` and will only run if real API keys are present:

```bash
# Run with real OpenAI API (requires OPENAI_API_KEY env var)
OPENAI_API_KEY=your_key pytest -m requires_api

# Run with real Gemini API (requires GEMINI_API_KEY env var)
GEMINI_API_KEY=your_key pytest -m requires_api
```

## Coverage

Generate coverage report:

```bash
pytest --cov=llm_ocr --cov-report=html
open htmlcov/index.html  # View in browser
```

Coverage report is also displayed in terminal after running tests.

## Continuous Integration

The test suite is designed to run in CI/CD pipelines. Example GitHub Actions workflow:

```yaml
- name: Run tests
  run: |
    pip install -r requirements-dev.txt
    pytest --cov=llm_ocr --cov-report=xml
```

## Writing New Tests

When adding new tests:

1. Use descriptive test names that explain what is being tested
2. Follow the Arrange-Act-Assert pattern
3. Use appropriate fixtures from `conftest.py`
4. Add markers for categorization (`@pytest.mark.unit`, `@pytest.mark.integration`, etc.)
5. Mock external dependencies (APIs, file system when appropriate)
6. Test both success and failure cases
7. Test edge cases and boundary conditions

## Common Fixtures

- `mock_api_key`: Provides a mock API key
- `sample_image_bytes`: Sample PNG image bytes for testing
- `sample_image_bytes_list`: List of sample image bytes
- `mock_openai_client`: Mocked OpenAI async client
- `mock_gemini_client`: Mocked Gemini async client
- `set_openai_env_key`: Sets OPENAI_API_KEY environment variable
- `set_gemini_env_key`: Sets GEMINI_API_KEY environment variable
- `unset_env_keys`: Removes API key environment variables
- `temp_output_dir`: Temporary directory for test outputs
- `tmp_path`: pytest's built-in temporary directory fixture
