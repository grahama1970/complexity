# GitGit Module Test Suite

This directory contains non-mocked tests for the `complexity.gitgit` module, ensuring that CLI commands work as expected.

## Test Structure

The test suite is organized into the following files:

- `conftest.py`: Common pytest fixtures and configuration
- `test_json_utils.py`: Tests for JSON utility functions
- `test_log_utils.py`: Tests for log utility functions
- `test_sparse_clone.py`: Tests for repository sparse cloning functionality
- `test_concat_summarize.py`: Tests for file concatenation and summarization
- `test_code_metadata.py`: Tests for code metadata extraction
- `test_cli.py`: Tests for the Typer CLI interface
- `test_integration.py`: End-to-end workflow integration tests
- `test_llm_summary.py`: Tests for LLM summarization (requires API keys)

## Running the Tests

### Basic Test Run

Run all tests except those marked as slow or requiring API keys:

```bash
pytest -v tests/gitgit/
```

### Run with Network Tests

Run all tests including those that make network calls:

```bash
pytest -v tests/gitgit/
```

### Run Specific Test Files

To run specific test files:

```bash
pytest -v tests/gitgit/test_json_utils.py
pytest -v tests/gitgit/test_sparse_clone.py
```

### Skip Slow Tests

To skip slow tests (like those making network calls):

```bash
pytest -v -m "not slow" tests/gitgit/
```

### Skip LLM API Tests

Some tests require API keys for LLM services. To skip these:

```bash
SKIP_NETWORK_TESTS=1 pytest -v tests/gitgit/
```

## Test Environment

The tests require:

1. Git CLI installed and available in PATH
2. (Optional) API keys for LLM services to test the summarization functionality:
   - Set `GOOGLE_API_KEY` for Google/Gemini models
   - Set `OPENAI_API_KEY` for OpenAI models

## Test Examples

### Example 1: Test JSON Utilities

```bash
pytest -v tests/gitgit/test_json_utils.py
```

### Example 2: Test CLI Interface

```bash
pytest -v tests/gitgit/test_cli.py
```

### Example 3: Run All Tests with Detailed Output

```bash
pytest -v tests/gitgit/
```