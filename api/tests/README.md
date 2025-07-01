# CinderSight API Tests

This directory contains comprehensive tests for the CinderSight API backend.

## Test Structure

```
tests/
├── __init__.py           # Makes tests a Python package
├── conftest.py          # Pytest configuration and fixtures
├── test_main.py         # Tests for FastAPI endpoints
├── test_predict.py      # Tests for prediction module
├── test_db.py           # Tests for database operations
└── README.md            # This file
```

## Test Categories

### 1. **API Endpoint Tests** (`test_main.py`)
- **Health endpoints**: Root and health check endpoints
- **Prediction endpoint**: POST /predict functionality
- **Error handling**: Invalid requests, wrong content types
- **Response validation**: Data structure and types

### 2. **Prediction Module Tests** (`test_predict.py`)
- **Basic functionality**: Core prediction logic
- **Input validation**: Different locations and dates
- **Response structure**: Data types and value ranges
- **Edge cases**: Invalid inputs and boundary conditions

### 3. **Database Tests** (`test_db.py`)
- **Covariates fetching**: Environmental data retrieval
- **Data validation**: Types, ranges, and structure
- **Location handling**: Different geographic coordinates
- **Date handling**: Various date formats and seasons

## Running Tests

### Basic Test Commands

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/test_main.py

# Run specific test class
pytest tests/test_main.py::TestHealthEndpoints

# Run specific test method
pytest tests/test_main.py::TestHealthEndpoints::test_root_endpoint
```

### Using the Test Runner Script

```bash
# Run all tests with verbose output
python run_tests.py -v

# Run with coverage report
python run_tests.py -c

# Run only fast tests (skip slow ones)
python run_tests.py -f

# Run only unit tests
python run_tests.py -u

# Run only integration tests
python run_tests.py -i

# Run specific test files
python run_tests.py tests/test_main.py tests/test_predict.py
```

### Test Markers

```bash
# Run only unit tests
pytest -m unit

# Run only integration tests
pytest -m integration

# Skip slow tests
pytest -m "not slow"

# Run tests with specific marker
pytest -m "unit and not slow"
```

## Test Fixtures

The `conftest.py` file provides reusable test fixtures:

- **`client`**: FastAPI TestClient instance
- **`sample_prediction_request`**: Sample request data
- **`sample_prediction_response`**: Sample response data

## Writing New Tests

### Test Naming Convention
- Test files: `test_*.py`
- Test classes: `Test*`
- Test methods: `test_*`

### Example Test Structure

```python
import pytest
from fastapi.testclient import TestClient

class TestNewFeature:
    """Test description for the new feature."""
    
    def test_basic_functionality(self, client: TestClient):
        """Test basic functionality of the new feature."""
        # Arrange
        test_data = {"key": "value"}
        
        # Act
        response = client.post("/new-endpoint", json=test_data)
        
        # Assert
        assert response.status_code == 200
        assert "expected_field" in response.json()
```

### Test Categories

Use markers to categorize your tests:

```python
import pytest

@pytest.mark.unit
def test_unit_function():
    """Unit test for a specific function."""
    pass

@pytest.mark.integration
def test_integration_feature():
    """Integration test for a feature."""
    pass

@pytest.mark.slow
def test_slow_operation():
    """Test that takes a long time to run."""
    pass
```

## Test Coverage

To check test coverage:

```bash
# Install coverage
pip install pytest-cov

# Run with coverage
pytest --cov=app --cov-report=html --cov-report=term

# View HTML coverage report
open htmlcov/index.html
```

## Continuous Integration

Tests are automatically run in GitHub Actions:
- **On pull requests**: All tests must pass
- **On main branch**: Tests run before deployment
- **Coverage reporting**: Available in CI logs

## Best Practices

1. **Test one thing at a time**: Each test should verify one specific behavior
2. **Use descriptive names**: Test names should explain what they're testing
3. **Arrange-Act-Assert**: Structure tests with clear sections
4. **Use fixtures**: Reuse common test data and setup
5. **Test edge cases**: Include tests for invalid inputs and boundary conditions
6. **Keep tests fast**: Avoid slow operations in unit tests
7. **Mock external dependencies**: Don't rely on external services in unit tests

## Debugging Tests

```bash
# Run with debug output
pytest -v -s

# Run single test with debugger
pytest tests/test_main.py::test_specific_test -s

# Run with print statements visible
pytest -s

# Stop on first failure
pytest -x
```

## Common Issues

### Import Errors
Make sure you're running tests from the `api` directory:
```bash
cd api
pytest
```

### Module Not Found
Ensure the `app` module is in your Python path:
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
pytest
```

### FastAPI TestClient Issues
Make sure you have `httpx` installed:
```bash
pip install httpx
``` 