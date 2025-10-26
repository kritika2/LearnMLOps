# Testing Guide for LearnMLOps

## Quick Start

### Install dependencies
```bash
pip install -r requirements.txt
```

### Run all tests
```bash
pytest
```

### Run with test runner script
```bash
./run_tests.sh
```

## Test Organization

```
LearnMLOps/
├── tests/
│   ├── __init__.py
│   ├── test_model_selector.py      # 45+ test cases
│   ├── test_smart_splitter.py      # 40+ test cases
│   ├── test_auto_tuner.py          # 35+ test cases
│   ├── test_shadow_evaluator.py    # 50+ test cases
│   └── README.md
├── conftest.py                      # Shared fixtures
├── pytest.ini                       # Pytest configuration
└── run_tests.sh                     # Test runner script
```

## Test Statistics

| Module | Test Cases | Coverage Areas |
|--------|------------|----------------|
| Model Selection | 45+ | Initialization, analysis, suggestions, evaluation, edge cases |
| Smart Splitter | 40+ | Strategies, quality, reproducibility, time series, groups |
| Auto Tuner | 35+ | Grid/random/Bayesian search, comparison, scoring, timeout |
| Shadow Evaluator | 50+ | Prediction, tracking, comparison, statistics, business metrics |
| **Total** | **170+** | Comprehensive coverage |

## Running Tests

### Basic Commands

```bash
# Run all tests
pytest

# Verbose output
pytest -v

# Show print statements
pytest -s

# Run specific file
pytest tests/test_model_selector.py

# Run specific class
pytest tests/test_model_selector.py::TestModelSelector

# Run specific test
pytest tests/test_model_selector.py::TestModelSelector::test_initialization

# Run tests matching keyword
pytest -k "classification"

# Run with coverage
pytest --cov=. --cov-report=html
```

### Using Test Markers

```bash
# Skip slow tests
pytest -m "not slow"

# Run only unit tests
pytest -m unit

# Run only integration tests
pytest -m integration

# Skip tests requiring Optuna
pytest -m "not requires_optuna"
```

### Parallel Execution

```bash
# Install pytest-xdist
pip install pytest-xdist

# Run tests in parallel
pytest -n auto
```

## Test Coverage by Module

### 1. Model Selection (`test_model_selector.py`)

**Core Functionality:**
- ✅ Model pool initialization (regression, classification, clustering, anomaly detection)
- ✅ Dataset analysis (samples, features, problem type detection)
- ✅ Size categorization (small/medium/large)
- ✅ Target type detection (continuous vs categorical)
- ✅ Model suggestions based on dataset characteristics
- ✅ Cross-validation evaluation

**Edge Cases:**
- ✅ Empty datasets
- ✅ Single feature datasets
- ✅ High-dimensional data (more features than samples)
- ✅ Mismatched dimensions
- ✅ Binary vs multiclass classification
- ✅ Very small datasets

### 2. Smart Splitter (`test_smart_splitter.py`)

**Splitting Strategies:**
- ✅ Standard random splitting
- ✅ Stratified splitting (for imbalanced data)
- ✅ Temporal splitting (for time series)
- ✅ Grouped splitting (for grouped data)
- ✅ Cross-validation (for small datasets)

**Quality Checks:**
- ✅ Split ratio verification
- ✅ Class balance maintenance
- ✅ Feature distribution similarity
- ✅ Temporal ordering preservation
- ✅ Group separation

**Reproducibility:**
- ✅ Random seed consistency
- ✅ Different random states produce different splits

### 3. Auto Tuner (`test_auto_tuner.py`)

**Optimization Strategies:**
- ✅ Grid search (exhaustive)
- ✅ Random search (sampling-based)
- ✅ Bayesian optimization (with Optuna)
- ✅ Strategy comparison

**Model Support:**
- ✅ Random Forest (classifier & regressor)
- ✅ Logistic Regression
- ✅ XGBoost (with parameter spaces defined)
- ✅ Auto-detection of model types

**Features:**
- ✅ Custom scoring metrics
- ✅ Different CV folds
- ✅ Timeout handling
- ✅ Optimization time tracking
- ✅ Tuning history recording

### 4. Shadow Evaluator (`test_shadow_evaluator.py`)

**Shadow Evaluation:**
- ✅ Parallel model execution
- ✅ Production model protection
- ✅ Shadow model failure handling
- ✅ Latency tracking
- ✅ Confidence score extraction

**Statistical Comparison:**
- ✅ Latency comparison (Wilcoxon/Mann-Whitney tests)
- ✅ Accuracy comparison (McNemar's test)
- ✅ Error rate analysis
- ✅ Agreement rate calculation

**Production Features:**
- ✅ Results buffering and flushing
- ✅ Business metrics recording
- ✅ Performance summaries
- ✅ Time window filtering

## Continuous Integration

### GitHub Actions Example

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, '3.10', '3.11']
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v2
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest-cov
    
    - name: Run tests
      run: |
        pytest --cov=. --cov-report=xml --cov-report=term
    
    - name: Upload coverage
      uses: codecov/codecov-action@v2
      with:
        file: ./coverage.xml
```

## Best Practices

### When Writing Tests

1. **Use Descriptive Names**: Test names should clearly describe what they test
   ```python
   def test_model_evaluation_with_small_dataset(self):
   ```

2. **Use Fixtures**: Share setup code across tests
   ```python
   @pytest.fixture
   def sample_data(self):
       return X, y
   ```

3. **Test One Thing**: Each test should verify one specific behavior
   
4. **Include Edge Cases**: Test boundary conditions and error scenarios
   
5. **Use Appropriate Assertions**: Be specific about what you're testing
   ```python
   assert result is not None
   assert 0 <= score <= 1
   assert len(predictions) == len(expected)
   ```

### When Running Tests

1. **Before Committing**: Always run tests locally
   ```bash
   pytest -v
   ```

2. **Check Coverage**: Aim for >80% coverage
   ```bash
   pytest --cov=. --cov-report=term
   ```

3. **Fix Warnings**: Address deprecation warnings early
   
4. **Performance**: Skip slow tests during development
   ```bash
   pytest -m "not slow"
   ```

## Troubleshooting

### Common Issues

**Import Errors**
```bash
# Solution: Run from project root
cd /path/to/LearnMLOps
pytest
```

**Missing Dependencies**
```bash
# Solution: Install all dependencies
pip install -r requirements.txt
```

**Optuna Tests Skipped**
```bash
# Solution: Install Optuna
pip install optuna>=3.0.0
```

**Slow Tests**
```bash
# Solution: Run in parallel
pip install pytest-xdist
pytest -n auto
```

## Contributing

When adding new features:

1. ✅ Write tests first (TDD approach)
2. ✅ Ensure all existing tests pass
3. ✅ Add appropriate markers
4. ✅ Update test documentation
5. ✅ Run full test suite before PR

## Test Metrics

- **Total Test Cases**: 170+
- **Expected Coverage**: >85%
- **Average Run Time**: ~30 seconds (without slow tests)
- **Average Run Time**: ~2-3 minutes (with all tests)

## Resources

- [Pytest Documentation](https://docs.pytest.org/)
- [Pytest Fixtures](https://docs.pytest.org/en/stable/fixture.html)
- [Pytest Markers](https://docs.pytest.org/en/stable/mark.html)
- [Coverage.py](https://coverage.readthedocs.io/)

