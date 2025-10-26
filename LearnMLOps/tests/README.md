# LearnMLOps Test Suite

This directory contains comprehensive unit tests for all LearnMLOps modules.

## Test Structure

- `test_model_selector.py` - Tests for the Model Selection module
- `test_smart_splitter.py` - Tests for the Smart Data Splitting module
- `test_auto_tuner.py` - Tests for the Hyperparameter Tuning module
- `test_shadow_evaluator.py` - Tests for the Shadow Evaluation module

## Running Tests

### Run all tests
```bash
pytest
```

### Run tests for a specific module
```bash
pytest tests/test_model_selector.py
pytest tests/test_smart_splitter.py
pytest tests/test_auto_tuner.py
pytest tests/test_shadow_evaluator.py
```

### Run tests with verbose output
```bash
pytest -v
```

### Run tests with coverage report
```bash
pytest --cov=. --cov-report=html
```

### Run specific test class
```bash
pytest tests/test_model_selector.py::TestModelSelector
```

### Run specific test method
```bash
pytest tests/test_model_selector.py::TestModelSelector::test_initialization
```

### Run tests matching a keyword
```bash
pytest -k "classification"
```

### Skip slow tests
```bash
pytest -m "not slow"
```

### Run only integration tests
```bash
pytest -m integration
```

## Test Markers

The test suite uses the following pytest markers:

- `@pytest.mark.slow` - Marks tests that take longer to run
- `@pytest.mark.integration` - Marks integration tests
- `@pytest.mark.unit` - Marks unit tests
- `@pytest.mark.requires_optuna` - Marks tests that require Optuna (skipped if not installed)

## Test Coverage

The test suite covers:

### ModelSelector (`test_model_selector.py`)
- ✅ Model initialization and configuration
- ✅ Dataset analysis (classification, regression, clustering)
- ✅ Dataset size categorization
- ✅ Continuous vs categorical target detection
- ✅ Model suggestions based on dataset characteristics
- ✅ Model evaluation with cross-validation
- ✅ Edge cases (empty datasets, single features, high-dimensional data)

### SmartSplitter (`test_smart_splitter.py`)
- ✅ Data splitting strategies (standard, stratified, temporal, grouped)
- ✅ Dataset analysis and recommendation
- ✅ Cross-validation for small datasets
- ✅ Time series splitting with temporal ordering
- ✅ Grouped data splitting
- ✅ Split quality evaluation
- ✅ Reproducibility with random seeds

### AutoTuner (`test_auto_tuner.py`)
- ✅ Grid search hyperparameter tuning
- ✅ Random search optimization
- ✅ Bayesian optimization (with Optuna)
- ✅ Model type detection
- ✅ Strategy comparison
- ✅ Different scoring metrics
- ✅ Cross-validation configurations
- ✅ Performance on classification and regression

### ShadowEvaluator (`test_shadow_evaluator.py`)
- ✅ Shadow model prediction execution
- ✅ Latency tracking and statistics
- ✅ Error handling for failing models
- ✅ Business metrics recording
- ✅ Performance summary generation
- ✅ Model comparison with statistical tests
- ✅ Buffer management and flushing

## Dependencies

The tests require the following packages (already in `requirements.txt`):
- pytest >= 7.0.0
- scikit-learn >= 1.3.0
- numpy >= 1.24.0
- pandas >= 2.0.0
- scipy >= 1.10.0

Optional:
- pytest-cov (for coverage reports)
- optuna >= 3.0.0 (for Bayesian optimization tests)

## Writing New Tests

When adding new tests, follow these guidelines:

1. **Test Structure**: Use classes to group related tests
2. **Fixtures**: Use pytest fixtures for shared setup
3. **Naming**: Name tests descriptively with `test_` prefix
4. **Documentation**: Add docstrings to test classes and methods
5. **Assertions**: Use clear, specific assertions
6. **Edge Cases**: Test boundary conditions and error cases
7. **Markers**: Add appropriate markers for test categorization

Example:
```python
class TestNewFeature:
    """Test suite for new feature"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing"""
        return X, y
    
    def test_basic_functionality(self, sample_data):
        """Test basic feature functionality"""
        X, y = sample_data
        result = new_feature(X, y)
        assert result is not None
```

## CI/CD Integration

These tests can be integrated into CI/CD pipelines:

```yaml
# Example GitHub Actions workflow
- name: Run tests
  run: |
    pip install -r requirements.txt
    pytest --cov=. --cov-report=xml
```

## Troubleshooting

### Import Errors
If you encounter import errors, make sure you're running pytest from the project root:
```bash
cd /path/to/LearnMLOps
pytest
```

### Missing Dependencies
Install all dependencies:
```bash
pip install -r requirements.txt
```

### Optuna Tests Failing
If Optuna tests are skipped or failing:
```bash
pip install optuna>=3.0.0
```

## Contributing

When contributing new features:
1. Write tests for all new functionality
2. Ensure all existing tests pass
3. Aim for >80% code coverage
4. Run tests locally before submitting PR

