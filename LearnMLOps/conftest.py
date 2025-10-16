"""
Pytest configuration file for LearnMLOps test suite
This file contains shared fixtures and configuration for all tests
"""

import pytest
import sys
import os
from pathlib import Path

# Add project directories to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / '01-model-selection'))
sys.path.insert(0, str(project_root / '02-data-splitting'))
sys.path.insert(0, str(project_root / '03-hyperparameter-tuning'))
sys.path.insert(0, str(project_root / '04-shadow-evaluation'))


def pytest_configure(config):
    """Configure pytest with custom settings"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )
    config.addinivalue_line(
        "markers", "requires_optuna: marks tests that require optuna package"
    )


@pytest.fixture(scope="session")
def random_seed():
    """Provide a consistent random seed for reproducible tests"""
    return 42


@pytest.fixture(scope="function")
def temp_output_dir(tmp_path):
    """Provide a temporary directory for test outputs"""
    output_dir = tmp_path / "test_output"
    output_dir.mkdir(exist_ok=True)
    return output_dir


@pytest.fixture(scope="session", autouse=True)
def cleanup_shadow_files():
    """Clean up shadow evaluation JSON files after tests"""
    yield
    # Cleanup after all tests
    import glob
    for file in glob.glob("shadow_results_*.json"):
        try:
            os.remove(file)
        except OSError:
            pass


@pytest.fixture
def suppress_warnings():
    """Suppress warnings during tests"""
    import warnings
    warnings.filterwarnings('ignore')
    yield
    warnings.filterwarnings('default')

