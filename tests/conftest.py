"""
pytest configuration file for the test suite.

This file contains fixtures and configuration that apply to all tests.
It is automatically discovered and loaded by pytest.
"""

import pytest
import numpy as np
import random
import os
import sys

# Add src directory to path for all tests
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


@pytest.fixture(autouse=True, scope="function")
def reset_random_seeds():
    """
    Automatically reset random seeds before each test function.
    
    This fixture ensures reproducible test results by resetting both
    NumPy's and Python's random number generators to a known state
    before each test runs.
    
    The seed value (42) is arbitrary but consistent, ensuring that:
    - Tests produce the same results across multiple runs
    - Debugging is easier (same conditions can be reproduced)
    - Performance comparisons are fair
    
    Scope: function - Seeds are reset before EACH test function
    Autouse: True - Applied automatically to ALL tests without explicit request
    """
    np.random.seed(42)
    random.seed(42)
    yield
    # Optional: Could reset again after test if needed
    # np.random.seed(42)
    # random.seed(42)


# Configure pytest markers
def pytest_configure(config):
    """
    Register custom pytest markers.
    
    These markers are used to categorize and selectively run tests.
    """
    config.addinivalue_line(
        "markers", "unit: Unit tests for individual components"
    )
    config.addinivalue_line(
        "markers", "integration: Integration tests for component interactions"
    )