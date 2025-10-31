"""
Pytest configuration and shared fixtures for InterPLM tests.

This file contains fixtures that are shared across multiple test directories.
For directory-specific fixtures, see conftest.py files in subdirectories.
"""

import pytest
import torch
import numpy as np

from interplm.sae.dictionary import ReLUSAE


# ============================================================================
# Shared SAE Fixtures (used by multiple test directories)
# ============================================================================

@pytest.fixture
def small_relu_sae():
    """Small ReLUSAE for testing (32 dim → 64 features).

    Used by test_sae/ and test_train/ tests.
    """
    return ReLUSAE(activation_dim=32, dict_size=64)


# ============================================================================
# Random Seed Fixture (auto-runs for all tests)
# ============================================================================

@pytest.fixture(autouse=True)
def set_random_seed():
    """Set random seeds for reproducibility.

    This fixture runs automatically for all tests (autouse=True).
    """
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)


# ============================================================================
# Pytest Configuration
# ============================================================================

def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "slow: tests that take a long time to run"
    )
    config.addinivalue_line(
        "markers", "integration: integration tests that test multiple components"
    )
