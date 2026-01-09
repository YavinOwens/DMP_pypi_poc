"""
Pytest configuration and fixtures
"""

from pathlib import Path

import pytest


@pytest.fixture
def test_data_dir():
    """Fixture providing path to test data directory"""
    return Path(__file__).parent / "data"


@pytest.fixture
def sample_config():
    """Fixture providing sample configuration"""
    return {
        "models": {
            "test-model": {
                "display": "Test Model",
                "cost_per_1M_input": 1.0,
                "cost_per_1M_output": 2.0,
                "enabled": True,
            }
        },
        "test_prompts": {
            "test_prompt": {
                "prompt": "This is a test prompt",
                "expected_elements": ["test"],
                "test_type": "test",
                "enabled": True,
            }
        },
    }
