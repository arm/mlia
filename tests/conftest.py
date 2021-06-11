"""Pytest conf module."""
from pathlib import Path

import pytest


@pytest.fixture
def test_models_path() -> Path:
    """Return test models path."""
    test_resources_path = Path(__file__).parent / "test_resources/models"
    return test_resources_path.absolute()
