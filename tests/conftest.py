# Copyright 2021, Arm Ltd.
"""Pytest conf module."""
from pathlib import Path

import pytest


@pytest.fixture
def test_models_path(test_resources_path: Path) -> Path:
    """Return test models path."""
    return (test_resources_path / "models").absolute()


@pytest.fixture
def test_resources_path() -> Path:
    """Return test resources path."""
    return Path(__file__).parent / "test_resources"
