# Copyright 2021, Arm Ltd.
"""Pytest conf module."""
from pathlib import Path
from typing import Any

import pytest

from tests.utils.common import DummyContext


@pytest.fixture
def test_models_path(
    test_resources_path: Path,  # pylint: disable=redefined-outer-name
) -> Path:
    """Return test models path."""
    return (test_resources_path / "models").absolute()


@pytest.fixture
def test_resources_path() -> Path:
    """Return test resources path."""
    return Path(__file__).parent / "test_resources"


@pytest.fixture
def dummy_context(tmpdir: str) -> DummyContext:
    """Return dummy context fixture."""
    return DummyContext(tmpdir)


def pytest_collection_modifyitems(config: Any, items: Any) -> None:
    """Configure tests collections."""
    mark_tests_as_skipped(config, items, "e2e")


def mark_tests_as_skipped(config: Any, items: Any, marker: str) -> None:
    """Disable tests marked by provided marker."""
    selected_markers = config.getoption("-m")
    marker_enabled = (
        selected_markers.find(marker) != -1
        and selected_markers.find(f"not {marker}") == -1
    )
    for item in items:
        if not marker_enabled and item.get_closest_marker(marker) is not None:
            item.add_marker(
                pytest.mark.skip(
                    reason="Tests with marker {} are disabled by default".format(marker)
                )
            )
