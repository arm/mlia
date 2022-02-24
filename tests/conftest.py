# Copyright (C) 2021-2022, Arm Ltd.
"""Pytest conf module."""
from pathlib import Path
from typing import Any

import pytest
from mlia.core.context import ExecutionContext


@pytest.fixture(name="test_models_path")
def fixture_test_models_path(test_resources_path: Path) -> Path:
    """Return test models path."""
    return test_resources_path / "models"


@pytest.fixture(name="test_resources_path")
def fixture_test_resources_path() -> Path:
    """Return test resources path."""
    return Path(__file__).parent / "test_resources"


@pytest.fixture(name="dummy_context")
def fixture_dummy_context(tmpdir: str) -> ExecutionContext:
    """Return dummy context fixture."""
    return ExecutionContext(working_dir=tmpdir)


def pytest_collection_modifyitems(config: Any, items: Any) -> None:
    """Configure tests collections."""
    mark_tests_as_skipped(config, items, "e2e")


def mark_tests_as_skipped(config: Any, items: Any, marker: str) -> None:
    """Disable tests marked by provided marker."""
    selected_markers = config.getoption("-m")

    marker_disabled = f"not {marker}" in selected_markers
    marker_enabled = marker in selected_markers and not marker_disabled

    for item in items:
        if not marker_enabled and item.get_closest_marker(marker) is not None:
            item.add_marker(
                pytest.mark.skip(reason=f"Tests with {marker=} are disabled by default")
            )
