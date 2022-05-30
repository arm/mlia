# SPDX-FileCopyrightText: Copyright 2022, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Pytest conf module."""
from pathlib import Path

import pytest

from mlia.core.context import ExecutionContext


@pytest.fixture(scope="session", name="test_resources_path")
def fixture_test_resources_path() -> Path:
    """Return test resources path."""
    return Path(__file__).parent / "test_resources"


@pytest.fixture(name="dummy_context")
def fixture_dummy_context(tmpdir: str) -> ExecutionContext:
    """Return dummy context fixture."""
    return ExecutionContext(working_dir=tmpdir)
