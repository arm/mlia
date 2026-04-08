# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Pytest-native MLIA e2e tests."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from mlia.testing import e2e as mlia_e2e
from mlia.testing.e2e import (
    COMMON_PATTERNS,
    COMPATIBILITY_PATTERNS,
    PERFORMANCE_PATTERNS,
)

pytestmark = pytest.mark.e2e


def assert_matches(pattern: str, output: str) -> None:
    """Assert that the e2e output contains the expected pattern."""
    assert re.search(pattern, output), f"Pattern: {pattern}\n\n{output}"


@mlia_e2e.parametrize(mlia_e2e.E2E_COMPATIBILITY)
def test_e2e_compatibility(
    case: mlia_e2e.E2ECase,
    tmp_path: Path,
) -> None:
    """Run one compatibility e2e case."""
    result = mlia_e2e.run_case(case, workdir=tmp_path)
    output = f"{result.stdout}\n{result.stderr}"
    assert result.returncode == 0, f"{case}\n\n{output}"
    for pattern in (*COMMON_PATTERNS, *COMPATIBILITY_PATTERNS):
        assert_matches(pattern, output)
    mlia_e2e.emit_e2e_results(result)


@mlia_e2e.parametrize(mlia_e2e.E2E_PERFORMANCE)
def test_e2e_performance(
    case: mlia_e2e.E2ECase,
    tmp_path: Path,
) -> None:
    """Run one performance e2e case."""
    result = mlia_e2e.run_case(case, workdir=tmp_path)
    output = f"{result.stdout}\n{result.stderr}"
    assert result.returncode == 0, f"{case}\n\n{output}"
    for pattern in (*COMMON_PATTERNS, *PERFORMANCE_PATTERNS):
        assert_matches(pattern, output)
    mlia_e2e.emit_e2e_results(result)
