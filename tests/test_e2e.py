# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Pytest-native MLIA e2e tests."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

from mlia.testing import e2e as mlia_e2e
from mlia.testing.e2e import (
    COMMON_PATTERNS,
    COMPATIBILITY_PATTERNS,
    PERFORMANCE_PATTERNS,
)

pytestmark = pytest.mark.e2e

NO_ARGS_HELP_TEXT = {
    "mlia": (
        "usage:",
        "ML Inference Advisor",
        "Supported Targets/Backends:",
        "-h, --help",
        "Commands:",
        "check",
    ),
    "mlia-backend": (
        "usage:",
        "ML Inference Advisor",
        "Supported Targets/Backends:",
        "-h, --help",
        "Commands:",
        "install",
        "uninstall",
        "list",
    ),
    "mlia-target": (
        "usage:",
        "ML Inference Advisor",
        "Supported Targets/Backends:",
        "-h, --help",
        "Commands:",
        "list",
        "List available target profiles",
    ),
}


def assert_matches(pattern: str, output: str) -> None:
    """Assert that the e2e output contains the expected pattern."""
    assert re.search(pattern, output), f"Pattern: {pattern}\n\n{output}"


@pytest.mark.parametrize(
    "command",
    ["mlia", "mlia-backend", "mlia-target"],
)
def test_e2e_no_arguments_show_help(command: str, tmp_path: Path) -> None:
    """Real console scripts should show help when called without arguments."""
    result = subprocess.run(  # nosec B603
        [command],
        cwd=tmp_path,
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 2
    for expected_text in NO_ARGS_HELP_TEXT[command]:
        assert expected_text in result.stdout
    assert not (tmp_path / "mlia-output").exists()


@pytest.mark.parametrize(
    "command",
    ["mlia", "mlia-backend", "mlia-target"],
)
def test_e2e_incorrect_arguments_show_error(command: str, tmp_path: Path) -> None:
    result = subprocess.run(  # nosec B603
        [command, "bongo"],
        cwd=tmp_path,
        capture_output=True,
        check=False,
        text=True,
    )
    assert result.returncode == 2
    assert "error: argument command: invalid choice: 'bongo'" in result.stderr


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
