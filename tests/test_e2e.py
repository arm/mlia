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

NO_ARGS_HELP_CASES = [
    pytest.param(
        ["mlia"],
        (
            "Usage:",
            "Commands",
            "check",
        ),
        id="root",
    ),
    pytest.param(
        ["mlia", "backend"],
        (
            "Usage:",
            "Commands",
            "install",
            "uninstall",
            "list",
        ),
        id="backend",
    ),
    pytest.param(
        ["mlia", "target"],
        (
            "Usage:",
            "Commands",
            "list",
        ),
        id="target",
    ),
]

LIST_COMMAND_CASES = [
    pytest.param(
        ["mlia", "backend", "list"],
        id="canonical-backend",
    ),
    pytest.param(
        ["mlia", "target", "list"],
        id="canonical-target",
    ),
    pytest.param(
        ["mlia-backend", "list"],
        id="legacy-backend",
    ),
    pytest.param(
        ["mlia-target", "list"],
        id="legacy-target",
    ),
]

INCORRECT_ARGUMENT_CASES = [
    pytest.param(
        ["mlia", "bongo"],
        ("No such command 'bongo'.",),
        id="root",
    ),
    pytest.param(
        ["mlia", "backend", "bongo"],
        ("No such command 'bongo'.",),
        id="backend",
    ),
    pytest.param(
        ["mlia", "target", "bongo"],
        ("No such command 'bongo'.",),
        id="target",
    ),
    pytest.param(
        ["mlia-backend", "bongo"],
        (
            "Warning: 'mlia-backend' is deprecated. Use 'mlia backend' instead.",
            "No such command 'bongo'.",
        ),
        id="legacy-backend",
    ),
    pytest.param(
        ["mlia-target", "bongo"],
        (
            "Warning: 'mlia-target' is deprecated. Use 'mlia target' instead.",
            "No such command 'bongo'.",
        ),
        id="legacy-target",
    ),
]


def assert_matches(pattern: str, output: str) -> None:
    """Assert that the e2e output contains the expected pattern."""
    assert re.search(pattern, output), f"Pattern: {pattern}\n\n{output}"


@pytest.mark.parametrize(("argv", "expected_text"), NO_ARGS_HELP_CASES)
def test_e2e_no_arguments_show_help(
    argv: list[str], expected_text: tuple[str, ...], tmp_path: Path
) -> None:
    """Real CLI entry points should show help when called without arguments."""
    result = subprocess.run(  # nosec B603
        argv,
        cwd=tmp_path,
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 2
    for text in expected_text:
        assert text in result.stderr
    assert result.stdout == ""
    assert not (tmp_path / "mlia-output").exists()


@pytest.mark.parametrize("argv", LIST_COMMAND_CASES)
def test_e2e_list_commands_do_not_create_output_directory(
    argv: list[str], tmp_path: Path
) -> None:
    """Management commands should not create an analysis output directory."""
    result = subprocess.run(  # nosec B603
        argv,
        cwd=tmp_path,
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"
    assert not (tmp_path / "mlia-output").exists()


def test_e2e_mistyped_arguments_show_suggestion(tmp_path: Path) -> None:
    """Should show suggestion when called without arguments."""
    result = subprocess.run(  # nosec B603
        ["mlia", "backned"],
        cwd=tmp_path,
        capture_output=True,
        check=False,
        text=True,
    )
    assert result.returncode == 2
    assert "Did you mean 'backend'?" in result.stderr


@pytest.mark.parametrize(("argv", "expected_stderr"), INCORRECT_ARGUMENT_CASES)
def test_e2e_incorrect_arguments_show_error(
    argv: list[str], expected_stderr: tuple[str, ...], tmp_path: Path
) -> None:
    result = subprocess.run(  # nosec B603
        argv,
        cwd=tmp_path,
        capture_output=True,
        check=False,
        text=True,
    )
    assert result.returncode == 2
    for text in expected_stderr:
        assert text in result.stderr


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
