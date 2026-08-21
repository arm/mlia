# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for CLI option exposure."""

from __future__ import annotations

import re

from typer.testing import CliRunner

import mlia.cli.main as cli_main

ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")


def _strip_ansi(value: str) -> str:
    """Remove ANSI escape sequences from captured CLI output."""
    return ANSI_ESCAPE_RE.sub("", value)


def test_check_help_lists_expected_options() -> None:
    """The check command help should expose the shared runtime options."""
    result = CliRunner().invoke(
        cli_main.mlia_app,
        ["check", "--help"],
        terminal_width=120,
    )
    help_output = _strip_ansi(result.stdout)

    assert result.exit_code == 0
    for option in (
        "--target-profile",
        "--output-dir",
        "--backend",
        "--performance",
        "--compatibility",
        "--json",
        "--noninteractive",
        "--debug",
    ):
        assert option in help_output


def test_check_accepts_eula_flag() -> None:
    """The check command parser should accept the EULA flag name."""
    result = CliRunner().invoke(
        cli_main.mlia_app,
        ["check", "--i-agree-to-the-contained-eula"],
        terminal_width=120,
    )

    assert result.exit_code == 2
    assert "Missing argument 'MODEL'" in _strip_ansi(result.output)


def test_backend_install_help_lists_expected_options() -> None:
    """The backend install help should expose the install-time flags."""
    result = CliRunner().invoke(
        cli_main.mlia_app,
        ["backend", "install", "--help"],
        terminal_width=120,
    )
    help_output = _strip_ansi(result.stdout)

    assert result.exit_code == 0
    for option in (
        "--path",
        "--accept-eula",
        "--noninteractive",
        "--force",
        "--debug",
    ):
        assert option in help_output


def test_backend_uninstall_help_lists_debug_option() -> None:
    """The backend uninstall help should expose the shared debug flag."""
    result = CliRunner().invoke(
        cli_main.mlia_app,
        ["backend", "uninstall", "--help"],
        terminal_width=120,
    )
    help_output = _strip_ansi(result.stdout)

    assert result.exit_code == 0
    assert "--debug" in help_output
