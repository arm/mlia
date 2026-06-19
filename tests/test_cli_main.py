# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for CLI entry point behavior."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from typer.testing import CliRunner

import mlia.cli.commands as cli_commands
import mlia.cli.main as cli_main
from mlia.api import BackendOptionSpec

ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")


def _strip_ansi(value: str) -> str:
    """Remove ANSI escape sequences from captured CLI output."""
    return ANSI_ESCAPE_RE.sub("", value)


def _backend_option_spec() -> BackendOptionSpec:
    """Return backend option metadata for CLI parser tests."""
    return {
        "module": "bingo_bongo_backend",
        "backend": "bingo-bongo-backend",
        "config_key": "system_config",
        "cli_option": "--system-config",
        "full_cli_option": "--bingo-bongo-backend.system-config",
        "dest": "bingo_bongo_backend_system_config",
        "type": Path,
        "help": "Overrides the --system-config backend option.",
    }


@pytest.mark.parametrize(
    ("app", "expected_text"),
    [
        (
            cli_main.mlia_app,
            (
                "Usage:",
                "Commands",
                "backend",
                "check",
                "target",
            ),
        ),
        (
            cli_main.backend_app,
            (
                "Usage:",
                "Commands",
                "install",
                "uninstall",
                "list",
            ),
        ),
        (
            cli_main.target_app,
            (
                "Usage:",
                "Commands",
                "list",
            ),
        ),
    ],
)
def test_no_arguments_show_help(app: Any, expected_text: tuple[str, ...]) -> None:
    """Calling a CLI app without arguments should show help."""
    result = CliRunner().invoke(app, [])

    assert result.exit_code == 2
    for text in expected_text:
        assert text in result.stdout


def test_main_calls_mlia_app(monkeypatch: pytest.MonkeyPatch) -> None:
    """Main entry point should call the root Typer app."""
    mlia_app = MagicMock()

    monkeypatch.setattr(cli_main, "mlia_app", mlia_app)
    monkeypatch.setattr(cli_main, "_configure_cli_colors", MagicMock(return_value=True))

    cli_main.main()
    mlia_app.assert_called_once_with(color=True)


def test_configure_cli_colors_enables_color_for_tty_without_no_color(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Colors should be enabled for TTY output when NO_COLOR is unset."""
    monkeypatch.delenv("NO_COLOR", raising=False)
    stream = MagicMock()
    stream.isatty.return_value = True
    monkeypatch.setattr(cli_main.sys, "stdout", stream)

    assert cli_main._configure_cli_colors() is True
    assert cli_commands.console.no_color is False


def test_configure_cli_colors_disables_color_when_stdout_is_not_tty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Colors should be disabled for non-TTY output."""
    monkeypatch.delenv("NO_COLOR", raising=False)
    stream = MagicMock()
    stream.isatty.return_value = False
    monkeypatch.setattr(cli_main.sys, "stdout", stream)

    assert cli_main._configure_cli_colors() is False
    assert cli_commands.console.no_color is True


def test_configure_cli_colors_disables_color_when_no_color_is_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Colors should be disabled when NO_COLOR is set."""
    monkeypatch.setenv("NO_COLOR", "1")
    stream = MagicMock()
    stream.isatty.return_value = True
    monkeypatch.setattr(cli_main.sys, "stdout", stream)

    assert not cli_main._configure_cli_colors()
    assert cli_commands.console.no_color


def test_check_without_arguments_shows_help_and_exit_code_2() -> None:
    """The check command should show help and exit with status 2 when empty."""
    result = CliRunner().invoke(cli_main.mlia_app, ["check"], terminal_width=120)

    assert result.exit_code == 2
    assert "Usage:" in result.stdout
    assert "Generate compatibility/performance advice for a model" in result.stdout


def test_check_help_lists_target_profile_option() -> None:
    """The check command help should list the target profile flag."""
    result = CliRunner().invoke(
        cli_main.mlia_app,
        ["check", "--help"],
        terminal_width=120,
    )
    help_output = _strip_ansi(result.stdout)

    assert result.exit_code == 0
    assert "--target-profile" in help_output


def test_check_help_lists_backend_option_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The check command help should list discovered backend-specific options."""
    discover_backend_option_specs = MagicMock(return_value=[_backend_option_spec()])
    monkeypatch.setattr(
        cli_commands,
        "discover_backend_option_specs",
        discover_backend_option_specs,
    )

    result = CliRunner().invoke(
        cli_main.mlia_app,
        ["check", "--help"],
        terminal_width=120,
    )
    help_output = _strip_ansi(result.stdout)

    assert result.exit_code == 0
    discover_backend_option_specs.assert_called()
    assert "--bingo-bongo-backend" in help_output
    assert "Overrides the --system-config" in help_output
    assert "backend option." in help_output


def test_root_help_lists_plugin_discovery_resources() -> None:
    """The root help should point users to plugin discovery resources."""
    result = CliRunner().invoke(
        cli_main.mlia_app,
        ["--help"],
        terminal_width=120,
    )
    help_output = _strip_ansi(result.stdout)

    assert result.exit_code == 0
    assert "Plugin discovery:" in help_output
    assert "mlia target list" in help_output
    assert "mlia backend list" in help_output
    assert "https://github.com/orgs/arm/repositories?q=mlia" in help_output


@pytest.mark.parametrize(
    ("args", "expected_text"),
    [
        (
            ["check", "--compatibility"],
            "Missing argument 'MODEL'",
        ),
        (
            ["check", "--i-agree-to-the-contained-eula"],
            "Missing argument 'MODEL'",
        ),
    ],
)
def test_check_accepts_updated_flag_names(args: list[str], expected_text: str) -> None:
    """Updated long option names should be accepted by the parser."""
    result = CliRunner().invoke(cli_main.mlia_app, args)

    assert result.exit_code == 2
    assert expected_text in result.output


def test_check_accepts_target_profile_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    """The check command should accept the target profile flag."""
    monkeypatch.setattr(cli_commands, "get_advice", MagicMock())
    monkeypatch.setattr(
        cli_commands,
        "validate_check_target_profile",
        MagicMock(),
    )

    result = CliRunner().invoke(
        cli_main.mlia_app,
        ["check", "model.tflite", "--target-profile", "ethos-u55-256"],
    )

    assert result.exit_code == 0


def test_check_passes_backend_options_from_discovered_cli_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The check command should forward dynamic backend options to the API."""
    get_advice = MagicMock()
    discover_backend_option_specs = MagicMock(return_value=[_backend_option_spec()])

    monkeypatch.setattr(
        cli_commands,
        "discover_backend_option_specs",
        discover_backend_option_specs,
    )
    monkeypatch.setattr(cli_commands, "get_advice", get_advice)
    monkeypatch.setattr(
        cli_commands,
        "validate_check_target_profile",
        MagicMock(return_value=True),
    )

    result = CliRunner().invoke(
        cli_main.mlia_app,
        [
            "check",
            "model.tflite",
            "--target-profile",
            "ethos-u55-256",
            "--bingo-bongo-backend.system-config",
            "backend.toml",
        ],
    )

    assert result.exit_code == 0
    discover_backend_option_specs.assert_called()
    get_advice.assert_called_once()
    assert get_advice.call_args.kwargs["backend_options"] == {
        "bingo-bongo-backend": {"system_config": Path("backend.toml")}
    }


def test_check_exits_cleanly_when_validation_skips_all_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The check command should exit 0 when validation reports no runnable checks."""
    get_advice = MagicMock()

    monkeypatch.setattr(cli_commands, "get_advice", get_advice)
    monkeypatch.setattr(
        cli_commands,
        "validate_check_target_profile",
        MagicMock(return_value=False),
    )

    result = CliRunner().invoke(
        cli_main.mlia_app,
        ["check", "model.tflite", "--target-profile", "tosa", "--performance"],
    )

    assert result.exit_code == 0
    get_advice.assert_not_called()


def test_main_dispatches_backend_list(monkeypatch: pytest.MonkeyPatch) -> None:
    """The backend list command should run through the main mlia entry point."""
    format_backend_info = MagicMock()

    monkeypatch.setattr(cli_commands, "setup_logging", MagicMock())
    monkeypatch.setattr(cli_commands, "format_backend_info", format_backend_info)

    result = CliRunner().invoke(cli_main.mlia_app, ["backend", "list"])

    assert result.exit_code == 0
    format_backend_info.assert_called_once_with()


def test_main_dispatches_target_list(monkeypatch: pytest.MonkeyPatch) -> None:
    """The target list command should run through the main mlia entry point."""
    format_target_info = MagicMock()

    monkeypatch.setattr(cli_commands, "setup_logging", MagicMock())
    monkeypatch.setattr(cli_commands, "format_target_info", format_target_info)

    result = CliRunner().invoke(cli_main.mlia_app, ["target", "list"])

    assert result.exit_code == 0
    format_target_info.assert_called_once_with()


def test_backend_main_warns_about_deprecated_entry_point(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Backend entry point should warn before calling the Typer app."""
    backend_app = MagicMock()
    secho = MagicMock()

    monkeypatch.setattr(cli_main, "backend_app", backend_app)
    monkeypatch.setattr(
        cli_main, "_configure_cli_colors", MagicMock(return_value=False)
    )
    monkeypatch.setattr(cli_main.typer, "secho", secho)

    cli_main.backend_main()

    secho.assert_called_once_with(
        cli_main.DEPRECATED_BACKEND_ENTRY_POINT,
        fg=cli_main.typer.colors.YELLOW,
        color=False,
        err=True,
    )
    backend_app.assert_called_once_with(color=False)
    assert capsys.readouterr().err == ""


def test_target_main_warns_about_deprecated_entry_point(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Target entry point should warn before calling the Typer app."""
    target_app = MagicMock()
    secho = MagicMock()

    monkeypatch.setattr(cli_main, "target_app", target_app)
    monkeypatch.setattr(
        cli_main, "_configure_cli_colors", MagicMock(return_value=False)
    )
    monkeypatch.setattr(cli_main.typer, "secho", secho)

    cli_main.target_main()

    secho.assert_called_once_with(
        cli_main.DEPRECATED_TARGET_ENTRY_POINT,
        fg=cli_main.typer.colors.YELLOW,
        color=False,
        err=True,
    )
    target_app.assert_called_once_with(color=False)
    assert capsys.readouterr().err == ""
