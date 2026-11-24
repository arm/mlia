# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for CLI entry point behavior."""

from __future__ import annotations

import json
import logging
import re
import sys
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any, TextIO, cast
from unittest.mock import MagicMock, call, patch

import click
import pytest
import typer.completion
from typer import rich_utils
from typer.testing import CliRunner

import mlia.api as mlia_api
import mlia.cli.commands as cli_commands
import mlia.cli.main as cli_main
import mlia.core.application_settings as cli_settings
from mlia.backend.options import BackendOptionSpec
from mlia.check import CheckError, CheckResult
from mlia.cli.settings import configure_typer_help
from mlia.core import check_validation as command_validators
from mlia.core.errors import ConfigurationError
from mlia.core.logging import close_configured_handlers
from mlia.core.settings import ApplicationSettings
from mlia.plugins.analysis import AnalysisPluginRegistry, AnalysisRunResult
from mlia.utils.boundaries.completion import install_shell_completion_with_notice

ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")
ORIGINAL_NO_PLUGINS_INSTALLED = cli_commands._no_plugins_installed
TYPER_HELP_STYLE_NAMES = (
    "STYLE_OPTION",
    "STYLE_SWITCH",
    "STYLE_NEGATIVE_OPTION",
    "STYLE_NEGATIVE_SWITCH",
    "STYLE_METAVAR",
    "STYLE_USAGE",
    "STYLE_REQUIRED_SHORT",
    "STYLE_REQUIRED_LONG",
    "STYLE_COMMANDS_TABLE_FIRST_COLUMN",
    "STYLE_OPTIONS_PANEL_BORDER",
    "STYLE_COMMANDS_PANEL_BORDER",
)


def _strip_ansi(value: str) -> str:
    """Remove ANSI escape sequences from captured CLI output."""
    return ANSI_ESCAPE_RE.sub("", value)


@pytest.fixture(autouse=True)
def _assume_plugins_available(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep unrelated CLI tests focused on their mocked command behavior."""
    monkeypatch.setattr(cli_commands, "_no_plugins_installed", lambda: False)


@pytest.mark.parametrize(
    "args",
    [
        ["check", "model.tflite"],
        ["check", "model.tflite", "--target-profile", "example"],
        ["check", "model.tflite", "--json"],
        ["target", "list"],
        ["backend", "list"],
    ],
)
def test_commands_explain_missing_plugins(
    monkeypatch: pytest.MonkeyPatch, args: list[str]
) -> None:
    """Commands provide installation guidance when no plugins are installed."""
    monkeypatch.setattr(cli_commands, "_no_plugins_installed", lambda: True)

    result = CliRunner().invoke(cli_main.mlia_app, args)

    output = _strip_ansi(result.output)
    assert "No MLIA plugins are installed." in output
    assert "MLIA requires plugins to analyse models." in output
    assert "model converters.\nFind available plugins" in output
    assert cli_commands.NO_PLUGINS_URL in output
    assert "No backends found." not in output
    assert "No plugins found." not in output
    if args[0] == "check":
        assert result.stdout == ""
        assert "No MLIA plugins are installed." in _strip_ansi(result.stderr)
    else:
        assert result.stderr == ""
        assert "No MLIA plugins are installed." in _strip_ansi(result.stdout)


@pytest.mark.parametrize(
    "installed_group",
    [
        cli_commands.TARGET_PLUGIN_GROUP,
        cli_commands.BACKEND_PLUGIN_GROUP,
        cli_commands.TRANSFORMER_PLUGIN_GROUP,
        cli_commands.ANALYSIS_PLUGIN_GROUP,
    ],
)
def test_any_supported_plugin_prevents_no_plugins_message(
    monkeypatch: pytest.MonkeyPatch, installed_group: str
) -> None:
    """An installed plugin in any supported group prevents a false warning."""
    monkeypatch.setattr(
        cli_commands,
        "list_entry_points",
        lambda group: [object()] if group == installed_group else [],
    )

    assert not ORIGINAL_NO_PLUGINS_INSTALLED()


def test_cli_entry_point_does_not_hide_missing_plugin_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unused CLI entry point cannot suppress missing-plugin guidance."""
    monkeypatch.setattr(
        cli_commands,
        "list_entry_points",
        lambda group: [object()] if group == "mlia.plugin.cli" else [],
    )

    assert ORIGINAL_NO_PLUGINS_INSTALLED()


def test_plan_preview_reports_invalid_target_without_traceback(tmp_path: Path) -> None:
    """An invalid target is a usage error, including with an explicit backend."""
    model = tmp_path / "model.tflite"
    model.write_bytes(b"model")
    result = CliRunner().invoke(
        cli_commands.mlia_app,
        [
            "check",
            str(model),
            "-t",
            "missing-target",
            "--backend",
            "missing-backend",
            "--show-plan",
            "--output-dir",
            str(tmp_path / "output"),
        ],
    )
    assert result.exit_code != 0
    assert "Error:" in result.output
    assert "missing-target" in result.output
    assert "Traceback" not in result.output
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize(
    "app",
    [cli_commands.mlia_app, cli_commands.backend_app, cli_commands.target_app],
)
def test_completion_options_have_clear_names_and_help(app: Any) -> None:
    """All CLI entry points advertise the renamed completion options."""
    help_result = CliRunner().invoke(app, ["--help"])

    assert help_result.exit_code == 0
    help_text = " ".join(_strip_ansi(help_result.stdout).split())
    assert "--enable-completion" in help_text
    assert "Set up Tab completion for your shell" in help_text
    assert "--print-completion-script" in help_text
    assert "Print the shell completion script" in help_text
    assert "--install-completion" not in help_text
    assert "--show-completion" not in help_text


def test_print_completion_script_uses_new_option(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The renamed option still prints a usable shell completion script."""
    monkeypatch.setattr(typer.completion, "_get_shell_name", lambda: "bash")

    result = CliRunner().invoke(cli_commands.mlia_app, ["--print-completion-script"])

    assert result.exit_code == 0
    assert "_root_completion()" in result.stdout


def test_shell_completion_notice_precedes_install(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Completion installation is announced before Typer changes shell state."""
    events = MagicMock()
    monkeypatch.setattr(
        "mlia.utils.boundaries.completion.log_boundary_action", events.notice
    )
    monkeypatch.setattr(
        "mlia.utils.boundaries.completion.install_callback", events.install
    )
    logger = logging.getLogger("test")
    ctx = MagicMock(resilient_parsing=False)
    param = MagicMock()

    result = install_shell_completion_with_notice(logger, ctx, param, True)

    assert result is events.install.return_value
    assert events.mock_calls == [
        call.notice(logger, "Installing shell completion."),
        call.install(ctx, param, True),
    ]


def test_enable_completion_uses_logged_gateway(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The installation notice is visible before Typer starts installing."""

    def start_installation(*_args: Any) -> None:
        click.echo("INSTALLATION_STARTED")
        raise SystemExit(0)

    installer = MagicMock(side_effect=start_installation)
    monkeypatch.setattr("mlia.utils.boundaries.completion.install_callback", installer)

    try:
        result = CliRunner().invoke(cli_commands.mlia_app, ["--enable-completion"])
    finally:
        close_configured_handlers()

    assert result.exit_code == 0
    assert result.stdout.index("Installing shell completion.") < result.stdout.index(
        "INSTALLATION_STARTED"
    )
    installer.assert_called_once()


def _backend_option_spec() -> BackendOptionSpec:
    """Return backend option metadata for CLI parser tests."""
    return {
        "module": "bingo_bongo_backend",
        "backend": "bingo-bongo-backend",
        "config_key": "system_config",
        "click_option": click.Option(
            [
                "--bingo-bongo-backend.system-config",
                "bingo_bongo_backend_system_config",
            ],
            default=None,
            type=Path,
            help="Overrides the --system-config backend option.",
        ),
    }


def _typed_backend_option_spec() -> BackendOptionSpec:
    """Return typed backend option metadata."""
    optimization_level = click.Option(
        ["--typed-backend.optimization-level", "typed_backend_optimization_level"],
        default=None,
        type=click.Choice(["0", "1", "2"]),
        help="Set optimization level.",
    )
    return {
        "module": "typed_backend",
        "backend": "typed-backend",
        "config_key": "optimization_level",
        "click_option": optimization_level,
    }


class DemoAnalysisPlugin:
    """Analysis plugin used by CLI integration tests."""

    name = "demo"

    def __init__(self) -> None:
        """Create a plugin with captured results."""
        self.results: list[AnalysisRunResult] = []

    def cli_options(self) -> list[click.Option]:
        """Return the demo plugin option."""
        return [
            click.Option(
                ["--demo-analysis"],
                is_flag=True,
                default=False,
                help="Run the demo post-analysis plugin.",
            )
        ]

    def enabled(self, args: Mapping[str, object]) -> bool:
        """Return whether the demo plugin is enabled."""
        return bool(args.get("demo_analysis"))

    def run(self, result: AnalysisRunResult) -> None:
        """Capture the analysis run result."""
        self.results.append(result)


class FileAnalysisPlugin(DemoAnalysisPlugin):
    """Read a Click-owned resource while the command context is still open."""

    contents: str = ""

    def cli_options(self) -> list[click.Option]:
        """Add a file input to the ordinary demo options."""
        return [
            *super().cli_options(),
            click.Option(["--plugin-input"], type=click.File("r")),
        ]

    def run(self, result: AnalysisRunResult) -> None:
        """Read the input during execution and retain the result for assertions."""
        self.contents = cast(TextIO, result.args["plugin_input"]).read()
        super().run(result)


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
@pytest.mark.parametrize("colors", ("", "1"))
def test_no_arguments_show_help(
    app: Any,
    expected_text: tuple[str, ...],
    colors: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Calling a CLI app without arguments should show help."""
    monkeypatch.setenv("NO_COLOR", colors)
    result = CliRunner().invoke(app, [])

    assert result.exit_code == 2
    assert result.stderr == ""
    assert "COMMAND [ARGS]..." in result.stdout
    for text in expected_text:
        assert text in result.stdout


@pytest.mark.parametrize(
    ("theme", "expected_colors"),
    [
        ("dark", ("107;155;248", "184;190;203", "246;155;79")),
        ("light", ("1;87;255", "64;69;79", "219;130;50")),
    ],
)
def test_help_uses_selected_mlia_colors(
    monkeypatch: pytest.MonkeyPatch,
    theme: cli_settings.ThemeName,
    expected_colors: tuple[str, ...],
) -> None:
    """Generated Rich help should render with the selected MLIA palette."""
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setattr(rich_utils, "FORCE_TERMINAL", True)
    monkeypatch.setattr(rich_utils, "COLOR_SYSTEM", "truecolor")
    for style_name in TYPER_HELP_STYLE_NAMES:
        monkeypatch.setattr(rich_utils, style_name, getattr(rich_utils, style_name))
    configure_typer_help(True, theme)

    result = CliRunner().invoke(cli_main.mlia_app, ["--help"], color=True)

    assert result.exit_code == 0
    for expected_color in expected_colors:
        assert f"38;2;{expected_color}" in result.stdout


def test_root_help_colors_quoted_commands_and_uses_one_blank_line(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Color quoted commands and separate the final note by one blank line."""
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setattr(rich_utils, "FORCE_TERMINAL", True)
    monkeypatch.setattr(rich_utils, "COLOR_SYSTEM", "truecolor")
    monkeypatch.setattr(cli_main.mlia_app, "rich_markup_mode", "rich")
    monkeypatch.setattr(cli_commands, "list_entry_points", lambda group: [])
    for style_name in TYPER_HELP_STYLE_NAMES:
        monkeypatch.setattr(rich_utils, style_name, getattr(rich_utils, style_name))
    configure_typer_help(True)

    result = CliRunner().invoke(cli_main.mlia_app, ["--help"], color=True)

    assert result.exit_code == 0
    assert re.search(r"'\x1b\[[0-9;]*mmlia target list", result.stdout)
    assert re.search(r"'\x1b\[[0-9;]*mmlia backend list", result.stdout)
    lines = _strip_ansi(result.stdout).splitlines()
    final_note = next(
        index
        for index, line in enumerate(lines)
        if line.startswith("No target plugins")
    )
    assert not lines[final_note - 1].strip()
    assert lines[final_note - 2].strip()


@pytest.mark.parametrize(
    ("theme", "required_color"),
    [("dark", "237;86;77"), ("light", "180;24;27")],
)
def test_help_uses_selected_required_color(
    monkeypatch: pytest.MonkeyPatch,
    theme: cli_settings.ThemeName,
    required_color: str,
) -> None:
    """Required options and arguments should use the selected theme color."""
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setattr(rich_utils, "FORCE_TERMINAL", True)
    monkeypatch.setattr(rich_utils, "COLOR_SYSTEM", "truecolor")
    for style_name in TYPER_HELP_STYLE_NAMES:
        monkeypatch.setattr(rich_utils, style_name, getattr(rich_utils, style_name))
    configure_typer_help(True, theme)

    check_help = CliRunner().invoke(cli_main.mlia_app, ["check", "--help"], color=True)
    backend_help = CliRunner().invoke(
        cli_main.backend_app, ["install", "--help"], color=True
    )

    assert check_help.exit_code == 0
    assert backend_help.exit_code == 0
    ansi_color = f"\x1b[38;2;{required_color}m"
    assert "[required]" not in _strip_ansi(check_help.stdout)
    assert f"{ansi_color}*" in backend_help.stdout


def test_required_help_has_no_ansi_when_color_is_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Required markers should have no ANSI color when color is disabled."""
    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.setattr(rich_utils, "FORCE_TERMINAL", True)
    monkeypatch.setattr(rich_utils, "STYLE_REQUIRED_SHORT", "#ED564D")
    monkeypatch.setattr(rich_utils, "STYLE_REQUIRED_LONG", "#ED564D")

    result = CliRunner().invoke(
        cli_main.backend_app, ["install", "--help"], color=False
    )

    assert result.exit_code == 0
    assert "*" in result.stdout
    assert "\x1b[38;" not in result.stdout


def test_main_calls_mlia_app(monkeypatch: pytest.MonkeyPatch) -> None:
    """Main entry point should call the root Typer app."""
    mlia_app = MagicMock()
    configure_typer_help = MagicMock()

    monkeypatch.setattr(cli_settings, "_color_enabled", lambda: None)
    monkeypatch.setattr(cli_main, "mlia_app", mlia_app)
    monkeypatch.setattr(cli_main, "configure_typer_help", configure_typer_help)
    monkeypatch.setattr(cli_settings.tomllib, "load", lambda x: {})

    cli_main.main()
    configure_typer_help.assert_called_once_with(True, "dark")
    mlia_app.assert_called_once_with(color=True)


def test_color_enabled_enables_color_for_tty_without_no_color(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Delegate TTY colors when the environment does not disable them."""
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.delenv("MLIA_NO_COLOR", raising=False)
    monkeypatch.setenv("TERM", "xterm-256color")
    stream = MagicMock()
    stream.isatty.return_value = True
    monkeypatch.setattr(cli_settings.sys, "stdout", stream)
    assert cli_settings._color_enabled() is None


def test_color_enabled_disables_color_when_stdout_is_not_tty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Colors should be disabled for non-TTY output."""
    monkeypatch.delenv("NO_COLOR", raising=False)
    stream = MagicMock()
    stream.isatty.return_value = False
    monkeypatch.setattr(cli_settings.sys, "stdout", stream)

    assert cli_settings._color_enabled() is False


def test_color_enabled_disables_color_when_no_color_is_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Colors should be disabled when NO_COLOR is set."""
    monkeypatch.setenv("NO_COLOR", "1")
    stream = MagicMock()
    stream.isatty.return_value = True
    monkeypatch.setattr(cli_settings.sys, "stdout", stream)

    assert not cli_settings._color_enabled()


@pytest.mark.parametrize(
    "colors",
    (
        "",
        "1",
    ),
)
def test_check_without_arguments_shows_help_and_exit_code_2(
    monkeypatch: pytest.MonkeyPatch, colors: str
) -> None:
    """The check command should show help and exit with status 2 when empty."""
    monkeypatch.setenv("NO_COLOR", colors)
    result = CliRunner().invoke(cli_main.mlia_app, ["check"], terminal_width=120)

    assert result.exit_code == 2
    assert "Usage:" in result.stdout
    assert "Check model compatibility with a target" in result.stdout


def test_check_help_includes_discovery_and_compatibility_example() -> None:
    """Check help should show how to discover profiles and start a check."""
    result = CliRunner().invoke(
        cli_main.mlia_app,
        ["check", "--help"],
        terminal_width=120,
    )
    help_output = " ".join(_strip_ansi(result.stdout).replace("│", " ").split())

    assert result.exit_code == 0
    assert "mlia target list" in help_output
    assert "mlia backend list" in help_output
    assert (
        "mlia check <model.ext> --target-profile <profile> --compatibility"
        in help_output
    )


def test_check_help_lists_profiling_data_option(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The check command help should list measured profiling input."""
    monkeypatch.setattr(
        cli_commands,
        "discover_backend_option_specs",
        MagicMock(return_value=[]),
    )
    result = CliRunner().invoke(
        cli_main.mlia_app,
        ["check", "--help"],
        terminal_width=120,
    )
    help_output = _strip_ansi(result.stdout)

    assert result.exit_code == 0
    assert "--profiling-data" in help_output
    assert "--out-dir" in help_output


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


@pytest.mark.parametrize(
    ("plugin", "profile", "backend", "analysis_flag", "action"),
    [
        (
            "ethos_u",
            "ethos-u85-2048",
            "vela",
            "--compatibility",
            "perform a compatibility check",
        ),
        (
            "neural_technology",
            "neural-technology",
            "nx-performance-estimator",
            "--performance",
            "estimate performance",
        ),
    ],
)
def test_root_help_shows_installed_plugin_example(
    monkeypatch: pytest.MonkeyPatch,
    plugin: str,
    profile: str,
    backend: str,
    analysis_flag: str,
    action: str,
) -> None:
    """Root help uses entry-point metadata without loading plugin modules."""
    entry_points = MagicMock(return_value=[SimpleNamespace(name=plugin)])
    monkeypatch.setattr(cli_commands, "list_entry_points", entry_points)

    result = CliRunner().invoke(cli_main.mlia_app, ["--help"])
    help_output = _strip_ansi(result.stdout)

    assert result.exit_code == 0
    assert (
        "Example Usage: Run 'mlia check model.ext "
        f"--target-profile {profile} --backend {backend} "
        f"{analysis_flag}' to {action}."
    ) in help_output
    entry_points.assert_called_once_with(cli_commands.TARGET_PLUGIN_GROUP)
    assert "Tip: Run 'mlia <command> --help'" in help_output
    assert "mlia target list" in help_output
    assert "mlia backend list" in help_output
    assert cli_commands.PLUGIN_URL in help_output


def test_root_help_has_one_blank_line_before_example(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep only one blank line between the Rich epilog and the example."""
    monkeypatch.setattr(
        cli_commands,
        "list_entry_points",
        lambda group: [SimpleNamespace(name="neural_technology")],
    )

    result = CliRunner().invoke(cli_main.mlia_app, ["--help"])

    assert result.exit_code == 0
    lines = _strip_ansi(result.stdout).splitlines()
    assert any(line.strip().startswith("Tip: Run") for line in lines)
    assert any(line.strip().startswith("Browse MLIA repositories:") for line in lines)
    example_line = next(
        index for index, line in enumerate(lines) if line.startswith("Example Usage:")
    )
    assert not lines[example_line - 1].strip()
    assert lines[example_line - 2].strip()


def test_root_help_prefers_neural_technology_plugin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Prefer Neural Technology when several known target plugins are installed."""
    monkeypatch.setattr(
        cli_commands,
        "list_entry_points",
        lambda group: [
            SimpleNamespace(name="ethos_u"),
            SimpleNamespace(name="neural_technology"),
        ],
    )

    result = CliRunner().invoke(cli_main.mlia_app, ["--help"])

    assert result.exit_code == 0
    assert "--target-profile neural-technology" in _strip_ansi(result.stdout)


@pytest.mark.parametrize("app", [cli_main.backend_app, cli_main.target_app])
def test_standalone_plugin_help_omits_root_example(app: Any) -> None:
    """Legacy standalone entry points should only show their own help."""
    result = CliRunner().invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "Example Usage: Run 'mlia check" not in result.stdout
    assert "No target plugins are installed." not in result.stdout
    assert "MLIA couldn't detect installed plugins." not in result.stdout
    assert "Explore your installed target plugins:" not in result.stdout


def test_root_help_without_target_plugins(monkeypatch: pytest.MonkeyPatch) -> None:
    """The root help should explain how to install target plugins."""
    monkeypatch.setattr(cli_commands, "list_entry_points", lambda group: [])
    result = CliRunner().invoke(cli_main.mlia_app, ["--help"])

    assert result.exit_code == 0
    assert "No target plugins are installed." in result.stdout
    assert cli_commands.PLUGIN_URL in result.stdout


def test_root_help_with_unknown_target_plugin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unknown target plugins receive guidance without loading them."""
    monkeypatch.setattr(
        cli_commands,
        "list_entry_points",
        lambda group: [SimpleNamespace(name="third_party_target")],
    )

    result = CliRunner().invoke(cli_main.mlia_app, ["--help"])

    assert result.exit_code == 0
    assert "Explore your installed target plugins:" in result.stdout
    assert "mlia target list" in result.stdout
    assert "mlia backend list" in result.stdout
    assert "Example Usage:" not in result.stdout


def test_root_help_skips_unknown_plugin_before_known_plugin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A known installed plugin can supply the example after an unknown one."""
    monkeypatch.setattr(
        cli_commands,
        "list_entry_points",
        lambda group: [
            SimpleNamespace(name="third_party_target"),
            SimpleNamespace(name="ethos_u"),
        ],
    )

    result = CliRunner().invoke(cli_main.mlia_app, ["--help"])

    assert result.exit_code == 0
    assert "--target-profile ethos-u85-2048 --backend vela" in result.stdout


def test_root_help_when_plugin_discovery_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    """The root help should offer discovery commands after an error."""
    monkeypatch.setattr(
        cli_commands, "list_entry_points", MagicMock(side_effect=RuntimeError("broken"))
    )
    result = CliRunner().invoke(cli_main.mlia_app, ["--help"])

    assert result.exit_code == 0
    assert "MLIA couldn't detect installed plugins." in result.stdout
    assert "mlia target list" in result.stdout
    assert "mlia backend list" in result.stdout


@pytest.mark.parametrize(
    ("args", "expected_text"),
    [
        (
            ["check", "--target-profile", "tosa", "--compatibility"],
            "Missing argument 'MODEL'",
        ),
        (
            [
                "check",
                "--target-profile",
                "tosa",
                "--i-agree-to-the-contained-eula",
            ],
            "Missing argument 'MODEL'",
        ),
    ],
)
def test_check_accepts_updated_flag_names(args: list[str], expected_text: str) -> None:
    """Updated long option names should be accepted by the parser."""
    result = CliRunner().invoke(cli_main.mlia_app, args)

    assert result.exit_code == 2
    assert result.stderr == expected_text + "\n"


def test_check_accepts_target_profile_flag(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Ordinary execution neither needs nor imports the optional TUI package."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, "mlia_tui", raising=False)
    (tmp_path / "mlia_tui.py").write_text(
        "raise AssertionError('TUI must not be imported')\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(mlia_api, "get_advice", MagicMock(return_value=None))
    monkeypatch.setattr(
        command_validators,
        "validate_check_target_profile",
        MagicMock(),
    )

    result = CliRunner().invoke(
        cli_main.mlia_app,
        ["check", "model.tflite", "--target-profile", "ethos-u55-256"],
    )

    assert result.exit_code == 0


def test_check_persists_canonical_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Every successful CLI analysis should persist its canonical JSON output."""
    output: dict[str, object] = {
        "schema_version": "1.2.0",
        "results": [{"kind": "compatibility", "status": "ok"}],
    }
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(mlia_api, "get_advice", MagicMock(return_value=output))
    monkeypatch.setattr(
        command_validators,
        "validate_check_target_profile",
        MagicMock(return_value=True),
    )

    result = CliRunner().invoke(
        cli_main.mlia_app,
        ["check", "model.tflite", "--target-profile", "ethos-u55-256"],
    )

    assert result.exit_code == 0
    output_path = tmp_path / "mlia-output" / "mlia-output.json"
    assert json.loads(output_path.read_text(encoding="utf-8")) == output


def test_check_passes_backend_options_from_discovered_cli_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The check command should forward dynamic backend options to the API."""
    get_advice = MagicMock(return_value=None)
    discover_backend_option_specs = MagicMock(return_value=[_backend_option_spec()])

    monkeypatch.setattr(
        cli_commands,
        "discover_backend_option_specs",
        discover_backend_option_specs,
    )
    monkeypatch.setattr(mlia_api, "get_advice", get_advice)
    monkeypatch.setattr(
        command_validators,
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


def test_check_rejects_unnamespaced_backend_click_option(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The check command should reject unnamespaced typed backend options."""
    get_advice = MagicMock()
    discover_backend_option_specs = MagicMock(
        return_value=[_typed_backend_option_spec()]
    )

    monkeypatch.setattr(
        cli_commands,
        "discover_backend_option_specs",
        discover_backend_option_specs,
    )
    monkeypatch.setattr(mlia_api, "get_advice", get_advice)
    monkeypatch.setattr(
        command_validators,
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
            "--optimization-level",
            "2",
        ],
    )

    assert result.exit_code != 0
    get_advice.assert_not_called()


def test_check_accepts_namespaced_backend_click_option(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The check command should accept namespaced long typed backend options."""
    get_advice = MagicMock(return_value=None)
    discover_backend_option_specs = MagicMock(
        return_value=[_typed_backend_option_spec()]
    )

    monkeypatch.setattr(
        cli_commands,
        "discover_backend_option_specs",
        discover_backend_option_specs,
    )
    monkeypatch.setattr(mlia_api, "get_advice", get_advice)
    monkeypatch.setattr(
        command_validators,
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
            "--typed-backend.optimization-level",
            "2",
        ],
    )

    assert result.exit_code == 0
    get_advice.assert_called_once()
    assert get_advice.call_args.kwargs["backend_options"] == {
        "typed-backend": {"optimization_level": "2"}
    }
    assert isinstance(get_advice.call_args.kwargs["settings"], ApplicationSettings)


def test_analysis_plugin_registry_is_scoped_to_each_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each command should load plugins into a fresh registry."""
    load_plugins = MagicMock()
    monkeypatch.setattr(cli_commands, "load_analysis_plugins", load_plugins)

    first = cli_commands._load_analysis_plugin_registry()
    second = cli_commands._load_analysis_plugin_registry()

    assert first is not second
    assert load_plugins.call_args_list[0].args == (first,)
    assert load_plugins.call_args_list[1].args == (second,)


@pytest.mark.parametrize(
    "model_and_plugin_args",
    [
        ["--demo-analysis", "model.tflite"],
        ["model.tflite", "--demo-analysis"],
    ],
)
def test_check_runs_enabled_analysis_plugins(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    model_and_plugin_args: list[str],
) -> None:
    """The check command should parse plugin options around the model argument."""
    plugin = DemoAnalysisPlugin()
    output: dict[str, object] = {"results": []}
    get_advice = MagicMock(return_value=output)
    discover_backend_option_specs = MagicMock(
        return_value=[_typed_backend_option_spec()]
    )
    config = tmp_path / "config.toml"
    config.write_text(
        "[filtering]\ncollapse = []\n\n[plugins.demo]\nenabled = true\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(cli_settings, "_CONFIG_PATH", config)

    monkeypatch.setattr(
        cli_commands,
        "discover_backend_option_specs",
        discover_backend_option_specs,
    )
    monkeypatch.setattr(mlia_api, "get_advice", get_advice)
    monkeypatch.setattr(
        command_validators,
        "validate_check_target_profile",
        MagicMock(return_value=True),
    )
    registry = AnalysisPluginRegistry()
    registry.register(plugin)
    monkeypatch.setattr(
        cli_commands,
        "_load_analysis_plugin_registry",
        MagicMock(return_value=registry),
    )

    result = CliRunner().invoke(
        cli_main.mlia_app,
        [
            "check",
            *model_and_plugin_args,
            "--target-profile",
            "ethos-u55-256",
            "--output-dir",
            str(tmp_path),
            "--typed-backend.optimization-level",
            "2",
        ],
    )

    assert result.exit_code == 0
    get_advice.assert_called_once()
    assert get_advice.call_args.kwargs["backend_options"] == {
        "typed-backend": {"optimization_level": "2"}
    }
    settings = get_advice.call_args.kwargs["settings"]
    assert isinstance(settings, ApplicationSettings)
    assert settings.filtering.collapse == ()
    assert len(plugin.results) == 1
    assert plugin.results[0].args["demo_analysis"] is True
    assert plugin.results[0].settings == {"enabled": True}
    assert plugin.results[0].output is output
    assert plugin.results[0].parameters["model"] == "model.tflite"


def test_check_preserves_click_file_options(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Backend and analysis options retain live resources until CLI completion."""
    source = tmp_path / "plugin-input.txt"
    source.write_text("plugin data", encoding="utf-8")
    plugin = FileAnalysisPlugin()
    registry = AnalysisPluginRegistry()
    registry.register(plugin)

    def read_backend_input(*_args: object, **kwargs: object) -> None:
        backend_options = cast(dict[str, dict[str, TextIO]], kwargs["backend_options"])
        assert backend_options["file-backend"]["input"].read() == "plugin data"

    get_advice = MagicMock(side_effect=read_backend_input)
    backend_option: BackendOptionSpec = {
        "module": "file_backend",
        "backend": "file-backend",
        "config_key": "input",
        "click_option": click.Option(
            ["--file-backend.input", "backend_input"], type=click.File("r")
        ),
    }
    monkeypatch.setattr(
        cli_commands,
        "discover_backend_option_specs",
        MagicMock(return_value=[backend_option]),
    )
    monkeypatch.setattr(
        cli_commands, "_load_analysis_plugin_registry", MagicMock(return_value=registry)
    )
    monkeypatch.setattr(mlia_api, "get_advice", get_advice)

    result = CliRunner().invoke(
        cli_main.mlia_app,
        [
            "check",
            "model.tflite",
            "--target-profile",
            "tosa",
            "--output-dir",
            str(tmp_path),
            "--demo-analysis",
            "--plugin-input",
            str(source),
            "--file-backend.input",
            str(source),
        ],
    )

    assert result.exit_code == 0, result.output
    assert plugin.contents == "plugin data"
    assert len(plugin.results) == 1
    assert cast(TextIO, plugin.results[0].args["plugin_input"]).closed
    backend_input = get_advice.call_args.kwargs["backend_options"]["file-backend"][
        "input"
    ]
    assert backend_input.name == str(source)
    assert backend_input.closed
    assert (
        get_advice.call_args.kwargs["settings"].backend_options["file-backend"]["input"]
        is backend_input
    )


def test_check_help_lists_analysis_plugin_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The check command help should include analysis-plugin options."""
    registry = AnalysisPluginRegistry()
    registry.register(DemoAnalysisPlugin())
    monkeypatch.setattr(
        cli_commands,
        "_load_analysis_plugin_registry",
        MagicMock(return_value=registry),
    )

    result = CliRunner().invoke(
        cli_main.mlia_app,
        ["check", "--help"],
        terminal_width=120,
    )

    assert result.exit_code == 0
    assert "--demo-analysis" in _strip_ansi(result.stdout)


def test_check_routes_profiling_data_through_standard_workflow(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Profiling input should use get_advice and enabled analysis plugins."""
    plugin = DemoAnalysisPlugin()
    output: dict[str, object] = {"results": []}
    get_advice = MagicMock(return_value=output)
    config = tmp_path / "config.toml"
    config.write_text("[plugins.demo]\nenabled = true\n", encoding="utf-8")
    monkeypatch.setattr(cli_settings, "_CONFIG_PATH", config)

    monkeypatch.setattr(mlia_api, "get_advice", get_advice)
    monkeypatch.setattr(
        command_validators,
        "validate_check_target_profile",
        MagicMock(return_value=True),
    )
    registry = AnalysisPluginRegistry()
    registry.register(plugin)
    monkeypatch.setattr(
        cli_commands,
        "_load_analysis_plugin_registry",
        MagicMock(return_value=registry),
    )

    result = CliRunner().invoke(
        cli_main.mlia_app,
        [
            "check",
            "--target-profile",
            "ethos-u55-256",
            "--performance",
            "--profiling-data",
            str(tmp_path),
            "--output-dir",
            str(tmp_path),
            "--demo-analysis",
        ],
    )

    assert result.exit_code == 0
    get_advice.assert_called_once()
    assert get_advice.call_args.args[:3] == (
        "ethos-u55-256",
        None,
        {"performance"},
    )
    assert get_advice.call_args.kwargs["profiling_data"] == [tmp_path]
    assert len(plugin.results) == 1
    assert plugin.results[0].args["demo_analysis"] is True
    assert plugin.results[0].output is output
    assert plugin.results[0].parameters["model"] is None
    assert plugin.results[0].parameters["profiling_data"] == [tmp_path]


def test_check_accepts_repeated_profiling_data_in_order(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Repeated profiling options should reach the API as an ordered list."""
    get_advice = MagicMock(return_value={"results": []})
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    monkeypatch.setattr(mlia_api, "get_advice", get_advice)
    monkeypatch.setattr(
        command_validators,
        "validate_check_target_profile",
        MagicMock(return_value=True),
    )

    result = CliRunner().invoke(
        cli_main.mlia_app,
        [
            "check",
            "model.vgf",
            "--target-profile",
            "ethos-u55-256",
            "--performance",
            "--profiling-data",
            str(first),
            "--profiling-data",
            str(second),
        ],
    )

    assert result.exit_code == 0
    assert get_advice.call_args.kwargs["profiling_data"] == [first, second]


def test_check_exits_cleanly_when_validation_skips_all_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The check command should exit 0 when validation reports no runnable checks."""
    get_advice = MagicMock()

    monkeypatch.setattr(mlia_api, "get_advice", get_advice)
    monkeypatch.setattr(
        command_validators,
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
    format_backend_info.assert_called_once()


def test_main_dispatches_target_list(monkeypatch: pytest.MonkeyPatch) -> None:
    """The target list command should run through the main mlia entry point."""
    format_target_info = MagicMock()

    monkeypatch.setattr(cli_commands, "setup_logging", MagicMock())
    monkeypatch.setattr(cli_commands, "format_target_info", format_target_info)

    result = CliRunner().invoke(cli_main.mlia_app, ["target", "list"])

    assert result.exit_code == 0
    format_target_info.assert_called_once()


def test_backend_main_warns_about_deprecated_entry_point(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Backend entry point should warn before calling the Typer app."""
    backend_app = MagicMock()
    configure_typer_help = MagicMock()
    secho = MagicMock()

    monkeypatch.setattr(cli_main, "backend_app", backend_app)
    monkeypatch.setattr(cli_main, "configure_typer_help", configure_typer_help)
    monkeypatch.setattr(cli_settings, "_color_enabled", MagicMock(return_value=False))
    monkeypatch.setattr(cli_main.typer, "secho", secho)

    cli_main.backend_main()

    configure_typer_help.assert_called_once_with(False, "dark")
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
    configure_typer_help = MagicMock()
    secho = MagicMock()

    monkeypatch.setattr(cli_main, "target_app", target_app)
    monkeypatch.setattr(cli_main, "configure_typer_help", configure_typer_help)
    monkeypatch.setattr(cli_settings, "_color_enabled", MagicMock(return_value=False))
    monkeypatch.setattr(cli_main.typer, "secho", secho)

    cli_main.target_main()

    configure_typer_help.assert_called_once_with(False, "dark")
    secho.assert_called_once_with(
        cli_main.DEPRECATED_TARGET_ENTRY_POINT,
        fg=cli_main.typer.colors.YELLOW,
        color=False,
        err=True,
    )
    target_app.assert_called_once_with(color=False)
    assert capsys.readouterr().err == ""


@pytest.mark.parametrize(
    "environment, result",
    [
        ({}, False),
        ({"DEBUG": ""}, False),
        ({"DEBUG": "1"}, True),
        ({"DEBUG": "true"}, True),
        ({"DEBUG": "release"}, False),
        ({"DEBUG": "0"}, False),
    ],
)
def test_debug_option_is_set_from_environment(
    monkeypatch: pytest.MonkeyPatch,
    environment: dict[str, str],
    result: bool,
) -> None:
    """
    The debug Typer option's default value should be set depending on the environment.
    """

    with patch.dict("os.environ", clear=True, **environment):
        assert cli_commands.debug_option().default is result


@pytest.mark.parametrize(
    ("args", "target", "model", "backends"),
    [
        pytest.param(
            ["check", "--backend", "vela", "model.tflite", "--tui", "--compatibility"],
            None,
            "model.tflite",
            ("vela",),
            id="missing-target",
        ),
        pytest.param(["check", "--tui"], None, None, None, id="discovery-only"),
        pytest.param(
            ["check", "--tui", "--target-profile", "example"],
            "example",
            None,
            None,
            id="missing-model",
        ),
    ],
)
def test_tui_accepts_incomplete_check_request(
    monkeypatch: pytest.MonkeyPatch,
    args: list[str],
    target: str | None,
    model: str | None,
    backends: tuple[str, ...] | None,
) -> None:
    """Editable fields can be completed after the optional interface opens."""
    launch = MagicMock(return_value=None)
    monkeypatch.setitem(sys.modules, "mlia_tui", SimpleNamespace(launch=launch))

    result = CliRunner().invoke(cli_main.mlia_app, args)

    assert result.exit_code == 0, result.stdout
    launch.assert_called_once()
    request = launch.call_args.args[0]
    assert (request.target_profile, request.model, request.backend) == (
        target,
        model,
        backends,
    )


@pytest.mark.parametrize(
    "args",
    [
        ["check", "model.tflite"],
        ["check", "--backend=--tui", "model.tflite"],
        ["check", "--", "--tui"],
    ],
)
def test_non_tui_check_still_requires_target_profile(
    monkeypatch: pytest.MonkeyPatch,
    args: list[str],
) -> None:
    """Relaxing interactive preparation does not relax CLI execution."""
    execute = MagicMock()
    monkeypatch.setattr(cli_commands, "execute_check", execute)
    registry = MagicMock(spec=AnalysisPluginRegistry)
    registry.cli_options.return_value = [click.Option(["--token"], prompt=True)]
    monkeypatch.setattr(
        cli_commands, "_load_analysis_plugin_registry", lambda: registry
    )

    result = CliRunner().invoke(cli_main.mlia_app, args)

    assert result.exit_code == 2
    assert "Missing option '--target-profile'" in _strip_ansi(result.output)
    assert "Token:" not in result.output
    execute.assert_not_called()


def test_tui_completed_check_emits_canonical_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """JSON comes from the completed result after the interface closes."""
    output: dict[str, object] = {"results": []}
    launch = MagicMock(return_value=CheckResult(output, None, "Rendered report"))
    monkeypatch.setitem(sys.modules, "mlia_tui", SimpleNamespace(launch=launch))

    result = CliRunner().invoke(cli_main.mlia_app, ["check", "--tui", "--json"])

    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout) == output
    assert result.stderr == ""


@pytest.mark.parametrize(
    "check_result",
    [None, CheckResult(None, None)],
    ids=["no-attempt", "completed-without-output"],
)
def test_tui_without_output_emits_no_json(
    monkeypatch: pytest.MonkeyPatch, check_result: CheckResult | None
) -> None:
    """Closing discovery or completing without output leaves stdout empty."""
    launch = MagicMock(return_value=check_result)
    monkeypatch.setitem(sys.modules, "mlia_tui", SimpleNamespace(launch=launch))

    result = CliRunner().invoke(cli_main.mlia_app, ["check", "--tui", "--json"])

    assert result.exit_code == 0, result.output
    assert result.output == ""


def test_tui_skipped_check_reports_reason_without_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A skipped check reports its reason on stderr even with JSON requested."""
    launch = MagicMock(
        return_value=CheckResult(
            None, None, outcome="skipped", skip_reason="No supported checks selected."
        )
    )
    monkeypatch.setitem(sys.modules, "mlia_tui", SimpleNamespace(launch=launch))

    result = CliRunner().invoke(cli_main.mlia_app, ["check", "--tui", "--json"])

    assert result.exit_code == 0
    assert result.stdout == ""
    assert result.stderr == "No supported checks selected.\n"


@pytest.mark.parametrize("exit_code", [1, 2, 130])
def test_tui_failed_check_preserves_exit_code_without_json(
    monkeypatch: pytest.MonkeyPatch, exit_code: int
) -> None:
    """The CLI preserves the failure status returned by the interface."""
    launch = MagicMock(side_effect=CheckError("Check failed.", exit_code=exit_code))
    monkeypatch.setitem(sys.modules, "mlia_tui", SimpleNamespace(launch=launch))

    result = CliRunner().invoke(cli_main.mlia_app, ["check", "--tui", "--json"])

    assert result.exit_code == exit_code
    assert result.stdout == ""
    assert result.stderr == "Error: Check failed.\n"


def test_tui_missing_package_is_actionable(monkeypatch: pytest.MonkeyPatch) -> None:
    """Only requesting the unavailable optional interface requires its installation."""
    monkeypatch.setitem(sys.modules, "mlia_tui", None)

    result = CliRunner().invoke(cli_main.mlia_app, ["check", "--tui"])

    assert result.exit_code == 1
    assert result.stdout == ""
    assert "pip install mlia-tui" in result.stderr


def test_tui_broken_dependency_is_not_reported_as_missing_package(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An installed frontend's import error retains the missing dependency."""
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, "mlia_tui", raising=False)
    (tmp_path / "mlia_tui.py").write_text(
        "raise ModuleNotFoundError('missing dependency', name='frontend_dependency')\n",
        encoding="utf-8",
    )

    result = CliRunner().invoke(cli_main.mlia_app, ["check", "--tui"])

    assert result.exit_code == 1
    assert isinstance(result.exception, ModuleNotFoundError)
    assert result.exception.name == "frontend_dependency"
    assert "pip install mlia-tui" not in result.output


def test_ordinary_check_reports_configuration_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Configuration errors produce an actionable CLI message without a traceback."""
    error = ConfigurationError("Invalid analysis configuration")
    monkeypatch.setattr(mlia_api, "get_advice", MagicMock(side_effect=error))

    result = CliRunner().invoke(
        cli_main.mlia_app,
        ["check", "model.tflite", "-t", "tosa", "--output-dir", str(tmp_path)],
    )

    assert result.exit_code == 1
    assert isinstance(result.exception, SystemExit)
    assert result.stderr == "Error: Invalid analysis configuration\n"


def test_post_analysis_failure_preserves_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Post-analysis errors retain ordinary CLI reporting and persisted results."""
    output: dict[str, object] = {"results": []}
    plugin = DemoAnalysisPlugin()
    monkeypatch.setattr(
        plugin, "run", MagicMock(side_effect=ConfigurationError("Plugin failed"))
    )
    registry = AnalysisPluginRegistry()
    registry.register(plugin)
    monkeypatch.setattr(
        cli_commands, "_load_analysis_plugin_registry", lambda: registry
    )
    monkeypatch.setattr(mlia_api, "get_advice", MagicMock(return_value=output))

    result = CliRunner().invoke(
        cli_main.mlia_app,
        [
            "check",
            "model.tflite",
            "-t",
            "tosa",
            "--output-dir",
            str(tmp_path),
            "--demo-analysis",
        ],
    )

    assert result.exit_code == 1
    assert result.stderr == "Error: Plugin failed\n"
    assert json.loads((tmp_path / "mlia-output/mlia-output.json").read_text()) == output


@pytest.mark.parametrize("tui", [False, True], ids=["cli", "tui"])
def test_check_companion_option_without_analysis_plugins(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, tui: bool
) -> None:
    """The companion is a core analysis input even without Model Explorer."""
    get_advice = MagicMock(return_value=None)
    launch = MagicMock(return_value=None)
    monkeypatch.setitem(sys.modules, "mlia_tui", SimpleNamespace(launch=launch))
    monkeypatch.setattr(
        cli_commands, "_load_analysis_plugin_registry", AnalysisPluginRegistry
    )
    monkeypatch.setattr(
        cli_commands, "discover_backend_option_specs", lambda *_args: []
    )
    monkeypatch.setattr(mlia_api, "get_advice", get_advice)
    monkeypatch.setattr(
        command_validators,
        "validate_check_target_profile",
        lambda *_args, **_kwargs: True,
    )
    companion = tmp_path / "model.pte"
    companion.touch()
    result = CliRunner().invoke(
        cli_main.mlia_app,
        [
            "check",
            "model.etrecord",
            "--target-profile",
            "neural-technology",
            "--performance",
            "--profiling-data",
            str(tmp_path),
            "--companion-model",
            str(companion),
            "--output-dir",
            str(tmp_path),
        ]
        + (["--tui"] if tui else []),
    )
    assert result.exit_code == 0, result.output
    if tui:
        launch.assert_called_once()
        assert launch.call_args.args[0].companion_model == companion
        get_advice.assert_not_called()
    else:
        assert get_advice.call_args.kwargs["companion_model"] == companion
        launch.assert_not_called()
