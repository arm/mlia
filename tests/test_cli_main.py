# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for CLI entry point behavior."""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import click
import pytest
from typer.testing import CliRunner

import mlia.api as mlia_api
import mlia.cli.command_validators as command_validators
import mlia.cli.commands as cli_commands
import mlia.cli.main as cli_main
import mlia.cli.settings as cli_settings
from mlia.backend.options import BackendOptionSpec
from mlia.core.settings import ApplicationSettings
from mlia.plugins.analysis import AnalysisPluginRegistry, AnalysisRunResult

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
        assert text in result.stdout or text in result.stderr


def test_main_calls_mlia_app(monkeypatch: pytest.MonkeyPatch) -> None:
    """Main entry point should call the root Typer app."""
    mlia_app = MagicMock()

    monkeypatch.setattr(cli_settings, "_color_enabled", lambda: None)
    monkeypatch.setattr(cli_main, "mlia_app", mlia_app)
    monkeypatch.setattr(cli_settings.tomllib, "load", lambda x: {})

    cli_main.main()
    mlia_app.assert_called_once_with(color=True)


def test_color_enabled_enables_color_for_tty_without_no_color(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With TTY ouput, colors should be delegated to lower configuration
    levels when NO_COLOR and MLIA_NO_COLOR is unset."""
    stream = MagicMock()
    stream.isatty.return_value = True
    monkeypatch.setattr(cli_settings.sys, "stdout", stream)
    monkeypatch.setattr(cli_settings, "get_environment", lambda: {})

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


def test_emit_standardized_output_uses_application_console(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Standardized output should use the configured application console."""
    settings = MagicMock()
    settings.console.encoding = "utf-8"
    context = MagicMock(output_format="plain_text")
    output: dict[str, object] = {"result": "[literal]"}
    render = MagicMock(return_value="[literal]")
    monkeypatch.setattr(cli_commands, "standardized_output_to_text", render)

    cli_commands.emit_standardized_output(settings, context, output)

    render.assert_called_once_with(output)
    settings.console.out.assert_called_once_with("[literal]", highlight=False)


def test_emit_standardized_output_replaces_unencodable_characters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CLI text output should respect the configured console encoding."""
    settings = MagicMock()
    settings.console.encoding = "cp1252"
    context = MagicMock(output_format="plain_text")
    output: dict[str, object] = {"result": "table"}
    render = MagicMock(return_value="┌─┐ café")
    monkeypatch.setattr(cli_commands, "standardized_output_to_text", render)

    cli_commands.emit_standardized_output(settings, context, output)

    settings.console.out.assert_called_once_with("??? café", highlight=False)


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
    assert expected_text in result.output


def test_check_accepts_target_profile_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    """The check command should accept the target profile flag."""
    monkeypatch.setattr(mlia_api, "get_advice", MagicMock())
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
    secho = MagicMock()

    monkeypatch.setattr(cli_main, "backend_app", backend_app)
    monkeypatch.setattr(cli_settings, "_color_enabled", MagicMock(return_value=False))
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
    monkeypatch.setattr(cli_settings, "_color_enabled", MagicMock(return_value=False))
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


@pytest.mark.parametrize(
    "environment, result",
    [
        ({}, False),
        ({"DEBUG": ""}, False),
        ({"DEBUG": "1"}, True),
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
