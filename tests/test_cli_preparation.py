# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for standalone preparation through the public CLI adapter."""

import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import IO, cast
from unittest.mock import MagicMock

import click
import pytest
from typer.testing import CliRunner

import mlia.cli.commands as commands
from mlia.check import CheckError
from mlia.cli.preparation import prepare_check_defaults
from mlia.core.settings import ApplicationSettings
from mlia.plugins.analysis import AnalysisPluginRegistry
from mlia.utils.boundaries import environment


@pytest.fixture
def plugin_options(monkeypatch: pytest.MonkeyPatch) -> list[click.Option]:
    """Use explicit plugin declarations independent of installed extensions."""
    options: list[click.Option] = []
    registry = MagicMock(spec=AnalysisPluginRegistry)
    registry.cli_options.return_value = options
    monkeypatch.setattr(commands, "_load_analysis_plugin_registry", lambda: registry)
    monkeypatch.setattr(commands, "discover_backend_option_specs", lambda: [])
    return options


def _converted_value(ctx: click.Context, _param: click.Parameter, value: int) -> int:
    """Exercise the same initialized context available to ordinary CLI plugins."""
    assert click.get_current_context() is ctx
    assert ctx.parent is not None
    assert ctx.parent.invoked_subcommand == "check"
    assert isinstance(ctx.obj, ApplicationSettings)
    return value + 1


@pytest.mark.parametrize(
    ("environment_name", "expected_threads"),
    [("UNUSED_THREADS", 2), ("TEST_THREADS", 8)],
)
def test_defaults_match_cli(
    monkeypatch: pytest.MonkeyPatch,
    plugin_options: list[click.Option],
    environment_name: str,
    expected_threads: int,
) -> None:
    """Both frontends convert dynamic defaults in initialized contexts."""
    default = MagicMock(return_value="4")
    callback = MagicMock(side_effect=_converted_value)
    plugin_options.append(
        click.Option(["--analysis-level"], type=int, default=default, callback=callback)
    )
    backend_option = click.Option(
        ["--sample.threads", "sample_threads"],
        type=int,
        default="2",
        envvar="TEST_THREADS",
    )
    monkeypatch.setattr(
        commands,
        "discover_backend_option_specs",
        lambda: [
            {
                "module": "sample",
                "backend": "sample",
                "config_key": "threads",
                "click_option": backend_option,
            }
        ],
    )
    monkeypatch.delenv("TEST_THREADS", raising=False)
    monkeypatch.setenv(environment_name, "8")
    launch = MagicMock(return_value=None)
    monkeypatch.setitem(sys.modules, "mlia_tui", SimpleNamespace(launch=launch))

    result = CliRunner().invoke(commands.mlia_app, ["check", "--tui"])
    assert result.exit_code == 0, result.output
    cli_request = launch.call_args.args[0]

    with prepare_check_defaults() as request:
        assert request == cli_request
        assert request.plugin_args == {"analysis_level": 5}
        assert request.settings.backend_options == {
            "sample": {"threads": expected_threads}
        }
    launch.assert_called_once_with(cli_request)


def test_prompt_uses_default_without_mutating_plugin_option(
    plugin_options: list[click.Option],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Standalone preparation resolves a prompted default without stdin."""
    option = click.Option(["--analysis-level"], type=int, default="4", prompt=True)
    plugin_options.append(option)
    with prepare_check_defaults() as request:
        assert request.plugin_args == {"analysis_level": 4}
    assert option.prompt == "Analysis level"
    execute = MagicMock()
    monkeypatch.setattr(commands, "execute_check", execute)

    result = CliRunner().invoke(
        commands.mlia_app, ["check", "model.tflite", "-t", "tosa"], input="7\n"
    )

    assert result.exit_code == 0, result.output
    assert "Analysis level" in result.output
    assert execute.call_args.args[0].plugin_args == {"analysis_level": 7}


def test_required_prompt_reports_cli_option(plugin_options: list[click.Option]) -> None:
    """Missing prompted input tells users which CLI option to supply."""
    plugin_options.append(
        click.Option(["--analysis-level"], required=True, prompt=True)
    )
    with pytest.raises(CheckError, match="--analysis-level.*mlia check --tui") as error:
        with prepare_check_defaults():
            pytest.fail("Missing required option was accepted")
    assert error.value.exit_code == 2


def test_file_resource_lifetime(
    plugin_options: list[click.Option], tmp_path: Path
) -> None:
    """Files remain available throughout the caller's context and close on exit."""
    path = tmp_path / "input.txt"
    path.write_text("input", encoding="utf-8")
    plugin_options.append(
        click.Option(["--analysis-file"], type=click.File(), default=str(path))
    )
    with prepare_check_defaults() as request:
        stream = cast(IO[str], request.plugin_args["analysis_file"])
        assert stream.read() == "input"
        assert not stream.closed
    assert stream.closed


def _reject_after_resource(
    _ctx: click.Context, _param: click.Parameter, _value: IO[str]
) -> None:
    """Reject a parsed file so its resource cleanup can be observed."""
    raise click.BadParameter("invalid file configuration")


def test_parser_failure_closes_resources(
    plugin_options: list[click.Option], tmp_path: Path
) -> None:
    """A callback failure during parsing still closes resources acquired earlier."""
    path = tmp_path / "input.txt"
    path.write_text("input", encoding="utf-8")
    callback = MagicMock(side_effect=_reject_after_resource)
    plugin_options.append(
        click.Option(
            ["--analysis-file"],
            type=click.File(),
            default=str(path),
            callback=callback,
        )
    )
    with pytest.raises(CheckError, match="--analysis-file.*mlia check --tui"):
        with prepare_check_defaults():
            pytest.fail("Invalid callback was accepted")
    assert callback.call_args.args[2].closed


def test_caller_errors_are_not_relabelled(plugin_options: list[click.Option]) -> None:
    """Errors raised by the UI keep their original identity."""
    error = click.BadParameter("UI failure")
    with pytest.raises(click.BadParameter) as raised:
        with prepare_check_defaults():
            raise error
    assert raised.value is error


@pytest.mark.parametrize("error", [click.Abort(), EOFError()])
def test_callback_input_failure_is_actionable(
    plugin_options: list[click.Option], error: Exception
) -> None:
    """Input failures during preparation direct users to the CLI."""
    plugin_options.append(
        click.Option(["--analysis-level"], callback=MagicMock(side_effect=error))
    )
    with pytest.raises(CheckError, match="option.*mlia check --tui") as raised:
        with prepare_check_defaults():
            pytest.fail("Aborted callback was accepted")
    assert raised.value.exit_code == 2


def test_defaults_use_root_default_map(
    monkeypatch: pytest.MonkeyPatch, plugin_options: list[click.Option]
) -> None:
    """Standalone preparation inherits the root configuration used by CLI parsing."""
    plugin_options.append(click.Option(["--analysis-level"], type=int, default="4"))
    context_settings = commands.mlia_app.info.context_settings
    assert context_settings is not None
    monkeypatch.setitem(
        context_settings,
        "default_map",
        {"check": {"analysis_level": "9"}},
    )
    launch = MagicMock(return_value=None)
    monkeypatch.setitem(sys.modules, "mlia_tui", SimpleNamespace(launch=launch))

    result = CliRunner().invoke(commands.mlia_app, ["check", "--tui"])

    assert result.exit_code == 0, result.output
    with prepare_check_defaults() as request:
        assert request == launch.call_args.args[0]
        assert request.plugin_args == {"analysis_level": 9}


def test_dotenv_defaults_are_scoped(
    monkeypatch: pytest.MonkeyPatch, plugin_options: list[click.Option]
) -> None:
    """Dotenv defaults and callback mutations do not escape preparation."""
    monkeypatch.setattr(
        environment,
        "dotenv_values",
        lambda: {"DOTENV_ONLY": "from-file", "EXISTING_VALUE": "from-file"},
    )
    monkeypatch.delenv("DOTENV_ONLY", raising=False)
    monkeypatch.delenv("CALLBACK_ADDED", raising=False)
    monkeypatch.setenv("EXISTING_VALUE", "from-shell")
    plugin_options.extend(
        [
            click.Option(["--dotenv-value"], envvar="DOTENV_ONLY"),
            click.Option(["--existing-value"], envvar="EXISTING_VALUE"),
        ]
    )

    with prepare_check_defaults() as request:
        assert request.plugin_args == {
            "dotenv_value": "from-file",
            "existing_value": "from-shell",
        }
        os.environ["CALLBACK_ADDED"] = "during-preparation"
        os.environ["EXISTING_VALUE"] = "changed"

    assert "DOTENV_ONLY" not in os.environ
    assert "CALLBACK_ADDED" not in os.environ
    assert os.environ["EXISTING_VALUE"] == "from-shell"


def test_dotenv_environment_is_restored_after_parse_failure(
    monkeypatch: pytest.MonkeyPatch, plugin_options: list[click.Option]
) -> None:
    """A failed callback cannot leak dotenv or callback environment changes."""
    monkeypatch.setattr(environment, "dotenv_values", lambda: {"DOTENV_ONLY": "yes"})
    monkeypatch.delenv("DOTENV_ONLY", raising=False)
    monkeypatch.delenv("CALLBACK_ADDED", raising=False)

    def reject(_ctx: click.Context, _param: click.Parameter, value: str) -> str:
        os.environ["CALLBACK_ADDED"] = value
        raise click.BadParameter("invalid environment configuration")

    plugin_options.append(
        click.Option(["--dotenv-value"], envvar="DOTENV_ONLY", callback=reject)
    )

    with pytest.raises(CheckError, match="invalid environment configuration"):
        with prepare_check_defaults():
            pytest.fail("Invalid callback was accepted")

    assert "DOTENV_ONLY" not in os.environ
    assert "CALLBACK_ADDED" not in os.environ
