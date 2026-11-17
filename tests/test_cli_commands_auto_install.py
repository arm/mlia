# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for CLI backend auto-install command behavior."""

from __future__ import annotations

from collections.abc import Callable
from unittest.mock import MagicMock

import pytest
import typer

import mlia.api as mlia_api
import mlia.check as check
from mlia.cli import commands as cli_commands
from mlia.cli.commands import mlia_app
from mlia.cli.settings import new_settings
from mlia.core import check_validation as command_validators
from mlia.core.settings import ApplicationSettings


def _get_app_settings() -> typer.Context:
    return typer.Context(
        typer.main.get_command(mlia_app),
        obj=new_settings(),
    )


def test_backend_list_does_not_install(monkeypatch: pytest.MonkeyPatch) -> None:
    install_backends = MagicMock()
    list_backends = MagicMock()

    monkeypatch.setattr(mlia_api, "list_backends", list_backends)
    monkeypatch.setattr(mlia_api, "install_backends", install_backends)
    monkeypatch.setattr(cli_commands, "setup_logging", MagicMock())

    cli_commands.backend_list(_get_app_settings())

    install_backends.assert_not_called()
    list_backends.assert_called()


def test_check_passes_backend_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    get_advice = MagicMock()
    monkeypatch.setattr(
        command_validators, "validate_check_target_profile", MagicMock()
    )
    monkeypatch.setattr(mlia_api, "get_advice", get_advice)
    monkeypatch.setattr(cli_commands, "setup_logging", MagicMock())
    monkeypatch.setattr(check, "_run_analysis_plugins", MagicMock())
    monkeypatch.setattr(check, "persist_standardized_output", MagicMock())

    settings = ApplicationSettings(
        backend_options={"bingo-bongo-backend": {"system_config": "backend.toml"}}
    )
    req = check.prepare_check(
        model="model.tflite",
        target_profile="ethos-u55-256",
        settings=settings,
    )
    check.execute_check(req)

    assert get_advice.call_args.kwargs["backend_options"] == settings.backend_options


@pytest.mark.parametrize(
    ("command", "args"),
    [
        (cli_commands.backend_install, (["backend"],)),
        (cli_commands.backend_uninstall, (["backend"],)),
        (cli_commands.backend_list, (_get_app_settings(),)),
        (cli_commands.target_list, (_get_app_settings(),)),
    ],
)
def test_contextless_commands_pass_debug_to_logging(
    command: Callable[..., None],
    args: tuple[object, ...],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    setup_logging = MagicMock()

    monkeypatch.setattr(cli_commands, "setup_logging", setup_logging)
    monkeypatch.setattr(mlia_api, "install_backends", MagicMock())
    monkeypatch.setattr(mlia_api, "uninstall_backends", MagicMock())
    monkeypatch.setattr(cli_commands, "format_backend_info", MagicMock())
    monkeypatch.setattr(cli_commands, "format_target_info", MagicMock())
    monkeypatch.setattr(check, "persist_standardized_output", MagicMock())
    monkeypatch.setattr(check, "_emit_standardized_output", MagicMock())

    command(*args, debug=True)

    setup_logging.assert_called_once_with(verbose=True, cli_mode=True)
