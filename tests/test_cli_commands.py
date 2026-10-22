# SPDX-FileCopyrightText: Copyright 2022-2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for cli.commands module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
import typer

import mlia.api
import mlia.cli.commands as cli_commands
from mlia.cli.commands import mlia_app
from mlia.cli.settings import new_settings


def _get_app_context() -> typer.Context:
    return typer.Context(
        typer.main.get_command(mlia_app),
        obj=new_settings(),
    )


def test_backend_command_action_list(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test `mlia backend list` command helper."""
    format_backend_info = MagicMock()

    monkeypatch.setattr(cli_commands, "format_backend_info", format_backend_info)
    monkeypatch.setattr(cli_commands, "setup_logging", MagicMock())

    ctx = _get_app_context()
    cli_commands.backend_list(ctx)

    format_backend_info.assert_called_once_with(ctx.obj)


def test_target_command_action_list(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test `mlia target list` command helper."""
    format_target_info = MagicMock()

    monkeypatch.setattr(cli_commands, "format_target_info", format_target_info)
    monkeypatch.setattr(cli_commands, "setup_logging", MagicMock())

    ctx = _get_app_context()
    cli_commands.target_list(ctx)

    format_target_info.assert_called_once_with(ctx.obj)


def test_backend_command_action_uninstall(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test `mlia backend uninstall` command helper."""
    uninstall_backends = MagicMock()

    monkeypatch.setattr(mlia.api, "uninstall_backends", uninstall_backends)
    monkeypatch.setattr(cli_commands, "setup_logging", MagicMock())

    cli_commands.backend_uninstall(["backend_name"])

    uninstall_backends.assert_called_once_with(["backend_name"])


@pytest.mark.parametrize(
    "accept_eula, noninteractive, force, backend_name, expected_accept_eula",
    [
        pytest.param(False, False, False, "backend_name", False, id="default"),
        pytest.param(True, False, False, "backend_name", True, id="accept-eula"),
        pytest.param(True, True, True, "BACKEND_NAME", True, id="all-flags"),
    ],
)
def test_backend_command_action_install(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    accept_eula: bool,
    noninteractive: bool,
    force: bool,
    backend_name: str,
    expected_accept_eula: bool,
) -> None:
    """Test `mlia backend install` command helper."""
    install_backends = MagicMock()

    monkeypatch.setattr(mlia.api, "install_backends", install_backends)
    monkeypatch.setattr(cli_commands, "setup_logging", MagicMock())

    cli_commands.backend_install(
        names=[backend_name],
        path=tmp_path,
        accept_eula=accept_eula,
        noninteractive=noninteractive,
        force=force,
    )

    install_backends.assert_called_once_with(
        names=[backend_name],
        path=tmp_path,
        accept_eula=expected_accept_eula,
        noninteractive=noninteractive,
        force=force,
    )
