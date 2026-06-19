# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for CLI backend auto-install command behavior."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from mlia.cli import commands as cli_commands
from mlia.core.context import ExecutionContext


def test_backend_list_does_not_install(monkeypatch: pytest.MonkeyPatch) -> None:
    install_backends = MagicMock()
    list_backends = MagicMock()

    monkeypatch.setattr(cli_commands, "list_backends", list_backends)
    monkeypatch.setattr(cli_commands, "install_backends", install_backends)
    monkeypatch.setattr(cli_commands, "setup_logging", MagicMock())

    cli_commands.backend_list()

    install_backends.assert_not_called()
    list_backends.assert_called()


def test_check_preserves_interactive_eula_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    get_advice = MagicMock()
    monkeypatch.setattr(cli_commands, "validate_check_target_profile", MagicMock())
    monkeypatch.setattr(cli_commands, "get_advice", get_advice)
    monkeypatch.setattr(cli_commands, "setup_logging", MagicMock())

    cli_commands.check(
        model="model.tflite",
        target_profile="ethos-u55-256",
        backend=["corstone-300"],
    )

    assert get_advice.call_args.kwargs["backends"] == ["corstone-300"]
    assert get_advice.call_args.kwargs["accept_eula"] is None
    assert isinstance(get_advice.call_args.kwargs["context"], ExecutionContext)


def test_check_noninteractive_without_eula_rejects_eula_install(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    get_advice = MagicMock()
    monkeypatch.setattr(cli_commands, "validate_check_target_profile", MagicMock())
    monkeypatch.setattr(cli_commands, "get_advice", get_advice)
    monkeypatch.setattr(cli_commands, "setup_logging", MagicMock())

    cli_commands.check(
        model="model.tflite",
        target_profile="ethos-u55-256",
        backend=["corstone-300"],
        noninteractive=True,
    )

    assert get_advice.call_args.kwargs["backends"] == ["corstone-300"]
    assert get_advice.call_args.kwargs["accept_eula"] is False


def test_check_noninteractive_with_eula_accepts_eula_install(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    get_advice = MagicMock()
    monkeypatch.setattr(cli_commands, "validate_check_target_profile", MagicMock())
    monkeypatch.setattr(cli_commands, "get_advice", get_advice)
    monkeypatch.setattr(cli_commands, "setup_logging", MagicMock())

    cli_commands.check(
        model="model.tflite",
        target_profile="ethos-u55-256",
        backend=["corstone-300"],
        noninteractive=True,
        i_agree_to_the_contained_eula=True,
    )

    assert get_advice.call_args.kwargs["backends"] == ["corstone-300"]
    assert get_advice.call_args.kwargs["accept_eula"] is True


def test_check_passes_backend_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    get_advice = MagicMock()
    backend_options: dict[str, dict[str, object]] = {
        "bingo-bongo-backend": {"system_config": "backend.toml"}
    }
    monkeypatch.setattr(cli_commands, "validate_check_target_profile", MagicMock())
    monkeypatch.setattr(cli_commands, "get_advice", get_advice)
    monkeypatch.setattr(cli_commands, "setup_logging", MagicMock())

    cli_commands.check(
        model="model.tflite",
        target_profile="ethos-u55-256",
        backend_options=backend_options,
    )

    assert get_advice.call_args.kwargs["backend_options"] == backend_options


def test_check_passes_cli_context_settings(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    get_advice = MagicMock()
    setup_logging = MagicMock()

    monkeypatch.setattr(cli_commands, "validate_check_target_profile", MagicMock())
    monkeypatch.setattr(cli_commands, "get_advice", get_advice)
    monkeypatch.setattr(cli_commands, "setup_logging", setup_logging)

    cli_commands.check(
        model="model.tflite",
        target_profile="ethos-u55-256",
        output_dir=tmp_path,
        json_output=True,
        debug=True,
    )

    context = get_advice.call_args.kwargs["context"]
    assert isinstance(context, ExecutionContext)
    assert context.output_dir == tmp_path / "mlia-output"
    assert context.output_format == "json"
    assert context.verbose is True
    setup_logging.assert_called_once_with(context.logs_path, True, "json")


@pytest.mark.parametrize(
    ("command", "args"),
    [
        (cli_commands.backend_install, (["backend"],)),
        (cli_commands.backend_uninstall, (["backend"],)),
        (cli_commands.backend_list, ()),
        (cli_commands.target_list, ()),
    ],
)
def test_contextless_commands_pass_debug_to_logging(
    command: Callable[..., None],
    args: tuple[object, ...],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    setup_logging = MagicMock()

    monkeypatch.setattr(cli_commands, "setup_logging", setup_logging)
    monkeypatch.setattr(cli_commands, "install_backends", MagicMock())
    monkeypatch.setattr(cli_commands, "uninstall_backends", MagicMock())
    monkeypatch.setattr(cli_commands, "format_backend_info", MagicMock())
    monkeypatch.setattr(cli_commands, "format_target_info", MagicMock())

    command(*args, debug=True)

    setup_logging.assert_called_once_with(verbose=True)
