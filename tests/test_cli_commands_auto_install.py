# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for CLI backend auto-install command behavior."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from mlia.cli import commands as cli_commands


def test_backend_list_does_not_install(monkeypatch: pytest.MonkeyPatch) -> None:
    manager = MagicMock()
    manager.download_and_install = MagicMock()
    manager.show_env_details = MagicMock()

    monkeypatch.setattr(
        cli_commands, "get_installation_manager", lambda *_, **__: manager
    )

    cli_commands.backend_list()

    manager.show_env_details.assert_called_once()
    manager.download_and_install.assert_not_called()


def test_check_preserves_interactive_eula_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    get_advice = MagicMock()
    monkeypatch.setattr(cli_commands, "validate_check_target_profile", MagicMock())
    monkeypatch.setattr(cli_commands, "get_advice", get_advice)

    cli_commands.check(
        MagicMock(),
        "ethos-u55-256",
        model="model.tflite",
        backend=["corstone-300"],
    )

    assert get_advice.call_args.kwargs["backends"] == ["corstone-300"]
    assert get_advice.call_args.kwargs["accept_eula"] is None


def test_check_noninteractive_without_eula_rejects_eula_install(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    get_advice = MagicMock()
    monkeypatch.setattr(cli_commands, "validate_check_target_profile", MagicMock())
    monkeypatch.setattr(cli_commands, "get_advice", get_advice)

    cli_commands.check(
        MagicMock(),
        "ethos-u55-256",
        model="model.tflite",
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

    cli_commands.check(
        MagicMock(),
        "ethos-u55-256",
        model="model.tflite",
        backend=["corstone-300"],
        noninteractive=True,
        i_agree_to_the_contained_eula=True,
    )

    assert get_advice.call_args.kwargs["backends"] == ["corstone-300"]
    assert get_advice.call_args.kwargs["accept_eula"] is True
