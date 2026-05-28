# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for CLI entry point behavior."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest

import mlia.cli.main as cli_main


def test_no_arguments_show_help(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Calling a CLI entry point without arguments should show help."""
    parser = MagicMock()

    monkeypatch.setattr(cli_main, "init_parser", lambda commands: parser)
    monkeypatch.setattr(sys, "argv", ["mlia"])

    result = cli_main.init_and_run([])

    assert result == 2
    parser.print_help.assert_called_once_with()
    parser.parse_args.assert_not_called()
