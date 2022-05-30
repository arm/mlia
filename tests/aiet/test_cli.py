# SPDX-FileCopyrightText: Copyright 2022, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Module for testing CLI top command."""
from typing import Any
from unittest.mock import ANY
from unittest.mock import MagicMock

from click.testing import CliRunner

from aiet.cli import cli


def test_cli(cli_runner: CliRunner) -> None:
    """Test CLI top level command."""
    result = cli_runner.invoke(cli)
    assert result.exit_code == 0
    assert "system" in cli.commands
    assert "application" in cli.commands


def test_cli_version(cli_runner: CliRunner) -> None:
    """Test version option."""
    result = cli_runner.invoke(cli, ["--version"])
    assert result.exit_code == 0
    assert "version" in result.output


def test_cli_verbose(cli_runner: CliRunner, monkeypatch: Any) -> None:
    """Test verbose option."""
    with monkeypatch.context() as mock_context:
        mock = MagicMock()
        # params[1] is the verbose option and we need to replace the
        # callback with a mock object
        mock_context.setattr(cli.params[1], "callback", mock)
        cli_runner.invoke(cli, ["-vvvv"])
        # 4 is the number -v called earlier
        mock.assert_called_once_with(ANY, ANY, 4)
