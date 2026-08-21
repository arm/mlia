# SPDX-FileCopyrightText: Copyright 2023, 2025-2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for process management functions."""

import sys
from subprocess import CalledProcessError  # nosec
from unittest.mock import MagicMock

import pytest

from mlia.utils.proc import Command, process_command_output


def test_process_command_output() -> None:
    """Test function process_command_output."""
    command = Command([sys.executable, "-c", "print('sample message', end='')"])

    output_consumer = MagicMock()
    process_command_output(command, [output_consumer])

    output_consumer.assert_called_once_with("sample message")


def test_process_command_output_subprocess_error() -> None:
    """Test function process_command_output."""

    command = Command([sys.executable, "-c", "raise SystemExit(1)"])

    output_consumer = MagicMock()
    with pytest.raises(CalledProcessError):
        process_command_output(command, [output_consumer])
