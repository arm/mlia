# SPDX-FileCopyrightText: Copyright 2023, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for process management functions."""
from unittest.mock import MagicMock

from mlia.utils.proc import Command
from mlia.utils.proc import process_command_output


def test_process_command_output() -> None:
    """Test function process_command_output."""
    command = Command(["echo", "-n", "sample message"])

    output_consumer = MagicMock()
    process_command_output(command, [output_consumer])

    output_consumer.assert_called_once_with("sample message")
