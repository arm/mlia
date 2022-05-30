# SPDX-FileCopyrightText: Copyright 2022, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Test for cli common module."""
from typing import Any

import pytest

from aiet.cli.common import print_command_details
from aiet.cli.common import raise_exception_at_signal


def test_print_command_details(capsys: Any) -> None:
    """Test print_command_details function."""
    command = {
        "command_strings": ["echo test"],
        "user_params": [
            {"name": "param_name", "description": "param_description"},
            {
                "name": "param_name2",
                "description": "param_description2",
                "alias": "alias2",
            },
        ],
    }
    print_command_details(command)
    captured = capsys.readouterr()
    assert "echo test" in captured.out
    assert "param_name" in captured.out
    assert "alias2" in captured.out


def test_raise_exception_at_signal() -> None:
    """Test raise_exception_at_signal graceful shutdown."""
    with pytest.raises(Exception) as err:
        raise_exception_at_signal(1, "")

    assert str(err.value) == "Middleware shutdown requested"
