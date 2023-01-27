# SPDX-FileCopyrightText: Copyright 2023, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for cli.command_validators module."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from unittest.mock import MagicMock

import pytest

from mlia.cli.command_validators import validate_backend
from mlia.cli.command_validators import validate_check_target_profile


@pytest.mark.parametrize(
    "target_profile, category, expected_warnings, sys_exits",
    [
        ["ethos-u55-256", {"compatibility", "performance"}, [], False],
        ["ethos-u55-256", {"compatibility"}, [], False],
        ["ethos-u55-256", {"performance"}, [], False],
        [
            "tosa",
            {"compatibility", "performance"},
            [
                (
                    "\nWARNING: Performance checks skipped as they cannot be "
                    "performed with target profile tosa."
                )
            ],
            False,
        ],
        [
            "tosa",
            {"performance"},
            [
                (
                    "\nWARNING: Performance checks skipped as they cannot be "
                    "performed with target profile tosa. No operation was performed."
                )
            ],
            True,
        ],
        ["tosa", "compatibility", [], False],
        [
            "cortex-a",
            {"performance"},
            [
                (
                    "\nWARNING: Performance checks skipped as they cannot be "
                    "performed with target profile cortex-a. "
                    "No operation was performed."
                )
            ],
            True,
        ],
        [
            "cortex-a",
            {"compatibility", "performance"},
            [
                (
                    "\nWARNING: Performance checks skipped as they cannot be "
                    "performed with target profile cortex-a."
                )
            ],
            False,
        ],
        ["cortex-a", "compatibility", [], False],
    ],
)
def test_validate_check_target_profile(
    caplog: pytest.LogCaptureFixture,
    target_profile: str,
    category: set[str],
    expected_warnings: list[str],
    sys_exits: bool,
) -> None:
    """Test outcomes of category dependent target profile validation."""
    # Capture if program terminates
    if sys_exits:
        with pytest.raises(SystemExit) as sys_ex:
            validate_check_target_profile(target_profile, category)
        assert sys_ex.value.code == 0
        return

    validate_check_target_profile(target_profile, category)

    log_records = caplog.records
    # Get all log records with level 30 (warning level)
    warning_messages = {x.message for x in log_records if x.levelno == 30}
    # Ensure the warnings coincide with the expected ones
    assert warning_messages == set(expected_warnings)


@pytest.mark.parametrize(
    "input_target_profile, input_backends, throws_exception,"
    "exception_message, output_backends",
    [
        [
            "tosa",
            ["Vela"],
            True,
            "Vela backend not supported with target-profile tosa.",
            None,
        ],
        [
            "tosa",
            ["Corstone-300, Vela"],
            True,
            "Corstone-300, Vela backend not supported with target-profile tosa.",
            None,
        ],
        [
            "cortex-a",
            ["Corstone-310", "tosa-checker"],
            True,
            "Corstone-310, tosa-checker backend not supported "
            "with target-profile cortex-a.",
            None,
        ],
        [
            "ethos-u55-256",
            ["tosa-checker", "Corstone-310"],
            True,
            "tosa-checker backend not supported with target-profile ethos-u55-256.",
            None,
        ],
        ["tosa", None, False, None, ["tosa-checker"]],
        ["cortex-a", None, False, None, ["ArmNNTFLiteDelegate"]],
        ["tosa", ["tosa-checker"], False, None, ["tosa-checker"]],
        ["cortex-a", ["ArmNNTFLiteDelegate"], False, None, ["ArmNNTFLiteDelegate"]],
        [
            "ethos-u55-256",
            ["Vela", "Corstone-300"],
            False,
            None,
            ["Vela", "Corstone-300"],
        ],
        [
            "ethos-u55-256",
            None,
            False,
            None,
            ["Vela", "Corstone-300"],
        ],
    ],
)
def test_validate_backend(
    monkeypatch: pytest.MonkeyPatch,
    input_target_profile: str,
    input_backends: list[str] | None,
    throws_exception: bool,
    exception_message: str,
    output_backends: list[str] | None,
) -> None:
    """Test backend validation with target-profiles and backends."""
    monkeypatch.setattr(
        "mlia.cli.config.get_available_backends",
        MagicMock(return_value=["Vela", "Corstone-300"]),
    )

    exit_stack = ExitStack()
    if throws_exception:
        exit_stack.enter_context(
            pytest.raises(argparse.ArgumentError, match=exception_message)
        )

    with exit_stack:
        assert validate_backend(input_target_profile, input_backends) == output_backends
