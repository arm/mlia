# SPDX-FileCopyrightText: Copyright 2023, 2025, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for cli.command_validators module."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from unittest.mock import MagicMock

import pytest

from mlia.cli.command_validators import normalize_string
from mlia.cli.command_validators import validate_backend
from mlia.cli.command_validators import validate_check_target_profile
from mlia.cli.command_validators import validate_optimize_target_profile


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
    (
        "input_target_profile",
        "input_backends",
        "throws_exception",
        "exception_message",
        "output_backends",
    ),
    [
        [
            "tosa",
            ["tosa-checker"],
            False,
            None,
            ["tosa-checker"],
        ],
        [
            "tosa",
            ["corstone-310"],
            True,
            "Backend corstone-310 not supported with target-profile tosa.",
            None,
        ],
        [
            "ethos-u85-1024",
            ["corstone-320"],
            False,
            None,
            ["corstone-320"],
        ],
        [
            "ethos-u55-256",
            ["corstone-320"],
            True,
            "Backend corstone-320 not supported with target-profile ethos-u55-256.",
            None,
        ],
        [
            "cortex-a",
            ["armnn-tflite-delegate"],
            False,
            None,
            ["armnn-tflite-delegate"],
        ],
        [
            "cortex-a",
            ["tosa-checker"],
            True,
            "Backend tosa-checker not supported with target-profile cortex-a.",
            None,
        ],
        [
            "ethos-u55-256",
            ["vela", "corstone-310"],
            False,
            None,
            ["vela", "corstone-310"],
        ],
        [
            "ethos-u65-256",
            ["vela", "corstone-310", "tosa-checker"],
            True,
            "Backend tosa-checker not supported with target-profile ethos-u65-256.",
            None,
        ],
    ],
)
def test_validate_backend(
    input_target_profile: str,
    input_backends: list[str],
    throws_exception: bool,
    exception_message: str,
    output_backends: list[str] | None,
) -> None:
    """Test backend validation with target-profiles and backends."""
    exit_stack = ExitStack()
    if throws_exception:
        exit_stack.enter_context(
            pytest.raises(argparse.ArgumentError, match=exception_message)
        )

    with exit_stack:
        assert validate_backend(input_target_profile, input_backends) == output_backends


def test_validate_backend_default_available() -> None:
    """Test default backend validation with available backend."""
    backends = validate_backend("cortex-a", None)
    assert backends
    assert backends == ["armnn-tflite-delegate"]


def test_validate_backend_default_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test default backend validation with unavailable backend."""
    monkeypatch.setattr(
        "mlia.cli.command_validators.default_backends",
        MagicMock(return_value=["UNKNOWN_BACKEND"]),
    )
    with pytest.raises(argparse.ArgumentError):
        validate_backend("cortex-a", None)


@pytest.mark.parametrize(
    "target_profile, sys_exit",
    [("ethos-u55-128", False), ("tosa", True), ("cortex-a", True)],
)
def test_validate_optimize_target_profile(
    target_profile: str,
    sys_exit: bool,
) -> None:
    """Tests if an incompatible target is passed for optimization."""
    if sys_exit:
        with pytest.raises(SystemExit) as sys_ex:
            validate_optimize_target_profile(target_profile)
        assert sys_ex.value.code == 1
        return

    validate_optimize_target_profile(target_profile)


@pytest.mark.parametrize(
    "input_string, expected_output",
    [
        ["", ""],
        ["lowercase", "lowercase"],
        ["UPPERCASE", "uppercase"],
        ["VELA", "vela"],
        ["check-no-hyphens", "checknohyphens"],
        ["MixedCase-With-Hyphens", "mixedcasewithhyphens"],
        ["ToSa-cHecker", "tosachecker"],
        ["corstone-310", "corstone310"],
        ["armnn-tflite-delegate", "armnntflitedelegate"],
        ["---multiple---hyphens---", "multiplehyphens"],
    ],
)
def test_normalize_string(input_string: str, expected_output: str) -> None:
    """Test normalize_string function with various inputs."""
    assert normalize_string(input_string) == expected_output


@pytest.mark.parametrize(
    "supported_backends, target, target_profile, backends, expected",
    [
        (
            ["armnn-tflite-delegate"],
            "cortex-a",
            "cortex-a",
            ["armnn-tflite-delegate"],
            ["armnn-tflite-delegate"],
        ),
        (
            ["Vela", "Corstone-310"],
            "ethos-u55",
            "ethos-u55-256",
            ["VELA", "corstone-310"],
            ["VELA", "corstone-310"],
        ),
    ],
    ids=["hyphen_normalization", "case_insensitive"],
)
def test_validate_backend_normalization(
    supported_backends: list[str],
    target: str,
    target_profile: str,
    backends: list[str],
    expected: list[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test backend validation with hyphen and case normalization."""
    monkeypatch.setattr(
        "mlia.cli.command_validators.supported_backends",
        MagicMock(return_value=supported_backends),
    )
    monkeypatch.setattr(
        "mlia.cli.command_validators.get_target",
        MagicMock(return_value=target),
    )

    result = validate_backend(target_profile, backends)
    assert result == expected


def test_validate_backend_multiple_incompatible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test validate_backend with multiple incompatible backends."""
    monkeypatch.setattr(
        "mlia.cli.command_validators.supported_backends",
        MagicMock(return_value=["vela"]),
    )
    monkeypatch.setattr(
        "mlia.cli.command_validators.get_target",
        MagicMock(return_value="ethos-u55"),
    )
    with pytest.raises(argparse.ArgumentError, match="not supported"):
        validate_backend("ethos-u55-256", ["tosa-checker", "corstone-320"])
