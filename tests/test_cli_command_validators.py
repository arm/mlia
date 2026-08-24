# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for CLI command validators."""

from __future__ import annotations

import pytest

from mlia.backend.config import BackendConfiguration, BackendType
from mlia.cli import command_validators
from mlia.core.errors import ConfigurationError


def test_validate_backend_returns_canonical_backend_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Backend validation should normalize user input to registry keys."""
    monkeypatch.setattr(
        command_validators, "get_target", lambda target_profile: "target"
    )
    monkeypatch.setattr(
        command_validators,
        "supported_backends",
        lambda target: ["corstone-300", "vela"],
    )

    assert command_validators.validate_backend(
        "target-profile",
        ["Corstone300", "Vela"],
    ) == ["corstone-300", "vela"]


def test_validate_backend_rejects_unsupported_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Backend validation should reject unknown backend names."""
    monkeypatch.setattr(
        command_validators, "get_target", lambda target_profile: "target"
    )
    monkeypatch.setattr(
        command_validators,
        "supported_backends",
        lambda target: ["corstone-300"],
    )

    with pytest.raises(ConfigurationError, match="not supported with target profile"):
        command_validators.validate_backend("target-profile", ["unknown"])


def test_validate_backend_returns_default_backends(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Backend validation should return defaults when no backend is provided."""
    monkeypatch.setattr(
        command_validators, "get_target", lambda target_profile: "target"
    )
    monkeypatch.setattr(
        command_validators,
        "default_backends",
        lambda target: ["default-backend"],
    )

    assert command_validators.validate_backend("target-profile", None) == [
        "default-backend"
    ]


@pytest.mark.parametrize(
    "input_string, expected_output",
    [
        ("", ""),
        ("lowercase", "lowercase"),
        ("UPPERCASE", "uppercase"),
        ("VELA", "vela"),
        ("check-no-hyphens", "checknohyphens"),
        ("MixedCase-With-Hyphens", "mixedcasewithhyphens"),
        ("corstone-310", "corstone310"),
        ("---multiple---hyphens---", "multiplehyphens"),
    ],
)
def test_normalize_string(input_string: str, expected_output: str) -> None:
    """Test normalize_string function with various inputs."""
    assert command_validators.normalize_string(input_string) == expected_output


def test_validate_backend_selects_unique_profiling_capable_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Profiling input should select the unique capable backend."""
    monkeypatch.setattr(command_validators, "get_target", lambda _profile: "target")
    monkeypatch.setattr(
        command_validators,
        "supported_backends",
        lambda _target: ["estimator", "profiler"],
    )
    monkeypatch.setattr(command_validators, "default_backends", lambda _target: [])
    monkeypatch.setattr(
        command_validators.backend_registry,
        "items",
        {
            "estimator": BackendConfiguration([], None, BackendType.BUILTIN, None),
            "profiler": BackendConfiguration(
                [],
                None,
                BackendType.BUILTIN,
                None,
                supports_profiling_data=True,
            ),
        },
    )

    assert command_validators.validate_backend(
        "target-profile", None, profiling_data=True
    ) == ["profiler"]


def test_validate_backend_rejects_profiling_backend_ambiguity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Multiple profiling-capable backends should require explicit selection."""
    monkeypatch.setattr(command_validators, "get_target", lambda _profile: "target")
    monkeypatch.setattr(
        command_validators, "supported_backends", lambda _target: ["one", "two"]
    )
    monkeypatch.setattr(command_validators, "default_backends", lambda _target: [])
    monkeypatch.setattr(
        command_validators.backend_registry,
        "items",
        {
            name: BackendConfiguration(
                [],
                None,
                BackendType.BUILTIN,
                None,
                supports_profiling_data=True,
            )
            for name in ("one", "two")
        },
    )

    with pytest.raises(ConfigurationError, match="Multiple backends"):
        command_validators.validate_backend("target-profile", None, profiling_data=True)


def test_validate_backend_rejects_estimator_for_profiling_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An explicitly selected estimator must not consume measured data."""
    monkeypatch.setattr(command_validators, "get_target", lambda _profile: "target")
    monkeypatch.setattr(
        command_validators,
        "supported_backends",
        lambda _target: ["estimator"],
    )
    monkeypatch.setattr(
        command_validators.backend_registry,
        "items",
        {"estimator": BackendConfiguration([], None, BackendType.BUILTIN, None)},
    )

    with pytest.raises(ConfigurationError, match="does not support profiling data"):
        command_validators.validate_backend(
            "target-profile", ["estimator"], profiling_data=True
        )


def test_validate_check_target_profile_returns_false_when_nothing_can_run(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Check validation should signal a clean no-op instead of exiting directly."""
    with caplog.at_level("WARNING"):
        result = command_validators.validate_check_target_profile(
            "tosa",
            {"performance"},
        )

    assert not result
    assert "No operation was performed." in caplog.text


def test_validate_check_target_profile_returns_true_when_some_checks_remain(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Check validation should keep the command running when work remains."""
    with caplog.at_level("WARNING"):
        result = command_validators.validate_check_target_profile(
            "tosa",
            {"compatibility", "performance"},
        )

    assert result
    assert "Performance checks skipped" in caplog.text
