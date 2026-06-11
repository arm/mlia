# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for CLI command validators."""

from __future__ import annotations

import pytest

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
