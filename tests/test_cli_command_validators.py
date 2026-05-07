# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for CLI command validators."""

from __future__ import annotations

import argparse

import pytest

from mlia.cli import command_validators


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
        lambda target: ["corstone-300", "nx-performance-estimator"],
    )

    assert command_validators.validate_backend(
        "target-profile",
        ["Corstone-300", "NXPerformanceEstimator"],
    ) == ["corstone-300", "nx-performance-estimator"]


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

    with pytest.raises(argparse.ArgumentError, match="not supported"):
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
