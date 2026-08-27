# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for validation of SBOMs embedded in wheels."""

from __future__ import annotations

import json
from pathlib import Path
from zipfile import ZipFile

import pytest

from hatch_build import create_sbom
from scripts.validate_wheel_sbom import validate_wheel_sboms


def _write_wheel(tmp_path: Path, sbom: dict | None) -> Path:
    """Write a minimal wheel-shaped ZIP with an optional SBOM."""
    wheel_path = tmp_path / "example-1.2.3-py3-none-any.whl"
    with ZipFile(wheel_path, "w") as wheel:
        wheel.writestr("example/__init__.py", "")
        if sbom is not None:
            wheel.writestr(
                "example-1.2.3.dist-info/sboms/example.cdx.json",
                json.dumps(sbom),
            )
    return wheel_path


def _valid_sbom() -> dict:
    """Create a valid SBOM for a test wheel."""
    return create_sbom(
        name="example",
        normalized_name="example",
        version="1.2.3",
        description="Example package",
        license_expression="Apache-2.0",
        urls={},
    )


def test_validate_wheel_sboms(tmp_path: Path) -> None:
    """A valid embedded CycloneDX SBOM should be accepted."""
    validate_wheel_sboms(_write_wheel(tmp_path, _valid_sbom()))


def test_validate_wheel_sboms_requires_sbom(tmp_path: Path) -> None:
    """A wheel without an embedded CycloneDX SBOM should be rejected."""
    with pytest.raises(ValueError, match="contains no CycloneDX JSON SBOM"):
        validate_wheel_sboms(_write_wheel(tmp_path, None))


def test_validate_wheel_sboms_rejects_invalid_sbom(tmp_path: Path) -> None:
    """An invalid embedded CycloneDX SBOM should be rejected."""
    sbom = _valid_sbom()
    sbom["version"] = "invalid"

    with pytest.raises(ValueError, match="example.cdx.json"):
        validate_wheel_sboms(_write_wheel(tmp_path, sbom))
