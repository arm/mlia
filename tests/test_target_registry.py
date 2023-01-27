# SPDX-FileCopyrightText: Copyright 2022-2023, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the target registry module."""
from __future__ import annotations

import pytest

from mlia.core.common import AdviceCategory
from mlia.target.registry import all_supported_backends
from mlia.target.registry import registry
from mlia.target.registry import supported_advice
from mlia.target.registry import supported_backends
from mlia.target.registry import supported_targets


@pytest.mark.parametrize(
    "expected_target", ("cortex-a", "ethos-u55", "ethos-u65", "tosa")
)
def test_target_registry(expected_target: str) -> None:
    """Test the target registry."""
    assert expected_target in registry.items, (
        f"Expected target '{expected_target}' not contained in registered "
        f"targets '{registry.items.keys()}'."
    )


@pytest.mark.parametrize(
    ("target_name", "expected_advices"),
    (
        ("cortex-a", [AdviceCategory.COMPATIBILITY]),
        (
            "ethos-u55",
            [
                AdviceCategory.COMPATIBILITY,
                AdviceCategory.OPTIMIZATION,
                AdviceCategory.PERFORMANCE,
            ],
        ),
        (
            "ethos-u65",
            [
                AdviceCategory.COMPATIBILITY,
                AdviceCategory.OPTIMIZATION,
                AdviceCategory.PERFORMANCE,
            ],
        ),
        ("tosa", [AdviceCategory.COMPATIBILITY]),
    ),
)
def test_supported_advice(
    target_name: str, expected_advices: list[AdviceCategory]
) -> None:
    """Test function supported_advice()."""
    supported = supported_advice(target_name)
    assert all(advice in expected_advices for advice in supported)
    assert all(advice in supported for advice in expected_advices)


@pytest.mark.parametrize(
    ("target_name", "expected_backends"),
    (
        ("cortex-a", ["ArmNNTFLiteDelegate"]),
        ("ethos-u55", ["Corstone-300", "Corstone-310", "Vela"]),
        ("ethos-u65", ["Corstone-300", "Corstone-310", "Vela"]),
        ("tosa", ["tosa-checker"]),
    ),
)
def test_supported_backends(target_name: str, expected_backends: list[str]) -> None:
    """Test function supported_backends()."""
    assert sorted(expected_backends) == sorted(supported_backends(target_name))


@pytest.mark.parametrize(
    ("advice", "expected_targets"),
    (
        (AdviceCategory.COMPATIBILITY, ["cortex-a", "ethos-u55", "ethos-u65", "tosa"]),
        (AdviceCategory.OPTIMIZATION, ["ethos-u55", "ethos-u65"]),
        (AdviceCategory.PERFORMANCE, ["ethos-u55", "ethos-u65"]),
    ),
)
def test_supported_targets(advice: AdviceCategory, expected_targets: list[str]) -> None:
    """Test function supported_targets()."""
    assert sorted(expected_targets) == sorted(supported_targets(advice))


def test_all_supported_backends() -> None:
    """Test function all_supported_backends."""
    assert all_supported_backends() == {
        "Vela",
        "tosa-checker",
        "ArmNNTFLiteDelegate",
        "Corstone-310",
        "Corstone-300",
    }
