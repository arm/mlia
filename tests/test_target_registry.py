# SPDX-FileCopyrightText: Copyright 2022, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the target registry module."""
from __future__ import annotations

import pytest

from mlia.core.common import AdviceCategory
from mlia.target.registry import registry
from mlia.target.registry import supported_advice
from mlia.target.registry import supported_backends
from mlia.target.registry import supported_targets


@pytest.mark.parametrize(
    "expected_target", ("Cortex-A", "Ethos-U55", "Ethos-U65", "TOSA")
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
        ("Cortex-A", [AdviceCategory.OPERATORS]),
        (
            "Ethos-U55",
            [
                AdviceCategory.OPERATORS,
                AdviceCategory.OPTIMIZATION,
                AdviceCategory.PERFORMANCE,
            ],
        ),
        (
            "Ethos-U65",
            [
                AdviceCategory.OPERATORS,
                AdviceCategory.OPTIMIZATION,
                AdviceCategory.PERFORMANCE,
            ],
        ),
        ("TOSA", [AdviceCategory.OPERATORS]),
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
        ("Cortex-A", ["ArmNNTFLiteDelegate"]),
        ("Ethos-U55", ["Corstone-300", "Corstone-310", "Vela"]),
        ("Ethos-U65", ["Corstone-300", "Corstone-310", "Vela"]),
        ("TOSA", ["TOSA-Checker"]),
    ),
)
def test_supported_backends(target_name: str, expected_backends: list[str]) -> None:
    """Test function supported_backends()."""
    assert sorted(expected_backends) == sorted(supported_backends(target_name))


@pytest.mark.parametrize(
    ("advice", "expected_targets"),
    (
        (AdviceCategory.OPERATORS, ["Cortex-A", "Ethos-U55", "Ethos-U65", "TOSA"]),
        (AdviceCategory.OPTIMIZATION, ["Ethos-U55", "Ethos-U65"]),
        (AdviceCategory.PERFORMANCE, ["Ethos-U55", "Ethos-U65"]),
    ),
)
def test_supported_targets(advice: AdviceCategory, expected_targets: list[str]) -> None:
    """Test function supported_targets()."""
    assert sorted(expected_targets) == sorted(supported_targets(advice))
