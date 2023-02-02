# SPDX-FileCopyrightText: Copyright 2022-2023, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the target registry module."""
from __future__ import annotations

import pytest

from mlia.core.common import AdviceCategory
from mlia.target.config import get_builtin_profile_path
from mlia.target.registry import all_supported_backends
from mlia.target.registry import default_backends
from mlia.target.registry import is_supported
from mlia.target.registry import profile
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
    ("backend", "target", "expected_result"),
    (
        ("ArmNNTFLiteDelegate", None, True),
        ("ArmNNTFLiteDelegate", "cortex-a", True),
        ("ArmNNTFLiteDelegate", "tosa", False),
        ("Corstone-310", None, True),
        ("Corstone-310", "ethos-u55", True),
        ("Corstone-310", "ethos-u65", True),
        ("Corstone-310", "cortex-a", False),
    ),
)
def test_is_supported(backend: str, target: str | None, expected_result: bool) -> None:
    """Test function is_supported()."""
    assert is_supported(backend, target) == expected_result


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


@pytest.mark.parametrize(
    ("target", "expected_default_backends", "is_subset_only"),
    [
        ["cortex-a", ["ArmNNTFLiteDelegate"], False],
        ["tosa", ["tosa-checker"], False],
        ["ethos-u55", ["Vela"], True],
        ["ethos-u65", ["Vela"], True],
    ],
)
def test_default_backends(
    target: str,
    expected_default_backends: list[str],
    is_subset_only: bool,
) -> None:
    """Test function default_backends()."""
    if is_subset_only:
        assert set(expected_default_backends).issubset(default_backends(target))
    else:
        assert default_backends(target) == expected_default_backends


@pytest.mark.parametrize(
    "target_profile", ("cortex-a", "tosa", "ethos-u55-128", "ethos-u65-256")
)
def test_profile(target_profile: str) -> None:
    """Test function profile()."""
    # Test loading by built-in profile name
    cfg = profile(target_profile)
    assert target_profile.startswith(cfg.target)

    # Test loading the file directly
    profile_file = get_builtin_profile_path(target_profile)
    cfg = profile(profile_file)
    assert target_profile.startswith(cfg.target)
