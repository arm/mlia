# SPDX-FileCopyrightText: Copyright 2022-2025, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the target registry module."""
from __future__ import annotations

import pytest

from mlia.core.common import AdviceCategory
from mlia.target.config import get_builtin_optimization_profile_path
from mlia.target.config import get_builtin_target_profile_path
from mlia.target.registry import all_supported_backends
from mlia.target.registry import default_backends
from mlia.target.registry import get_optimization_profile
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
        ("armnn-tflite-delegate", None, True),
        ("armnn-tflite-delegate", "cortex-a", True),
        ("armnn-tflite-delegate", "tosa", False),
        ("corstone-310", None, True),
        ("corstone-310", "ethos-u55", True),
        ("corstone-310", "ethos-u65", True),
        ("corstone-310", "cortex-a", False),
        ("corstone-320", None, True),
        ("corstone-320", "ethos-u55", False),
        ("corstone-320", "ethos-u85", True),
        ("corstone-320", "cortex-a", False),
    ),
)
def test_is_supported(backend: str, target: str | None, expected_result: bool) -> None:
    """Test function is_supported()."""
    assert is_supported(backend, target) == expected_result


@pytest.mark.parametrize(
    ("target_name", "expected_backends"),
    (
        ("cortex-a", ["armnn-tflite-delegate"]),
        ("ethos-u55", ["corstone-300", "corstone-310", "vela"]),
        ("ethos-u65", ["corstone-300", "corstone-310", "vela"]),
        ("ethos-u85", ["corstone-320", "vela"]),
        ("tosa", ["tosa-checker"]),
    ),
)
def test_supported_backends(target_name: str, expected_backends: list[str]) -> None:
    """Test function supported_backends()."""
    assert sorted(expected_backends) == sorted(supported_backends(target_name))


@pytest.mark.parametrize(
    ("advice", "expected_targets"),
    (
        (
            AdviceCategory.COMPATIBILITY,
            ["cortex-a", "ethos-u55", "ethos-u65", "ethos-u85", "tosa"],
        ),
        (AdviceCategory.OPTIMIZATION, ["ethos-u55", "ethos-u65", "ethos-u85"]),
        (AdviceCategory.PERFORMANCE, ["ethos-u55", "ethos-u65", "ethos-u85"]),
    ),
)
def test_supported_targets(advice: AdviceCategory, expected_targets: list[str]) -> None:
    """Test function supported_targets()."""
    assert sorted(expected_targets) == sorted(supported_targets(advice))


def test_all_supported_backends() -> None:
    """Test function all_supported_backends."""
    assert all_supported_backends() == {
        "vela",
        "tosa-checker",
        "armnn-tflite-delegate",
        "corstone-320",
        "corstone-310",
        "corstone-300",
    }


@pytest.mark.parametrize(
    ("target", "expected_default_backends", "is_subset_only"),
    [
        ["cortex-a", ["armnn-tflite-delegate"], False],
        ["tosa", ["tosa-checker"], False],
        ["ethos-u55", ["vela"], True],
        ["ethos-u65", ["vela"], True],
        ["ethos-u85", ["vela"], True],
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
    profile_file = get_builtin_target_profile_path(target_profile)
    cfg = profile(profile_file)
    assert target_profile.startswith(cfg.target)


@pytest.mark.parametrize("optimization_profile", ["optimization"])
def test_optimization_profile(optimization_profile: str) -> None:
    """Test function optimization_profile()."""

    get_optimization_profile(optimization_profile)

    profile_file = get_builtin_optimization_profile_path(optimization_profile)
    get_optimization_profile(profile_file)


@pytest.mark.parametrize("optimization_profile", ["non_valid_file"])
def test_optimization_profile_non_valid_file(optimization_profile: str) -> None:
    """Test function optimization_profile()."""
    with pytest.raises(
        ValueError,
        match=f"optimization Profile '{optimization_profile}' is neither "
        "a valid built-in optimization profile name or a valid file path.",
    ):
        get_optimization_profile(optimization_profile)
