# SPDX-FileCopyrightText: Copyright 2022-2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the backend registry module."""

from __future__ import annotations

from functools import partial

import pytest

from mlia.backend.config import BackendType, System
from mlia.backend.registry import (
    get_supported_backends,
    get_supported_systems,
    registry,
)
from mlia.core.common import AdviceCategory


@pytest.mark.parametrize(
    ("backend", "advices", "systems", "type_"),
    (
        (
            "corstone-300",
            [
                AdviceCategory.COMPATIBILITY,
                AdviceCategory.PERFORMANCE,
                AdviceCategory.OPTIMIZATION,
            ],
            [System.LINUX_AMD64, System.LINUX_AARCH64],
            BackendType.CUSTOM,
        ),
        (
            "corstone-310",
            [
                AdviceCategory.COMPATIBILITY,
                AdviceCategory.PERFORMANCE,
                AdviceCategory.OPTIMIZATION,
            ],
            [System.LINUX_AMD64, System.LINUX_AARCH64],
            BackendType.CUSTOM,
        ),
        (
            "corstone-320",
            [
                AdviceCategory.COMPATIBILITY,
                AdviceCategory.PERFORMANCE,
                AdviceCategory.OPTIMIZATION,
            ],
            [System.LINUX_AMD64, System.LINUX_AARCH64],
            BackendType.CUSTOM,
        ),
        (
            "tosa-checker",
            [AdviceCategory.COMPATIBILITY],
            [System.LINUX_AMD64],
            BackendType.WHEEL,
        ),
        (
            "vela",
            [
                AdviceCategory.COMPATIBILITY,
                AdviceCategory.PERFORMANCE,
                AdviceCategory.OPTIMIZATION,
            ],
            [
                System.LINUX_AMD64,
                System.LINUX_AARCH64,
                System.WINDOWS_AMD64,
                System.WINDOWS_AARCH64,
            ],
            BackendType.WHEEL,
        ),
    ),
)
def test_backend_registry(
    backend: str,
    advices: list[AdviceCategory],
    systems: list[System] | None,
    type_: BackendType,
) -> None:
    """Test the backend registry."""
    sorted_by_name = partial(sorted, key=lambda x: x.name)

    assert backend in registry.items
    cfg = registry.items[backend]
    assert sorted_by_name(advices) == sorted_by_name(cfg.supported_advice), (
        f"Advices differs: {advices} != {cfg.supported_advice}"
    )
    if systems is None:
        assert cfg.supported_systems is None
    else:
        assert cfg.supported_systems is not None
        assert sorted_by_name(systems) == sorted_by_name(cfg.supported_systems), (
            f"Supported systems differs: {advices} != {cfg.supported_advice}"
        )
    assert cfg.type == type_


_SUPPORTED_BACKENDS = {
    "corstone-300",
    "corstone-310",
    "corstone-320",
    "tosa-checker",
    "vela",
}


def test_get_supported_backends() -> None:
    """Test function get_supported_backends."""
    assert get_supported_backends() == sorted(_SUPPORTED_BACKENDS)


def test_get_supported_systems() -> None:
    """Test function get_supported_systems."""
    assert get_supported_systems()
