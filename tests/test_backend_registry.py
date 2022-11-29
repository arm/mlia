# SPDX-FileCopyrightText: Copyright 2022-2023, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the backend registry module."""
from __future__ import annotations

from functools import partial

import pytest

from mlia.backend.config import BackendType
from mlia.backend.config import System
from mlia.backend.registry import registry
from mlia.core.common import AdviceCategory


@pytest.mark.parametrize(
    ("backend", "advices", "systems", "type_"),
    (
        (
            "ArmNNTFLiteDelegate",
            [AdviceCategory.COMPATIBILITY],
            None,
            BackendType.BUILTIN,
        ),
        (
            "Corstone-300",
            [AdviceCategory.PERFORMANCE, AdviceCategory.OPTIMIZATION],
            [System.LINUX_AMD64],
            BackendType.CUSTOM,
        ),
        (
            "Corstone-310",
            [AdviceCategory.PERFORMANCE, AdviceCategory.OPTIMIZATION],
            [System.LINUX_AMD64],
            BackendType.CUSTOM,
        ),
        (
            "TOSA-Checker",
            [AdviceCategory.COMPATIBILITY],
            [System.LINUX_AMD64],
            BackendType.WHEEL,
        ),
        (
            "Vela",
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
            BackendType.BUILTIN,
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
    assert sorted_by_name(advices) == sorted_by_name(
        cfg.supported_advice
    ), f"Advices differs: {advices} != {cfg.supported_advice}"
    if systems is None:
        assert cfg.supported_systems is None
    else:
        assert cfg.supported_systems is not None
        assert sorted_by_name(systems) == sorted_by_name(
            cfg.supported_systems
        ), f"Supported systems differs: {advices} != {cfg.supported_advice}"
    assert cfg.type == type_
