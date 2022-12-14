# SPDX-FileCopyrightText: Copyright 2022, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the backend config module."""
from __future__ import annotations

import pytest

from mlia.backend.config import BackendConfiguration
from mlia.backend.config import BackendType
from mlia.backend.config import System
from mlia.core.common import AdviceCategory
from mlia.target.config import IPConfiguration
from mlia.target.config import TargetInfo
from mlia.utils.registry import Registry


def test_ip_config() -> None:
    """Test the class 'IPConfiguration'."""
    cfg = IPConfiguration("AnyTarget")
    assert cfg.target == "AnyTarget"


@pytest.mark.parametrize(
    ("advice", "check_system", "supported"),
    (
        (None, False, True),
        (None, True, True),
        (AdviceCategory.OPERATORS, True, True),
        (AdviceCategory.OPTIMIZATION, True, False),
    ),
)
def test_target_info(
    monkeypatch: pytest.MonkeyPatch,
    advice: AdviceCategory | None,
    check_system: bool,
    supported: bool,
) -> None:
    """Test the class 'TargetInfo'."""
    info = TargetInfo(["backend"])

    backend_registry = Registry[BackendConfiguration]()
    backend_registry.register(
        "backend",
        BackendConfiguration(
            [AdviceCategory.OPERATORS],
            [System.CURRENT],
            BackendType.BUILTIN,
        ),
    )
    monkeypatch.setattr("mlia.target.config.backend_registry", backend_registry)

    assert info.is_supported(advice, check_system) == supported
    assert bool(info.filter_supported_backends(advice, check_system)) == supported
