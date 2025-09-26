# SPDX-FileCopyrightText: Copyright 2022-2023, 2025, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the backend config module."""
from mlia.backend.config import BackendConfiguration
from mlia.backend.config import BackendType
from mlia.backend.config import System
from mlia.core.common import AdviceCategory

UNSUPPORTED_SYSTEM = next(sys for sys in System if not sys.is_compatible())


def test_system() -> None:
    """Test the class 'System'."""
    assert System.CURRENT.is_compatible()
    assert not UNSUPPORTED_SYSTEM.is_compatible()
    assert UNSUPPORTED_SYSTEM != System.CURRENT
    assert System.LINUX_AMD64 != System.LINUX_AARCH64
    assert System.CURRENT != {"different": "type"}  # Type mismatch


def test_backend_config() -> None:
    """Test the class 'BackendConfiguration'."""
    cfg = BackendConfiguration(
        [AdviceCategory.COMPATIBILITY],
        [System.CURRENT],
        BackendType.CUSTOM,
        None,
    )
    assert cfg.supported_advice == [AdviceCategory.COMPATIBILITY]
    assert cfg.supported_systems == [System.CURRENT]
    assert cfg.type == BackendType.CUSTOM
    assert str(cfg)
    assert cfg.is_supported()
    assert cfg.is_supported(advice=AdviceCategory.COMPATIBILITY)
    assert not cfg.is_supported(advice=AdviceCategory.PERFORMANCE)
    assert cfg.is_supported(check_system=True)
    assert cfg.is_supported(check_system=False)
    cfg.supported_systems = None
    assert cfg.is_supported(check_system=True)
    assert cfg.is_supported(check_system=False)
    cfg.supported_systems = [UNSUPPORTED_SYSTEM]
    assert not cfg.is_supported(check_system=True)
    assert cfg.is_supported(check_system=False)
    assert not cfg.is_supported(advice=AdviceCategory.COMPATIBILITY, check_system=True)
    assert cfg.is_supported(advice=AdviceCategory.COMPATIBILITY, check_system=False)
    assert not cfg.is_supported(advice=AdviceCategory.PERFORMANCE, check_system=False)
