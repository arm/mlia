# SPDX-FileCopyrightText: Copyright 2022-2023, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for config module."""
from __future__ import annotations

from contextlib import ExitStack as does_not_raise
from typing import Any
from unittest.mock import MagicMock

import pytest

from mlia.backend.vela.compiler import VelaCompilerOptions
from mlia.target.ethos_u.config import EthosUConfiguration
from mlia.target.ethos_u.config import get_target
from mlia.utils.filesystem import get_vela_config


def test_compiler_options_default_init() -> None:
    """Test compiler options default init."""
    opts = VelaCompilerOptions()

    assert opts.config_files is None
    assert opts.system_config == "internal-default"
    assert opts.memory_mode == "internal-default"
    assert opts.accelerator_config is None
    assert opts.max_block_dependency == 3
    assert opts.arena_cache_size is None
    assert opts.tensor_allocator == "HillClimb"
    assert opts.cpu_tensor_alignment == 16
    assert opts.optimization_strategy == "Performance"
    assert opts.output_dir == "output"


def test_ethosu_target() -> None:
    """Test Ethos-U target configuration init."""
    default_config = EthosUConfiguration("ethos-u55-256")

    assert default_config.target == "ethos-u55"
    assert default_config.mac == 256
    assert default_config.compiler_options is not None


def test_get_target() -> None:
    """Test function get_target."""
    with pytest.raises(Exception, match="No target profile given"):
        get_target(None)  # type: ignore

    with pytest.raises(Exception, match=r"File not found:*"):
        get_target("unknown")

    u65_device = get_target("ethos-u65-512")

    assert isinstance(u65_device, EthosUConfiguration)
    assert u65_device.target == "ethos-u65"
    assert u65_device.mac == 512
    assert u65_device.compiler_options.accelerator_config == "ethos-u65-512"
    assert u65_device.compiler_options.memory_mode == "Dedicated_Sram"
    assert u65_device.compiler_options.config_files == str(get_vela_config())


@pytest.mark.parametrize(
    "profile_data, expected_error",
    [
        [
            {},
            pytest.raises(
                Exception,
                match="Mandatory fields missing from target profile: "
                r"\['mac', 'memory_mode', 'system_config', 'target'\]",
            ),
        ],
        [
            {"target": "ethos-u65", "mac": 512},
            pytest.raises(
                Exception,
                match="Mandatory fields missing from target profile: "
                r"\['memory_mode', 'system_config'\]",
            ),
        ],
        [
            {
                "target": "ethos-u65",
                "mac": 2,
                "system_config": "Ethos_U65_Embedded",
                "memory_mode": "Shared_Sram",
            },
            pytest.raises(
                Exception,
                match=r"Mac value for selected device should be in \[256, 512\]",
            ),
        ],
        [
            {
                "target": "ethos-u55",
                "mac": 1,
                "system_config": "Ethos_U55_High_End_Embedded",
                "memory_mode": "Shared_Sram",
            },
            pytest.raises(
                Exception,
                match="Mac value for selected device should be "
                r"in \[32, 64, 128, 256\]",
            ),
        ],
        [
            {
                "target": "ethos-u65",
                "mac": 512,
                "system_config": "Ethos_U65_Embedded",
                "memory_mode": "Shared_Sram",
            },
            does_not_raise(),
        ],
    ],
)
def test_ethosu_configuration(
    monkeypatch: pytest.MonkeyPatch, profile_data: dict[str, Any], expected_error: Any
) -> None:
    """Test creating Ethos-U configuration."""
    monkeypatch.setattr(
        "mlia.target.ethos_u.config.get_profile", MagicMock(return_value=profile_data)
    )

    with expected_error:
        EthosUConfiguration("target")
