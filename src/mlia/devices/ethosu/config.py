# SPDX-FileCopyrightText: Copyright 2022, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Ethos-U configuration."""
from __future__ import annotations

import logging
from typing import Any

from mlia.devices.config import IPConfiguration
from mlia.tools.vela_wrapper import resolve_compiler_config
from mlia.tools.vela_wrapper import VelaCompilerOptions
from mlia.utils.filesystem import get_profile
from mlia.utils.filesystem import get_vela_config


logger = logging.getLogger(__name__)


class EthosUConfiguration(IPConfiguration):
    """Ethos-U configuration."""

    def __init__(self, target_profile: str) -> None:
        """Init Ethos-U target configuration."""
        target_data = get_profile(target_profile)
        _check_target_data_complete(target_data)

        target = target_data["target"]
        super().__init__(target)

        mac = target_data["mac"]
        _check_device_options_valid(target, mac)

        self.mac = mac
        self.compiler_options = VelaCompilerOptions(
            system_config=target_data["system_config"],
            memory_mode=target_data["memory_mode"],
            config_files=str(get_vela_config()),
            accelerator_config=f"{self.target}-{mac}",  # type: ignore
        )

    @property
    def resolved_compiler_config(self) -> dict[str, Any]:
        """Resolve compiler configuration."""
        return resolve_compiler_config(self.compiler_options)

    def __str__(self) -> str:
        """Return string representation."""
        return (
            f"Ethos-U target={self.target} "
            f"mac={self.mac} "
            f"compiler_options={self.compiler_options}"
        )

    def __repr__(self) -> str:
        """Return string representation."""
        return f"<Ethos-U configuration target={self.target}>"


def get_target(target_profile: str) -> EthosUConfiguration:
    """Get target instance based on provided params."""
    if not target_profile:
        raise Exception("No target profile given")

    return EthosUConfiguration(target_profile)


def _check_target_data_complete(target_data: dict[str, Any]) -> None:
    """Check if profile contains all needed data."""
    mandatory_keys = {"target", "mac", "system_config", "memory_mode"}
    missing_keys = sorted(mandatory_keys - target_data.keys())

    if missing_keys:
        raise Exception(f"Mandatory fields missing from target profile: {missing_keys}")


def _check_device_options_valid(target: str, mac: int) -> None:
    """Check if mac is valid for selected device."""
    target_mac_ranges = {
        "ethos-u55": [32, 64, 128, 256],
        "ethos-u65": [256, 512],
    }

    if target not in target_mac_ranges:
        raise Exception(f"Unsupported target: {target}")

    target_mac_range = target_mac_ranges[target]
    if mac not in target_mac_range:
        raise Exception(
            f"Mac value for selected device should be in {target_mac_range}"
        )
