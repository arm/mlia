# Copyright 2021, Arm Ltd.
"""Ethos-U configuration."""
import logging
from typing import Any
from typing import Dict

from mlia.devices.config import IPConfiguration
from mlia.tools.vela_wrapper import resolve_compiler_config
from mlia.tools.vela_wrapper import VelaCompilerOptions
from mlia.utils.filesystem import get_profile
from mlia.utils.filesystem import get_vela_config


logger = logging.getLogger(__name__)


class EthosUConfiguration(IPConfiguration):
    """Ethos-U configuration."""

    def __init__(self, target: str, **kwargs: Any) -> None:
        """Init Ethos-U target configuration."""
        target_data = get_profile(target)
        _check_target_data_complete(target_data)

        ip_class = target_data["device"]
        super().__init__(ip_class)

        mac = target_data["mac"]
        _check_device_options_valid(ip_class, mac)

        self.mac = mac
        self.compiler_options = VelaCompilerOptions(
            system_config=target_data["system_config"],
            memory_mode=target_data["memory_mode"],
            config_files=str(get_vela_config()),
            accelerator_config=f"{ip_class}-{mac}",  # type: ignore
            **kwargs,
        )

    @property
    def resolved_compiler_config(self) -> Dict[str, Any]:
        """Resolve compiler configuration."""
        return resolve_compiler_config(self.compiler_options)

    def __str__(self) -> str:
        """Return string representation."""
        return (
            f"Ethos-U ip_class={self.ip_class} "
            f"mac={self.mac} "
            f"compiler_options={self.compiler_options}"
        )

    def __repr__(self) -> str:
        """Return string representation."""
        return f"<Ethos-U configuration ip_class={self.ip_class}>"


def get_target(**kwargs: Any) -> EthosUConfiguration:
    """Get target instance based on provided params."""
    target = kwargs.pop("target", None)

    if not target:
        raise Exception("No target given")

    return EthosUConfiguration(target, **kwargs)


def _check_target_data_complete(target_data: Dict[str, Any]) -> None:
    """Check if profile contains all needed data."""
    mandatory_keys = {"device", "mac", "system_config", "memory_mode"}
    missing_keys = sorted(mandatory_keys - target_data.keys())

    if missing_keys:
        raise Exception(f"Mandatory fields missing from target profile: {missing_keys}")


def _check_device_options_valid(device: str, mac: int) -> None:
    """Check if mac is valid for selected device."""
    target_mac_ranges = {
        "ethos-u55": [32, 64, 128, 256],
        "ethos-u65": [256, 512],
    }

    if device not in target_mac_ranges:
        raise Exception(f"Unsupported device: {device}")

    target_mac_range = target_mac_ranges[device]
    if mac not in target_mac_range:
        raise Exception(
            f"Mac value for selected device should be in {target_mac_range}"
        )
