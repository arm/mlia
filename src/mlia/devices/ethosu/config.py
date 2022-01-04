# Copyright 2021, Arm Ltd.
"""IP configuration."""
# pylint: disable=too-few-public-methods,too-many-instance-attributes
# pylint: disable=too-many-arguments
import logging
from typing import Any
from typing import Dict

from mlia.tools.vela_wrapper import VelaCompilerOptions
from mlia.utils.filesystem import get_profiles_data
from mlia.utils.filesystem import get_supported_profile_names
from mlia.utils.filesystem import get_vela_config


logger = logging.getLogger(__name__)


class IPConfiguration:
    """Base class for IP configuration."""


class EthosUConfiguration(IPConfiguration):
    """EthosU configuration."""

    def __init__(self, target: str, **kwargs: Any) -> None:
        """Init EthosU target configuration."""
        target_data = get_profiles_data()[target]

        _check_target_data_complete(target_data)

        device = target_data["device"]
        mac = target_data["mac"]

        _check_device_options_valid(device, mac)

        accelerator_config = f"{device}-{mac}"

        system_config = target_data["system_config"]
        memory_mode = target_data["memory_mode"]
        config_files = str(get_vela_config())

        self.ip_class = device
        self.mac = mac
        self.compiler_options = VelaCompilerOptions(
            system_config=system_config,
            memory_mode=memory_mode,
            config_files=config_files,
            accelerator_config=accelerator_config,  # type: ignore
            **kwargs,
        )

    def __str__(self) -> str:
        """Return string representation."""
        return (
            f"EthosU ip_class={self.ip_class} "
            f"mac={self.mac} "
            f"compiler_options= {self.compiler_options}"
        )


def get_target(**kwargs: Any) -> IPConfiguration:
    """Get target instance based on provided params."""
    target = kwargs.pop("target", None)

    if not target:
        raise Exception("No target given!")

    if target not in get_supported_profile_names():
        raise Exception(f"Unsupported target: {target}")

    return EthosUConfiguration(target, **kwargs)


def _check_target_data_complete(target_data: Dict[str, Any]) -> None:
    mandatory_keys = set(["device", "mac", "system_config", "memory_mode"])
    missing_keys = mandatory_keys - target_data.keys()
    if missing_keys:
        raise Exception("Mandatory fields missing from target profile: {missing_keys}")


def _check_device_options_valid(device: str, mac: int) -> None:
    if device == "ethos-u55":
        target_mac_range = [32, 64, 128, 256]
        if mac not in target_mac_range:
            raise Exception(
                f"Mac value for selected device should be in {target_mac_range}"
            )
        return
    if device == "ethos-u65":
        target_mac_range = [256, 512]
        if mac not in target_mac_range:
            raise Exception(
                f"Mac value for selected device should be in {target_mac_range}"
            )
        return
    raise Exception(f"Unsupported device: {device}")
