# Copyright 2021, Arm Ltd.
"""IP configuration."""
# pylint: disable=too-few-public-methods,too-many-instance-attributes
# pylint: disable=too-many-arguments
import logging
from typing import Any
from typing import Literal

from mlia.tools.vela_wrapper import VelaCompilerOptions
from mlia.utils.filesystem import get_vela_config


logger = logging.getLogger(__name__)


class IPConfiguration:
    """Base class for IP configuration."""


class EthosUConfiguration(IPConfiguration):
    """EthosU configuration."""

    def __init__(
        self,
        ip_class: Literal["ethos-u55", "ethos-u65"],
        mac: int,
        compiler_options: VelaCompilerOptions,
    ):
        """Init EthosU configuration."""
        self.ip_class = ip_class
        self.mac = mac
        self.compiler_options = compiler_options

    def __str__(self) -> str:
        """Return string representation."""
        return (
            f"EthosU ip_class={self.ip_class} "
            f"mac={self.mac} "
            f"compiler_options= {self.compiler_options}"
        )


class EthosU55(EthosUConfiguration):
    """EthosU55 configuration."""

    def __init__(self, mac: Literal[32, 64, 128, 256] = 256, **kwargs: Any) -> None:
        """Init EthosU55 configuration."""
        if not mac or mac not in (32, 64, 128, 256):
            raise Exception("Wrong or empty MAC value")

        super().__init__(
            ip_class="ethos-u55",
            mac=mac,
            compiler_options=VelaCompilerOptions(
                config_files=[str(get_vela_config())],
                accelerator_config=f"ethos-u55-{mac}",  # type: ignore
                **kwargs,
            ),
        )


class EthosU65(EthosUConfiguration):
    """EthosU65 configuration."""

    def __init__(self, mac: Literal[256, 512] = 256, **kwargs: Any) -> None:
        """Init EthosU65 configuration."""
        if not mac or mac not in (256, 512):
            raise Exception("Wrong or empty MAC value")

        super().__init__(
            ip_class="ethos-u65",
            mac=mac,
            compiler_options=VelaCompilerOptions(
                config_files=[str(get_vela_config())],
                accelerator_config=f"ethos-u65-{mac}",  # type: ignore
                **kwargs,
            ),
        )


def get_device(**kwargs: Any) -> IPConfiguration:
    """Get device instance based on provided params."""
    device = kwargs.pop("device", None)

    if not device:
        raise Exception("Device is not provided")

    if device.lower() == "ethos-u55":
        return EthosU55(**kwargs)

    if device.lower() == "ethos-u65":
        return EthosU65(**kwargs)

    raise Exception(f"Unsupported device: {device}")
