# SPDX-FileCopyrightText: Copyright 2022-2023,2026 Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Backend config module."""

from __future__ import annotations

import platform
from enum import Enum, auto
from typing import cast

from mlia.backend.install import Installation
from mlia.core.common import AdviceCategory


class System(Enum):
    """Enum of system configurations (e.g. OS and architecture)."""

    LINUX_AMD64 = ("Linux", "x86_64")
    LINUX_AARCH64 = ("Linux", "aarch64")
    WINDOWS_AMD64 = ("Windows", "AMD64")
    WINDOWS_AARCH64 = ("Windows", "ARM64")
    DARWIN_AARCH64 = ("Darwin", "arm64")
    CURRENT = (platform.system(), platform.machine())

    def __init__(
        self, system: str = platform.system(), machine: str = platform.machine()
    ) -> None:
        """Set the system parameters (defaults to the current system)."""
        self.system = system.lower()
        self.machine = machine.lower()

    def __eq__(self, other: object) -> bool:
        """
        Compare two System instances for equality.

        Raises a TypeError if the input is not a System.
        """
        if isinstance(other, self.__class__):
            return self.system == other.system and self.machine == other.machine
        return False

    def is_compatible(self) -> bool:
        """Check if this system is compatible with the current system."""
        return self == self.CURRENT


class BackendType(Enum):
    """Define the type of the backend (builtin, wheel file etc)."""

    BUILTIN = auto()
    WHEEL = auto()
    CUSTOM = auto()


class BackendConfiguration:
    """Base class for backend configurations."""

    def __init__(
        self,
        supported_advice: list[AdviceCategory],
        supported_systems: list[System] | None,
        backend_type: BackendType,
        installation: Installation | None,
        selectable: bool = True,
        cli_options: dict[str, str] | None = None,
        is_deprecated: bool = False,
        deprecated_message: str | None = None,
    ) -> None:
        """Set up basic information about the backend."""
        self.supported_advice = supported_advice
        self.supported_systems = supported_systems
        self.type = backend_type
        self.installation = installation
        self.selectable = selectable
        self.cli_options = cli_options or {}
        self.is_deprecated = is_deprecated
        self.deprecated_message = deprecated_message

    def __str__(self) -> str:
        """List supported advice."""
        return ", ".join(cast(str, adv.name).lower() for adv in self.supported_advice)

    def __eq__(self, other: object) -> bool:
        """Check equality with another BackendConfiguration."""
        if isinstance(other, BackendConfiguration):
            return (
                self.supported_advice == other.supported_advice
                and self.supported_systems == other.supported_systems
                and self.type == other.type
                and self.installation == other.installation
                and self.cli_options == other.cli_options
                and self.is_deprecated == other.is_deprecated
                and self.deprecated_message == other.deprecated_message
            )
        raise NotImplementedError

    def is_supported(
        self, advice: AdviceCategory | None = None, check_system: bool = False
    ) -> bool:
        """Check backend supports the current system and advice."""
        is_system_supported = (
            not self.supported_systems
            or not check_system
            or any(sys.is_compatible() for sys in self.supported_systems)
        )
        is_advice_supported = advice is None or advice in self.supported_advice
        return is_system_supported and is_advice_supported
