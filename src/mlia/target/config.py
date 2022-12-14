# SPDX-FileCopyrightText: Copyright 2022, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""IP configuration module."""
from __future__ import annotations

from dataclasses import dataclass

from mlia.backend.registry import registry as backend_registry
from mlia.core.common import AdviceCategory


class IPConfiguration:  # pylint: disable=too-few-public-methods
    """Base class for IP configuration."""

    def __init__(self, target: str) -> None:
        """Init IP configuration instance."""
        self.target = target


@dataclass
class TargetInfo:
    """Collect information about supported targets."""

    supported_backends: list[str]

    def __str__(self) -> str:
        """List supported backends."""
        return ", ".join(sorted(self.supported_backends))

    def is_supported(
        self, advice: AdviceCategory | None = None, check_system: bool = False
    ) -> bool:
        """Check if any of the supported backends support this kind of advice."""
        return any(
            backend_registry.items[name].is_supported(advice, check_system)
            for name in self.supported_backends
        )

    def filter_supported_backends(
        self, advice: AdviceCategory | None = None, check_system: bool = False
    ) -> list[str]:
        """Get the list of supported backends filtered by the given arguments."""
        return [
            name
            for name in self.supported_backends
            if backend_registry.items[name].is_supported(advice, check_system)
        ]
