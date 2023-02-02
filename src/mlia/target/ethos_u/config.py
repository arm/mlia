# SPDX-FileCopyrightText: Copyright 2022-2023, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Ethos-U configuration."""
from __future__ import annotations

import logging
from typing import Any

from mlia.backend.corstone import is_corstone_backend
from mlia.backend.manager import get_available_backends
from mlia.backend.vela.compiler import resolve_compiler_config
from mlia.backend.vela.compiler import VelaCompilerOptions
from mlia.target.config import TargetProfile
from mlia.utils.filesystem import get_vela_config

logger = logging.getLogger(__name__)


class EthosUConfiguration(TargetProfile):
    """Ethos-U configuration."""

    def __init__(self, **kwargs: Any) -> None:
        """Init Ethos-U target configuration."""
        target = kwargs["target"]
        super().__init__(target)

        mac = kwargs["mac"]

        self.mac = mac
        self.compiler_options = VelaCompilerOptions(
            system_config=kwargs["system_config"],
            memory_mode=kwargs["memory_mode"],
            config_files=str(get_vela_config()),
            accelerator_config=f"{self.target}-{mac}",  # type: ignore
        )

    def verify(self) -> None:
        """Check the parameters."""
        super().verify()

        target_mac_ranges = {
            "ethos-u55": [32, 64, 128, 256],
            "ethos-u65": [256, 512],
        }

        if self.target not in target_mac_ranges:
            raise ValueError(f"Unsupported target: {self.target}")

        target_mac_range = target_mac_ranges[self.target]
        if self.mac not in target_mac_range:
            raise ValueError(
                f"Mac value for selected device should be in {target_mac_range}."
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


def get_default_ethos_u_backends(
    supported_backends_priority_order: list[str],
) -> list[str]:
    """Return default backends for Ethos-U targets."""
    available_backends = get_available_backends()

    default_backends = []
    corstone_added = False
    for backend in supported_backends_priority_order:
        if backend not in available_backends:
            continue
        if is_corstone_backend(backend):
            if corstone_added:
                continue  # only add one Corstone backend
            corstone_added = True
        default_backends.append(backend)
    return default_backends
