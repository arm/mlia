# SPDX-FileCopyrightText: Copyright 2022-2023, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Environment configuration functions."""
from __future__ import annotations

import logging
from functools import lru_cache
from typing import List
from typing import Optional
from typing import TypedDict

from mlia.backend.corstone.install import get_corstone_installations
from mlia.backend.install import supported_backends
from mlia.backend.manager import DefaultInstallationManager
from mlia.backend.manager import InstallationManager
from mlia.backend.tosa_checker.install import get_tosa_backend_installation

logger = logging.getLogger(__name__)

DEFAULT_PRUNING_TARGET = 0.5
DEFAULT_CLUSTERING_TARGET = 32


def get_installation_manager(noninteractive: bool = False) -> InstallationManager:
    """Return installation manager."""
    backends = get_corstone_installations()
    backends.append(get_tosa_backend_installation())

    return DefaultInstallationManager(backends, noninteractive=noninteractive)


@lru_cache
def get_available_backends() -> list[str]:
    """Return list of the available backends."""
    available_backends = ["Vela", "tosa-checker", "armnn-tflitedelegate"]

    # Add backends using backend manager
    manager = get_installation_manager()
    available_backends.extend(
        backend
        for backend in supported_backends()
        if manager.backend_installed(backend)
    )

    return available_backends


# List of mutually exclusive Corstone backends ordered by priority
_CORSTONE_EXCLUSIVE_PRIORITY = ("Corstone-310", "Corstone-300")
_NON_ETHOS_U_BACKENDS = ("tosa-checker", "armnn-tflitedelegate")


def get_ethos_u_default_backends() -> list[str]:
    """Get default backends for evaluation."""
    backends = get_available_backends()

    # Filter backends to only include one Corstone backend
    for corstone in _CORSTONE_EXCLUSIVE_PRIORITY:
        if corstone in backends:
            backends = [
                backend
                for backend in backends
                if backend == corstone or backend not in _CORSTONE_EXCLUSIVE_PRIORITY
            ]
            break

    # Filter out non ethos-u backends
    backends = [x for x in backends if x not in _NON_ETHOS_U_BACKENDS]
    return backends


def is_corstone_backend(backend: str) -> bool:
    """Check if the given backend is a Corstone backend."""
    return backend in _CORSTONE_EXCLUSIVE_PRIORITY


BackendCompatibility = TypedDict(
    "BackendCompatibility",
    {
        "partial-match": bool,
        "backends": List[str],
        "default-return": Optional[List[str]],
        "use-custom-return": bool,
        "custom-return": Optional[List[str]],
    },
)


def get_default_backends() -> dict[str, list[str]]:
    """Return default backends for all targets."""
    ethos_u_defaults = get_ethos_u_default_backends()
    return {
        "ethos-u55": ethos_u_defaults,
        "ethos-u65": ethos_u_defaults,
        "tosa": ["tosa-checker"],
        "cortex-a": ["armnn-tflitedelegate"],
    }
