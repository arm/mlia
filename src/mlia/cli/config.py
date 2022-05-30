# SPDX-FileCopyrightText: Copyright 2022, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Environment configuration functions."""
import logging
from functools import lru_cache
from typing import List

import mlia.tools.aiet_wrapper as aiet
from mlia.tools.metadata.common import DefaultInstallationManager
from mlia.tools.metadata.common import InstallationManager
from mlia.tools.metadata.corstone import get_corstone_installations

logger = logging.getLogger(__name__)


def get_installation_manager(noninteractive: bool = False) -> InstallationManager:
    """Return installation manager."""
    backends = get_corstone_installations()

    return DefaultInstallationManager(backends, noninteractive=noninteractive)


@lru_cache
def get_available_backends() -> List[str]:
    """Return list of the available backends."""
    available_backends = ["Vela"]

    # Add backends using AIET
    manager = get_installation_manager()
    available_backends.extend(
        (
            backend
            for backend in aiet.supported_backends()
            if manager.backend_installed(backend)
        )
    )

    return available_backends


# List of mutually exclusive Corstone backends ordered by priority
_CORSTONE_EXCLUSIVE_PRIORITY = ("Corstone-310", "Corstone-300")


def get_default_backends() -> List[str]:
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

    return backends


def is_corstone_backend(backend: str) -> bool:
    """Check if the given backend is a Corstone backend."""
    return backend in _CORSTONE_EXCLUSIVE_PRIORITY
