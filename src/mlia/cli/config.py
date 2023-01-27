# SPDX-FileCopyrightText: Copyright 2022-2023, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Environment configuration functions."""
from __future__ import annotations

import logging

from mlia.backend.manager import get_installation_manager
from mlia.target.registry import all_supported_backends

logger = logging.getLogger(__name__)

DEFAULT_PRUNING_TARGET = 0.5
DEFAULT_CLUSTERING_TARGET = 32


def get_available_backends() -> list[str]:
    """Return list of the available backends."""
    available_backends = ["Vela", "ArmNNTFLiteDelegate"]

    # Add backends using backend manager
    manager = get_installation_manager()
    available_backends.extend(
        backend
        for backend in all_supported_backends()
        if manager.backend_installed(backend)
    )

    return available_backends


# List of mutually exclusive Corstone backends ordered by priority
_CORSTONE_EXCLUSIVE_PRIORITY = ("Corstone-310", "Corstone-300")
_NON_ETHOS_U_BACKENDS = ("tosa-checker", "ArmNNTFLiteDelegate")


def get_ethos_u_default_backends(backends: list[str]) -> list[str]:
    """Get Ethos-U default backends for evaluation."""
    return [x for x in backends if x not in _NON_ETHOS_U_BACKENDS]


def get_default_backends() -> list[str]:
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


def get_default_backends_dict() -> dict[str, list[str]]:
    """Return default backends for all targets."""
    default_backends = get_default_backends()
    ethos_u_defaults = get_ethos_u_default_backends(default_backends)

    return {
        "ethos-u55": ethos_u_defaults,
        "ethos-u65": ethos_u_defaults,
        "tosa": ["tosa-checker"],
        "cortex-a": ["ArmNNTFLiteDelegate"],
    }
