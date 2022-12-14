# SPDX-FileCopyrightText: Copyright 2022, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Target module."""
from __future__ import annotations

from mlia.backend.registry import registry as backend_registry
from mlia.core.common import AdviceCategory
from mlia.target.config import TargetInfo
from mlia.utils.registry import Registry

# All supported targets are required to be registered here.
registry = Registry[TargetInfo]()


def supported_advice(target: str) -> list[AdviceCategory]:
    """Get a list of supported advice for the given target."""
    advice: set[AdviceCategory] = set()
    for supported_backend in registry.items[target].supported_backends:
        advice.update(backend_registry.items[supported_backend].supported_advice)
    return list(advice)


def supported_backends(target: str) -> list[str]:
    """Get a list of backends supported by the given target."""
    return registry.items[target].filter_supported_backends(check_system=False)


def supported_targets(advice: AdviceCategory) -> list[str]:
    """Get a list of all targets supporting the given advice category."""
    return [
        name
        for name, info in registry.items.items()
        if info.is_supported(advice, check_system=False)
    ]
