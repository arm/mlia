# SPDX-FileCopyrightText: Copyright 2022-2023, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Target module."""
from __future__ import annotations

from typing import cast

from mlia.backend.config import BackendType
from mlia.backend.manager import DefaultInstallationManager
from mlia.backend.registry import registry as backend_registry
from mlia.cli.config import get_installation_manager
from mlia.core.common import AdviceCategory
from mlia.core.reporting import Column
from mlia.core.reporting import Table
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


def table() -> Table:
    """Get a table representation of registered targets with backends."""

    def get_status(backend: str) -> str:
        if backend_registry.items[backend].type == BackendType.BUILTIN:
            return BackendType.BUILTIN.name
        mgr = cast(DefaultInstallationManager, get_installation_manager())
        return "INSTALLED" if mgr.already_installed(backend) else "NOT INSTALLED"

    def get_advice(target: str) -> tuple[str, str, str]:
        supported = supported_advice(target)
        return tuple(  # type: ignore
            "YES" if advice in supported else "NO"
            for advice in (
                AdviceCategory.COMPATIBILITY,
                AdviceCategory.PERFORMANCE,
                AdviceCategory.OPTIMIZATION,
            )
        )

    rows = [
        (
            name,
            Table(
                columns=[Column("Backend"), Column("Status")],
                rows=[
                    (backend, get_status(backend))
                    for backend in info.supported_backends
                ],
                name="Backends",
            ),
            "/".join(get_advice(name)),
        )
        for name, info in registry.items.items()
    ]

    return Table(
        columns=[
            Column("Target"),
            Column("Backend(s)"),
            Column("Advice: comp/perf/opt"),
        ],
        rows=rows,
        name="Supported Targets/Backends",
        notes="Comp/Perf/Opt: Advice categories compatibility/performance/optimization",
    )
