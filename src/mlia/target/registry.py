# SPDX-FileCopyrightText: Copyright 2022-2023, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Target module."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import cast

from mlia.backend.config import BackendType
from mlia.backend.manager import get_installation_manager
from mlia.backend.registry import registry as backend_registry
from mlia.core.common import AdviceCategory
from mlia.core.reporting import Column
from mlia.core.reporting import Table
from mlia.target.config import BUILTIN_SUPPORTED_PROFILE_NAMES
from mlia.target.config import get_builtin_profile_path
from mlia.target.config import is_builtin_profile
from mlia.target.config import load_profile
from mlia.target.config import TargetInfo
from mlia.target.config import TargetProfile
from mlia.utils.registry import Registry


class TargetRegistry(Registry[TargetInfo]):
    """Registry for targets."""

    def register(
        self, name: str, item: TargetInfo, pretty_name: str | None = None
    ) -> bool:
        """Register an item: returns `False` if already registered."""
        assert all(
            backend in backend_registry.items for backend in item.supported_backends
        )
        return super().register(name, item, pretty_name)


# All supported targets are required to be registered here.
registry = TargetRegistry()


def builtin_profile_names() -> list[str]:
    """Return a list of built-in profile names (not file paths)."""
    return BUILTIN_SUPPORTED_PROFILE_NAMES


@lru_cache
def profile(target_profile: str | Path) -> TargetProfile:
    """Get the target profile data (built-in or custom file)."""
    if not target_profile:
        raise ValueError("No valid target profile was provided.")
    if is_builtin_profile(target_profile):
        profile_file = get_builtin_profile_path(cast(str, target_profile))
        profile_ = create_target_profile(profile_file)
    else:
        profile_file = Path(target_profile)
        if profile_file.is_file():
            profile_ = create_target_profile(profile_file)
        else:
            raise ValueError(
                f"Profile '{target_profile}' is neither a valid built-in "
                "target profile name or a valid file path."
            )

    return profile_


def get_target(target_profile: str | Path) -> str:
    """Return target for the provided target_profile."""
    return profile(target_profile).target


@lru_cache
def create_target_profile(path: Path) -> TargetProfile:
    """Create a new instance of a TargetProfile from the file."""
    profile_data = load_profile(path)
    target = profile_data["target"]
    target_info = registry.items[target]
    return target_info.target_profile_cls.load_json_data(profile_data)


def supported_advice(target: str) -> list[AdviceCategory]:
    """Get a list of supported advice for the given target."""
    advice: set[AdviceCategory] = set()
    for supported_backend in registry.items[target].supported_backends:
        advice.update(backend_registry.items[supported_backend].supported_advice)
    return list(advice)


def supported_backends(target: str) -> list[str]:
    """Get a list of backends supported by the given target."""
    return registry.items[target].filter_supported_backends(check_system=False)


def default_backends(target: str) -> list[str]:
    """Get a list of default backends for the given target."""
    return registry.items[target].default_backends


def get_backend_to_supported_targets() -> dict[str, list]:
    """Get a dict that maps a list of supported targets given backend."""
    targets = dict(registry.items)
    supported_backends_dict: dict[str, list] = {}
    for target, info in targets.items():
        target_backends = info.supported_backends
        for backend in target_backends:
            supported_backends_dict.setdefault(backend, []).append(target)
    return supported_backends_dict


def is_supported(backend: str, target: str | None = None) -> bool:
    """Check if the backend (and optionally target) is supported."""
    backends = get_backend_to_supported_targets()
    if target is None:
        if backend in backends:
            return True
        return False
    try:
        return target in backends[backend]
    except KeyError:
        return False


def supported_targets(advice: AdviceCategory) -> list[str]:
    """Get a list of all targets supporting the given advice category."""
    return [
        name
        for name, info in registry.items.items()
        if info.is_supported(advice, check_system=False)
    ]


def all_supported_backends() -> set[str]:
    """Return set of all supported backends by all targets."""
    return {
        backend
        for item in registry.items.values()
        for backend in item.supported_backends
    }


def table() -> Table:
    """Get a table representation of registered targets with backends."""

    def get_status(backend: str) -> str:
        if backend_registry.items[backend].type == BackendType.BUILTIN:
            return BackendType.BUILTIN.name
        mgr = get_installation_manager()
        return "INSTALLED" if mgr.backend_installed(backend) else "NOT INSTALLED"

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
            f"{registry.pretty_name(name)}\n<{name}>",
            "\n".join(
                f"{backend_registry.pretty_name(backend)}\n<{backend}>"
                for backend in info.supported_backends
            ),
            "\n\n".join(get_status(backend) for backend in info.supported_backends),
            "/".join(get_advice(name)),
        )
        for name, info in registry.items.items()
    ]

    return Table(
        columns=[
            Column("Target"),
            Column("Backend(s)"),
            Column("Status"),
            Column("Advice: comp/perf/opt"),
        ],
        rows=rows,
        name="Supported Targets/Backends",
        notes="Comp/Perf/Opt: Advice categories compatibility/performance/optimization",
    )
