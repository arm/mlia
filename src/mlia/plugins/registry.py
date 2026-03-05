# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Plugin registry utilities."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Iterable, cast

if sys.version_info < (3, 10):
    import importlib_metadata as metadata
else:
    from importlib import metadata

from packaging.requirements import Requirement
from packaging.version import Version


@dataclass(frozen=True)
class PluginInfo:
    """Metadata about an available plugin entry point."""

    name: str
    group: str
    value: str
    dist_name: str | None
    dist_version: str | None


def list_entry_points(group: str) -> list[PluginInfo]:
    """Return available entry points for a group."""
    entry_points = metadata.entry_points(group=group)
    plugins: list[PluginInfo] = []
    for entry_point in entry_points:
        dist = entry_point.dist
        plugins.append(
            PluginInfo(
                name=entry_point.name,
                group=group,
                value=entry_point.value,
                dist_name=dist.name if dist else None,
                dist_version=dist.version if dist else None,
            )
        )
    return plugins


def resolve_entry_point(group: str, name: str) -> metadata.EntryPoint | None:
    """Resolve a single entry point by name."""
    entry_points = metadata.entry_points(group=group)
    for entry_point in entry_points:
        if entry_point.name == name:
            return entry_point
    return None


def load_entry_point(group: str, name: str) -> object | None:
    """Load a single entry point by name."""
    entry_point = resolve_entry_point(group, name)
    if not entry_point:
        return None
    return cast(object, entry_point.load())


def _find_core_requirement(requirements: Iterable[str]) -> Requirement | None:
    for req_str in requirements:
        req = Requirement(req_str)
        if req.name == "mlia":
            return req
    return None


def check_core_compatibility(
    entry_point: metadata.EntryPoint,
) -> tuple[bool, str | None]:
    """Check whether a plugin entry point is compatible with the core package."""
    if not entry_point.dist:
        return True, None

    dist = entry_point.dist
    requires = dist.requires or []
    core_req = _find_core_requirement(requires)
    if core_req is None:
        return (
            False,
            (
                f"Plugin '{entry_point.name}' (version '{dist.version}') requires "
                "core '<missing>', but no core version range was declared."
            ),
        )

    try:
        core_version = Version(metadata.version("mlia"))
    except metadata.PackageNotFoundError:
        return (
            False,
            f"Core package 'mlia' is not installed for plugin '{entry_point.name}'.",
        )

    if not core_req.specifier.contains(core_version, prereleases=True):
        return (
            False,
            (
                f"Plugin '{entry_point.name}' (version '{dist.version}') requires "
                f"core '{core_req.specifier}', but core '{core_version}' is installed."
            ),
        )

    return True, None
