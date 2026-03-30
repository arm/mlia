# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Helpers for dynamic constants modules."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from functools import lru_cache

from mlia.backend.registry import registry as backend_registry
from mlia.plugins.plugins import load_backend_plugins, load_target_plugins
from mlia.target.registry import profiles_by_target
from mlia.target.registry import registry as target_registry

_IDENTIFIER_RE = re.compile(r"[^0-9A-Za-z]+")


@dataclass(frozen=True)
class ConstantEntry:
    """A constant exported by a dynamic module."""

    symbol: str
    value: str


def normalize_constant_name(value: str) -> str:
    """Convert a canonical MLIA identifier into a Python constant name."""
    normalized = _IDENTIFIER_RE.sub("_", value).strip("_").upper()
    if not normalized:
        raise ValueError(f"Cannot derive a constant name from {value!r}.")
    if normalized[0].isdigit():
        normalized = f"_{normalized}"
    return normalized


@lru_cache
def backend_entries() -> tuple[ConstantEntry, ...]:
    """Return exported backend constants for the installed plugin set."""
    load_backend_plugins(backend_registry)
    return _build_entries(sorted(backend_registry.items))


@lru_cache
def target_entries() -> tuple[ConstantEntry, ...]:
    """Return exported target constants for the installed plugin set."""
    load_backend_plugins(backend_registry)
    load_target_plugins(target_registry)
    return _build_entries(sorted(target_registry.items))


@lru_cache
def target_profile_entries() -> tuple[ConstantEntry, ...]:
    """Return exported target profile constants for the installed plugin set."""
    load_backend_plugins(backend_registry)
    load_target_plugins(target_registry)
    profile_names = sorted(
        profile_name
        for profile_names in profiles_by_target().values()
        for profile_name in profile_names
    )
    return _build_entries(profile_names)


@lru_cache
def exported_symbols(kind: str) -> tuple[str, ...]:
    """Return exported symbol names for the requested constants kind."""
    entries = _entries_for_kind(kind)
    return tuple(entry.symbol for entry in entries)


@lru_cache
def exported_value(kind: str, symbol: str) -> str:
    """Return the canonical value for the requested exported symbol."""
    for entry in _entries_for_kind(kind):
        if entry.symbol == symbol:
            return entry.value
    raise AttributeError(f"module has no attribute {symbol!r}")


@lru_cache
def exported_mapping(kind: str) -> dict[str, str]:
    """Return the exported symbol-to-value mapping for the given kind."""
    return {entry.symbol: entry.value for entry in _entries_for_kind(kind)}


def _entries_for_kind(kind: str) -> tuple[ConstantEntry, ...]:
    if kind == "targets":
        return target_entries()
    if kind == "backends":
        return backend_entries()
    if kind == "target_profiles":
        return target_profile_entries()
    raise ValueError(f"Unknown constants kind: {kind}")


def _build_entries(values: Iterable[str]) -> tuple[ConstantEntry, ...]:
    symbol_to_value: dict[str, str] = {}
    for value in values:
        symbol = normalize_constant_name(value)
        existing = symbol_to_value.get(symbol)
        if existing is not None and existing != value:
            raise ValueError(
                f"Constant symbol collision for {symbol!r}: {existing!r} vs {value!r}."
            )
        symbol_to_value[symbol] = value
    return tuple(
        ConstantEntry(symbol=symbol, value=value)
        for symbol, value in sorted(symbol_to_value.items())
    )
