# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Target profile identifiers exposed by installed plugins."""

from __future__ import annotations

from mlia._constants import exported_symbols, exported_value

_KIND = "target_profiles"
__all__ = list(exported_symbols(_KIND))


def __getattr__(name: str) -> str:
    """Return the canonical identifier for a dynamically exported constant."""
    return exported_value(_KIND, name)


def __dir__() -> list[str]:
    """Expose dynamic constant symbols through standard introspection."""
    return sorted(set(globals()) | set(__all__))
