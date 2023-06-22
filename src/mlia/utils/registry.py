# SPDX-FileCopyrightText: Copyright 2022-2023, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Generic registry class."""
from __future__ import annotations

from typing import Generic
from typing import TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    """Generic registry for name-config pairs."""

    def __init__(self) -> None:
        """Create an empty registry."""
        self.items: dict[str, T] = {}
        self.pretty_names: dict[str, str] = {}

    def __str__(self) -> str:
        """List all registered items."""
        return "\n".join(
            f"- {self.pretty_names[name] if name in self.pretty_names else name}: "
            f"{item}"
            for name, item in sorted(self.items.items(), key=lambda v: v[0])
        )

    def register(self, name: str, item: T, pretty_name: str | None = None) -> bool:
        """Register an item: returns `False` if already registered."""
        if name in self.items:
            return False  # already registered
        self.items[name] = item
        if pretty_name:
            self.pretty_names[name] = pretty_name
        return True

    def pretty_name(self, name: str) -> str:
        """Get the pretty name (if available) or return the name as is otherwise."""
        return self.pretty_names[name] if name in self.pretty_names else name

    def names(self) -> list[str]:
        """Sorted list of registered item names."""
        return sorted(list(self.items.keys()))
