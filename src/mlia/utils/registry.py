# SPDX-FileCopyrightText: Copyright 2022, Arm Limited and/or its affiliates.
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

    def __str__(self) -> str:
        """List all registered items."""
        return "\n".join(
            f"- {name}: {item}"
            for name, item in sorted(self.items.items(), key=lambda v: v[0])
        )

    def register(self, name: str, item: T) -> bool:
        """Register an item: returns `False` if already registered."""
        if name in self.items:
            return False  # already registered
        self.items[name] = item
        return True
