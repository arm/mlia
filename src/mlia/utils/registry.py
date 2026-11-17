# SPDX-FileCopyrightText: Copyright 2022-2023, 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Generic registry class."""

from __future__ import annotations

from collections.abc import Callable
from threading import RLock
from typing import Generic, TypeVar

T = TypeVar("T")

_LOAD_LOCK = RLock()


class Registry(Generic[T]):
    """Generic registry for name-config pairs."""

    def __init__(self, loader: Callable[[], None] | None = None) -> None:
        """Create an empty registry."""
        self._items: dict[str, T] = {}
        self._loader = loader
        self._loading = False
        self._loaded = False
        self.plugin_interface_versions: dict[str, str | None] = {}
        self.pretty_names: dict[str, str] = {}

    def ensure_loaded(self) -> bool:
        """Run the configured loader once; return whether it has completed.

        Return False inside the loader so its registration calls can proceed,
        and for registries without a configured loader.
        """
        if self._loader is None:
            return self._loaded
        with _LOAD_LOCK:
            if self._loader is None or self._loading:
                return self._loaded
            self._loading = True
            try:
                self._loader()
                self._loaded = True
                self._loader = None
                return True
            finally:
                self._loading = False

    @property
    def items(self) -> dict[str, T]:
        """Return registered items, loading this registry's providers if needed."""
        self.ensure_loaded()
        return self._items

    @items.setter
    def items(self, value: dict[str, T]) -> None:
        """Replace items explicitly, preserving the mutable registry interface."""
        self.ensure_loaded()
        self._items = value

    def __str__(self) -> str:
        """List all registered items."""
        return "\n".join(
            f"- {self.pretty_names[name] if name in self.pretty_names else name}: "
            f"{item}"
            for name, item in sorted(self.items.items(), key=lambda v: v[0])
        )

    def register(self, name: str, item: T, pretty_name: str | None = None) -> bool:
        """Register an item: returns `False` if already registered."""
        self.ensure_loaded()
        if name in self._items:
            return False  # already registered
        self._items[name] = item
        if pretty_name:
            self.pretty_names[name] = pretty_name
        return True

    def pretty_name(self, name: str) -> str:
        """Get the pretty name (if available) or return the name as is otherwise."""
        self.ensure_loaded()
        return self.pretty_names[name] if name in self.pretty_names else name

    def names(self) -> list[str]:
        """Sorted list of registered item names."""
        return sorted(list(self.items.keys()))

    def get(self, name: str) -> T | None:
        """Get a named item from the items dict."""
        return self.items.get(name)
