# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Converter plugin registry utilities."""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

logger = logging.getLogger("mlia")

ConverterFn = Callable[[Path, Path], Path]


class ConverterRegistry:
    """Registry for converter callables."""

    def __init__(self) -> None:
        """Initialize registry."""
        self._converters: dict[str, ConverterFn] = {}

    def register(self, name: str, converter: ConverterFn) -> None:
        """Register a converter by name."""
        if name in self._converters:
            logger.warning(
                "Converter '%s' already registered; keeping the existing entry.", name
            )
            return
        self._converters[name] = converter

    def get(self, name: str) -> ConverterFn | None:
        """Get a converter by name."""
        return self._converters.get(name)

    def list(self) -> list[str]:
        """Return registered converter names."""
        return sorted(self._converters.keys())
