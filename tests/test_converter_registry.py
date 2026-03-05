# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for converter registry."""

from pathlib import Path

from mlia.plugins.converter_registry import ConverterRegistry


def test_converter_registry_register_get_list() -> None:
    registry = ConverterRegistry()

    def converter(input_path: Path, output_dir: Path) -> Path:  # pragma: no cover
        return output_dir / input_path.name

    registry.register("example", converter)

    assert registry.get("example") is converter
    assert registry.list() == ["example"]
