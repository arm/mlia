# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Focused tests for backend option discovery."""

from __future__ import annotations

from pathlib import Path

import pytest

from mlia.api import discover_backend_option_specs
from mlia.backend.config import BackendConfiguration, BackendType
from mlia.backend.registry import registry as backend_registry
from mlia.core.common import AdviceCategory


def test_discover_backend_option_specs_empty_registry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No registered backends means no backend option metadata."""
    monkeypatch.setattr(backend_registry, "items", {})

    assert discover_backend_option_specs() == []


def test_discover_backend_option_specs_skips_backends_without_cli_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Backends without declared CLI options should be ignored."""
    monkeypatch.setattr(
        backend_registry,
        "items",
        {
            "vela": BackendConfiguration(
                supported_advice=[AdviceCategory.COMPATIBILITY],
                supported_systems=None,
                backend_type=BackendType.BUILTIN,
                installation=None,
            )
        },
    )

    assert discover_backend_option_specs() == []


def test_discover_backend_option_specs_extracts_cli_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Discovery should expose BackendConfiguration.cli_options as API metadata."""
    monkeypatch.setattr(
        backend_registry,
        "items",
        {
            "bingo-bongo-backend": BackendConfiguration(
                supported_advice=[AdviceCategory.PERFORMANCE],
                supported_systems=None,
                backend_type=BackendType.CUSTOM,
                installation=None,
                cli_options={
                    "system_config": "--system-config",
                    "compiler_config": "--compiler-config",
                },
            )
        },
    )

    assert discover_backend_option_specs() == [
        {
            "module": "bingo_bongo_backend",
            "backend": "bingo-bongo-backend",
            "config_key": "system_config",
            "cli_option": "--system-config",
            "full_cli_option": "--bingo-bongo-backend.system-config",
            "dest": "bingo_bongo_backend_system_config",
            "type": Path,
            "help": "Overrides the --system-config backend option.",
        },
        {
            "module": "bingo_bongo_backend",
            "backend": "bingo-bongo-backend",
            "config_key": "compiler_config",
            "cli_option": "--compiler-config",
            "full_cli_option": "--bingo-bongo-backend.compiler-config",
            "dest": "bingo_bongo_backend_compiler_config",
            "type": Path,
            "help": "Overrides the --compiler-config backend option.",
        },
    ]
