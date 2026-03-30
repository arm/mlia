# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Focused tests for CLI backend option discovery."""

from __future__ import annotations

import types
from pathlib import Path

import pytest

from mlia.cli.options import discover_backend_option_specs


def test_discover_backend_option_specs_skips_non_packages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Non-package backend entries should be ignored."""
    monkeypatch.setattr(
        "mlia.cli.options.pkgutil.iter_modules",
        lambda _path: [("ignored", "not_a_pkg", False)],
    )

    assert discover_backend_option_specs() == []


def test_discover_backend_option_specs_skips_import_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Import failures while probing backend modules should be ignored."""
    monkeypatch.setattr(
        "mlia.cli.options.pkgutil.iter_modules",
        lambda _path: [("ignored", "vela", True)],
    )
    monkeypatch.setattr(
        "mlia.cli.options.importlib.import_module",
        lambda _name: (_ for _ in ()).throw(ImportError("boom")),
    )

    assert discover_backend_option_specs() == []


def test_discover_backend_option_specs_skips_modules_without_mapping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Imported backend modules without CLI mappings should be ignored."""
    monkeypatch.setattr(
        "mlia.cli.options.pkgutil.iter_modules",
        lambda _path: [("ignored", "vela", True)],
    )
    monkeypatch.setattr(
        "mlia.cli.options.importlib.import_module",
        lambda _name: types.SimpleNamespace(),
    )

    assert discover_backend_option_specs() == []


def test_discover_backend_option_specs_extracts_config_mapping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Discovery should expose CONFIG_TO_CLI_OPTION entries as API metadata."""
    module = types.SimpleNamespace(CONFIG_TO_CLI_OPTION={"config_file": "--config"})
    monkeypatch.setattr(
        "mlia.cli.options.pkgutil.iter_modules",
        lambda _path: [("ignored", "vela", True)],
    )

    def import_module(name: str) -> object:
        if name.endswith(".config"):
            return module
        raise ImportError("skip")

    monkeypatch.setattr("mlia.cli.options.importlib.import_module", import_module)

    assert discover_backend_option_specs() == [
        {
            "module": "vela",
            "backend": "vela",
            "config_key": "config_file",
            "cli_option": "--config",
            "full_cli_option": "--vela.config",
            "dest": "vela_config_file",
            "type": Path,
            "help": "Overrides the --config backend option.",
        }
    ]
