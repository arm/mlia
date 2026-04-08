# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Import-time regression tests for pytest-native MLIA e2e helpers."""

from __future__ import annotations

import importlib
from typing import Any

import pytest

import mlia.cli.main
import mlia.testing.e2e


def test_importing_e2e_helpers_does_not_load_cli_plugins(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reloading the module should not initialize CLI plugins eagerly."""

    def fail_load_cli_plugins(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("CLI plugins should not load during module import")

    monkeypatch.setattr(mlia.cli.main, "load_cli_plugins", fail_load_cli_plugins)

    importlib.reload(mlia.testing.e2e)
