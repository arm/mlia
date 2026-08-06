# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

"""Settings module.

This module contains data structures the house the settings for the running instance of
MLIA.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from rich.console import Console


@dataclass(frozen=True)
class ApplicationSettings:
    """Settings for the running MLIA application."""

    console: Console = field(default_factory=Console)
    color: bool = True
    backend_options: dict[str, Any] = field(default_factory=dict)
    core_settings: dict[str, Any] = field(default_factory=dict)
    plugin_settings: dict[str, dict[str, Any]] = field(default_factory=dict)
