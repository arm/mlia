# SPDX-FileCopyrightText: Copyright 2023, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: LicenseRef-LICENSE
"""NGP Graph Compiler backend configuration."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class NGPGraphCompilerConfig:
    """Configuration for the NGP Graph Compiler."""

    system_config: Path
    compiler_config: Path


CONFIG_TO_CLI_OPTION = {
    "system_config": "--system_config",
    "compiler_config": "--compiler_config",
}
