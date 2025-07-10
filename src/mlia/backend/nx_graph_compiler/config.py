# SPDX-FileCopyrightText: Copyright 2023, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: LicenseRef-LICENSE
"""NGP Graph Compiler backend configuration."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class NGPGraphCompilerConfig:
    """Configuration for the NGP Graph Compiler."""

    DEFAULT = Path("default")

    system_config: str | Path
    compiler_config: str | Path

    def set_config_dir(self, config_dir: Path) -> None:
        """Prepend config file paths (if relative) with the given config dir."""

        def make_absolute(config: str | Path) -> Path:
            if config == "default":
                return NGPGraphCompilerConfig.DEFAULT

            config_path = Path(config)
            if config_path.is_absolute():
                return config_path
            return config_dir / config_path

        self.system_config = make_absolute(self.system_config)
        self.compiler_config = make_absolute(self.compiler_config)


CONFIG_TO_CLI_OPTION = {
    "system_config": "--system_config",
    "compiler_config": "--compiler_config",
}
