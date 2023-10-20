# SPDX-FileCopyrightText: Copyright 2023, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: LicenseRef-LICENSE
"""Tests for NGP Graph Compiler config."""
from __future__ import annotations

from pathlib import Path

from mlia.backend.ngp_graph_compiler.config import CONFIG_TO_CLI_OPTION
from mlia.backend.ngp_graph_compiler.config import NGPGraphCompilerConfig


def test_ngp_graph_compiler_config() -> None:
    """Test for class NGPGraphCompilerConfig."""
    sys_cfg, compiler_cfg = Path("system-config"), Path("compiler-config")
    cfg = NGPGraphCompilerConfig(sys_cfg, compiler_cfg)
    assert cfg.system_config == sys_cfg
    assert cfg.system_config == sys_cfg

    assert set(CONFIG_TO_CLI_OPTION) == set(vars(cfg))
