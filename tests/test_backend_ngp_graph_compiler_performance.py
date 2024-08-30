# SPDX-FileCopyrightText: Copyright 2023-2024, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: LicenseRef-LICENSE
"""Tests for NGP Graph Compiler performance estimation."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from mlia.backend.ngp_graph_compiler.config import NGPGraphCompilerConfig
from mlia.backend.ngp_graph_compiler.performance import GRAPH_COMPILER_COMMAND_ARGS
from mlia.backend.ngp_graph_compiler.performance import NGPGraphCompilerOutputFiles
from mlia.backend.ngp_graph_compiler.performance import (
    NGPGraphCompilerPerformanceEstimator,
)
from mlia.target.hydra.config import HydraConfiguration


def test_ngp_graph_compiler_output_files(tmp_path: Path) -> None:
    """Test for class NGPGraphCompilerConfig."""
    output_files = NGPGraphCompilerOutputFiles.from_output_dir(tmp_path, "model_xyz")
    with pytest.raises(FileNotFoundError):
        output_files.check_exists()
    for file in vars(output_files).values():
        assert isinstance(file, Path)
        file.touch()
    output_files.check_exists()


def test_ngp_graph_compiler_performance_estimator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test class NGPGraphCompilerPerformanceEstimator."""
    hydra_cfg = HydraConfiguration.load_profile("hydra")
    pco_mock = MagicMock()
    mock_repo = MagicMock()
    mock_repo.get_backend_settings = MagicMock(return_value=(tmp_path / "backend", {}))
    monkeypatch.setattr(
        "mlia.backend.ngp_graph_compiler.performance.get_backend_repository",
        MagicMock(return_value=mock_repo),
    )
    monkeypatch.setattr(
        "mlia.backend.ngp_graph_compiler.performance.VulkanModelConverter",
        MagicMock(return_value=MagicMock(return_value=tmp_path / "vgf_file")),
    )
    monkeypatch.setattr(
        "mlia.backend.ngp_graph_compiler.performance.process_command_output",
        pco_mock,
    )
    monkeypatch.setattr(
        "mlia.backend.ngp_graph_compiler.performance.NGPPerformanceDatabaseParser",
        MagicMock(),
    )
    monkeypatch.setattr(
        "mlia.backend.ngp_graph_compiler.performance.NGPDebugDatabaseParser",
        MagicMock(),
    )
    monkeypatch.setattr(
        "mlia.backend.ngp_graph_compiler.performance."
        "NGPGraphCompilerOutputFiles.check_exists",
        MagicMock(return_value=True),
    )
    operator_types_mapping = {
        "Identity": "identity_op_type",
        "model/re_lu_7/Relu": "RELU",
    }

    estimator = NGPGraphCompilerPerformanceEstimator(
        tmp_path, hydra_cfg.backend_config, operator_types_mapping
    )
    metrics = estimator.estimate(tmp_path / "model.tflite")
    assert isinstance(metrics.backend_config, NGPGraphCompilerConfig)
    assert all(isinstance(file, Path) for file in vars(metrics.output_files).values())

    assert pco_mock.called
    cmd = pco_mock.call_args[0][0].cmd
    assert any(argument in cmd for argument in GRAPH_COMPILER_COMMAND_ARGS)

    json_dump_path = Path(tmp_path / "ngp_performance_statistics.json")
    assert json_dump_path.exists()
