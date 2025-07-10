# SPDX-FileCopyrightText: Copyright 2023-2025, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: LicenseRef-LICENSE
"""Tests for Neural Accelerator Graph Compiler performance estimation."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from mlia.backend.nx_graph_compiler.config import NXGraphCompilerConfig
from mlia.backend.nx_graph_compiler.performance import GC_OUTPUT_CONTROL_PARAMS
from mlia.backend.nx_graph_compiler.performance import NXGraphCompilerOutputFiles
from mlia.backend.nx_graph_compiler.performance import (
    NXGraphCompilerPerformanceEstimator,
)
from mlia.target.neural_technology.config import NeuralTechnologyConfiguration


def test_nx_graph_compiler_output_files(tmp_path: Path) -> None:
    """Test for class NXGraphCompilerConfig."""
    output_files = NXGraphCompilerOutputFiles.from_output_dir(tmp_path, "model_xyz")
    with pytest.raises(FileNotFoundError):
        output_files.check_exists()
    for file in vars(output_files).values():
        assert isinstance(file, Path)
        file.touch()
    output_files.check_exists()


def test_nx_graph_compiler_performance_estimator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test class NXGraphCompilerPerformanceEstimator."""
    neural_technology_cfg = NeuralTechnologyConfiguration.load_profile(
        "neural-technology"
    )
    pco_mock = MagicMock()
    mock_repo = MagicMock()
    mock_repo.get_backend_settings = MagicMock(return_value=(tmp_path / "backend", {}))
    monkeypatch.setattr(
        "mlia.backend.nx_graph_compiler.performance.get_backend_repository",
        MagicMock(return_value=mock_repo),
    )
    monkeypatch.setattr(
        "mlia.backend.nx_graph_compiler.performance.VulkanModelConverter",
        MagicMock(return_value=MagicMock(return_value=tmp_path / "vgf_file")),
    )
    monkeypatch.setattr(
        "mlia.backend.nx_graph_compiler.performance.process_command_output",
        pco_mock,
    )
    monkeypatch.setattr(
        "mlia.backend.nx_graph_compiler.performance.NXPerformanceDatabaseParser",
        MagicMock(),
    )
    monkeypatch.setattr(
        "mlia.backend.nx_graph_compiler.performance.NXDebugDatabaseParser",
        MagicMock(),
    )
    monkeypatch.setattr(
        "mlia.backend.nx_graph_compiler.performance."
        "NXGraphCompilerOutputFiles.check_exists",
        MagicMock(return_value=True),
    )
    operator_types_mapping = {
        "Identity": "identity_op_type",
        "model/re_lu_7/Relu": "RELU",
    }

    estimator = NXGraphCompilerPerformanceEstimator(
        tmp_path, neural_technology_cfg.backend_config, operator_types_mapping
    )
    metrics = estimator.estimate(tmp_path / "model.tflite")
    assert isinstance(metrics.backend_config, NXGraphCompilerConfig)
    assert all(isinstance(file, Path) for file in vars(metrics.output_files).values())

    assert pco_mock.called
    cmd = pco_mock.call_args[0][0].cmd
    assert any(argument in cmd for argument in GC_OUTPUT_CONTROL_PARAMS)

    json_dump_path = Path(tmp_path / "nx_performance_statistics.json")
    assert json_dump_path.exists()
