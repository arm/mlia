# SPDX-FileCopyrightText: Copyright 2023, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: LicenseRef-LICENSE
"""Tests for the Argo performance module."""
from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from mlia.backend.argo.performance import ArgoPerformanceEstimator
from mlia.backend.argo.performance import run_argo_in_subprocess


def test_estimate_performance_docker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test for function estimate_performance() using docker."""
    test_model_path = tmp_path / "test_model.tflite"
    # nested_tmp_path is used as temporary output of the Argo data. It should
    # be different from the output dir passed to estimate_performance() to check
    # that the files are copied correctly.
    nested_tmp_path = tmp_path / "tmp"
    nested_tmp_path.mkdir()
    trace_output_path = nested_tmp_path / f"{test_model_path.stem}_chrome_trace.json"

    monkeypatch.setattr(
        "mlia.backend.argo.performance.mkdtemp",
        MagicMock(return_value=str(nested_tmp_path)),
    )
    monkeypatch.setattr(
        "mlia.backend.argo.performance.get_argo_backend_path",
        MagicMock(return_value=None),
    )
    docker_mock = MagicMock()
    docker_client_mock = MagicMock()

    def mock_run(*_: Any, **__: Any) -> bytes:
        """Mock the docker run by creating an output file and returning a str."""
        trace_output_path.touch()
        return bytes("### DOCKER OUTPUT ###", encoding="utf-8")

    docker_run_mock = MagicMock(
        side_effect=mock_run,
    )
    docker_client_mock.containers.run = docker_run_mock
    docker_mock.from_env = MagicMock(return_value=docker_client_mock)
    monkeypatch.setattr("mlia.backend.argo.performance.docker", docker_mock)

    perf = ArgoPerformanceEstimator(output_dir=tmp_path, backend_config={})
    metrics_file = perf._run_argo(test_model_path)  # pylint: disable=protected-access
    assert metrics_file.is_file()
    assert docker_run_mock.call_count == 1


def test_estimate_performance_subprocess(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test for function estimate_performance() running local Argo."""
    test_model_path = tmp_path / "test_model.tflite"
    # nested_tmp_path is used as temporary output of the Argo data. It should
    # be different from the output dir passed to estimate_performance() to check
    # that the files are copied correctly.
    nested_tmp_path = tmp_path / "tmp"
    nested_tmp_path.mkdir()
    trace_output_path = nested_tmp_path / f"{test_model_path.stem}_chrome_trace.json"

    monkeypatch.setattr(
        "mlia.backend.argo.performance.mkdtemp",
        MagicMock(return_value=str(nested_tmp_path)),
    )
    monkeypatch.setattr(
        "mlia.backend.argo.performance.get_argo_backend_path",
        MagicMock(return_value=Path("touch")),
    )
    monkeypatch.setattr(
        "mlia.backend.argo.performance.create_argo_command",
        MagicMock(return_value=[str(trace_output_path)]),
    )
    mock_run_in_subproc = MagicMock(wraps=run_argo_in_subprocess)
    monkeypatch.setattr(
        "mlia.backend.argo.performance.run_argo_in_subprocess", mock_run_in_subproc
    )

    perf = ArgoPerformanceEstimator(output_dir=tmp_path, backend_config={})
    metrics_file = perf._run_argo(test_model_path)  # pylint: disable=protected-access
    assert metrics_file.is_file()
    mock_run_in_subproc.assert_called_once()
