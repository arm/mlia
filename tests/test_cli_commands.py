# SPDX-FileCopyrightText: Copyright 2022-2024, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for cli.commands module."""
from __future__ import annotations

import re
from contextlib import ExitStack as does_not_raise
from pathlib import Path
from typing import Any
from unittest.mock import call
from unittest.mock import MagicMock

import pytest

from mlia.backend.manager import DefaultInstallationManager
from mlia.backend.vela.performance import LayerwisePerfInfo
from mlia.cli.commands import backend_install
from mlia.cli.commands import backend_list
from mlia.cli.commands import backend_uninstall
from mlia.cli.commands import check
from mlia.cli.commands import optimize
from mlia.core.context import ExecutionContext
from mlia.target.ethos_u.config import EthosUConfiguration
from mlia.target.ethos_u.performance import MemoryUsage
from mlia.target.ethos_u.performance import NPUCycles
from mlia.target.ethos_u.performance import PerformanceMetrics


def test_operators_expected_parameters(sample_context: ExecutionContext) -> None:
    """Test operators command wrong parameters."""
    with pytest.raises(Exception, match="Model is not provided"):
        check(sample_context, "ethos-u55-256")


def test_performance_unknown_target(
    sample_context: ExecutionContext, test_tflite_model: Path
) -> None:
    """Test that command should fail if unknown target passed."""
    with pytest.raises(
        Exception,
        match=(
            r"Profile 'unknown' is neither a valid built-in target profile "
            r"name or a valid file path."
        ),
    ):
        check(
            sample_context,
            model=str(test_tflite_model),
            target_profile="unknown",
            performance=True,
        )


@pytest.mark.parametrize(
    "target_profile, pruning, clustering, pruning_target, clustering_target, "
    "rewrite, rewrite_target, rewrite_start, rewrite_end, expected_error",
    [
        [
            "ethos-u55-256",
            True,
            False,
            0.5,
            None,
            False,
            None,
            "node_a",
            "node_b",
            does_not_raise(),
        ],
        [
            "ethos-u55-256",
            False,
            False,
            None,
            None,
            True,
            "fully_connected",
            "sequential/flatten/Reshape",
            "StatefulPartitionedCall:0",
            does_not_raise(),
        ],
        [
            "ethos-u55-256",
            True,
            False,
            0.5,
            None,
            True,
            "fully_connected",
            "sequential/flatten/Reshape",
            "StatefulPartitionedCall:0",
            pytest.raises(
                Exception,
                match=(r"Only 'rewrite' is supported for TensorFlow Lite files."),
            ),
        ],
        [
            "ethos-u65-512",
            False,
            True,
            0.5,
            32,
            False,
            None,
            None,
            None,
            does_not_raise(),
        ],
        [
            "ethos-u55-256",
            False,
            False,
            0.5,
            None,
            True,
            "random",
            "node_x",
            "node_y",
            pytest.raises(
                Exception,
                match=re.escape(
                    "Invalid rewrite target: 'random'. "
                    "Supported rewrites: ['fully_connected']"
                ),
            ),
        ],
        [
            "ethos-u55-256",
            False,
            False,
            0.5,
            None,
            True,
            None,
            "node_m",
            "node_n",
            pytest.raises(
                Exception,
                match=(
                    r"To perform rewrite, rewrite-target, "
                    r"rewrite-start and rewrite-end must be set."
                ),
            ),
        ],
        [
            "ethos-u55-256",
            False,
            False,
            "invalid",
            None,
            True,
            "remove",
            None,
            "node_end",
            pytest.raises(
                Exception,
                match=(
                    r"To perform rewrite, rewrite-target, "
                    r"rewrite-start and rewrite-end must be set."
                ),
            ),
        ],
    ],
)
def test_opt_valid_optimization_target(  # pylint: disable=too-many-locals,too-many-arguments
    target_profile: str,
    sample_context: ExecutionContext,
    pruning: bool,
    clustering: bool,
    pruning_target: float | None,
    clustering_target: int | None,
    rewrite: bool,
    rewrite_target: str | None,
    rewrite_start: str | None,
    rewrite_end: str | None,
    expected_error: Any,
    monkeypatch: pytest.MonkeyPatch,
    test_keras_model: Path,
    test_tflite_model_fp32: Path,
    test_tfrecord_fp32: Path,
) -> None:
    """Test that command should not fail with valid optimization targets."""
    mock_performance_estimation(monkeypatch)

    model_type = test_tflite_model_fp32 if rewrite else test_keras_model
    data = test_tfrecord_fp32 if rewrite else None

    with expected_error:
        optimize(
            ctx=sample_context,
            target_profile=target_profile,
            model=str(model_type),
            pruning=pruning,
            clustering=clustering,
            pruning_target=pruning_target,
            clustering_target=clustering_target,
            rewrite=rewrite,
            rewrite_target=rewrite_target,
            rewrite_start=rewrite_start,
            rewrite_end=rewrite_end,
            dataset=data,
        )


def mock_performance_estimation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mock performance estimation."""
    metrics = PerformanceMetrics(
        EthosUConfiguration.load_profile("ethos-u55-256"),
        NPUCycles(1, 2, 3, 4, 5, 6),
        MemoryUsage(1, 2, 3, 4),
        LayerwisePerfInfo(layerwise_info=[]),
    )
    monkeypatch.setattr(
        "mlia.target.ethos_u.data_collection.EthosUPerformanceEstimator.estimate",
        MagicMock(return_value=metrics),
    )


@pytest.fixture(name="installation_manager_mock")
def fixture_mock_installation_manager(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Mock installation manager."""
    install_manager_mock = MagicMock(spec=DefaultInstallationManager)
    monkeypatch.setattr(
        "mlia.cli.commands.get_installation_manager",
        MagicMock(return_value=install_manager_mock),
    )
    return install_manager_mock


def test_backend_command_action_list(installation_manager_mock: MagicMock) -> None:
    """Test mlia-backend command list."""
    backend_list()

    installation_manager_mock.show_env_details.assert_called_once()


@pytest.mark.parametrize(
    "backend_name",
    [
        "backend_name",
        "BACKEND_NAME",
        "BaCkend_NAme",
    ],
)
def test_backend_command_action_uninstall(
    installation_manager_mock: MagicMock,
    backend_name: str,
) -> None:
    """Test mlia-backend command uninstall."""
    backend_uninstall(backend_name)

    installation_manager_mock.uninstall.assert_called_once()


@pytest.mark.parametrize(
    "i_agree_to_the_contained_eula, force, backend_name, expected_calls",
    [
        [False, False, "backend_name", [call("backend_name", True, False)]],
        [True, False, "backend_name", [call("backend_name", False, False)]],
        [True, True, "BACKEND_NAME", [call("BACKEND_NAME", False, True)]],
    ],
)
def test_backend_command_action_add_download(
    installation_manager_mock: MagicMock,
    i_agree_to_the_contained_eula: bool,
    force: bool,
    backend_name: str,
    expected_calls: Any,
) -> None:
    """Test mlia-backend command "install" with download option."""
    backend_install(
        name=backend_name,
        i_agree_to_the_contained_eula=i_agree_to_the_contained_eula,
        force=force,
    )

    assert installation_manager_mock.download_and_install.mock_calls == expected_calls


@pytest.mark.parametrize(
    "backend_name, force",
    [
        ["backend_name", False],
        ["backend_name", True],
        ["BACKEND_NAME", True],
    ],
)
def test_backend_command_action_install_from_path(
    installation_manager_mock: MagicMock,
    tmp_path: Path,
    backend_name: str,
    force: bool,
) -> None:
    """Test mlia-backend command "install" with backend path."""
    backend_install(path=tmp_path, name=backend_name, force=force)
    installation_manager_mock.install_from.assert_called_once()
