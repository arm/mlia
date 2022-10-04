# SPDX-FileCopyrightText: Copyright 2022, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for cli.commands module."""
from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import call
from unittest.mock import MagicMock

import pytest

from mlia.cli.commands import backend
from mlia.cli.commands import operators
from mlia.cli.commands import optimization
from mlia.cli.commands import performance
from mlia.core.context import ExecutionContext
from mlia.devices.ethosu.config import EthosUConfiguration
from mlia.devices.ethosu.performance import MemoryUsage
from mlia.devices.ethosu.performance import NPUCycles
from mlia.devices.ethosu.performance import PerformanceMetrics
from mlia.tools.metadata.common import InstallationManager


def test_operators_expected_parameters(sample_context: ExecutionContext) -> None:
    """Test operators command wrong parameters."""
    with pytest.raises(Exception, match="Model is not provided"):
        operators(sample_context, "ethos-u55-256")


def test_performance_unknown_target(
    sample_context: ExecutionContext, test_tflite_model: Path
) -> None:
    """Test that command should fail if unknown target passed."""
    with pytest.raises(Exception, match="Unable to find target profile unknown"):
        performance(
            sample_context, model=str(test_tflite_model), target_profile="unknown"
        )


@pytest.mark.parametrize(
    "target_profile, optimization_type, optimization_target, expected_error",
    [
        [
            "ethos-u55-256",
            None,
            "0.5",
            pytest.raises(Exception, match="Optimization type is not provided"),
        ],
        [
            "ethos-u65-512",
            "unknown",
            "16",
            pytest.raises(Exception, match="Unsupported optimization type: unknown"),
        ],
        [
            "ethos-u55-256",
            "pruning",
            None,
            pytest.raises(Exception, match="Optimization target is not provided"),
        ],
        [
            "ethos-u65-512",
            "clustering",
            None,
            pytest.raises(Exception, match="Optimization target is not provided"),
        ],
        [
            "unknown",
            "clustering",
            "16",
            pytest.raises(Exception, match="Unable to find target profile unknown"),
        ],
    ],
)
def test_opt_expected_parameters(
    sample_context: ExecutionContext,
    target_profile: str,
    monkeypatch: pytest.MonkeyPatch,
    optimization_type: str,
    optimization_target: str,
    expected_error: Any,
    test_keras_model: Path,
) -> None:
    """Test that command should fail if no or unknown optimization type provided."""
    mock_performance_estimation(monkeypatch)

    with expected_error:
        optimization(
            ctx=sample_context,
            target_profile=target_profile,
            model=str(test_keras_model),
            optimization_type=optimization_type,
            optimization_target=optimization_target,
        )


@pytest.mark.parametrize(
    "target_profile, optimization_type, optimization_target",
    [
        ["ethos-u55-256", "pruning", "0.5"],
        ["ethos-u65-512", "clustering", "32"],
        ["ethos-u55-256", "pruning,clustering", "0.5,32"],
    ],
)
def test_opt_valid_optimization_target(
    target_profile: str,
    sample_context: ExecutionContext,
    optimization_type: str,
    optimization_target: str,
    monkeypatch: pytest.MonkeyPatch,
    test_keras_model: Path,
) -> None:
    """Test that command should not fail with valid optimization targets."""
    mock_performance_estimation(monkeypatch)

    optimization(
        ctx=sample_context,
        target_profile=target_profile,
        model=str(test_keras_model),
        optimization_type=optimization_type,
        optimization_target=optimization_target,
    )


def mock_performance_estimation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mock performance estimation."""
    metrics = PerformanceMetrics(
        EthosUConfiguration("ethos-u55-256"),
        NPUCycles(1, 2, 3, 4, 5, 6),
        MemoryUsage(1, 2, 3, 4, 5),
    )
    monkeypatch.setattr(
        "mlia.devices.ethosu.data_collection.EthosUPerformanceEstimator.estimate",
        MagicMock(return_value=metrics),
    )


@pytest.fixture(name="installation_manager_mock")
def fixture_mock_installation_manager(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Mock installation manager."""
    install_manager_mock = MagicMock(spec=InstallationManager)
    monkeypatch.setattr(
        "mlia.cli.commands.get_installation_manager",
        MagicMock(return_value=install_manager_mock),
    )
    return install_manager_mock


def test_backend_command_action_status(installation_manager_mock: MagicMock) -> None:
    """Test backend command "status"."""
    backend(backend_action="status")

    installation_manager_mock.show_env_details.assert_called_once()


@pytest.mark.parametrize(
    "i_agree_to_the_contained_eula, backend_name, expected_calls",
    [
        [False, None, [call(None, True)]],
        [True, None, [call(None, False)]],
        [False, "backend_name", [call("backend_name", True)]],
        [True, "backend_name", [call("backend_name", False)]],
    ],
)
def test_backend_command_action_add_downoad(
    installation_manager_mock: MagicMock,
    i_agree_to_the_contained_eula: bool,
    backend_name: str | None,
    expected_calls: Any,
) -> None:
    """Test backend command "install" with download option."""
    backend(
        backend_action="install",
        download=True,
        name=backend_name,
        i_agree_to_the_contained_eula=i_agree_to_the_contained_eula,
    )

    assert installation_manager_mock.download_and_install.mock_calls == expected_calls


@pytest.mark.parametrize("backend_name", [None, "backend_name"])
def test_backend_command_action_install_from_path(
    installation_manager_mock: MagicMock,
    tmp_path: Path,
    backend_name: str | None,
) -> None:
    """Test backend command "install" with backend path."""
    backend(backend_action="install", path=tmp_path, name=backend_name)

    installation_manager_mock.install_from(tmp_path, backend_name)


def test_backend_command_action_install_only_one_action(
    installation_manager_mock: MagicMock,  # pylint: disable=unused-argument
    tmp_path: Path,
) -> None:
    """Test that only one of action type allowed."""
    with pytest.raises(
        Exception,
        match="Please select only one action: download or "
        "provide path to the backend installation",
    ):
        backend(backend_action="install", download=True, path=tmp_path)
