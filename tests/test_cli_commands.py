# SPDX-FileCopyrightText: Copyright 2022-2025, Arm Limited and/or its affiliates.
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
from mlia.cli.commands import target_list
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
    "target_profile, pruning, clustering, optimization_profile, pruning_target, "
    "clustering_target, rewrite, rewrite_target, rewrite_start, rewrite_end ,"
    "expected_error",
    [
        [
            "ethos-u55-256",
            True,
            False,
            None,
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
            None,
            True,
            "fully-connected",
            "sequential/flatten/Reshape",
            "StatefulPartitionedCall:0",
            does_not_raise(),
        ],
        [
            "ethos-u55-256",
            False,
            False,
            "src/mlia/resources/optimization_profiles/"
            "optimization-fully-connected-pruning.toml",
            None,
            None,
            True,
            "fully-connected-sparsity",
            "sequential/flatten/Reshape",
            "StatefulPartitionedCall:0",
            does_not_raise(),
        ],
        [
            "ethos-u55-256",
            True,
            False,
            None,
            0.5,
            None,
            True,
            "fully-connected",
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
            None,
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
            None,
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
                    "Supported rewrites: ['conv2d', "
                    "'conv2d-clustering', 'conv2d-sparsity', "
                    "'conv2d-unstructured-sparsity', "
                    "'depthwise-separable-conv2d', "
                    "'depthwise-separable-conv2d-clustering', "
                    "'depthwise-separable-conv2d-sparsity', "
                    "'depthwise-separable-conv2d-unstructured-sparsity', "
                    "'fully-connected', 'fully-connected-clustering', "
                    "'fully-connected-sparsity', "
                    "'fully-connected-unstructured-sparsity']"
                ),
            ),
        ],
        [
            "ethos-u55-256",
            False,
            False,
            None,
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
            None,
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
        [
            "ethos-u55-256",
            False,
            False,
            "src/mlia/resources/optimization_profiles/"
            "optimization-fully-connected-clustering.toml",
            None,
            None,
            True,
            "fully-connected-clustering",
            "sequential/flatten/Reshape",
            "StatefulPartitionedCall:0",
            does_not_raise(),
        ],
        [
            "ethos-u55-256",
            False,
            False,
            "src/mlia/resources/optimization_profiles/"
            "optimization-fully-connected-unstructured-pruning.toml",
            None,
            None,
            True,
            "fully-connected-unstructured-sparsity",
            "sequential/flatten/Reshape",
            "StatefulPartitionedCall:0",
            does_not_raise(),
        ],
        [
            "ethos-u55-256",
            False,
            False,
            "src/mlia/resources/optimization_profiles/"
            "optimization-conv2d-pruning.toml",
            None,
            None,
            True,
            "conv2d-sparsity",
            "sequential/conv1/Relu;sequential/conv1/Conv2D",
            "sequential/conv2/Relu;sequential/conv2/Conv2D",
            does_not_raise(),
        ],
        [
            "ethos-u55-256",
            False,
            False,
            "src/mlia/resources/optimization_profiles/"
            "optimization-conv2d-unstructured-pruning.toml",
            None,
            None,
            True,
            "conv2d-unstructured-sparsity",
            "sequential/conv1/Relu;sequential/conv1/Conv2D",
            "sequential/conv2/Relu;sequential/conv2/Conv2D",
            does_not_raise(),
        ],
        [
            "ethos-u55-256",
            False,
            False,
            "src/mlia/resources/optimization_profiles/"
            "optimization-conv2d-clustering.toml",
            None,
            None,
            True,
            "conv2d-clustering",
            "sequential/conv1/Relu;sequential/conv1/Conv2D",
            "sequential/conv2/Relu;sequential/conv2/Conv2D",
            does_not_raise(),
        ],
        [
            "ethos-u55-256",
            False,
            False,
            "src/mlia/resources/optimization_profiles/optimization-conv2d.toml",
            None,
            None,
            True,
            "conv2d",
            "sequential/conv1/Relu;sequential/conv1/Conv2D",
            "sequential/conv2/Relu;sequential/conv2/Conv2D",
            does_not_raise(),
        ],
        [
            "ethos-u55-256",
            False,
            False,
            "src/mlia/resources/optimization_profiles/"
            "optimization-depthwise-separable-conv2d-pruning.toml",
            None,
            None,
            True,
            "depthwise-separable-conv2d-sparsity",
            "sequential/conv1/Relu;sequential/conv1/Conv2D",
            "sequential/conv2/Relu;sequential/conv2/Conv2D",
            does_not_raise(),
        ],
        [
            "ethos-u55-256",
            False,
            False,
            "src/mlia/resources/optimization_profiles/"
            "optimization-depthwise-separable-conv2d-unstructured-pruning.toml",
            None,
            None,
            True,
            "depthwise-separable-conv2d",
            "sequential/conv1/Relu;sequential/conv1/Conv2D",
            "sequential/conv2/Relu;sequential/conv2/Conv2D",
            does_not_raise(),
        ],
        [
            "ethos-u55-256",
            False,
            False,
            "src/mlia/resources/optimization_profiles/"
            "optimization-depthwise-separable-conv2d-clustering.toml",
            None,
            None,
            True,
            "depthwise-separable-conv2d-clustering",
            "sequential/conv1/Relu;sequential/conv1/Conv2D",
            "sequential/conv2/Relu;sequential/conv2/Conv2D",
            does_not_raise(),
        ],
        [
            "ethos-u55-256",
            False,
            False,
            "src/mlia/resources/optimization_profiles/"
            "optimization-depthwise-separable-conv2d.toml",
            None,
            None,
            True,
            "depthwise-separable-conv2d",
            "sequential/conv1/Relu;sequential/conv1/Conv2D",
            "sequential/conv2/Relu;sequential/conv2/Conv2D",
            does_not_raise(),
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
    optimization_profile: str | None,
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
            optimization_profile=optimization_profile,
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

    # Mock get_available_backends to include vela so tests pass
    monkeypatch.setattr(
        "mlia.cli.command_validators.get_available_backends",
        MagicMock(return_value=["vela"]),
    )

    # Defaults must match the above list so validate_backend() will succeed
    def _mock_default_backends(target: str) -> list[str]:
        if target in ("ethos-u55", "ethos-u65"):
            return ["vela"]
        raise AssertionError(
            f"Unexpected target passed to default_backends(): {target!r}"
        )

    monkeypatch.setattr(
        "mlia.cli.command_validators.default_backends",
        MagicMock(side_effect=_mock_default_backends),
    )

    # Mock resolve_compiler_config to avoid vela.ini file issues
    monkeypatch.setattr(
        "mlia.backend.vela.compiler.resolve_compiler_config",
        MagicMock(return_value=MagicMock()),
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


def test_target_command_action_list(caplog: pytest.LogCaptureFixture) -> None:
    """Test mlia-target command list."""
    caplog.set_level(level=20)
    target_list()

    # Verify that the target profiles were logged
    assert "Available Target Profiles" in caplog.text


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
    backend_uninstall([backend_name])

    installation_manager_mock.uninstall.assert_called_once()


@pytest.mark.parametrize(
    "i_agree_to_the_contained_eula, force, backend_name, expected_calls",
    [
        [False, False, "backend_name", [call(["backend_name"], True, False)]],
        [True, False, "backend_name", [call(["backend_name"], False, False)]],
        [True, True, "BACKEND_NAME", [call(["BACKEND_NAME"], False, True)]],
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
        names=[backend_name],
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
    backend_install(path=tmp_path, names=[backend_name], force=force)
    installation_manager_mock.install_from.assert_called_once()


def test_backend_command_action_install_no_names_with_path(
    installation_manager_mock: MagicMock,
    tmp_path: Path,
) -> None:
    """Test backend_install raises ValueError with no names & path."""
    with pytest.raises(ValueError, match="backend name"):
        backend_install(path=tmp_path, names=[])
    installation_manager_mock.install_from.assert_not_called()


def test_backend_command_action_install_multiple_names_with_path(
    installation_manager_mock: MagicMock,
    tmp_path: Path,
) -> None:
    """Test backend_install raises ValueError when multiple names & path."""
    with pytest.raises(ValueError, match="backend name"):
        backend_install(path=tmp_path, names=["backend1", "backend2"])
    installation_manager_mock.install_from.assert_not_called()


@pytest.mark.parametrize(
    "compatibility, performance, expected_category",
    [
        [True, True, {"compatibility", "performance"}],
        [True, False, {"compatibility"}],
        [False, True, {"performance"}],
        [False, False, {"compatibility"}],
    ],
)
def test_check_category_combinations(
    sample_context: ExecutionContext,
    test_tflite_model: Path,
    monkeypatch: pytest.MonkeyPatch,
    compatibility: bool,
    performance: bool,
    expected_category: set[str],
) -> None:
    """Test check() with different category combinations."""
    mock_performance_estimation(monkeypatch)

    # Mock get_advice to capture what category is passed
    get_advice_mock = MagicMock()
    monkeypatch.setattr("mlia.cli.commands.get_advice", get_advice_mock)

    # Mock validators
    monkeypatch.setattr("mlia.cli.commands.validate_check_target_profile", MagicMock())
    monkeypatch.setattr(
        "mlia.cli.commands.validate_backend", MagicMock(return_value=None)
    )

    check(
        sample_context,
        target_profile="ethos-u55-256",
        model=str(test_tflite_model),
        compatibility=compatibility,
        performance=performance,
    )

    # Verify get_advice was called with the expected category
    get_advice_mock.assert_called_once()
    call_args = get_advice_mock.call_args
    assert call_args[0][2] == expected_category
