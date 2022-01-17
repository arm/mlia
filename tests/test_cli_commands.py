# Copyright 2021, Arm Ltd.
"""Tests for cli.commands module."""
# pylint: disable=no-self-use,too-many-arguments
from pathlib import Path
from typing import Any

import pytest
from mlia.cli.commands import optimization
from mlia.cli.commands import performance
from mlia.core.context import ExecutionContext

from tests.test_cli_main import mock_performance_estimation


# temporary disable all tests in this module
pytestmark = pytest.mark.skip


def test_performance_unknown_target(dummy_context: ExecutionContext) -> None:
    """Test that command should fail if unknown target passed."""
    with pytest.raises(Exception, match="Unsupported target: unknown"):
        performance(dummy_context, model="some_model.tflite", target="unknown")


@pytest.mark.parametrize(
    "target, optimization_type, optimization_target, expected_error",
    [
        [
            "U55-256",
            None,
            "0.5",
            pytest.raises(Exception, match="Optimization type is not provided"),
        ],
        [
            "U65-512",
            "unknown",
            "16",
            pytest.raises(Exception, match="Unsupported optimization type: unknown"),
        ],
        [
            "U55-256",
            "pruning",
            None,
            pytest.raises(Exception, match="Optimization target is not provided"),
        ],
        [
            "U65-512",
            "clustering",
            None,
            pytest.raises(Exception, match="Optimization target is not provided"),
        ],
        [
            "unknown",
            "clustering",
            "16",
            pytest.raises(Exception, match="Unsupported target: unknown"),
        ],
    ],
)
def test_opt_expected_parameters(
    dummy_context: ExecutionContext,
    target: str,
    optimization_type: str,
    optimization_target: str,
    expected_error: Any,
    test_models_path: Path,
) -> None:
    """Test that command should fail if no or unknown optimization type provided."""
    model = test_models_path / "simple_model.h5"

    with expected_error:
        optimization(
            ctx=dummy_context,
            target=target,
            model=str(model),
            optimization_type=optimization_type,
            optimization_target=optimization_target,
        )


@pytest.mark.parametrize(
    "target, optimization_type, optimization_target",
    [
        ["U55-256", "pruning", "0.5"],
        ["U65-512", "clustering", "32"],
        ["U55-256", "pruning,clustering", "0.5,32"],
    ],
)
def test_opt_valid_optimization_target(
    dummy_context: ExecutionContext,
    target: str,
    optimization_type: str,
    optimization_target: str,
    monkeypatch: Any,
    tmp_path: Path,
    test_models_path: Path,
) -> None:
    """Test that command should not fail with valid optimization targets."""
    model = test_models_path / "simple_model.h5"

    mock_performance_estimation(monkeypatch)

    optimization(
        ctx=dummy_context,
        target=target,
        model=str(model),
        optimization_type=optimization_type,
        optimization_target=optimization_target,
    )

    assert (tmp_path / "models/original_model.tflite").is_file()
    assert (tmp_path / "models/optimized_model.tflite").is_file()
