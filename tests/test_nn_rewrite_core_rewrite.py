# SPDX-FileCopyrightText: Copyright 2023, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for module mlia.nn.rewrite.core.rewrite."""
from __future__ import annotations

from contextlib import ExitStack as does_not_raise
from pathlib import Path
from typing import Any
from typing import cast

import pytest

from mlia.nn.rewrite.core.rewrite import DynamicallyLoadedRewrite
from mlia.nn.rewrite.core.rewrite import Rewrite
from mlia.nn.rewrite.core.rewrite import RewriteCallable
from mlia.nn.rewrite.core.rewrite import RewriteConfiguration
from mlia.nn.rewrite.core.rewrite import RewriteRegistry
from mlia.nn.rewrite.core.rewrite import RewritingOptimizer
from mlia.nn.tensorflow.config import TFLiteModel
from tests.utils.rewrite import MockTrainingParameters


def mock_rewrite_function(*_: Any) -> Any:
    """Mock function to test autoloading of rewrite functions."""


def test_rewrite() -> None:
    """Test the Rewrite class."""

    def bad_rewrite_func() -> Any:
        raise NotImplementedError()

    rewrite = Rewrite("BAD_REWRITE", rewrite_fn=cast(RewriteCallable, bad_rewrite_func))
    with pytest.raises(RuntimeError):
        rewrite((1, 2), (1, 2))


@pytest.mark.parametrize(
    "rewrite_name, expected_error",
    [
        ("fully_connected", does_not_raise()),
        ("random", does_not_raise()),
    ],
)
def test_rewrite_configuration(
    test_tflite_model_fp32: Path, rewrite_name: str, expected_error: Any
) -> None:
    """Test get_rewrite function only supports rewrite type fully_connected."""
    with expected_error:
        config_obj = RewriteConfiguration(
            rewrite_name,
            ["sample_node_start", "sample_node_end"],
            None,
        )

        assert config_obj.optimization_target in str(config_obj)

        rewriter_obj = RewritingOptimizer(test_tflite_model_fp32, config_obj)
        assert rewriter_obj.optimizer_configuration.optimization_target == rewrite_name
        assert isinstance(rewriter_obj, RewritingOptimizer)


def test_rewriting_optimizer(
    test_tflite_model_fp32: Path,
    test_tfrecord_fp32: Path,
) -> None:
    """Test fc_layer rewrite process with rewrite type fully_connected."""
    config_obj = RewriteConfiguration(
        "fully_connected",
        ["sequential/flatten/Reshape", "StatefulPartitionedCall:0"],
        test_tfrecord_fp32,
        train_params=MockTrainingParameters(),
    )

    test_obj = RewritingOptimizer(test_tflite_model_fp32, config_obj)
    test_obj.apply_optimization()
    trained_model = test_obj.get_model()

    assert isinstance(trained_model, TFLiteModel)

    cfg = test_obj.optimization_config()
    assert isinstance(cfg, str)
    assert cfg


def test_register_rewrite_function() -> None:
    """Test adding rewrite functions and verify the are reported via the registry."""
    registry = RewriteRegistry()

    rewrite1 = Rewrite("r1", cast(RewriteCallable, lambda: 1))
    rewrite2 = Rewrite("r2", cast(RewriteCallable, lambda: 2))

    registry.register_rewrite(rewrite1)
    registry.register_rewrite(rewrite2)
    assert registry.names() == ["r1", "r2"]


def test_builtin_rewrite_names() -> None:
    """Test if all builtin rewrites are properly registered and returned."""
    assert RewritingOptimizer.builtin_rewrite_names() == ["fully_connected"]


def test_rewrite_function_autoload() -> None:
    """Test rewrite function loading."""
    function_name = "tests.test_nn_rewrite_core_rewrite.mock_rewrite_function"
    rewrite = DynamicallyLoadedRewrite(name="mock_rewrite", function_name=function_name)
    assert rewrite.name == "mock_rewrite"

    assert rewrite.function is not mock_rewrite_function
    assert rewrite.load_function(function_name) is mock_rewrite_function
    assert rewrite.function is mock_rewrite_function


def test_rewrite_function_autoload_fail() -> None:
    """Test rewrite function loading failure."""
    function_name = "invalid_module.invalid_function"
    rewrite = DynamicallyLoadedRewrite(
        name="mock_rewrite",
        function_name="invalid_module.invalid_function",
    )
    assert rewrite.name == "mock_rewrite"

    with pytest.raises(Exception) as exc_info:
        rewrite.load_function(function_name)

    message = exc_info.value.args[0]

    assert message == (
        "Unable to load rewrite function 'invalid_module.invalid_function'"
        " for 'mock_rewrite'."
    )
