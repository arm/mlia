# SPDX-FileCopyrightText: Copyright 2023-2024, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for module mlia.nn.rewrite.core.rewrite."""
from __future__ import annotations

from contextlib import ExitStack as does_not_raise
from pathlib import Path
from typing import Any
from typing import cast
from unittest.mock import MagicMock

import pytest

from mlia.nn.rewrite.core.rewrite import FullyConnectedRewrite
from mlia.nn.rewrite.core.rewrite import RewriteCallable
from mlia.nn.rewrite.core.rewrite import RewriteConfiguration
from mlia.nn.rewrite.core.rewrite import RewriteRegistry
from mlia.nn.rewrite.core.rewrite import RewritingOptimizer
from mlia.nn.rewrite.core.rewrite import Sparsity24Rewrite
from mlia.nn.rewrite.core.rewrite import TrainingParameters
from mlia.nn.rewrite.core.train import train_in_dir
from mlia.nn.tensorflow.config import TFLiteModel
from tests.utils.rewrite import MockTrainingParameters


def test_rewrite() -> None:
    """Test a derived Rewrite class."""

    def bad_rewrite_func() -> Any:
        raise NotImplementedError()

    rewrite = Sparsity24Rewrite(
        "BAD_REWRITE", rewrite_fn=cast(RewriteCallable, bad_rewrite_func)
    )
    with pytest.raises(RuntimeError):
        rewrite((1, 2), (1, 2))


@pytest.mark.parametrize(
    "rewrite_name, rewrite_class",
    [
        ("fully-connected", FullyConnectedRewrite),
        ("fully-connected-sparsity24", Sparsity24Rewrite),
    ],
)
def test_rewrite_selection(
    rewrite_name: str,
    rewrite_class: Any,
) -> None:
    """Check that the correct rewrite class is instantiated through the registry"""
    config_obj = RewriteConfiguration(
        rewrite_name,
        ["sample_node_start", "sample_node_end"],
    )

    rewrite = RewritingOptimizer.registry.items[config_obj.optimization_target]
    assert rewrite.name == rewrite_name
    assert isinstance(rewrite, rewrite_class)


@pytest.mark.parametrize(
    "rewrite_name, expected_error",
    [
        ("fully-connected", does_not_raise()),
        ("fully-connected-sparsity24", does_not_raise()),
        ("random", does_not_raise()),
    ],
)
def test_rewrite_configuration(
    test_tflite_model_fp32: Path, rewrite_name: str, expected_error: Any
) -> None:
    """Test get_rewrite function only supports rewrite types
    fully-connected and fully-connected-sparsity24."""
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
    """Test fc_layer rewrite process with rewrite type fully-connected."""
    config_obj = RewriteConfiguration(
        "fully-connected",
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
    """Test adding rewrite functions and verify they are reported via the registry."""
    registry = RewriteRegistry()

    rewrite1 = FullyConnectedRewrite("r1", cast(RewriteCallable, lambda: 1))
    rewrite2 = Sparsity24Rewrite("r2", cast(RewriteCallable, lambda: 2))

    registry.register_rewrite(rewrite1)
    registry.register_rewrite(rewrite2)
    assert registry.names() == ["r1", "r2"]


def test_builtin_rewrite_names() -> None:
    """Test if all builtin rewrites are properly registered and returned."""
    assert RewritingOptimizer.builtin_rewrite_names() == [
        "fully-connected",
        "fully-connected-sparsity24",
    ]


def test_rewrite_configuration_train_params(
    test_tflite_model_fp32: Path,
    test_tfrecord_fp32: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test if we pass training parameters to the
    rewrite configuration function they are passed to train_in_dir."""
    train_params = TrainingParameters(
        batch_size=64, steps=24000, learning_rate=1e-5, show_progress=True
    )

    config_obj = RewriteConfiguration(
        "fully-connected",
        ["sequential/flatten/Reshape", "StatefulPartitionedCall:0"],
        test_tfrecord_fp32,
        train_params=train_params,
    )

    rewriter_obj = RewritingOptimizer(test_tflite_model_fp32, config_obj)
    train_mock = MagicMock(side_effect=train_in_dir)
    monkeypatch.setattr("mlia.nn.rewrite.core.train.train_in_dir", train_mock)
    rewriter_obj.apply_optimization()

    train_mock.assert_called_once()
    assert train_mock.call_args.kwargs["train_params"] == train_params
