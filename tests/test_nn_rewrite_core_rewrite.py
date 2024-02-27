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
from keras.api._v2 import keras  # Temporary workaround for now: MLIA-1107
from tensorflow_model_optimization.python.core.clustering.keras.cluster_wrapper import (  # pylint: disable=no-name-in-module
    ClusterWeights,
)

from mlia.nn.rewrite.core.rewrite import ClusteringRewrite
from mlia.nn.rewrite.core.rewrite import FullyConnectedRewrite
from mlia.nn.rewrite.core.rewrite import Rewrite
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
    "rewrite_name, callbacks_length, instance",
    [
        ("fully-connected", 0, Rewrite),
        ("fully-connected-clustering", 0, ClusteringRewrite),
        ("fully-connected-sparsity24", 1, Sparsity24Rewrite),
    ],
)
def test_rewrite_selection(
    rewrite_name: str, callbacks_length: int, instance: Rewrite
) -> None:
    """Test that the correct rewrite class is instantiated."""
    rewrite = RewritingOptimizer.registry.items[rewrite_name]
    assert rewrite.name == rewrite_name
    assert isinstance(rewrite, instance)  # type: ignore
    assert len(rewrite.training_callbacks()) == callbacks_length


@pytest.mark.parametrize(
    "rewrite_name, expected_error",
    [
        ("fully-connected", does_not_raise()),
        ("fully-connected-sparsity24", does_not_raise()),
        ("fully-connected-clustering", does_not_raise()),
        ("random", does_not_raise()),
    ],
)
def test_rewrite_configuration(
    test_tflite_model_fp32: Path, rewrite_name: str, expected_error: Any
) -> None:
    """Test get_rewrite function only supports rewrite types
    fully-connected, fully-connected-clustering and fully-connected-sparsity24."""
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


@pytest.mark.parametrize(
    "rewrite_type, expected_layers",
    [
        ["fully-connected", [keras.layers.Reshape, keras.layers.Dense]],
        ["fully-connected-clustering", [ClusterWeights, ClusterWeights]],
    ],
)
def test_rewriting_optimizer(
    test_tflite_model_fp32: Path,
    test_tfrecord_fp32: Path,
    rewrite_type: str,
    expected_layers: list[object],
) -> None:
    """Test fc_layer rewrite process with rewrite type fully-connected."""
    config_obj = RewriteConfiguration(
        rewrite_type,
        ["sequential/flatten/Reshape", "StatefulPartitionedCall:0"],
        test_tfrecord_fp32,
        train_params=MockTrainingParameters(),
    )

    test_obj = RewritingOptimizer(test_tflite_model_fp32, config_obj)
    rewrite_function = RewritingOptimizer.registry.items[
        test_obj.optimizer_configuration.optimization_target
    ]
    # Input, output shape does not matter, just need the test the layers are as expected
    rewrite_model = rewrite_function(input_shape=(28, 28, 1), output_shape=12)
    for idx, layer in enumerate(rewrite_model.layers):
        assert isinstance(layer, expected_layers[idx])  # type: ignore

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
        "fully-connected-clustering",
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
