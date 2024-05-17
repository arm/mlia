# SPDX-FileCopyrightText: Copyright 2023-2024, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for module mlia.nn.rewrite.core.rewrite."""
from __future__ import annotations

import re
from contextlib import ExitStack as does_not_raise
from pathlib import Path
from typing import Any
from typing import cast
from unittest.mock import MagicMock

import numpy as np
import pytest
import tensorflow_model_optimization as tfmot
from keras.api._v2 import keras  # Temporary workaround for now: MLIA-1107
from tensorflow_model_optimization.python.core.clustering.keras.cluster_wrapper import (  # pylint: disable=no-name-in-module
    ClusterWeights,
)
from tensorflow_model_optimization.python.core.sparsity.keras.pruning_wrapper import (  # pylint: disable=no-name-in-module
    PruneLowMagnitude,
)

from mlia.nn.rewrite.core.rewrite import ClusteringRewrite
from mlia.nn.rewrite.core.rewrite import GenericRewrite
from mlia.nn.rewrite.core.rewrite import Rewrite
from mlia.nn.rewrite.core.rewrite import RewriteCallable
from mlia.nn.rewrite.core.rewrite import RewriteConfiguration
from mlia.nn.rewrite.core.rewrite import RewriteRegistry
from mlia.nn.rewrite.core.rewrite import RewritingOptimizer
from mlia.nn.rewrite.core.rewrite import Sparsity24Rewrite
from mlia.nn.rewrite.core.rewrite import TrainingParameters
from mlia.nn.rewrite.core.train import train_in_dir
from mlia.nn.rewrite.library.clustering import conv2d_clustering_rewrite
from mlia.nn.rewrite.library.clustering import fc_clustering_rewrite
from mlia.nn.rewrite.library.sparsity import conv2d_sparsity_rewrite
from mlia.nn.rewrite.library.sparsity import fc_sparsity_rewrite
from mlia.nn.tensorflow.config import TFLiteModel
from tests.utils.rewrite import MockTrainingParameters


class TestRewrite(Rewrite):
    """Test rewrite class."""

    def quantize(self, model: keras.Model) -> keras.Model:
        """Return a quantized model if required."""
        return tfmot.quantization.keras.quantize_model(model)

    def preserved_quantize(self, model: keras.Model) -> keras.Model:
        """Not needed."""
        return model

    def training_callbacks(self) -> list:
        """Return default rewrite callbacks."""
        return []

    def post_process(self, model: keras.Model) -> keras.Model:
        """Return default post-processing rewrite options."""
        return model

    def check_optimization(self, model: keras.Model, **kwargs: dict) -> bool:
        """Not needed here."""
        return True


def mock_rewrite_function(*_: Any) -> Any:
    """Mock function to test autoloading of rewrite functions."""


def test_rewrite() -> None:
    """Test a derived Rewrite class."""

    def bad_rewrite_func() -> Any:
        raise NotImplementedError()

    rewrite = TestRewrite(
        "BAD_REWRITE", rewrite_fn=cast(RewriteCallable, bad_rewrite_func)
    )
    with pytest.raises(RuntimeError):
        rewrite((1, 2), (1, 2))


@pytest.mark.parametrize(
    "rewrite_name, callbacks_length, instance",
    [
        ("fully-connected", 0, GenericRewrite),
        ("fully-connected-clustering", 0, ClusteringRewrite),
        ("fully-connected-sparsity24", 1, Sparsity24Rewrite),
        ("conv2d-clustering", 0, ClusteringRewrite),
        ("conv2d-sparsity24", 1, Sparsity24Rewrite),
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
        ("conv2d-clustering", does_not_raise()),
        ("conv2d-sparsity24", does_not_raise()),
        ("random", does_not_raise()),
    ],
)
def test_rewrite_configuration(
    test_tflite_model_fp32: Path, rewrite_name: str, expected_error: Any
) -> None:
    """Test get_rewrite function only supports rewrite type fully-connected,
    fully-connected-clustering, fully-connected-sparsity24, conv2d-clustering
    and conv2d-sparsity24."""
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


def train_rewrite_model(
    input_shape: tuple | np.ndarray,
    output_shape: int | np.ndarray,
    rewrite_model: keras.Model,
) -> keras.Model:
    """Helper function to quickly train a rewrite model."""
    rewrite_model.compile(
        optimizer=keras.optimizers.Nadam(learning_rate=0.01),
        loss=keras.losses.MeanSquaredError(),
        metrics=["mae"],
    )
    if isinstance(output_shape, int):
        output_shape_list = [output_shape]
    else:
        output_shape_list = output_shape.tolist()
    rewrite_model.fit(
        x=np.random.rand(16, *input_shape),
        y=np.random.rand(16, *output_shape_list),
        batch_size=1,
        epochs=1,
        callbacks=[tfmot.sparsity.keras.UpdatePruningStep()],
    )
    return rewrite_model


def test_rewrite_fully_connected_clustering(caplog: pytest.LogCaptureFixture) -> None:
    """Check that fully connected clustering rewrite model
    has the set number of clusters."""

    rewrite = ClusteringRewrite("fully-connected-clustering", fc_clustering_rewrite)
    model = rewrite(input_shape=(28, 28), output_shape=10)
    model = rewrite.post_process(model)
    assert rewrite.check_optimization(model, number_of_clusters=4)
    rewrite.check_optimization(model, number_of_clusters=2)
    log_records = caplog.records
    warning_messages = [x.message for x in log_records if x.levelno == 30]
    assert (
        re.search(
            r"\nWARNING: Expected 2 cluster\(s\), found \d+ cluster\(s\) "
            r"in layer dense_?\d? for weight kernel:0 \n",
            warning_messages[0],
        )
        is not None
    )


def test_rewrite_conv2d_clustering(caplog: pytest.LogCaptureFixture) -> None:
    """Check that conv2d clustering rewrite model has the set number of clusters."""

    rewrite = ClusteringRewrite("conv2d-clustering", conv2d_clustering_rewrite)
    model = rewrite(
        input_shape=np.array([28, 28, 3]), output_shape=np.array([14, 14, 3])
    )
    model = rewrite.post_process(model)
    assert rewrite.check_optimization(model, number_of_clusters=4)
    rewrite.check_optimization(model, number_of_clusters=2)
    log_records = caplog.records
    warning_messages = [x.message for x in log_records if x.levelno == 30]
    assert (
        re.search(
            r"\nWARNING: Expected 2 cluster\(s\), found \d+ cluster\(s\) "
            r"in layer conv2d_?\d? for weight kernel:0 \n",
            warning_messages[0],
        )
        is not None
    )


def test_rewrite_clustering_error_handling() -> None:
    """
    Check that the clustering rewrite check_optimization
    function returns the current error.
    """

    rewrite = ClusteringRewrite("fully-connected-clustering", fc_clustering_rewrite)
    model = rewrite(input_shape=(28, 28), output_shape=10)
    with pytest.raises(
        ValueError,
        match=(r"Expected check_optimization to have argument number_of_clusters"),
    ):
        rewrite.check_optimization(model, bad_arg_name=25)


def test_rewrite_fully_connected_sparsity(caplog: pytest.LogCaptureFixture) -> None:
    """
    Check that sparse fully connected
    rewrite model is correctly sparse.
    """

    rewrite = Sparsity24Rewrite("fully-connected-sparsity24", fc_sparsity_rewrite)
    input_shape = (28, 28)
    output_shape = 10
    model = rewrite(input_shape=tuple(input_shape), output_shape=output_shape)
    model = rewrite.post_process(model)
    assert not rewrite.check_optimization(model)
    log_records = caplog.records
    warning_messages = [x.message for x in log_records if x.levelno == 30]
    assert (
        re.search(
            r"\nWARNING: Could not find \(2,4\) sparsity, in "
            r"layer dense_?\d? for weight dense_?\d?\/kernel:0 \n",
            warning_messages[0],
        )
        is not None
    )
    model = rewrite(input_shape=input_shape, output_shape=output_shape)
    train_rewrite_model(
        input_shape=input_shape, output_shape=output_shape, rewrite_model=model
    )
    model = rewrite.post_process(model)
    assert rewrite.check_optimization(model)


def test_rewrite_conv2d_sparsity(caplog: pytest.LogCaptureFixture) -> None:
    """Check that sparse conv2d rewrite model is correctly sparse."""

    rewrite = Sparsity24Rewrite("conv2d-sparsity24", conv2d_sparsity_rewrite)
    input_shape = np.array([28, 28, 3])
    output_shape = np.array([14, 14, 3])
    model = rewrite(input_shape=input_shape, output_shape=output_shape)
    model = rewrite.post_process(model)
    assert not rewrite.check_optimization(model)
    log_records = caplog.records
    warning_messages = [x.message for x in log_records if x.levelno == 30]
    assert (
        re.search(
            r"\nWARNING: Could not find \(2,4\) sparsity, in "
            r"layer conv2d_?\d? for weight conv2d_?\d?\/kernel:0 \n",
            warning_messages[0],
        )
        is not None
    )
    model = rewrite(input_shape=input_shape, output_shape=output_shape)
    train_rewrite_model(
        input_shape=input_shape, output_shape=output_shape, rewrite_model=model
    )
    model = rewrite.post_process(model)
    assert rewrite.check_optimization(model)


@pytest.mark.parametrize(
    "rewrite_type, expected_layers, quant",
    [
        ["fully-connected", [keras.layers.Reshape, keras.layers.Dense], False],
        ["fully-connected-clustering", [ClusterWeights, ClusterWeights], False],
        ["fully-connected-clustering", [ClusterWeights, ClusterWeights], True],
        ["fully-connected-sparsity24", [PruneLowMagnitude, PruneLowMagnitude], False],
        ["fully-connected-sparsity24", [PruneLowMagnitude, PruneLowMagnitude], True],
        ["conv2d-clustering", [ClusterWeights, ClusterWeights, ClusterWeights], False],
        ["conv2d-clustering", [ClusterWeights, ClusterWeights, ClusterWeights], True],
        [
            "conv2d-sparsity24",
            [PruneLowMagnitude, PruneLowMagnitude, PruneLowMagnitude],
            False,
        ],
        [
            "conv2d-sparsity24",
            [PruneLowMagnitude, PruneLowMagnitude, PruneLowMagnitude],
            True,
        ],
    ],
)
def test_rewriting_optimizer(  # pylint: disable=too-many-locals
    test_tflite_model_fp32: Path,
    test_tfrecord_fp32: Path,
    test_tflite_model: Path,
    test_tfrecord: Path,
    rewrite_type: str,
    expected_layers: list[object],
    quant: bool,
) -> None:
    """Test the rewrite process with all rewrite types."""

    tfrecord = test_tfrecord if quant else test_tfrecord_fp32
    tflite_model = test_tflite_model if quant else test_tflite_model_fp32

    config_obj = RewriteConfiguration(
        rewrite_type,
        ["sequential/flatten/Reshape", "StatefulPartitionedCall:0"]
        if "fully-connected" in rewrite_type
        else [
            "sequential/conv1/Relu;sequential/conv1/Conv2D",
            "sequential/conv2/Relu;sequential/conv2/Conv2D",
        ],
        tfrecord,
        train_params=MockTrainingParameters(),
    )

    test_obj = RewritingOptimizer(tflite_model, config_obj)
    rewrite_function = RewritingOptimizer.registry.items[
        test_obj.optimizer_configuration.optimization_target
    ]
    # Input, output shape does not matter, just need the test the layers are as expected
    rewrite_model = (
        rewrite_function(input_shape=(28, 28, 1), output_shape=12)
        if "fully-connected" in rewrite_type
        else rewrite_function(
            input_shape=np.array([28, 28, 3]), output_shape=np.array([14, 14, 3])
        )
    )
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

    rewrite1 = TestRewrite("r1", cast(RewriteCallable, lambda: 1))
    rewrite2 = TestRewrite("r2", cast(RewriteCallable, lambda: 2))

    registry.register_rewrite(rewrite1)
    registry.register_rewrite(rewrite2)
    assert registry.names() == ["r1", "r2"]


def test_builtin_rewrite_names() -> None:
    """Test if all builtin rewrites are properly registered and returned."""
    assert RewritingOptimizer.builtin_rewrite_names() == [
        "conv2d-clustering",
        "conv2d-sparsity24",
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
