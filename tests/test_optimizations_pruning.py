# Copyright 2021, Arm Ltd.
"""Test for module optimizations/pruning."""
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
import pytest
import tensorflow as tf
from mlia.config import TFLiteModel
from mlia.optimizations.pruning import Pruner
from mlia.optimizations.pruning import PruningConfiguration
from mlia.utils import general as general_utils
from mlia.utils import tflite_metrics
from numpy.core.numeric import isclose

from tests.utils.generate_keras_model import generate_keras_model


def _get_dataset() -> Tuple[np.array, np.array]:
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), _ = mnist.load_data()
    x_train = x_train / 255.0

    # Use subset of 60000 examples to keep unit test speed fast.
    x_train = x_train[0:1]
    y_train = y_train[0:1]

    return x_train, y_train


def _train_model(model: tf.keras.Model) -> None:
    num_epochs = 1

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer="adam", loss=loss_fn, metrics=["accuracy"])

    x_train, y_train = _get_dataset()

    model.fit(x_train, y_train, epochs=num_epochs)


def _test_sparsity(
    metrics: tflite_metrics.TFLiteMetrics,
    target_sparsity: float,
    layers_to_prune: Optional[List[str]],
) -> None:
    pruned_sparsity_dict = metrics.sparsity_per_layer()
    num_sparse_layers = 0
    num_optimizable_layers = len(pruned_sparsity_dict)
    error_margin = 0.03
    if layers_to_prune:
        expected_num_sparse_layers = len(layers_to_prune)
    else:
        expected_num_sparse_layers = num_optimizable_layers
    for layer_name in pruned_sparsity_dict:
        if abs(pruned_sparsity_dict[layer_name] - target_sparsity) < error_margin:
            num_sparse_layers = num_sparse_layers + 1
    # make sure we are having exactly as many sparse layers as we wanted
    assert num_sparse_layers == expected_num_sparse_layers


@pytest.mark.parametrize("target_sparsity", (0.5, 0.9))
@pytest.mark.parametrize("mock_data", (False, True))
@pytest.mark.parametrize("layers_to_prune", (["conv1"], ["conv1", "conv2"], None))
def test_prune_simple_model_fully(
    target_sparsity: float, mock_data: bool, layers_to_prune: Optional[List[str]]
) -> None:
    """Simple mnist test to see if pruning works correctly."""
    x_train, y_train = _get_dataset()
    batch_size = 1
    num_epochs = 1

    base_model = generate_keras_model()
    _train_model(base_model)

    tflite_base_model = TFLiteModel(
        general_utils.save_tflite_model(general_utils.convert_to_tflite(base_model))
    )
    base_compressed_size = tflite_metrics.get_gzipped_file_size(
        tflite_base_model.model_path
    )
    base_tflite_metrics = tflite_metrics.TFLiteMetrics(tflite_base_model.model_path)

    # make sure sparsity is zero before pruning
    base_sparsity_dict = base_tflite_metrics.sparsity_per_layer()
    for layer_name in base_sparsity_dict:
        assert isclose(base_sparsity_dict[layer_name], 0, atol=1e-2)

    if mock_data:
        pruner = Pruner(
            base_model,
            PruningConfiguration(
                target_sparsity,
                layers_to_prune,
            ),
        )

    else:
        pruner = Pruner(
            base_model,
            PruningConfiguration(
                target_sparsity,
                layers_to_prune,
                x_train,
                y_train,
                batch_size,
                num_epochs,
            ),
        )

    pruner.apply_optimization()
    pruned_model = pruner.get_model()

    tflite_pruned_model = TFLiteModel(
        general_utils.save_tflite_model(general_utils.convert_to_tflite(pruned_model))
    )
    pruned_compressed_size = tflite_metrics.get_gzipped_file_size(
        tflite_pruned_model.model_path
    )
    pruned_tflite_metrics = tflite_metrics.TFLiteMetrics(tflite_pruned_model.model_path)

    _test_sparsity(pruned_tflite_metrics, target_sparsity, layers_to_prune)

    assert base_compressed_size >= pruned_compressed_size
