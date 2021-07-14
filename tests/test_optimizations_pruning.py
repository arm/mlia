# Copyright 2021, Arm Ltd.
"""Test for module optimizations/pruning."""
from math import isclose
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
import pytest
import tensorflow as tf
from mlia.optimizations.pruning import Pruner
from mlia.optimizations.pruning import PruningConfiguration
from mlia.utils import general as test_utils
from mlia.utils import tflite_metrics

from tests.utils.generate_keras_model import generate_keras_model


def _get_dataset() -> Tuple[np.array, np.array]:
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), _ = mnist.load_data()
    x_train = x_train / 255.0

    # Use subset of 60000 examples to keep unit test speed fast.
    x_train = x_train[0:1000]
    y_train = y_train[0:1000]

    return x_train, y_train


def _train_model(model: tf.keras.Model) -> None:
    num_epochs = 1

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer="adam", loss=loss_fn, metrics=["accuracy"])

    x_train, y_train = _get_dataset()

    model.fit(x_train, y_train, epochs=num_epochs)


def _test_sparsity_per_layers(
    metrics: tflite_metrics.TFLiteMetrics,
    desired_sparsity: float,
    layers_to_prune: Optional[List[str]],
) -> None:
    sparsity_per_layer = metrics.sparsity_per_layer()

    if layers_to_prune is None:
        layers_to_prune = ["conv1", "conv2"]

    # To make mypy happy.
    assert layers_to_prune is not None

    for name, sparsity in sparsity_per_layer.items():
        if name in layers_to_prune:
            assert isclose(
                sparsity, desired_sparsity, abs_tol=0.01
            ), "Layer '{}' has incorrect sparsity.".format(name)


@pytest.mark.parametrize("target_sparsity", (0.1, 0.9))
@pytest.mark.parametrize("mock_data", (False, True))
@pytest.mark.parametrize("layers_to_prune", (["conv1"], ["conv1", "conv2"], None))
def test_prune_simple_model_fully(
    target_sparsity: int, mock_data: bool, layers_to_prune: Optional[List[str]]
) -> None:
    """Simple mnist test to see if pruning works correctly."""
    x_train, y_train = _get_dataset()
    initial_sparsity = 0.0
    batch_size = 1000
    num_epochs = 10

    base_model = generate_keras_model()
    _train_model(base_model)
    base_model_path = test_utils.save_keras_model(base_model)
    base_compressed_size = tflite_metrics.get_gzipped_file_size(base_model_path)

    tflite_base_model = test_utils.convert_to_tflite(base_model)
    tflite_base_path = test_utils.save_tflite_model(tflite_base_model)
    base_metrics = tflite_metrics.TFLiteMetrics(tflite_base_path)

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
    pruned_model_path = test_utils.save_keras_model(pruned_model)
    pruned_compressed_size = tflite_metrics.get_gzipped_file_size(pruned_model_path)

    tflite_pruned_model = test_utils.convert_to_tflite(pruned_model)
    tflite_pruned_path = test_utils.save_tflite_model(tflite_pruned_model)
    pruned_metrics = tflite_metrics.TFLiteMetrics(tflite_pruned_path)

    _test_sparsity_per_layers(base_metrics, initial_sparsity, layers_to_prune)
    _test_sparsity_per_layers(pruned_metrics, target_sparsity, layers_to_prune)

    assert base_compressed_size > pruned_compressed_size
