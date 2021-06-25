"""Test for module optimizations/pruning."""
from math import isclose
from typing import List
from typing import Union

import numpy as np
import pytest
import tensorflow as tf
from mlia.optimizations import tflite_metrics
from mlia.optimizations import utils
from mlia.optimizations.pruning import Pruner
from mlia.optimizations.tflite_metrics import TFLiteMetrics


def _build_model() -> tf.keras.Model:
    """Build a simple CNN model."""
    # Create a dummy model
    keras_model = tf.keras.Sequential(
        [
            tf.keras.Input(shape=(28, 28)),
            tf.keras.layers.Reshape((28, 28, 1)),
            tf.keras.layers.Conv2D(
                filters=12, kernel_size=(3, 3), activation="relu", name="conv1"
            ),
            tf.keras.layers.MaxPool2D(2, 2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(10),
        ]
    )

    return keras_model


def _get_dataset() -> Union[np.array, np.array]:
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
    metrics: TFLiteMetrics, desired_sparsity: float, layers_to_prune: List[str]
) -> None:
    sparsity_per_layer = metrics.sparsity_per_layer()
    for name, sparsity in sparsity_per_layer.items():
        if name in layers_to_prune:
            assert isclose(
                sparsity, desired_sparsity, abs_tol=0.01
            ), "Layer '{}' has incorrect sparsity.".format(name)


@pytest.mark.parametrize("target_sparsity", (0.1, 0.5, 0.9))
def test_prune_simple_model_fully(target_sparsity: int) -> None:
    """Simple mnist test to see if pruning works correctly."""
    x_train, y_train = _get_dataset()
    initial_sparsity = 0.0
    batch_size = 1000
    num_epochs = 10
    layers_to_prune = ["conv1"]

    base_model = _build_model()
    _train_model(base_model)
    base_model_path = utils.save_keras_model(base_model)
    base_compressed_size = tflite_metrics.get_gzipped_file_size(base_model_path)

    tflite_base_model = utils.convert_to_tflite(base_model)
    tflite_base_path = utils.save_tflite_model(tflite_base_model)
    base_metrics = TFLiteMetrics(tflite_base_path)

    pruner = Pruner(
        base_model,
        x_train,
        y_train,
        target_sparsity,
        batch_size,
        num_epochs,
        layers_to_prune,
    )
    pruner.apply_pruning()
    pruned_model = pruner.get_model()
    pruned_model_path = utils.save_keras_model(pruned_model)
    pruned_compressed_size = tflite_metrics.get_gzipped_file_size(pruned_model_path)

    tflite_pruned_model = utils.convert_to_tflite(pruned_model)
    tflite_pruned_path = utils.save_tflite_model(tflite_pruned_model)
    pruned_metrics = TFLiteMetrics(tflite_pruned_path)

    _test_sparsity_per_layers(base_metrics, initial_sparsity, layers_to_prune)
    _test_sparsity_per_layers(pruned_metrics, target_sparsity, layers_to_prune)

    assert base_compressed_size > pruned_compressed_size
