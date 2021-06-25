"""Test for module optimizations/clustering."""
from math import isclose
from typing import List
from typing import Union

import numpy as np
import pytest
import tensorflow as tf
from mlia.optimizations.clustering import Clusterer
from mlia.optimizations.pruning import Pruner

from tests.utils import general as test_utils
from tests.utils import tflite_metrics


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


def _prune_model(
    model: tf.keras.Model, target_sparsity: float, layers_to_prune: List[str]
) -> tf.keras.Model:
    x_train, y_train = _get_dataset()
    batch_size = 1000
    num_epochs = 10

    pruner = Pruner(
        model,
        x_train,
        y_train,
        target_sparsity,
        batch_size,
        num_epochs,
        layers_to_prune,
    )
    pruner.apply_pruning()
    pruned_model = pruner.get_model()

    return pruned_model


def _test_sparsity_per_layers(
    metrics: tflite_metrics.TFLiteMetrics,
    desired_sparsity: float,
    layers_to_prune: List[str],
) -> None:
    sparsity_per_layer = metrics.sparsity_per_layer()
    for name, sparsity in sparsity_per_layer.items():
        if name in layers_to_prune:
            assert isclose(
                sparsity, desired_sparsity, abs_tol=0.01
            ), "Layer '{}' has incorrect sparsity.".format(name)


@pytest.mark.parametrize("sparsity_aware", (False, True))
@pytest.mark.parametrize("target_num_clusters", (32, 16, 8, 4))
def test_cluster_simple_model_fully(
    target_num_clusters: int, sparsity_aware: bool
) -> None:
    """Simple mnist test to see if clustering works correctly."""
    layers_to_cluster = ["conv1"]
    target_sparsity = 0.5

    base_model = _build_model()
    _train_model(base_model)

    if sparsity_aware:
        base_model = _prune_model(base_model, target_sparsity, layers_to_cluster)

    base_model_path = test_utils.save_keras_model(base_model)
    base_compressed_size = tflite_metrics.get_gzipped_file_size(base_model_path)

    tflite_base_model = test_utils.convert_to_tflite(base_model)
    tflite_base_path = test_utils.save_tflite_model(tflite_base_model)
    base_metrics = tflite_metrics.TFLiteMetrics(tflite_base_path)

    base_clusters_per_axis = base_metrics.num_unique_weights(
        tflite_metrics.ReportClusterMode.NUM_CLUSTERS_PER_AXIS
    )

    for key, value in base_clusters_per_axis.items():
        if "conv1" in key:
            assert value[0] > target_num_clusters

    clusterer = Clusterer(
        base_model,
        target_num_clusters,
        layers_to_cluster,
    )
    clusterer.apply_clustering()
    clustered_model = clusterer.get_model()
    clustered_model_path = test_utils.save_keras_model(clustered_model)
    clustered_compressed_size = tflite_metrics.get_gzipped_file_size(
        clustered_model_path
    )

    tflite_clustered_model = test_utils.convert_to_tflite(clustered_model)
    tflite_clustered_path = test_utils.save_tflite_model(tflite_clustered_model)
    clustered_metrics = tflite_metrics.TFLiteMetrics(tflite_clustered_path)

    clustered_clusters_per_axis = clustered_metrics.num_unique_weights(
        tflite_metrics.ReportClusterMode.NUM_CLUSTERS_PER_AXIS
    )

    for key, value in clustered_clusters_per_axis.items():
        if "conv1" in key:
            # Note: <= needed because of a bug.
            # Should be changed once latest stable tensorflow version updated with fix.
            # see: https://github.com/tensorflow/model-optimization/pull/702
            assert value[0] <= target_num_clusters

    if sparsity_aware:
        _test_sparsity_per_layers(clustered_metrics, target_sparsity, layers_to_cluster)

    assert base_compressed_size > clustered_compressed_size
