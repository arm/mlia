# Copyright 2021, Arm Ltd.
"""Test for module optimizations/clustering."""
from math import isclose
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
import pytest
import tensorflow as tf
from mlia.optimizations.clustering import Clusterer
from mlia.optimizations.clustering import ClusteringConfiguration
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


def _prune_model(
    model: tf.keras.Model, target_sparsity: float, layers_to_prune: Optional[List[str]]
) -> tf.keras.Model:
    x_train, y_train = _get_dataset()
    batch_size = 1000
    num_epochs = 10

    pruner = Pruner(
        model,
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

    return pruned_model


def _test_sparsity_per_layers(
    metrics: tflite_metrics.TFLiteMetrics,
    desired_sparsity: float,
    layers_to_prune: Optional[List[str]],
) -> None:
    sparsity_per_layer = metrics.sparsity_per_layer()
    if layers_to_prune is None:
        layers_to_prune = ["conv1", "conv2"]
    assert layers_to_prune is not None
    for name, sparsity in sparsity_per_layer.items():
        if name in layers_to_prune:
            assert isclose(
                sparsity, desired_sparsity, abs_tol=0.01
            ), "Layer '{}' has incorrect sparsity.".format(name)


@pytest.mark.parametrize("sparsity_aware", (False, True))
@pytest.mark.parametrize("target_num_clusters", (32, 4))
@pytest.mark.parametrize("layers_to_cluster", (["conv1"], ["conv1", "conv2"], None))
def test_cluster_simple_model_fully(
    target_num_clusters: int,
    sparsity_aware: bool,
    layers_to_cluster: Optional[List[str]],
) -> None:
    """Simple mnist test to see if clustering works correctly."""
    target_sparsity = 0.5

    base_model = generate_keras_model()
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
        ClusteringConfiguration(
            target_num_clusters,
            layers_to_cluster,
        ),
    )
    clusterer.apply_optimization()
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
