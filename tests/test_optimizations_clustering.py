# Copyright 2021, Arm Ltd.
"""Test for module optimizations/clustering."""
import pathlib
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
    x_train = x_train[0:1]
    y_train = y_train[0:1]

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
    batch_size = 1
    num_epochs = 1

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


def _test_num_unique_weights(
    metrics: tflite_metrics.TFLiteMetrics,
    target_num_clusters: int,
    layers_to_cluster: Optional[List[str]],
) -> None:
    clustered_uniqueness_dict = metrics.num_unique_weights(
        tflite_metrics.ReportClusterMode.NUM_CLUSTERS_PER_AXIS
    )
    num_clustered_layers = 0
    num_optimizable_layers = len(clustered_uniqueness_dict)
    if layers_to_cluster:
        expected_num_clustered_layers = len(layers_to_cluster)
    else:
        expected_num_clustered_layers = num_optimizable_layers
    for layer_name in clustered_uniqueness_dict:
        # the +1 is there temporarily because of a bug that's been fixed
        # but the fix hasn't been merged yet.
        # Will need to be removed in the future.
        if clustered_uniqueness_dict[layer_name][0] <= (target_num_clusters + 1):
            num_clustered_layers = num_clustered_layers + 1
    # make sure we are having exactly as many clustered layers as we wanted
    assert num_clustered_layers == expected_num_clustered_layers


def _test_sparsity(
    metrics: tflite_metrics.TFLiteMetrics,
    target_sparsity: float,
    layers_to_cluster: Optional[List[str]],
) -> None:
    pruned_sparsity_dict = metrics.sparsity_per_layer()
    num_sparse_layers = 0
    num_optimizable_layers = len(pruned_sparsity_dict)
    error_margin = 0.03
    if layers_to_cluster:
        expected_num_sparse_layers = len(layers_to_cluster)
    else:
        expected_num_sparse_layers = num_optimizable_layers
    for layer_name in pruned_sparsity_dict:
        if abs(pruned_sparsity_dict[layer_name] - target_sparsity) < error_margin:
            num_sparse_layers = num_sparse_layers + 1
    # make sure we are having exactly as many sparse layers as we wanted
    assert num_sparse_layers == expected_num_sparse_layers


@pytest.mark.skip(reason="Test fails randomly, further investigation is needed")
@pytest.mark.parametrize("target_num_clusters", (32, 4))
@pytest.mark.parametrize("sparsity_aware", (False, True))
@pytest.mark.parametrize("layers_to_cluster", (["conv1"], ["conv1", "conv2"], None))
def test_cluster_simple_model_fully(
    target_num_clusters: int,
    sparsity_aware: bool,
    layers_to_cluster: Optional[List[str]],
    tmp_path: pathlib.Path,
) -> None:
    """Simple mnist test to see if clustering works correctly."""
    target_sparsity = 0.5

    base_model = generate_keras_model()
    _train_model(base_model)

    if sparsity_aware:
        base_model = _prune_model(base_model, target_sparsity, layers_to_cluster)

    clusterer = Clusterer(
        base_model,
        ClusteringConfiguration(
            target_num_clusters,
            layers_to_cluster,
        ),
    )
    clusterer.apply_optimization()
    clustered_model = clusterer.get_model()

    temp_file = tmp_path / "test_cluster_simple_model_fully_after.tflite"
    tflite_clustered_model = test_utils.convert_to_tflite(clustered_model)
    test_utils.save_tflite_model(tflite_clustered_model, temp_file)
    clustered_tflite_metrics = tflite_metrics.TFLiteMetrics(str(temp_file))

    _test_num_unique_weights(
        clustered_tflite_metrics, target_num_clusters, layers_to_cluster
    )

    if sparsity_aware:
        _test_sparsity(clustered_tflite_metrics, target_sparsity, layers_to_cluster)
