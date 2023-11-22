# SPDX-FileCopyrightText: Copyright 2022-2023, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Test for module optimizations/clustering."""
from __future__ import annotations

import math
from pathlib import Path

import pytest
import tensorflow as tf
from flaky import flaky

from mlia.nn.tensorflow.optimizations.clustering import Clusterer
from mlia.nn.tensorflow.optimizations.clustering import ClusteringConfiguration
from mlia.nn.tensorflow.optimizations.pruning import Pruner
from mlia.nn.tensorflow.optimizations.pruning import PruningConfiguration
from mlia.nn.tensorflow.tflite_metrics import ReportClusterMode
from mlia.nn.tensorflow.tflite_metrics import TFLiteMetrics
from mlia.nn.tensorflow.utils import convert_to_tflite
from mlia.nn.tensorflow.utils import save_tflite_model
from tests.utils.common import get_dataset
from tests.utils.common import train_model


def _prune_model(
    model: tf.keras.Model, target_sparsity: float, layers_to_prune: list[str] | None
) -> tf.keras.Model:
    x_train, y_train = get_dataset()
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
    metrics: TFLiteMetrics,
    target_num_clusters: int,
    layers_to_cluster: list[str] | None,
) -> None:
    clustered_uniqueness = metrics.num_unique_weights(
        ReportClusterMode.NUM_CLUSTERS_PER_AXIS
    )

    num_clustered_layers = 0
    for layer_num_clusters in clustered_uniqueness.values():
        if layer_num_clusters[0] <= target_num_clusters:
            num_clustered_layers += 1

    expected_num_clustered_layers = len(layers_to_cluster or clustered_uniqueness)
    assert num_clustered_layers == expected_num_clustered_layers


def _test_sparsity(
    metrics: TFLiteMetrics,
    target_sparsity: float,
    layers_to_cluster: list[str] | None,
) -> None:
    error_margin = 0.03
    pruned_sparsity = metrics.sparsity_per_layer()

    num_sparse_layers = 0
    for layer_sparsity in pruned_sparsity.values():
        if math.isclose(layer_sparsity, target_sparsity, abs_tol=error_margin):
            num_sparse_layers += 1

    # make sure we are having exactly as many sparse layers as we wanted
    expected_num_sparse_layers = len(layers_to_cluster or pruned_sparsity)
    assert num_sparse_layers == expected_num_sparse_layers


# This test fails sporadically for stochastic reasons, due to a threshold not being met.
# Re-running the test will help. We are yet to find a more deterministic approach
# to run the test, and in the meantime we classify it as a known issue.
# Additionally, flaky is (as of 2023) untyped and thus we need to silence the
# warning from mypy.
@flaky  # type: ignore
@pytest.mark.parametrize("target_num_clusters", (32, 4))
@pytest.mark.parametrize("sparsity_aware", (False, True))
@pytest.mark.parametrize("layers_to_cluster", (["conv1"], ["conv1", "conv2"], None))
def test_cluster_simple_model_fully(
    target_num_clusters: int,
    sparsity_aware: bool,
    layers_to_cluster: list[str] | None,
    tmp_path: Path,
    test_keras_model: Path,
) -> None:
    """Simple MNIST test to see if clustering works correctly."""
    target_sparsity = 0.5

    base_model = tf.keras.models.load_model(str(test_keras_model))
    train_model(base_model)

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
    tflite_clustered_model = convert_to_tflite(clustered_model)
    save_tflite_model(tflite_clustered_model, temp_file)
    clustered_tflite_metrics = TFLiteMetrics(str(temp_file))

    _test_num_unique_weights(
        clustered_tflite_metrics, target_num_clusters, layers_to_cluster
    )

    if sparsity_aware:
        _test_sparsity(clustered_tflite_metrics, target_sparsity, layers_to_cluster)
