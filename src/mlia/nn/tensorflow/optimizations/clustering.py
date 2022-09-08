# SPDX-FileCopyrightText: Copyright 2022, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""
Contains class Clusterer that clusters unique weights per layer to a specified number.

In order to do this, we need to have a base model and corresponding training data.
We also have to specify a subset of layers we want to cluster. For more details,
please refer to the documentation for TensorFlow Model Optimization Toolkit.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import tensorflow as tf
import tensorflow_model_optimization as tfmot
from tensorflow_model_optimization.python.core.clustering.keras.experimental import (  # pylint: disable=no-name-in-module
    cluster as experimental_cluster,
)

from mlia.nn.tensorflow.optimizations.common import Optimizer
from mlia.nn.tensorflow.optimizations.common import OptimizerConfiguration


@dataclass
class ClusteringConfiguration(OptimizerConfiguration):
    """Clustering configuration."""

    optimization_target: int
    layers_to_optimize: list[str] | None = None

    def __str__(self) -> str:
        """Return string representation of the configuration."""
        return f"clustering: {self.optimization_target}"


class Clusterer(Optimizer):
    """
    Clusterer class.

    Used to cluster a model to a specified number of unique weights per layer.

    Sample usage:
        clusterer = Clusterer(
            base_model,
            optimizer_configuration)

    clusterer.apply_clustering()
    clustered_model = clusterer.get_model()
    """

    def __init__(
        self, model: tf.keras.Model, optimizer_configuration: ClusteringConfiguration
    ):
        """Init Clusterer instance."""
        self.model = model
        self.optimizer_configuration = optimizer_configuration

    def optimization_config(self) -> str:
        """Return string representation of the optimization config."""
        return str(self.optimizer_configuration)

    def _setup_clustering_params(self) -> dict[str, Any]:
        CentroidInitialization = tfmot.clustering.keras.CentroidInitialization
        return {
            "number_of_clusters": self.optimizer_configuration.optimization_target,
            "cluster_centroids_init": CentroidInitialization.LINEAR,
            "preserve_sparsity": True,
        }

    def _apply_clustering_to_layer(
        self, layer: tf.keras.layers.Layer
    ) -> tf.keras.layers.Layer:
        layers_to_optimize = self.optimizer_configuration.layers_to_optimize
        assert layers_to_optimize, "List of the layers to optimize is empty"

        if layer.name not in layers_to_optimize:
            return layer

        clustering_params = self._setup_clustering_params()
        return experimental_cluster.cluster_weights(layer, **clustering_params)

    def _init_for_clustering(self) -> None:
        # Use `tf.keras.models.clone_model` to apply `apply_clustering_to_layer`
        # to the layers of the model
        if not self.optimizer_configuration.layers_to_optimize:
            clustering_params = self._setup_clustering_params()
            clustered_model = experimental_cluster.cluster_weights(
                self.model, **clustering_params
            )
        else:
            clustered_model = tf.keras.models.clone_model(
                self.model, clone_function=self._apply_clustering_to_layer
            )

        self.model = clustered_model

    def _strip_clustering(self) -> None:
        self.model = tfmot.clustering.keras.strip_clustering(self.model)

    def apply_optimization(self) -> None:
        """Apply all steps of clustering at once."""
        self._init_for_clustering()
        self._strip_clustering()

    def get_model(self) -> tf.keras.Model:
        """Get model."""
        return self.model
