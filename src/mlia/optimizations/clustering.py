# Copyright 2021, Arm Ltd.
"""
Contains class Clusterer that clusters unique weights per layer to a specified number.

In order to do this, we need to have a base model and corresponding training data.
We also have to specify a subset of layers we want to cluster.
"""
from typing import List
from typing import Optional

import tensorflow as tf
import tensorflow_model_optimization as tfmot
from mlia.optimizations.common import Optimizer
from mlia.optimizations.common import OptimizerConfiguration
from tensorflow_model_optimization.python.core.clustering.keras.experimental import (
    cluster as experimental_cluster,
)


class ClusteringConfiguration(OptimizerConfiguration):
    """Clustering configuration."""

    def __init__(
        self,
        optimization_target: int,
        layers_to_optimize: Optional[List[str]] = None,
    ):
        """Init clustering configuration."""
        self.optimization_target = optimization_target
        self.layers_to_optimize = layers_to_optimize

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

    def _setup_clustering_params(self) -> dict:
        CentroidInitialization = tfmot.clustering.keras.CentroidInitialization

        clustering_params = {
            "number_of_clusters": self.optimizer_configuration.optimization_target,
            "cluster_centroids_init": CentroidInitialization.LINEAR,
            "preserve_sparsity": True,
        }

        return clustering_params

    def _apply_clustering_to_layer(
        self, layer: tf.keras.layers.Layer
    ) -> tf.keras.layers.Layer:
        clustering_params = self._setup_clustering_params()

        # To make mypy happy.
        assert self.optimizer_configuration.layers_to_optimize is not None

        if layer.name in self.optimizer_configuration.layers_to_optimize:
            layer = experimental_cluster.cluster_weights(layer, **clustering_params)

        return layer

    def _init_for_clustering(self) -> None:
        # Use `tf.keras.models.clone_model` to apply `apply_clustering_to_layer`
        # to the layers of the model.
        if self.optimizer_configuration.layers_to_optimize is None:
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
