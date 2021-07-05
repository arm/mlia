# Copyright 2021, Arm Ltd.
"""
Contains class Clusterer that clusters unique weights per layer to a specified number.

In order to do this, we need to have a base model and corresponding training data.
We also have to specify a subset of layers we want to cluster.
"""
from typing import List

import tensorflow as tf
import tensorflow_model_optimization as tfmot
from tensorflow_model_optimization.python.core.clustering.keras.experimental import (
    cluster as experimental_cluster,
)


class Clusterer:
    """
    Clusterer class.

    Used to cluster a model to a specified number of unique weights per layer.

    Sample usage:
    clusterer = Clusterer(
        base_model,
        target_num_clusters,
        layers_to_cluster,
    )
    clusterer.apply_clustering()
    clustered_model = clusterer.get_model()
    """

    def __init__(
        self,
        model: tf.keras.Model,
        target_num_clusters: int,
        layers_to_cluster: List[str],
    ):
        """Init Clusterer instance."""
        self.model = model
        self.target_num_clusters = target_num_clusters
        self.layers_to_cluster = layers_to_cluster

    def get_model(self) -> tf.keras.Model:
        """Return the model instance from the clusterer."""
        return self.model

    def _setup_clustering_params(self) -> dict:
        CentroidInitialization = tfmot.clustering.keras.CentroidInitialization

        clustering_params = {
            "number_of_clusters": self.target_num_clusters,
            "cluster_centroids_init": CentroidInitialization.LINEAR,
            "preserve_sparsity": True,
        }

        return clustering_params

    def _apply_clustering_to_layer(
        self, layer: tf.keras.layers.Layer
    ) -> tf.keras.layers.Layer:
        clustering_params = self._setup_clustering_params()

        if layer.name in self.layers_to_cluster:
            layer = experimental_cluster.cluster_weights(layer, **clustering_params)

        return layer

    def _init_for_clustering(self) -> None:
        # Use `tf.keras.models.clone_model` to apply `apply_clustering_to_layer`
        # to the layers of the model.
        clustered_model = tf.keras.models.clone_model(
            self.model, clone_function=self._apply_clustering_to_layer
        )

        self.model = clustered_model

    def _strip_clustering(self) -> None:
        self.model = tfmot.clustering.keras.strip_clustering(self.model)

    def apply_clustering(self) -> None:
        """Apply all steps of clustering at once."""
        self._init_for_clustering()
        self._strip_clustering()
