# SPDX-FileCopyrightText: Copyright 2024, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Example rewrite with one fully connected clustered layer."""
from typing import Any

import tensorflow_model_optimization as tfmot
from keras.api._v2 import keras  # Temporary workaround for now: MLIA-1107

from mlia.nn.rewrite.library.helper_functions import compute_conv2d_parameters


def fc_clustering_rewrite(input_shape: Any, output_shape: Any) -> keras.Model:
    """Generate TensorFlow Lite model for clustering rewrite."""
    rewrite_params = {
        "number_of_clusters": 4,
        "cluster_centroids_init": tfmot.clustering.keras.CentroidInitialization.LINEAR,
    }
    model = tfmot.clustering.keras.cluster_weights(
        to_cluster=keras.Sequential(
            [
                keras.layers.InputLayer(input_shape=input_shape),
                keras.layers.Flatten(),
                keras.layers.Dense(units=output_shape),
            ]
        ),
        **rewrite_params
    )
    return model


def conv2d_clustering_rewrite(input_shape: Any, output_shape: Any) -> keras.Model:
    """Generate TensorFlow Lite model for clustering rewrite."""
    rewrite_params = {
        "number_of_clusters": 4,
        "cluster_centroids_init": tfmot.clustering.keras.CentroidInitialization.LINEAR,
    }
    conv2d_parameters = compute_conv2d_parameters(
        input_shape=input_shape, output_shape=output_shape
    )
    model = tfmot.clustering.keras.cluster_weights(
        to_cluster=keras.Sequential(
            [
                keras.layers.InputLayer(input_shape=input_shape),
                keras.layers.Conv2D(**conv2d_parameters),
                keras.layers.BatchNormalization(),
                keras.layers.ReLU(),
            ]
        ),
        **rewrite_params
    )
    return model
