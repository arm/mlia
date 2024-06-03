# SPDX-FileCopyrightText: Copyright 2024, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Rewrite functions used to return layers ready for clustering."""
from typing import Any

import tensorflow_model_optimization as tfmot
from keras.api._v2 import keras  # Temporary workaround for now: MLIA-1107

from mlia.nn.rewrite.library.helper_functions import compute_conv2d_parameters
from mlia.nn.rewrite.library.helper_functions import get_activation_function


def fc_clustering_rewrite(
    input_shape: Any,
    output_shape: Any,
    num_clusters: int = 2,
    cluster_centroids_init: tfmot.clustering.keras.CentroidInitialization = tfmot.clustering.keras.CentroidInitialization(  # pylint: disable=line-too-long
        "CentroidInitialization.LINEAR"
    ),
) -> keras.Model:
    """Fully connected TensorFlow Lite model ready for clustering."""
    rewrite_params = {
        "number_of_clusters": num_clusters,
        "cluster_centroids_init": cluster_centroids_init,
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


def conv2d_clustering_rewrite(  # pylint: disable=dangerous-default-value
    input_shape: Any,
    output_shape: Any,
    num_clusters: int = 2,
    cluster_centroids_init: tfmot.clustering.keras.CentroidInitialization = tfmot.clustering.keras.CentroidInitialization(  # pylint: disable=line-too-long
        "CentroidInitialization.LINEAR"
    ),
    activation: str = "relu",
    kernel_size: list[int] = [3, 3],
) -> keras.Model:
    """Conv2d TensorFlow Lite model ready for clustering."""
    rewrite_params = {
        "number_of_clusters": num_clusters,
        "cluster_centroids_init": cluster_centroids_init,
    }
    conv2d_parameters = compute_conv2d_parameters(
        input_shape=input_shape,
        output_shape=output_shape,
        kernel_size_input=kernel_size,
    )
    activation_function, activation_function_extra_args = get_activation_function(
        activation
    )
    model = tfmot.clustering.keras.cluster_weights(
        to_cluster=keras.Sequential(
            [
                keras.layers.InputLayer(input_shape=input_shape),
                keras.layers.Conv2D(**conv2d_parameters),
                keras.layers.BatchNormalization(),
                activation_function(**activation_function_extra_args),
            ]
        ),
        **rewrite_params
    )
    return model
