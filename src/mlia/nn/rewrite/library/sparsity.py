# SPDX-FileCopyrightText: Copyright 2024, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Rewrite functions used to return layers ready for sparse pruning."""
from typing import Any

import tensorflow_model_optimization as tfmot
from keras.api._v2 import keras  # Temporary workaround for now: MLIA-1107

from mlia.nn.rewrite.library.helper_functions import compute_conv2d_parameters


def fc_sparsity_rewrite(input_shape: Any, output_shape: Any) -> keras.Model:
    """Fully connected TensorFlow Lite model ready for sparse pruning."""
    model = tfmot.sparsity.keras.prune_low_magnitude(
        to_prune=keras.Sequential(
            [
                keras.layers.InputLayer(input_shape=input_shape),
                keras.layers.Reshape([-1]),
                keras.layers.Dense(output_shape),
            ]
        ),
        sparsity_m_by_n=(2, 4),
    )

    return model


def conv2d_sparsity_rewrite(input_shape: Any, output_shape: Any) -> keras.Model:
    """Conv2d TensorFlow Lite model ready for sparse pruning."""
    conv2d_parameters = compute_conv2d_parameters(
        input_shape=input_shape, output_shape=output_shape
    )
    model = tfmot.sparsity.keras.prune_low_magnitude(
        to_prune=keras.Sequential(
            [
                keras.layers.InputLayer(input_shape=input_shape),
                keras.layers.Conv2D(**conv2d_parameters),
                keras.layers.BatchNormalization(),
                keras.layers.ReLU(),
            ]
        ),
        sparsity_m_by_n=(2, 4),
    )

    return model
