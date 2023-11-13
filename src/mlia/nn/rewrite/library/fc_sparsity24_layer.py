# SPDX-FileCopyrightText: Copyright 2024, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Example rewrite with one fully connected 2:4 sparsity layer."""
from typing import Any

import tensorflow_model_optimization as tfmot
from keras.api._v2 import keras  # Temporary workaround for now: MLIA-1107


def get_keras_model(input_shape: Any, output_shape: Any) -> keras.Model:
    """Generate TensorFlow Lite model for rewrite."""
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
