# SPDX-FileCopyrightText: Copyright 2023-2024, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Rewrite function used to return regular layers."""
from typing import Any

from keras.api._v2 import keras  # Temporary workaround for now: MLIA-1107


def fc_rewrite(input_shape: Any, output_shape: Any) -> keras.Model:
    """Fully connected TensorFlow Lite model for rewrite."""
    model = keras.Sequential(
        (
            keras.layers.InputLayer(input_shape=input_shape),
            keras.layers.Reshape([-1]),
            keras.layers.Dense(output_shape),
        )
    )
    return model
