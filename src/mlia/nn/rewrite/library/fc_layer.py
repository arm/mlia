# SPDX-FileCopyrightText: Copyright 2023, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Example rewrite with one fully connected layer."""
from typing import Any

import tensorflow as tf


def get_keras_model(input_shape: Any, output_shape: Any) -> tf.keras.Model:
    """Generate TensorFlow Lite model for rewrite."""
    model = tf.keras.Sequential(
        (
            tf.keras.layers.InputLayer(input_shape=input_shape),
            tf.keras.layers.Reshape([-1]),
            tf.keras.layers.Dense(output_shape),
        )
    )
    return model
