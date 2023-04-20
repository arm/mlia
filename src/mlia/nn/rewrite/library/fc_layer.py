# SPDX-FileCopyrightText: Copyright 2023, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Example rewrite with one fully connected layer."""
from typing import Any

import tensorflow as tf


def get_keras_model(input_shape: Any, output_shape: Any) -> tf.keras.Model:
    """Generate tflite model for rewrite."""
    input_tensor = tf.keras.layers.Input(
        shape=input_shape, name="MbileNet/avg_pool/AvgPool"
    )
    output_tensor = tf.keras.layers.Dense(output_shape, name="MobileNet/fc1/BiasAdd")(
        input_tensor
    )
    model = tf.keras.Model(input_tensor, output_tensor)
    return model
