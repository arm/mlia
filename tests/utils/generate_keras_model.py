# Copyright 2021, Arm Ltd.
"""Generate keras model module."""
import tensorflow as tf


def generate_keras_model() -> tf.keras.Model:
    """Build a simple CNN model."""
    keras_model = tf.keras.Sequential(
        [
            tf.keras.Input(shape=(28, 28)),
            tf.keras.layers.Reshape((28, 28, 1)),
            tf.keras.layers.Conv2D(
                filters=12, kernel_size=(3, 3), activation="relu", name="conv1"
            ),
            tf.keras.layers.Conv2D(
                filters=12, kernel_size=(3, 3), activation="relu", name="conv2"
            ),
            tf.keras.layers.MaxPool2D(2, 2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(10),
        ]
    )

    return keras_model
