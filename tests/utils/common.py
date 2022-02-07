# Copyright 2021, Arm Ltd.
"""Common test utils module."""
from typing import Tuple

import numpy as np
import tensorflow as tf


def get_dataset() -> Tuple[np.array, np.array]:
    """Return sample dataset."""
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), _ = mnist.load_data()
    x_train = x_train / 255.0

    # Use subset of 60000 examples to keep unit test speed fast.
    x_train = x_train[0:1]
    y_train = y_train[0:1]

    return x_train, y_train


def train_model(model: tf.keras.Model) -> None:
    """Train model using sample dataset."""
    num_epochs = 1

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer="adam", loss=loss_fn, metrics=["accuracy"])

    x_train, y_train = get_dataset()

    model.fit(x_train, y_train, epochs=num_epochs)
