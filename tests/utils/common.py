# SPDX-FileCopyrightText: Copyright 2022-2024, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Common test utils module."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from keras.api._v2 import keras  # Temporary workaround for now: MLIA-1107


def get_dataset() -> tuple[np.ndarray, np.ndarray]:
    """Return sample dataset."""
    mnist = keras.datasets.mnist
    (x_train, y_train), _ = mnist.load_data()
    x_train = x_train / 255.0

    # Use subset of 60000 examples to keep unit test speed fast.
    x_train = x_train[0:1]
    y_train = y_train[0:1]

    return x_train, y_train


def train_model(model: keras.Model) -> None:
    """Train model using sample dataset."""
    num_epochs = 1

    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer="adam", loss=loss_fn, metrics=["accuracy"])

    x_train, y_train = get_dataset()

    model.fit(x_train, y_train, epochs=num_epochs)


def check_expected_permissions(path: Path, expected_permissions_mask: int) -> None:
    """Check expected permissions for the provided path."""
    path_mode = path.stat().st_mode
    permissions_mask = path_mode & 0o777

    assert permissions_mask == expected_permissions_mask
