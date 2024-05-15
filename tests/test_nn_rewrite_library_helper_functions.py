# SPDX-FileCopyrightText: Copyright 2024, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for module mlia.nn.rewrite.library.helper_functions."""
from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from keras.api._v2 import keras  # Temporary workaround for now: MLIA-1107

from mlia.nn.rewrite.library.helper_functions import compute_conv2d_parameters


def compute_conv_output(
    input_data: np.ndarray, input_shape: np.ndarray, conv_parameters: dict[str, Any]
) -> np.ndarray:
    """Compute the output of a conv layer for testing."""
    test_model = keras.Sequential(
        [
            keras.layers.InputLayer(input_shape=input_shape),
            keras.layers.Conv2D(**conv_parameters),
        ]
    )
    output = test_model(input_data)
    return np.array(output.shape[1:])


@pytest.mark.parametrize(
    "input_shape, output_shape",
    [
        (np.array([32, 32, 3]), np.array([16, 16, 3])),
        (np.array([32, 32, 3]), np.array([8, 8, 3])),
        (np.array([32, 32, 3]), np.array([8, 16, 3])),
        (np.array([25, 10, 3]), np.array([13, 5, 3])),
        (np.array([25, 10, 3]), np.array([7, 5, 3])),
        (np.array([25, 10, 3]), np.array([6, 4, 3])),
        (np.array([25, 10, 3]), np.array([5, 5, 3])),
    ],
)
def test_compute_conv2d_parameters(
    input_shape: np.ndarray, output_shape: np.ndarray
) -> None:
    """Test to check compute_conv2d_parameters works as expected."""
    conv_parameters = compute_conv2d_parameters(
        input_shape=input_shape, output_shape=output_shape
    )
    computed_output_shape = compute_conv_output(
        np.random.rand(1, *input_shape), input_shape, conv_parameters
    )
    assert np.equal(computed_output_shape, output_shape).all()
