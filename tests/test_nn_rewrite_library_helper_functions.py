# SPDX-FileCopyrightText: Copyright 2024, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for module mlia.nn.rewrite.library.helper_functions."""
from __future__ import annotations

from contextlib import ExitStack as does_not_raise
from typing import Any

import numpy as np
import pytest
from keras.api._v2 import keras  # Temporary workaround for now: MLIA-1107

from mlia.nn.rewrite.library.helper_functions import ACTIVATION_FUNCTION_LIST
from mlia.nn.rewrite.library.helper_functions import compute_conv2d_parameters
from mlia.nn.rewrite.library.helper_functions import get_activation_function


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
    "input_shape, output_shape, kernel_size",
    [
        (np.array([32, 32, 3]), np.array([16, 16, 3]), [3, 3]),
        (np.array([32, 32, 3]), np.array([8, 8, 3]), [3, 3]),
        (np.array([32, 32, 3]), np.array([8, 16, 3]), [3, 3]),
        (np.array([25, 10, 3]), np.array([13, 5, 3]), [3, 3]),
        (np.array([25, 10, 3]), np.array([7, 5, 3]), [3, 3]),
        (np.array([25, 10, 3]), np.array([6, 4, 3]), [3, 3]),
        (np.array([25, 10, 3]), np.array([5, 5, 3]), [3, 3]),
        (np.array([32, 32, 3]), np.array([16, 16, 3]), [1, 3]),
        (np.array([32, 32, 3]), np.array([16, 16, 3]), [1, 1]),
        (np.array([32, 32, 3]), np.array([16, 16, 3]), [5, 5]),
    ],
)
def test_compute_conv2d_parameters(
    input_shape: np.ndarray, output_shape: np.ndarray, kernel_size: list[int]
) -> None:
    """Test to check compute_conv2d_parameters works as expected."""
    conv_parameters = compute_conv2d_parameters(
        input_shape=input_shape,
        output_shape=output_shape,
        kernel_size_input=kernel_size,
    )
    computed_output_shape = compute_conv_output(
        np.random.rand(1, *input_shape), input_shape, conv_parameters
    )
    assert np.equal(computed_output_shape, output_shape).all()


@pytest.mark.parametrize(
    "activation, expected_function_type, expected_extra_args, expected_error",
    [
        ("relu", keras.layers.ReLU, {}, does_not_raise()),
        ("relu6", keras.layers.ReLU, {"max_value": 6}, does_not_raise()),
        ("none", keras.layers.Identity, {}, does_not_raise()),
        (
            "wrong_key",
            keras.layers.Identity,
            {},
            pytest.raises(
                KeyError,
                match=(
                    "Expected activation function to be "
                    rf"in \{ACTIVATION_FUNCTION_LIST}\, found wrong_key"
                ),
            ),
        ),
    ],
)
def test_get_activation_functions(
    activation: str,
    expected_function_type: type[keras.layers.Layer],
    expected_extra_args: dict,
    expected_error: Any,
) -> None:
    """
    Check the get_activation_function returns
    the expected layer and extra arguments.
    """
    with expected_error:
        activation_function, activation_function_extra_args = get_activation_function(
            activation
        )
        assert isinstance(
            activation_function(**activation_function_extra_args),
            expected_function_type,
        )
        assert expected_extra_args == activation_function_extra_args
