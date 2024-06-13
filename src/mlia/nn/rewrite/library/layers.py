# SPDX-FileCopyrightText: Copyright 2023-2024, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Rewrite function used to return regular layers."""
from typing import Any

from keras.api._v2 import keras  # Temporary workaround for now: MLIA-1107

from mlia.nn.rewrite.library.helper_functions import compute_conv2d_parameters
from mlia.nn.rewrite.library.helper_functions import get_activation_function


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


def conv2d_rewrite(  # pylint: disable=dangerous-default-value
    input_shape: Any,
    output_shape: Any,
    activation: str = "relu",
    kernel_size: list[int] = [3, 3],
    layer_type: type[keras.layers.Layer] = keras.layers.Conv2D,
) -> keras.Model:
    """Fully connected TensorFlow Lite model for rewrite."""
    conv2d_parameters = compute_conv2d_parameters(
        input_shape=input_shape,
        output_shape=output_shape,
        kernel_size_input=kernel_size,
    )
    activation_function, activation_function_extra_args = get_activation_function(
        activation
    )
    activation_func_found = (  # pylint: disable=duplicate-code
        [activation_function(**activation_function_extra_args)]
        if activation_function
        else []
    )
    model = keras.Sequential(
        (
            keras.layers.InputLayer(input_shape=input_shape),
            layer_type(**conv2d_parameters),
            keras.layers.BatchNormalization(),
            *activation_func_found,
        )
    )
    return model
