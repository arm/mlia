# SPDX-FileCopyrightText: Copyright 2024-2025, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Helper functions for the rewrite library."""
import math
from typing import Any

import numpy as np
import tf_keras as keras

ACTIVATION_FUNCTION_PRESETS = {
    "relu": {"layer_func": keras.layers.ReLU, "extra_args": {}},
    "relu6": {"layer_func": keras.layers.ReLU, "extra_args": {"max_value": 6}},
    "none": {"layer_func": None, "extra_args": {}},
}
ACTIVATION_FUNCTION_LIST = [
    act_func for act_func, _ in ACTIVATION_FUNCTION_PRESETS.items()
]


def get_activation_function(
    activation: str = "relu",
) -> tuple[type, dict]:
    """Get the activation function from a key."""
    if activation not in ACTIVATION_FUNCTION_LIST:
        raise KeyError(
            "Expected activation function to be "
            f"in {ACTIVATION_FUNCTION_LIST}, found {activation}"
        )
    activation_function = ACTIVATION_FUNCTION_PRESETS[activation]["layer_func"]
    activation_function_extra_args = ACTIVATION_FUNCTION_PRESETS[activation][
        "extra_args"
    ]
    return activation_function, activation_function_extra_args


def compute_conv2d_parameters(  # pylint: disable=dangerous-default-value
    input_shape: np.ndarray,
    output_shape: np.ndarray,
    kernel_size_input: list[int] = [3, 3],
) -> dict[str, Any]:
    """Compute needed kernel size and strides for a given input and output_shape."""
    input_shape = input_shape.tolist()
    output_shape = output_shape.tolist()
    assert len(kernel_size_input) == 2, "Kernel size should have 2 entries"
    assert len(input_shape) == 3
    assert len(output_shape) == 3
    kernel_size = tuple(kernel_size_input)
    num_filters = output_shape[-1]
    padding = "valid"
    stride_h = round(input_shape[0] / output_shape[0])
    check_output_size_h = math.floor((input_shape[0] - kernel_size[0]) / stride_h) + 1
    stride_w = round(input_shape[1] / output_shape[1])
    check_output_size_w = math.floor((input_shape[1] - kernel_size[1]) / stride_w) + 1
    if check_output_size_h != output_shape[0] or check_output_size_w != output_shape[1]:
        padding = "same"
    return {
        "filters": num_filters,
        "kernel_size": kernel_size,
        "padding": padding,
        "strides": (stride_h, stride_w),
    }
