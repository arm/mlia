# SPDX-FileCopyrightText: Copyright 2024, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Helper functions for the rewrite library."""
import math
from typing import Any

import numpy as np


def compute_conv2d_parameters(
    input_shape: np.ndarray, output_shape: np.ndarray
) -> dict[str, Any]:
    """Compute needed kernel size and strides for a given input and output_shape."""
    input_shape = input_shape.tolist()
    output_shape = output_shape.tolist()
    assert len(input_shape) == 3
    assert len(output_shape) == 3
    num_filters = (output_shape[-1] - input_shape[-1]) + input_shape[-1]
    padding = "valid"
    kernel_size = (3, 3)
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
