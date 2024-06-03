# SPDX-FileCopyrightText: Copyright 2024, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Rewrite functions used to return layers ready for sparse pruning."""
from typing import Any

import tensorflow_model_optimization as tfmot
from keras.api._v2 import keras  # Temporary workaround for now: MLIA-1107

from mlia.nn.rewrite.library.helper_functions import compute_conv2d_parameters
from mlia.nn.rewrite.library.helper_functions import get_activation_function


def fc_sparsity_rewrite(
    input_shape: Any, output_shape: Any, sparsity_m: int = 2, sparsity_n: int = 4
) -> keras.Model:
    """Fully connected TensorFlow Lite model ready for sparse pruning."""
    model = tfmot.sparsity.keras.prune_low_magnitude(
        to_prune=keras.Sequential(
            [
                keras.layers.InputLayer(input_shape=input_shape),
                keras.layers.Reshape([-1]),
                keras.layers.Dense(output_shape),
            ]
        ),
        sparsity_m_by_n=(
            sparsity_m,
            sparsity_n,
        ),
    )

    return model


def conv2d_sparsity_rewrite(  # pylint: disable=dangerous-default-value
    input_shape: Any,
    output_shape: Any,
    sparsity_m: int = 2,
    sparsity_n: int = 4,
    activation: str = "relu",
    kernel_size: list[int] = [3, 3],
) -> keras.Model:
    """Conv2d TensorFlow Lite model ready for sparse pruning."""
    conv2d_parameters = compute_conv2d_parameters(
        input_shape=input_shape,
        output_shape=output_shape,
        kernel_size_input=kernel_size,
    )
    activation_function, activation_function_extra_args = get_activation_function(
        activation
    )
    activation_func_found = (
        [activation_function(**activation_function_extra_args)]
        if activation_function
        else []
    )
    model = tfmot.sparsity.keras.prune_low_magnitude(
        to_prune=keras.Sequential(
            [
                keras.layers.InputLayer(input_shape=input_shape),
                keras.layers.Conv2D(**conv2d_parameters),
                keras.layers.BatchNormalization(),
                *activation_func_found,
            ]
        ),
        sparsity_m_by_n=(
            sparsity_m,
            sparsity_n,
        ),
    )
    return model
