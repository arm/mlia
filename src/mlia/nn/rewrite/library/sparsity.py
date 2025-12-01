# SPDX-FileCopyrightText: Copyright 2024-2025, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Rewrite functions used to return layers ready for sparse pruning."""
from __future__ import annotations

from typing import Any

import tensorflow_model_optimization as tfmot
import tf_keras as keras

from mlia.nn.rewrite.library.helper_functions import compute_conv2d_parameters
from mlia.nn.rewrite.library.helper_functions import get_activation_function


def fc_sparsity_unstructured_rewrite(
    input_shape: Any,
    output_shape: Any,
    initial_sparsity: float = 0.5,
    final_sparsity: float = 0.5,
    begin_step: int = 0,
    end_step: int = 48000,
) -> keras.Model:
    """Fully connected TensorFlow Lite model ready for unstructured sparse pruning."""
    model = tfmot.sparsity.keras.prune_low_magnitude(
        to_prune=keras.Sequential(
            [
                keras.layers.InputLayer(input_shape=input_shape),
                keras.layers.Reshape([-1]),
                keras.layers.Dense(output_shape),
            ]
        ),
        pruning_schedule=tfmot.sparsity.keras.PolynomialDecay(
            initial_sparsity=initial_sparsity,
            final_sparsity=final_sparsity,
            begin_step=begin_step,
            end_step=end_step,
        ),
    )

    return model


def conv2d_sparsity_unstructured_rewrite(  # pylint: disable=dangerous-default-value, too-many-arguments
    input_shape: Any,
    output_shape: Any,
    initial_sparsity: float = 0.5,
    final_sparsity: float = 0.5,
    begin_step: int = 0,
    end_step: int = 48000,
    activation: str = "relu",
    kernel_size: list[int] = [3, 3],
    layer_type: type[keras.layers.Layer] = keras.layers.Conv2D,
) -> keras.Model:
    """Conv2d TensorFlow Lite model ready for unstructured sparse pruning."""
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
                layer_type(**conv2d_parameters),
                keras.layers.BatchNormalization(),
                *activation_func_found,
            ]
        ),
        pruning_schedule=tfmot.sparsity.keras.PolynomialDecay(
            initial_sparsity=initial_sparsity,
            final_sparsity=final_sparsity,
            begin_step=begin_step,
            end_step=end_step,
        ),
    )

    return model


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
    layer_type: type[keras.layers.Layer] = keras.layers.Conv2D,
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
    activation_func_found = (  # pylint: disable=duplicate-code
        [activation_function(**activation_function_extra_args)]
        if activation_function
        else []
    )
    model = tfmot.sparsity.keras.prune_low_magnitude(
        to_prune=keras.Sequential(
            [
                keras.layers.InputLayer(input_shape=input_shape),
                layer_type(**conv2d_parameters),
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
