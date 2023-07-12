# SPDX-FileCopyrightText: Copyright 2023, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Contains functionality for quantization and de-quantization."""
from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
import tensorflow as tf


@dataclass
class QuantizationParameters:
    """Collection of TensorFlow Lite quantization parameters.

    Can be directly initialized from TensorFlow Lite tensor details, e.g.

        ```
        QuantizationParameters(
            **interpreter.get_input_details()[0]["quantization_parameters"]
        )
        ```
    """

    scales: np.ndarray
    zero_points: np.ndarray
    quantized_dimension: int


def is_quantized(quant_params: QuantizationParameters) -> bool:
    """Check if the quantization parameters describe a quantized tensor."""
    return quant_params.scales.size > 0


def dequantize(
    quantized_tensor: np.ndarray | tf.Tensor, quant_params: QuantizationParameters
) -> np.ndarray:
    """De-quantize the input tensor using the given quantization parameters."""
    assert isinstance(quantized_tensor, (tf.Tensor, np.ndarray))
    assert (
        not isinstance(quantized_tensor, tf.Tensor)
        or quantized_tensor.dtype.is_quantized
    ) and (
        not isinstance(quantized_tensor, np.ndarray)
        or issubclass(quantized_tensor.dtype.type, np.integer)
    ), (
        f"Input tensor for de-quantization is of type {quantized_tensor.dtype}, "
        "but it must be int."
    )

    dequantized_tensor = np.subtract(
        quantized_tensor, quant_params.zero_points, dtype=np.float32
    )
    dequantized_tensor = np.multiply(
        dequantized_tensor, quant_params.scales, dtype=np.float32
    )
    return dequantized_tensor


def quantize(
    tensor: np.ndarray | tf.Tensor, quant_params: QuantizationParameters
) -> np.ndarray:
    """Quantize the given float input tensor to int8."""
    assert isinstance(tensor, (tf.Tensor, np.ndarray))
    assert (not isinstance(tensor, tf.Tensor) or tensor.dtype.is_floating) and (
        not isinstance(tensor, np.ndarray) or issubclass(tensor.dtype.type, np.floating)
    ), f"Input tensor for quantization is of type {tensor.dtype}, but it must be float."

    quantized_tensor = (tensor / quant_params.scales) + quant_params.zero_points
    quantized_tensor = np.clip(  # type: ignore
        quantized_tensor, -128, 127, dtype=np.int8, casting="unsafe"
    )
    return cast(np.ndarray, quantized_tensor)
