# SPDX-FileCopyrightText: Copyright 2023, 2025, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for module optimizations/quantization."""
from __future__ import annotations

from itertools import chain
from pathlib import Path
from typing import Generator

import numpy as np
from numpy import isclose

from mlia.nn.tensorflow.config import TFLiteModel
from mlia.nn.tensorflow.optimizations.quantization import dequantize
from mlia.nn.tensorflow.optimizations.quantization import is_quantized
from mlia.nn.tensorflow.optimizations.quantization import QuantizationParameters
from mlia.nn.tensorflow.optimizations.quantization import quantize


def model_io_quant_params(model_path: Path) -> Generator:
    """Generate QuantizationParameters for all model inputs and outputs."""
    model = TFLiteModel(model_path=model_path)
    for details in chain(model.input_details, model.output_details):
        yield QuantizationParameters(**details["quantization_parameters"])


def test_is_quantized(test_tflite_model: Path) -> None:
    """Test function is_quantized() with a quantized model."""
    for quant_params in model_io_quant_params(test_tflite_model):
        assert is_quantized(quant_params)


def test_is_not_quantized(test_tflite_model_fp32: Path) -> None:
    """Test function is_quantized() with an unquantized model."""
    for quant_params in model_io_quant_params(test_tflite_model_fp32):
        assert not is_quantized(quant_params)


def test_quantize() -> None:
    """Test function quantize()."""
    ref_dequant = np.array((0.0, 0.1, 0.2, 0.3))
    ref_quant = np.array((0, 10, 20, 30), dtype=np.int8)
    quant_params = QuantizationParameters(
        scales=np.array([0.01]), zero_points=np.array([0.0]), quantized_dimension=0
    )

    quant = quantize(ref_dequant, quant_params)
    assert quant.dtype == np.int8
    assert np.all(quant == ref_quant)

    dequant = dequantize(quant, quant_params)
    assert dequant.dtype == np.float32
    assert np.all(isclose(dequant, ref_dequant, atol=0.03))
