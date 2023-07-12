# SPDX-FileCopyrightText: Copyright 2023, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for module mlia.nn.rewrite.graph_edit.record."""
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

from mlia.nn.rewrite.core.extract import ExtractPaths
from mlia.nn.rewrite.core.graph_edit.record import record_model
from mlia.nn.rewrite.core.utils.numpy_tfrecord import numpytf_read


def data_matches_outputs(
    name: str,
    tensor: tf.Tensor,
    model_outputs: list,
    dequantized_output: bool,
) -> bool:
    """Check that the name and the tensor match any of the model outputs."""
    for model_output in model_outputs:
        if model_output["name"] == name:
            # If the name is a match, tensor shape and type have to match!
            tensor_shape = tensor.shape.as_list()
            tensor_type = tensor.dtype.as_numpy_dtype
            return all(
                (
                    tensor_shape == model_output["shape"].tolist(),
                    tensor_type == np.float32
                    if dequantized_output
                    else model_output["dtype"],
                )
            )
    return False


def check_record_model(
    test_tflite_model: Path,
    tmp_path: Path,
    test_tfrecord: Path,
    batch_size: int,
    dequantize_output: bool,
) -> None:
    """Test the function record_model()."""
    output_file = ExtractPaths.tfrec.output(tmp_path)
    record_model(
        input_filename=str(test_tfrecord),
        model_filename=str(test_tflite_model),
        output_filename=str(output_file),
        batch_size=batch_size,
        dequantize_output=dequantize_output,
    )
    output_file = ExtractPaths.tfrec.output(tmp_path, dequantize_output)
    assert output_file.is_file()

    # Now load model and the data and make sure that the written data matches
    # any of the model outputs
    interpreter = tf.lite.Interpreter(str(test_tflite_model))
    model_outputs = interpreter.get_output_details()
    dataset = numpytf_read(str(output_file))
    for data in dataset:
        for name, tensor in data.items():
            assert data_matches_outputs(name, tensor, model_outputs, dequantize_output)


@pytest.mark.parametrize("batch_size", (None, 1, 2))
@pytest.mark.parametrize("dequantize_output", (True, False))
def test_record_model(
    test_tflite_model: Path,
    tmp_path: Path,
    test_tfrecord: Path,
    batch_size: int,
    dequantize_output: bool,
) -> None:
    """Test the function record_model()."""
    check_record_model(
        test_tflite_model, tmp_path, test_tfrecord, batch_size, dequantize_output
    )
