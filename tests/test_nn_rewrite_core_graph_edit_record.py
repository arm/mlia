# SPDX-FileCopyrightText: Copyright 2023, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for module mlia.nn.rewrite.graph_edit.record."""
from pathlib import Path

import pytest
import tensorflow as tf

from mlia.nn.rewrite.core.graph_edit.record import record_model
from mlia.nn.rewrite.core.utils.numpy_tfrecord import NumpyTFReader


@pytest.mark.parametrize("batch_size", (None, 1, 2))
def test_record_model(
    test_tflite_model: Path,
    tmp_path: Path,
    test_tfrecord: Path,
    batch_size: int,
) -> None:
    """Test the function record_model()."""
    output_file = tmp_path / "out.tfrecord"
    record_model(
        input_filename=str(test_tfrecord),
        model_filename=str(test_tflite_model),
        output_filename=str(output_file),
        batch_size=batch_size,
    )
    assert output_file.is_file()

    def data_matches_outputs(name: str, tensor: tf.Tensor, model_outputs: list) -> bool:
        """Check that the name and the tensor match any of the model outputs."""
        for model_output in model_outputs:
            if model_output["name"] == name:
                # If the name is a match, tensor shape and type have to match!
                tensor_shape = tensor.shape.as_list()
                tensor_type = tensor.dtype.as_numpy_dtype
                return all(
                    (
                        tensor_shape == model_output["shape"].tolist(),
                        tensor_type == model_output["dtype"],
                    )
                )
        return False

    # Now load model and the data and make sure that the written data matches
    # any of the model outputs
    interpreter = tf.lite.Interpreter(str(test_tflite_model))
    model_outputs = interpreter.get_output_details()
    dataset = NumpyTFReader(str(output_file))
    for data in dataset:
        for name, tensor in data.items():
            assert data_matches_outputs(name, tensor, model_outputs)
