# SPDX-FileCopyrightText: Copyright 2023, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for module mlia.nn.rewrite.utils."""
from pathlib import Path

import pytest
import tensorflow as tf
from tensorflow.lite.python.schema_py_generated import ModelT

from mlia.nn.rewrite.core.utils.utils import load
from mlia.nn.rewrite.core.utils.utils import save
from tests.utils.rewrite import models_are_equal


def test_load_save(test_tflite_model: Path, tmp_path: Path) -> None:
    """Test the load/save functions for TensorFlow Lite models."""
    with pytest.raises(FileNotFoundError):
        load("THIS_IS_NOT_A_REAL_FILE")

    model = load(test_tflite_model)
    assert isinstance(model, ModelT)
    assert model.subgraphs

    output_file = tmp_path / "test.tflite"
    assert not output_file.is_file()
    save(model, output_file)
    assert output_file.is_file()

    model_copy = load(str(output_file))
    assert models_are_equal(model, model_copy)

    # Double check that the TensorFlow Lite Interpreter can still load the file.
    tf.lite.Interpreter(model_path=str(output_file))
