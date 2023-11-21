# SPDX-FileCopyrightText: Copyright 2022-2023, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Test for module utils/test_utils."""
import re
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

from mlia.nn.tensorflow.tflite_convert import convert_to_tflite
from mlia.nn.tensorflow.utils import check_tflite_datatypes
from mlia.nn.tensorflow.utils import get_tf_tensor_shape
from mlia.nn.tensorflow.utils import get_tflite_model_type_map
from mlia.nn.tensorflow.utils import is_keras_model
from mlia.nn.tensorflow.utils import is_tflite_model
from mlia.nn.tensorflow.utils import save_keras_model


def test_save_keras_model(tmp_path: Path, test_keras_model: Path) -> None:
    """Test saving Keras model."""
    keras_model = tf.keras.models.load_model(str(test_keras_model))

    temp_file = tmp_path / "test_model_saving.h5"
    save_keras_model(keras_model, temp_file)
    loaded_model = tf.keras.models.load_model(temp_file)

    assert loaded_model.summary() == keras_model.summary()


def test_save_tflite_model(tmp_path: Path, test_keras_model: Path) -> None:
    """Test saving TensorFlow Lite model."""
    keras_model = tf.keras.models.load_model(str(test_keras_model))

    temp_file = tmp_path / "test_model_saving.tflite"
    convert_to_tflite(keras_model, output_path=temp_file)

    interpreter = tf.lite.Interpreter(model_path=str(temp_file))
    assert interpreter


@pytest.mark.parametrize(
    "model_path, expected_result",
    [
        [Path("sample_model.tflite"), True],
        [Path("strange_model.tflite.tfl"), False],
        [Path("sample_model.h5"), False],
        [Path("sample_model"), False],
    ],
)
def test_is_tflite_model(model_path: Path, expected_result: bool) -> None:
    """Test function is_tflite_model."""
    result = is_tflite_model(model_path)
    assert result == expected_result


@pytest.mark.parametrize(
    "model_path, expected_result",
    [
        [Path("sample_model.h5"), True],
        [Path("strange_model.h5.keras"), False],
        [Path("sample_model.tflite"), False],
        [Path("sample_model"), False],
    ],
)
def test_is_keras_model(model_path: Path, expected_result: bool) -> None:
    """Test function is_keras_model."""
    result = is_keras_model(model_path)
    assert result == expected_result


def test_get_tf_tensor_shape(test_tf_model: Path) -> None:
    """Test get_tf_tensor_shape with test model."""
    assert get_tf_tensor_shape(str(test_tf_model)) == [1, 28, 28, 1]


def test_tflite_model_type_map(
    test_tflite_model_fp32: Path, test_tflite_model: Path
) -> None:
    """Test the model type map function."""
    assert get_tflite_model_type_map(test_tflite_model_fp32) == {
        "serving_default_input:0": np.float32
    }
    assert get_tflite_model_type_map(test_tflite_model) == {
        "serving_default_input:0": np.int8
    }


def test_check_tflite_datatypes(
    test_tflite_model_fp32: Path, test_tflite_model: Path
) -> None:
    """Test the model type map function."""
    check_tflite_datatypes(test_tflite_model_fp32, np.float32)
    check_tflite_datatypes(test_tflite_model, np.int8)

    with pytest.raises(
        Exception,
        match=re.escape(
            "unexpected data types: ['float32']. Only ['int8'] are allowed"
        ),
    ):
        check_tflite_datatypes(test_tflite_model_fp32, np.int8)
