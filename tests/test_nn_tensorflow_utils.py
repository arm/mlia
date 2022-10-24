# SPDX-FileCopyrightText: Copyright 2022, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Test for module utils/test_utils."""
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

from mlia.nn.tensorflow.utils import convert_to_tflite
from mlia.nn.tensorflow.utils import get_tf_tensor_shape
from mlia.nn.tensorflow.utils import is_keras_model
from mlia.nn.tensorflow.utils import is_tflite_model
from mlia.nn.tensorflow.utils import representative_dataset
from mlia.nn.tensorflow.utils import save_keras_model
from mlia.nn.tensorflow.utils import save_tflite_model


def test_generate_representative_dataset() -> None:
    """Test function for generating representative dataset."""
    dataset = representative_dataset([1, 3, 3], 5)
    data = list(dataset())

    assert len(data) == 5
    for elem in data:
        assert isinstance(elem, list)
        assert len(elem) == 1

        ndarray = elem[0]
        assert ndarray.dtype == np.float32
        assert isinstance(ndarray, np.ndarray)


def test_generate_representative_dataset_wrong_shape() -> None:
    """Test that only shape with batch size=1 is supported."""
    with pytest.raises(Exception, match="Only the input batch_size=1 is supported!"):
        representative_dataset([2, 3, 3], 5)


def test_convert_saved_model_to_tflite(test_tf_model: Path) -> None:
    """Test converting SavedModel to TensorFlow Lite."""
    result = convert_to_tflite(test_tf_model.as_posix())
    assert isinstance(result, bytes)


def test_convert_keras_to_tflite(test_keras_model: Path) -> None:
    """Test converting Keras model to TensorFlow Lite."""
    keras_model = tf.keras.models.load_model(str(test_keras_model))
    result = convert_to_tflite(keras_model)
    assert isinstance(result, bytes)


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

    tflite_model = convert_to_tflite(keras_model)

    temp_file = tmp_path / "test_model_saving.tflite"
    save_tflite_model(tflite_model, temp_file)

    interpreter = tf.lite.Interpreter(model_path=str(temp_file))
    assert interpreter


def test_convert_unknown_model_to_tflite() -> None:
    """Test that unknown model type cannot be converted to TensorFlow Lite."""
    with pytest.raises(
        ValueError, match="Unable to create TensorFlow Lite converter for 123"
    ):
        convert_to_tflite(123)


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
