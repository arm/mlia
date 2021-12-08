# Copyright 2021, Arm Ltd.
"""Test for module utils/test_utils."""
import shutil
from pathlib import Path
from typing import Optional

import pytest
import tensorflow as tf
from mlia.utils.general import convert_to_tflite
from mlia.utils.general import extract_if_archived
from mlia.utils.general import get_tf_tensor_shape
from mlia.utils.general import is_keras_model
from mlia.utils.general import is_tflite_model
from mlia.utils.general import save_keras_model
from mlia.utils.general import save_tflite_model


def test_convert_to_tflite(test_models_path: Path) -> None:
    """Test converting keras model to tflite."""
    model_path = str(test_models_path / "simple_model.h5")
    keras_model = tf.keras.models.load_model(model_path)
    tflite_model = convert_to_tflite(keras_model)

    assert tflite_model


def test_save_keras_model(tmp_path: Path, test_models_path: Path) -> None:
    """Test saving keras model."""
    model_path = str(test_models_path / "simple_model.h5")
    keras_model = tf.keras.models.load_model(model_path)

    temp_file = tmp_path / "test_model_saving.h5"
    save_keras_model(keras_model, temp_file)
    loaded_model = tf.keras.models.load_model(temp_file)

    assert loaded_model.summary() == keras_model.summary()


def test_save_tflite_model(tmp_path: Path, test_models_path: Path) -> None:
    """Test saving tflite model."""
    model_path = str(test_models_path / "simple_model.h5")
    keras_model = tf.keras.models.load_model(model_path)

    tflite_model = convert_to_tflite(keras_model)

    temp_file = tmp_path / "test_model_saving.tflite"
    save_tflite_model(tflite_model, temp_file)

    interpreter = tf.lite.Interpreter(model_path=str(temp_file))
    assert interpreter


@pytest.mark.parametrize(
    "fmt",
    [None, "zip", "tar", "gztar", "bztar", "xztar"],
)
def test_extract_if_archived(
    fmt: Optional[str], tmp_path: Path, test_models_path: Path
) -> None:
    """Test function extract_if_archived."""
    model_name = "keras_model_simple_mnist_convnet_non_quantized"
    model_path = test_models_path / model_name

    if fmt:
        model_path = Path(shutil.make_archive(str(tmp_path), fmt, model_path))
        assert is_keras_model(model_path) is False

    assert is_keras_model(extract_if_archived(model_path, save_path=tmp_path))


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


def test_get_tf_tensor_shape(test_models_path: Path) -> None:
    """Test get_tf_tensor_shape with tf_model_simple_3_layers_model."""
    model_path = test_models_path / "tf_model_simple_3_layers_model"
    assert get_tf_tensor_shape(str(model_path)) == [1, 1]
