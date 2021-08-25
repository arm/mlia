# Copyright 2021, Arm Ltd.
"""Test for module utils/test_utils."""
import pathlib

import tensorflow as tf
from mlia.utils import general as test_utils

from tests.utils.generate_keras_model import generate_keras_model


def test_convert_to_tflite() -> None:
    """Test converting keras model to tflite."""
    tflite_model = test_utils.convert_to_tflite(generate_keras_model())

    assert tflite_model


def test_save_keras_model(tmp_path: pathlib.Path) -> None:
    """Test saving keras model."""
    model = generate_keras_model()
    temp_file = tmp_path / "test_optimization_command.h5"
    test_utils.save_keras_model(model, temp_file)
    loaded_model = tf.keras.models.load_model(temp_file)

    assert loaded_model.summary() == model.summary()


def test_save_tflite_model(tmp_path: pathlib.Path) -> None:
    """Test saving tflite model."""
    temp_file = tmp_path / "test_optimization_command.tflite"
    tflite_model = test_utils.convert_to_tflite(generate_keras_model())
    test_utils.save_tflite_model(tflite_model, temp_file)
    interpreter = tf.lite.Interpreter(model_path=str(temp_file))

    assert interpreter
