# Copyright 2021, Arm Ltd.
"""Test for module utils/test_utils."""
import tensorflow as tf
from mlia.utils import general as test_utils

from tests.utils.generate_keras_model import generate_keras_model


def test_convert_to_tflite() -> None:
    """Test converting keras model to tflite."""
    tflite_model = test_utils.convert_to_tflite(generate_keras_model())

    assert tflite_model


def test_save_keras_model() -> None:
    """Test saving keras model."""
    model = generate_keras_model()
    saved_model = test_utils.save_keras_model(model)
    loaded_model = tf.keras.models.load_model(saved_model)

    assert loaded_model.summary() == model.summary()


def test_save_tflite_model() -> None:
    """Test saving tflite model."""
    tflite_model_path = test_utils.save_tflite_model(
        test_utils.convert_to_tflite(generate_keras_model())
    )
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)

    assert interpreter
