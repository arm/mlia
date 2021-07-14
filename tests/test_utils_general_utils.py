# Copyright 2021, Arm Ltd.
"""Test for module utils/test_utils."""
import pytest
import tensorflow as tf
from mlia.utils import general as test_utils


@pytest.fixture(scope="function")
def dummy_keras_model() -> tf.keras.Model:
    """Create a dummy model."""
    keras_model = tf.keras.Sequential(
        [
            tf.keras.Input(shape=(8, 8, 3)),
            tf.keras.layers.Conv2D(4, 3),
            tf.keras.layers.DepthwiseConv2D(3),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(8),
        ]
    )
    return keras_model


def test_convert_to_tflite(dummy_keras_model: tf.keras.Model) -> None:
    """Test converting keras model to tflite."""
    tflite_model = test_utils.convert_to_tflite(dummy_keras_model)

    assert tflite_model


def test_save_keras_model(dummy_keras_model: tf.keras.Model) -> None:
    """Test saving keras model."""
    saved_path = test_utils.save_keras_model(dummy_keras_model)
    loaded_model = tf.keras.models.load_model(saved_path)

    assert loaded_model.summary() == dummy_keras_model.summary()


def test_save_tflite_model(dummy_keras_model: tf.keras.Model) -> None:
    """Test saving tflite model."""
    dummy_tflite_model = test_utils.convert_to_tflite(dummy_keras_model)
    saved_path = test_utils.save_tflite_model(dummy_tflite_model)
    interpreter = tf.lite.Interpreter(model_path=saved_path)

    assert interpreter
