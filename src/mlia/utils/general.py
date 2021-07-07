# Copyright 2021, Arm Ltd.
"""Collection of useful functions for optimizations."""
import tempfile
from typing import Any

import tensorflow as tf


def convert_to_tflite(model: tf.keras.Model) -> Any:
    """Convert keras model to tflite."""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    return tflite_model


def save_keras_model(model: tf.keras.Model) -> str:
    """Save keras model in temporary file and return the name of that file."""
    _, keras_model_file = tempfile.mkstemp(".h5")

    # Checkpoint: saving the optimizer is necessary.
    model.save(keras_model_file, include_optimizer=True)

    return keras_model_file


def save_tflite_model(model: tf.keras.Model) -> str:
    """Save tflite model in temporary file and return the name of that file."""
    _, tflite_model_file = tempfile.mkstemp(".tflite")
    with open(tflite_model_file, "wb") as file:
        file.write(model)

    return tflite_model_file


def deep_clone_model(model: tf.keras.Model) -> tf.keras.Model:
    """Create a clone of a model, this ensures separation between optimizations."""
    # Check if custom objects are required (i.e. QAT)
    cloned_model = tf.keras.models.clone_model(model)
    cloned_model.set_weights(model.get_weights())

    return cloned_model
