# SPDX-FileCopyrightText: Copyright 2022, Arm Limited and/or its affiliates.
# SPDX-FileCopyrightText: Copyright The TensorFlow Authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Collection of useful functions for optimizations."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable
from typing import Iterable

import numpy as np
import tensorflow as tf
from tensorflow.lite.python.interpreter import Interpreter

from mlia.utils.logging import redirect_output


def representative_dataset(model: tf.keras.Model) -> Callable:
    """Sample dataset used for quantization."""
    input_shape = model.input_shape

    def dataset() -> Iterable:
        for _ in range(100):
            if input_shape[0] != 1:
                raise Exception("Only the input batch_size=1 is supported!")
            data = np.random.rand(*input_shape)
            yield [data.astype(np.float32)]

    return dataset


def get_tf_tensor_shape(model: str) -> list:
    """Get input shape for the TensorFlow tensor model."""
    # Loading the model
    loaded = tf.saved_model.load(model)
    # The model signature must have 'serving_default' as a key
    if "serving_default" not in loaded.signatures.keys():
        raise Exception(
            "Unsupported TensorFlow model signature, must have 'serving_default'"
        )
    # Get the signature inputs
    inputs_tensor_info = loaded.signatures["serving_default"].inputs
    dims = []
    # Build a list of all inputs shape sizes
    for input_key in inputs_tensor_info:
        if input_key.get_shape():
            dims.extend(list(input_key.get_shape()))
    return dims


def representative_tf_dataset(model: str) -> Callable:
    """Sample dataset used for quantization."""
    if not (input_shape := get_tf_tensor_shape(model)):
        raise Exception("Unable to get input shape")

    def dataset() -> Iterable:
        for _ in range(100):
            data = np.random.rand(*input_shape)
            yield [data.astype(np.float32)]

    return dataset


def convert_to_tflite(model: tf.keras.Model, quantized: bool = False) -> Interpreter:
    """Convert Keras model to TensorFlow Lite."""
    if not isinstance(model, tf.keras.Model):
        raise Exception("Invalid model type")

    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    if quantized:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset(model)
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8

    with redirect_output(logging.getLogger("tensorflow")):
        tflite_model = converter.convert()

    return tflite_model


def convert_tf_to_tflite(model: str, quantized: bool = False) -> Interpreter:
    """Convert TensorFlow model to TensorFlow Lite."""
    if not isinstance(model, str):
        raise Exception("Invalid model type")

    converter = tf.lite.TFLiteConverter.from_saved_model(model)

    if quantized:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_tf_dataset(model)
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8

    with redirect_output(logging.getLogger("tensorflow")):
        tflite_model = converter.convert()

    return tflite_model


def save_keras_model(model: tf.keras.Model, save_path: str | Path) -> None:
    """Save Keras model at provided path."""
    # Checkpoint: saving the optimizer is necessary.
    model.save(save_path, include_optimizer=True)


def save_tflite_model(model: tf.lite.TFLiteConverter, save_path: str | Path) -> None:
    """Save TensorFlow Lite model at provided path."""
    with open(save_path, "wb") as file:
        file.write(model)


def is_tflite_model(model: str | Path) -> bool:
    """Check if model type is supported by TensorFlow Lite API.

    TensorFlow Lite model is indicated by the model file extension .tflite
    """
    model_path = Path(model)
    return model_path.suffix == ".tflite"


def is_keras_model(model: str | Path) -> bool:
    """Check if model type is supported by Keras API.

    Keras model is indicated by:
        1. if it's a directory (meaning saved model),
             it should contain keras_metadata.pb file
        2. or if the model file extension is .h5/.hdf5
    """
    model_path = Path(model)

    if model_path.is_dir():
        return (model_path / "keras_metadata.pb").exists()
    return model_path.suffix in (".h5", ".hdf5")


def is_tf_model(model: str | Path) -> bool:
    """Check if model type is supported by TensorFlow API.

    TensorFlow model is indicated if its directory (meaning saved model)
    doesn't contain keras_metadata.pb file
    """
    model_path = Path(model)
    return model_path.is_dir() and not is_keras_model(model)
