# SPDX-FileCopyrightText: Copyright 2022-2023, Arm Limited and/or its affiliates.
# SPDX-FileCopyrightText: Copyright The TensorFlow Authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Collection of useful functions for optimizations."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any
from typing import Callable
from typing import cast
from typing import Iterable

import numpy as np
import tensorflow as tf

from mlia.utils.logging import redirect_output


def representative_dataset(
    input_shape: Any, sample_count: int = 100, input_dtype: type = np.float32
) -> Callable:
    """Sample dataset used for quantization."""
    if input_shape[0] != 1:
        raise ValueError("Only the input batch_size=1 is supported!")

    def dataset() -> Iterable:
        for _ in range(sample_count):
            data = np.random.rand(*input_shape)
            yield [data.astype(input_dtype)]

    return dataset


def get_tf_tensor_shape(model: str) -> list:
    """Get input shape for the TensorFlow tensor model."""
    loaded = tf.saved_model.load(model)

    try:
        default_signature_key = tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY
        default_signature = loaded.signatures[default_signature_key]
        inputs_tensor_info = default_signature.inputs
    except KeyError as err:
        raise KeyError(f"Signature '{default_signature_key}' not found.") from err

    return [
        dim
        for input_key in inputs_tensor_info
        if (shape := input_key.get_shape())
        for dim in shape
    ]


def convert_to_tflite(model: tf.keras.Model | str, quantized: bool = False) -> bytes:
    """Convert Keras model to TensorFlow Lite."""
    converter = get_tflite_converter(model, quantized)

    with redirect_output(logging.getLogger("tensorflow")):
        return cast(bytes, converter.convert())


def save_keras_model(
    model: tf.keras.Model, save_path: str | Path, include_optimizer: bool = True
) -> None:
    """Save Keras model at provided path."""
    model.save(save_path, include_optimizer=include_optimizer)


def save_tflite_model(tflite_model: bytes, save_path: str | Path) -> None:
    """Save TensorFlow Lite model at provided path."""
    with open(save_path, "wb") as file:
        file.write(tflite_model)


def is_tflite_model(model: str | Path) -> bool:
    """Check if path contains TensorFlow Lite model."""
    model_path = Path(model)

    return model_path.suffix == ".tflite"


def is_keras_model(model: str | Path) -> bool:
    """Check if path contains a Keras model."""
    model_path = Path(model)

    if model_path.is_dir():
        return model_path.joinpath("keras_metadata.pb").exists()

    return model_path.suffix in (".h5", ".hdf5")


def is_saved_model(model: str | Path) -> bool:
    """Check if path contains SavedModel model."""
    model_path = Path(model)

    return model_path.is_dir() and not is_keras_model(model)


def get_tflite_converter(
    model: tf.keras.Model | str | Path, quantized: bool = False
) -> tf.lite.TFLiteConverter:
    """Configure TensorFlow Lite converter for the provided model."""
    if isinstance(model, (str, Path)):
        # converter's methods accept string as input parameter
        model = str(model)

    if isinstance(model, tf.keras.Model):
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        input_shape = model.input_shape
    elif isinstance(model, str) and is_saved_model(model):
        converter = tf.lite.TFLiteConverter.from_saved_model(model)
        input_shape = get_tf_tensor_shape(model)
    elif isinstance(model, str) and is_keras_model(model):
        keras_model = tf.keras.models.load_model(model)
        input_shape = keras_model.input_shape
        converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    else:
        raise ValueError(f"Unable to create TensorFlow Lite converter for {model}")

    if quantized:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset(input_shape)
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8

    return converter
