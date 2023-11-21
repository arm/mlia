# SPDX-FileCopyrightText: Copyright 2022-2023, Arm Limited and/or its affiliates.
# SPDX-FileCopyrightText: Copyright The TensorFlow Authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Collection of useful functions for optimizations."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import tensorflow as tf


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


def get_tflite_model_type_map(model_filename: str | Path) -> dict[str, type]:
    """Get type map from tflite model."""
    model_type_map: dict[str, Any] = {}
    interpreter = tf.lite.Interpreter(str(model_filename))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    model_type_map = {
        input_detail["name"]: input_detail["dtype"] for input_detail in input_details
    }
    return model_type_map


def check_tflite_datatypes(model_filename: str | Path, *allowed_types: type) -> None:
    """Check if the model only has the given allowed datatypes."""
    type_map = get_tflite_model_type_map(model_filename)
    types = set(type_map.values())
    allowed = set(allowed_types)
    unexpected = types - allowed

    def cls_to_str(types: set[type]) -> list[str]:
        return [t.__name__ for t in types]

    if len(unexpected) > 0:
        raise TypeError(
            f"Model {model_filename} has "
            f"unexpected data types: {cls_to_str(unexpected)}. "
            f"Only {cls_to_str(allowed)} are allowed."
        )
