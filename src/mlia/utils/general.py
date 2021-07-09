# Copyright 2021, Arm Ltd.
"""Collection of useful functions for optimizations."""
import os
import tempfile
from contextlib import contextmanager
from contextlib import redirect_stderr
from contextlib import redirect_stdout
from typing import Any
from typing import Callable
from typing import Generator
from typing import Iterable
from typing import Optional

import numpy as np
import tensorflow as tf


def representative_dataset(model: Any) -> Callable:
    """Sample dataset used for quantization."""

    def dataset() -> Iterable:
        for _ in range(100):
            data = np.random.rand(1, *model.input_shape[1:])
            yield [data.astype(np.float32)]

    return dataset


def convert_to_tflite(model: tf.keras.Model, quantized: bool = False) -> Any:
    """Convert keras model to tflite."""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    if quantized:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset(model)
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8

    tflite_model = converter.convert()

    return tflite_model


def save_keras_model(model: tf.keras.Model, save_path: Optional[str] = None) -> str:
    """Save keras model in temporary file and return the name of that file."""
    if not save_path:
        _, save_path = tempfile.mkstemp(".h5")

    # Checkpoint: saving the optimizer is necessary.
    model.save(save_path, include_optimizer=True)

    return save_path


def save_tflite_model(model: tf.keras.Model, save_path: Optional[str] = None) -> str:
    """Save tflite model in temporary file and return the name of that file."""
    if not save_path:
        _, save_path = tempfile.mkstemp(".tflite")

    with open(save_path, "wb") as file:
        file.write(model)

    return save_path


def deep_clone_model(model: tf.keras.Model) -> tf.keras.Model:
    """Create a clone of a model, this ensures separation between optimizations."""
    # Check if custom objects are required (i.e. QAT)
    cloned_model = tf.keras.models.clone_model(model)
    cloned_model.set_weights(model.get_weights())

    return cloned_model


@contextmanager
def suppress_any_output() -> Generator[Any, Any, Any]:
    """Context manager for suppressing output."""
    with open(os.devnull, "w") as dev_null:
        with redirect_stderr(dev_null), redirect_stdout(dev_null):
            yield
