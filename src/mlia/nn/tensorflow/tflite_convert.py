# SPDX-FileCopyrightText: Copyright 2022-2024, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Support module to call TFLiteConverter."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any
from typing import Callable
from typing import cast
from typing import Iterable

import numpy as np
import tensorflow as tf
from keras.api._v2 import keras  # Temporary workaround for now: MLIA-1107

from mlia.nn.tensorflow.utils import get_tf_tensor_shape
from mlia.nn.tensorflow.utils import is_keras_model
from mlia.nn.tensorflow.utils import is_saved_model
from mlia.nn.tensorflow.utils import save_tflite_model
from mlia.utils.logging import redirect_output
from mlia.utils.proc import Command
from mlia.utils.proc import command_output


logger = logging.getLogger(__name__)


def representative_dataset(
    input_shape: Any, sample_count: int = 100, input_dtype: type = np.float32
) -> Callable:
    """Sample dataset used for quantization."""

    def dataset() -> Iterable:
        for _ in range(sample_count):
            data = np.random.rand(1, *input_shape[1:])
            yield [data.astype(input_dtype)]

    return dataset


def get_tflite_converter(
    model: keras.Model | str | Path, quantized: bool = False
) -> tf.lite.TFLiteConverter:
    """Configure TensorFlow Lite converter for the provided model."""
    if isinstance(model, (str, Path)):
        # converter's methods accept string as input parameter
        model = str(model)

    if isinstance(model, keras.Model):
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        input_shape = model.input_shape
    elif isinstance(model, str) and is_saved_model(model):
        converter = tf.lite.TFLiteConverter.from_saved_model(model)
        input_shape = get_tf_tensor_shape(model)
    elif isinstance(model, str) and is_keras_model(model):
        keras_model = keras.models.load_model(model)
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


def convert_to_tflite_bytes(model: keras.Model | str, quantized: bool = False) -> bytes:
    """Convert Keras model to TensorFlow Lite."""
    converter = get_tflite_converter(model, quantized)

    with redirect_output(logging.getLogger("tensorflow")):
        output_bytes = cast(bytes, converter.convert())

    return output_bytes


def _convert_to_tflite(
    model: keras.Model | str,
    quantized: bool = False,
    output_path: Path | None = None,
) -> bytes:
    """Convert Keras model to TensorFlow Lite."""
    output_bytes = convert_to_tflite_bytes(model, quantized)

    if output_path:
        save_tflite_model(output_bytes, output_path)

    return output_bytes


def convert_to_tflite(
    model: keras.Model | str,
    quantized: bool = False,
    output_path: Path | None = None,
    input_path: Path | None = None,
    subprocess: bool = False,
) -> None:
    """Convert Keras model to TensorFlow Lite.

    Optionally runs TFLiteConverter in a subprocess,
    this is added mainly to work around issues when redirecting
    Tensorflow's output using SDK calls, didn't make an effect,
    which would produce unwanted output for MLIA.

    In the subprocess mode, the model should be passed as a
    file path, or via a dedicated 'input_path' parameter.

    If 'output_path' is provided, the result model be saved under
    that path.
    """
    if not subprocess:
        _convert_to_tflite(model, quantized, output_path)
        return

    if input_path is None:
        if isinstance(model, str):
            input_path = Path(model)
        else:
            raise RuntimeError(
                f"Input path is required for {model}"
                " when converter is called in subprocess."
            )

    args = ["python", __file__, str(input_path)]
    if output_path:
        args.append("--output")
        args.append(str(output_path))
    if quantized:
        args.append("--quantize")

    command = Command(args)

    for line in command_output(command):
        logger.debug("TFLiteConverter: %s", line)


def main(argv: list[str] | None = None) -> int:
    """Entry point to run this module as a standalone executable."""
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--quantize", default=False, action="store_true")
    args = parser.parse_args(argv)

    if not Path(args.input).exists():
        raise ValueError(f"Input file doesn't exist: [{args.input}]")

    logger.debug(
        "Invoking TFLiteConverter on [%s] -> [%s], quantize: [%s]",
        args.input,
        args.output,
        args.quantize,
    )
    _convert_to_tflite(args.input, args.quantize, args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
