# Copyright 2021, Arm Ltd.
"""Collection of useful functions for optimizations."""
import logging
from contextlib import contextmanager
from contextlib import ExitStack
from contextlib import redirect_stderr
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any
from typing import Callable
from typing import Generator
from typing import Iterable
from typing import Optional
from typing import Union

import numpy as np
import tensorflow as tf
from tensorflow.lite.python.interpreter import Interpreter
from tensorflow.python.tools import (  # pylint: disable=no-name-in-module
    saved_model_utils,
)


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
    """Get input shape for the TF tensor model.

    Based on saved_model_cli tool
    """
    tag_sets = saved_model_utils.get_saved_model_tag_sets(model)
    for tag_set in sorted(tag_sets):
        tag_set = ",".join(tag_set)
        meta_graph = saved_model_utils.get_meta_graph_def(model, tag_set)
        signature_def_map = meta_graph.signature_def
        for signature_def_key in sorted(signature_def_map.keys()):
            inputs_tensor_info = meta_graph.signature_def[signature_def_key].inputs
            for _input_key, input_tensor in sorted(inputs_tensor_info.items()):
                dims = [dim.size for dim in input_tensor.tensor_shape.dim]
                return dims
    return []


def representative_tf_dataset(model: str) -> Callable:
    """Sample dataset used for quantization."""
    if not (
        input_shape := get_tf_tensor_shape(model)  # pylint: disable=superfluous-parens
    ):
        raise Exception("Unable to get input shape")

    def dataset() -> Iterable:
        for _ in range(100):
            data = np.random.rand(*input_shape)
            yield [data.astype(np.float32)]

    return dataset


def convert_to_tflite(model: tf.keras.Model, quantized: bool = False) -> Interpreter:
    """Convert keras model to tflite."""
    if not isinstance(model, tf.keras.Model):
        raise Exception("Invalid model type")

    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    if quantized:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset(model)
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8

    with redirect_output(logging.getLogger("tensorflow")):
        tflite_model = converter.convert()

    return tflite_model


def convert_tf_to_tflite(model: str, quantized: bool = False) -> Interpreter:
    """Convert TF model to tflite."""
    if not isinstance(model, str):
        raise Exception("Invalid model type")

    converter = tf.lite.TFLiteConverter.from_saved_model(model)

    if quantized:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_tf_dataset(model)
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8

    with redirect_output(logging.getLogger("tensorflow")):
        tflite_model = converter.convert()

    return tflite_model


def save_keras_model(model: tf.keras.Model, save_path: Union[str, Path]) -> None:
    """Save keras model at provided path."""
    # Checkpoint: saving the optimizer is necessary.
    model.save(save_path, include_optimizer=True)


def save_tflite_model(
    model: tf.lite.TFLiteConverter, save_path: Union[str, Path]
) -> None:
    """Save tflite model at provided path."""
    with open(save_path, "wb") as file:
        file.write(model)


def deep_clone_model(model: tf.keras.Model) -> tf.keras.Model:
    """Create a clone of a model, this ensures separation between optimizations."""
    # Check if custom objects are required (i.e. QAT)
    cloned_model = tf.keras.models.clone_model(model)
    cloned_model.set_weights(model.get_weights())

    return cloned_model


class LoggerWriter:
    """Redirect printed messages to the logger."""

    def __init__(self, logger: logging.Logger, level: int):
        """Init logger writer."""
        self.logger = logger
        self.level = level

    def write(self, message: str) -> None:
        """Write message."""
        if message.strip() != "":
            self.logger.log(self.level, message)

    def flush(self) -> None:
        """Flush buffers."""


@contextmanager
def redirect_output(
    logger: logging.Logger,
    stdout_level: int = logging.INFO,
    stderr_level: int = logging.INFO,
) -> Generator[None, None, None]:
    """Redirect standard output to the logger."""
    stdout_to_log = LoggerWriter(logger, stdout_level)
    stderr_to_log = LoggerWriter(logger, stderr_level)

    with ExitStack() as exit_stack:
        exit_stack.enter_context(redirect_stdout(stdout_to_log))  # type: ignore
        exit_stack.enter_context(redirect_stderr(stderr_to_log))  # type: ignore

        yield


def is_list_of(data: Any, cls: type, elem_num: Optional[int] = None) -> bool:
    """Check if data is a list of object of the same class."""
    return (
        isinstance(data, (tuple, list))
        and all(isinstance(item, cls) for item in data)
        and (elem_num is None or len(data) == elem_num)
    )


def is_tflite_model(model: Union[Path, str]) -> bool:
    """Check if model type is supported by TFLite API.

    TFLite is model is indicated by the model file extension .tflite
    """
    model_path = Path(model)
    return model_path.suffix == ".tflite"


def is_keras_model(model: Union[Path, str]) -> bool:
    """Check if model type is supported by Keras API.

    Keras is model is indicated by:
        1. if its directory (meaning saved model),
             it should contain keras_metadata.pb file
        2. or if the model file extension .h5/.hdf5
    """
    model_path = Path(model)
    if model_path.is_dir():
        return (model_path / "keras_metadata.pb").exists()
    return model_path.suffix == ".h5" or model_path.suffix == ".hdf5"


def is_tf_model(model: Union[Path, str]) -> bool:
    """Check if model type is supported by TF API.

    TF is model is indicated if its directory (meaning saved model),
    that doesn't contain keras_metadata.pb file
    """
    model_path = Path(model)
    return model_path.is_dir() and not is_keras_model(model)
