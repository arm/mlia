# SPDX-FileCopyrightText: Copyright 2022-2023, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Model configuration."""
from __future__ import annotations

import logging
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import tensorflow as tf

from mlia.core.context import Context
from mlia.nn.tensorflow.tflite_graph import load_fb
from mlia.nn.tensorflow.tflite_graph import save_fb
from mlia.nn.tensorflow.utils import convert_to_tflite
from mlia.nn.tensorflow.utils import is_keras_model
from mlia.nn.tensorflow.utils import is_saved_model
from mlia.nn.tensorflow.utils import is_tflite_model
from mlia.nn.tensorflow.utils import save_tflite_model
from mlia.utils.logging import log_action

logger = logging.getLogger(__name__)


class ModelConfiguration:
    """Base class for model configuration."""

    def __init__(self, model_path: str | Path) -> None:
        """Init model configuration instance."""
        self.model_path = str(model_path)

    def convert_to_tflite(
        self, tflite_model_path: str | Path, quantized: bool = False
    ) -> TFLiteModel:
        """Convert model to TensorFlow Lite format."""
        raise NotImplementedError()

    def convert_to_keras(self, keras_model_path: str | Path) -> KerasModel:
        """Convert model to Keras format."""
        raise NotImplementedError()


class KerasModel(ModelConfiguration):
    """Keras model configuration.

    Supports all models supported by Keras API: saved model, H5, HDF5
    """

    def get_keras_model(self) -> tf.keras.Model:
        """Return associated Keras model."""
        return tf.keras.models.load_model(self.model_path)

    def convert_to_tflite(
        self, tflite_model_path: str | Path, quantized: bool = False
    ) -> TFLiteModel:
        """Convert model to TensorFlow Lite format."""
        with log_action("Converting Keras to TensorFlow Lite ..."):
            converted_model = convert_to_tflite(self.get_keras_model(), quantized)

        save_tflite_model(converted_model, tflite_model_path)
        logger.debug(
            "Model %s converted and saved to %s", self.model_path, tflite_model_path
        )

        return TFLiteModel(tflite_model_path)

    def convert_to_keras(self, keras_model_path: str | Path) -> KerasModel:
        """Convert model to Keras format."""
        return self


class TFLiteModel(ModelConfiguration):  # pylint: disable=abstract-method
    """TensorFlow Lite model configuration."""

    def __init__(
        self,
        model_path: str | Path,
        batch_size: int | None = None,
        num_threads: int | None = None,
    ) -> None:
        """Initiate a TFLite Model."""
        super().__init__(model_path)
        if not num_threads:
            num_threads = None
        if not batch_size:
            self.interpreter = tf.lite.Interpreter(
                model_path=self.model_path, num_threads=num_threads
            )
        else:  # if a batch size is specified, modify the TFLite model to use this size
            with tempfile.TemporaryDirectory() as tmp:
                flatbuffer = load_fb(self.model_path)
                for subgraph in flatbuffer.subgraphs:
                    for tensor in list(subgraph.inputs) + list(subgraph.outputs):
                        subgraph.tensors[tensor].shape = np.array(
                            [batch_size] + list(subgraph.tensors[tensor].shape[1:]),
                            dtype=np.int32,
                        )
                tempname = Path(tmp, "rewrite_tmp.tflite")
                save_fb(flatbuffer, tempname)
                self.interpreter = tf.lite.Interpreter(
                    model_path=str(tempname), num_threads=num_threads
                )

        try:
            self.interpreter.allocate_tensors()
        except RuntimeError:
            self.interpreter = tf.lite.Interpreter(
                model_path=self.model_path, num_threads=num_threads
            )
            self.interpreter.allocate_tensors()

        # Get input and output tensors.
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        details = list(self.input_details) + list(self.output_details)
        self.handle_from_name = {d["name"]: d["index"] for d in details}
        self.shape_from_name = {d["name"]: d["shape"] for d in details}
        self.batch_size = next(iter(self.shape_from_name.values()))[0]

    def __call__(self, named_input: dict) -> dict:
        """Execute the model on one or a batch of named inputs \
            (a dict of name: numpy array)."""
        input_len = next(iter(named_input.values())).shape[0]
        full_steps = input_len // self.batch_size
        remainder = input_len % self.batch_size

        named_ys = defaultdict(list)
        for i in range(full_steps):
            for name, x_batch in named_input.items():
                x_tensor = x_batch[i : i + self.batch_size]  # noqa: E203
                self.interpreter.set_tensor(self.handle_from_name[name], x_tensor)
            self.interpreter.invoke()
            for output_detail in self.output_details:
                named_ys[output_detail["name"]].append(
                    self.interpreter.get_tensor(output_detail["index"])
                )
        if remainder:
            for name, x_batch in named_input.items():
                x_tensor = np.zeros(  # pylint: disable=invalid-name
                    self.shape_from_name[name]
                ).astype(x_batch.dtype)
                x_tensor[:remainder] = x_batch[-remainder:]
                self.interpreter.set_tensor(self.handle_from_name[name], x_tensor)
            self.interpreter.invoke()
            for output_detail in self.output_details:
                named_ys[output_detail["name"]].append(
                    self.interpreter.get_tensor(output_detail["index"])[:remainder]
                )
        return {k: np.concatenate(v) for k, v in named_ys.items()}

    def input_tensors(self) -> list:
        """Return name from input details."""
        return [d["name"] for d in self.input_details]

    def output_tensors(self) -> list:
        """Return name from output details."""
        return [d["name"] for d in self.output_details]

    def convert_to_tflite(
        self, tflite_model_path: str | Path, quantized: bool = False
    ) -> TFLiteModel:
        """Convert model to TensorFlow Lite format."""
        return self


class TfModel(ModelConfiguration):  # pylint: disable=abstract-method
    """TensorFlow model configuration.

    Supports models supported by TensorFlow API (not Keras)
    """

    def convert_to_tflite(
        self, tflite_model_path: str | Path, quantized: bool = False
    ) -> TFLiteModel:
        """Convert model to TensorFlow Lite format."""
        converted_model = convert_to_tflite(self.model_path, quantized)
        save_tflite_model(converted_model, tflite_model_path)

        return TFLiteModel(tflite_model_path)


def get_model(model: str | Path) -> ModelConfiguration:
    """Return the model object."""
    if is_tflite_model(model):
        return TFLiteModel(model)

    if is_keras_model(model):
        return KerasModel(model)

    if is_saved_model(model):
        return TfModel(model)

    raise ValueError(
        "The input model format is not supported "
        "(supported formats: TensorFlow Lite, Keras, TensorFlow saved model)!"
    )


def get_tflite_model(model: str | Path, ctx: Context) -> TFLiteModel:
    """Convert input model to TensorFlow Lite and returns TFLiteModel object."""
    dst_model_path = ctx.get_model_path("converted_model.tflite")
    src_model = get_model(model)

    return src_model.convert_to_tflite(dst_model_path, quantized=True)


def get_keras_model(model: str | Path, ctx: Context) -> KerasModel:
    """Convert input model to Keras and returns KerasModel object."""
    keras_model_path = ctx.get_model_path("converted_model.h5")
    converted_model = get_model(model)

    return converted_model.convert_to_keras(keras_model_path)
