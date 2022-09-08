# SPDX-FileCopyrightText: Copyright 2022, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Model configuration."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import cast
from typing import List

import tensorflow as tf

from mlia.core.context import Context
from mlia.nn.tensorflow.utils import convert_tf_to_tflite
from mlia.nn.tensorflow.utils import convert_to_tflite
from mlia.nn.tensorflow.utils import is_keras_model
from mlia.nn.tensorflow.utils import is_tf_model
from mlia.nn.tensorflow.utils import is_tflite_model
from mlia.nn.tensorflow.utils import save_tflite_model

logger = logging.getLogger(__name__)


class ModelConfiguration:
    """Base class for model configuration."""

    def __init__(self, model_path: str | Path) -> None:
        """Init model configuration instance."""
        self.model_path = str(model_path)

    def convert_to_tflite(
        self, tflite_model_path: str | Path, quantized: bool = False
    ) -> TFLiteModel:
        """Convert model to TFLite format."""
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
        """Convert model to TFLite format."""
        logger.info("Converting Keras to TFLite ...")

        converted_model = convert_to_tflite(self.get_keras_model(), quantized)
        logger.info("Done\n")

        save_tflite_model(converted_model, tflite_model_path)
        logger.debug(
            "Model %s converted and saved to %s", self.model_path, tflite_model_path
        )

        return TFLiteModel(tflite_model_path)

    def convert_to_keras(self, keras_model_path: str | Path) -> KerasModel:
        """Convert model to Keras format."""
        return self


class TFLiteModel(ModelConfiguration):  # pylint: disable=abstract-method
    """TFLite model configuration."""

    def input_details(self) -> list[dict]:
        """Get model's input details."""
        interpreter = tf.lite.Interpreter(model_path=self.model_path)
        return cast(List[dict], interpreter.get_input_details())

    def convert_to_tflite(
        self, tflite_model_path: str | Path, quantized: bool = False
    ) -> TFLiteModel:
        """Convert model to TFLite format."""
        return self


class TfModel(ModelConfiguration):  # pylint: disable=abstract-method
    """TensorFlow model configuration.

    Supports models supported by TensorFlow API (not Keras)
    """

    def convert_to_tflite(
        self, tflite_model_path: str | Path, quantized: bool = False
    ) -> TFLiteModel:
        """Convert model to TFLite format."""
        converted_model = convert_tf_to_tflite(self.model_path, quantized)
        save_tflite_model(converted_model, tflite_model_path)

        return TFLiteModel(tflite_model_path)


def get_model(model: str | Path) -> ModelConfiguration:
    """Return the model object."""
    if is_tflite_model(model):
        return TFLiteModel(model)

    if is_keras_model(model):
        return KerasModel(model)

    if is_tf_model(model):
        return TfModel(model)

    raise Exception(
        "The input model format is not supported"
        "(supported formats: TFLite, Keras, TensorFlow saved model)!"
    )


def get_tflite_model(model: str | Path, ctx: Context) -> TFLiteModel:
    """Convert input model to TFLite and returns TFLiteModel object."""
    tflite_model_path = ctx.get_model_path("converted_model.tflite")
    converted_model = get_model(model)

    return converted_model.convert_to_tflite(tflite_model_path, True)


def get_keras_model(model: str | Path, ctx: Context) -> KerasModel:
    """Convert input model to Keras and returns KerasModel object."""
    keras_model_path = ctx.get_model_path("converted_model.h5")
    converted_model = get_model(model)

    return converted_model.convert_to_keras(keras_model_path)
