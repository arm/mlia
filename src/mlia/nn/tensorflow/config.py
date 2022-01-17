# Copyright 2021, Arm Ltd.
"""Model configuration."""
# pylint: disable=too-few-public-methods,too-many-instance-attributes
import logging
from pathlib import Path
from typing import cast
from typing import Dict
from typing import List
from typing import Union

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

    def __init__(self, model_path: Union[str, Path]) -> None:
        """Init model configuration instance."""
        self.model_path = str(model_path)

    def convert_to_tflite(
        self, tflite_model_path: Union[str, Path], quantized: bool = False
    ) -> "TFLiteModel":
        """Convert model to TFLite format."""
        raise NotImplementedError()

    def convert_to_keras(self, keras_model_path: Union[str, Path]) -> "KerasModel":
        """Convert model to Keras format."""
        raise NotImplementedError()


class KerasModel(ModelConfiguration):
    """Keras model congiguration.

    Supports all models supported by keras API: saved model, H5, HDF5
    """

    def get_keras_model(self) -> tf.keras.Model:
        """Return associated keras model."""
        return tf.keras.models.load_model(self.model_path)

    def convert_to_tflite(
        self, tflite_model_path: Union[str, Path], quantized: bool = False
    ) -> "TFLiteModel":
        """Convert model to TFLite format."""
        logger.info("Converting Keras to TFLite...")

        converted_model = convert_to_tflite(self.get_keras_model(), quantized)
        logger.info("Done")

        save_tflite_model(converted_model, tflite_model_path)
        logger.info(
            "Model %s converted and saved to %s", self.model_path, tflite_model_path
        )

        return TFLiteModel(tflite_model_path)

    def convert_to_keras(self, keras_model_path: Union[str, Path]) -> "KerasModel":
        """Do nothing."""
        return self


class TFLiteModel(ModelConfiguration):  # pylint: disable=abstract-method
    """TFLite model configuration."""

    def input_details(self) -> List[Dict]:
        """Get model's input details."""
        interpreter = tf.lite.Interpreter(model_path=self.model_path)
        return cast(List[Dict], interpreter.get_input_details())

    def get_tflite_model(self) -> tf.lite.Interpreter:
        """Return associated tflite model."""
        tf.lite.Interpreter(model_path=self.model_path)

    def convert_to_tflite(
        self, tflite_model_path: Union[str, Path], quantized: bool = False
    ) -> "TFLiteModel":
        """Do nothing."""
        return self


class TfModel(ModelConfiguration):  # pylint: disable=abstract-method
    """Tensor Flow model configuration.

    Supports models supported by TF API (not Keras)
    """

    def convert_to_tflite(
        self, tflite_model_path: Union[str, Path], quantized: bool = False
    ) -> "TFLiteModel":
        """Convert model to TFLite format."""
        converted_model = convert_tf_to_tflite(self.model_path, quantized)
        save_tflite_model(converted_model, tflite_model_path)

        return TFLiteModel(tflite_model_path)


def get_model(model: Union[Path, str]) -> "ModelConfiguration":
    """Return the model object."""
    if is_tflite_model(model):
        return TFLiteModel(model)

    if is_keras_model(model):
        return KerasModel(model)

    if is_tf_model(model):
        return TfModel(model)

    raise Exception(
        "The input model format is not supported"
        "(supported formats: tflite, Keras, TF saved model)!"
    )


def get_tflite_model(model: Union[str, Path], ctx: Context) -> "TFLiteModel":
    """Convert input model to tflite and returns TFLiteModel object."""
    tflite_model_path = str(ctx.get_model_path("converted_model.tflite"))
    converted_model = get_model(model)
    return converted_model.convert_to_tflite(tflite_model_path, True)


def get_keras_model(model: Union[str, Path], ctx: Context) -> "KerasModel":
    """Convert input model to Keras and returns KerasModel object."""
    keras_model_path = str(ctx.get_model_path("converted_model.h5"))

    converted_model = get_model(model)
    return converted_model.convert_to_keras(keras_model_path)
