# Copyright 2021, Arm Ltd.
"""Model and IP configuration."""
# pylint: disable=too-few-public-methods,too-many-instance-attributes
# pylint: disable=too-many-arguments
import logging
from abc import ABC
from abc import abstractmethod
from pathlib import Path
from typing import cast
from typing import Dict
from typing import List
from typing import Union

import tensorflow as tf
from mlia.utils.general import convert_tf_to_tflite
from mlia.utils.general import convert_to_tflite
from mlia.utils.general import extract_if_archived
from mlia.utils.general import is_keras_model
from mlia.utils.general import is_tf_model
from mlia.utils.general import is_tflite_model
from mlia.utils.general import save_tflite_model

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

    def convert_to_keras(self, keras_model_path: Union[str, Path]) -> "KerasModel":
        """Print error."""
        raise Exception(
            "TfModel cannot be converted into Keras Model. \n"
            "Please ensure that you are saving the model with "
            "model.save() or tf.keras.models.save_model(), "
            "NOT tf.saved_model.save(). \n"
            "To confirm, there should be a file named "
            "'keras_metadata.pb' in the SavedModel directory."
        )


def get_model(model: Union[Path, str], ctx: "Context") -> "ModelConfiguration":
    """Return the model object."""
    extracted_model_path = ctx.get_model_path("extracted_model")
    model = extract_if_archived(model, extracted_model_path)
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


def get_tflite_model(model: str, ctx: "Context") -> "TFLiteModel":
    """Convert input model to tflite and returns TFLiteModel object."""
    tflite_model_path = str(ctx.get_model_path("converted_model.tflite"))
    converted_model = get_model(model, ctx)
    return converted_model.convert_to_tflite(tflite_model_path, True)


def get_keras_model(model: str, ctx: "Context") -> "KerasModel":
    """Convert input model to Keras and returns KerasModel object."""
    keras_model_path = str(ctx.get_model_path("converted_model.h5"))

    converted_model = get_model(model, ctx)
    return converted_model.convert_to_keras(keras_model_path)


class Context(ABC):
    """Abstract class for the execution context."""

    @abstractmethod
    def get_model_path(self, model_filename: str) -> Path:
        """Return path for the model."""
