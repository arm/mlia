# Copyright 2021, Arm Ltd.
"""Script for generating sample TFLite models."""
import argparse
from functools import wraps
from pathlib import Path
from typing import Any
from typing import Callable
from typing import Iterable
from typing import Optional

import numpy as np
import tensorflow as tf
from typing_extensions import TypedDict

tf.keras.backend.set_image_data_format("channels_last")

models = {}


class ModelCreatorAttrs(TypedDict):
    """Model creator attributes."""

    model_creator: Callable
    quantize: bool


def test_model(compile_model: bool = True, quantize: bool = True) -> Callable:
    """Mark function as model creator."""

    def wrapper(model_creator: Callable) -> Callable:
        """Wrap model creator function."""

        @wraps(model_creator)
        def model_creator_wrapper(*args: Any, **kwargs: Any) -> tf.keras.Model:
            model = model_creator(*args, **kwargs)

            if compile_model:
                model.compile(optimizer="sgd", loss="mean_squared_error")

            return model

        models[model_creator.__name__] = ModelCreatorAttrs(
            model_creator=model_creator_wrapper, quantize=quantize
        )
        return model_creator_wrapper

    return wrapper


@test_model()
def simple_3_layers_model() -> tf.keras.Model:
    """Generate simple model with 3 layers."""
    return tf.keras.models.Sequential(
        [
            tf.keras.layers.Dense(units=1, input_shape=[1]),
            tf.keras.layers.Dense(units=16, activation="relu"),
            tf.keras.layers.Dense(units=1),
        ]
    )


@test_model()
def simple_conv_model() -> tf.keras.Model:
    """Generate simple model with Conv2d operator."""
    return tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(28, 28, 1)),
            tf.keras.layers.Conv2D(
                filters=12,
                kernel_size=(3, 3),
                activation=tf.nn.relu,
            ),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(10),
        ]
    )


@test_model(quantize=False)
def simple_mnist_convnet_non_quantized() -> tf.keras.Model:
    """Generate simple MNIST model.

    This example is taken from https://keras.io/examples/vision/mnist_convnet.
    """
    return tf.keras.Sequential(
        [
            tf.keras.Input(shape=(28, 28, 1)),
            tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(10, activation="softmax"),
        ]
    )


def get_model_path(
    model_name: str, output_dir: Optional[str], ext: str = "tflite"
) -> Path:
    """Get model path."""
    model_path = Path(f"{model_name}.{ext}")
    if not output_dir:
        return model_path

    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(exist_ok=True)

    return output_dir_path / model_path


def save_tflite_model(
    tflite_model: Any, model_name: str, output_dir: Optional[str]
) -> None:
    """Save TFLite model."""
    tflite_file_path = get_model_path(model_name, output_dir)
    with open(tflite_file_path, "wb") as tflite_file:
        print(f"Model {model_name} saved to {tflite_file_path}")
        tflite_file.write(tflite_model)


def save_keras_model(
    model: tf.keras.Model, model_name: str, output_dir: Optional[str]
) -> None:
    """Save Keras model."""
    keras_file_path = get_model_path(model_name, output_dir, "h5")
    model.save(keras_file_path, include_optimizer=True)


def representative_dataset(model: Any) -> Callable:
    """Sample dataset used for quantization."""

    def dataset() -> Iterable:
        for _ in range(100):
            data = np.random.rand(1, *model.input_shape[1:])
            yield [data.astype(np.float32)]

    return dataset


def gen_models(
    output_dir: Optional[str], specific_model: Optional[str], save_keras: bool
) -> None:
    """Generate test models."""
    for model_name, model_creator_attrs in models.items():
        if specific_model and model_name != specific_model:
            continue

        model_creator = model_creator_attrs["model_creator"]
        quantize = model_creator_attrs["quantize"]

        print(f"==> Generate {model_name} ...")
        model = model_creator()
        if save_keras:
            save_keras_model(model, model_name, output_dir)

        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        if quantize:
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.representative_dataset = representative_dataset(model)
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8

        tflite_model = converter.convert()
        save_tflite_model(tflite_model, model_name, output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir", help="Path to the output directory where models will be saved"
    )
    parser.add_argument(
        "--model-name",
        help="Name of the particular model to generate",
        choices=models.keys(),
    )
    parser.add_argument(
        "--save-keras",
        action="store_true",
        default=False,
        help="Save Keras model in addition to the TFLite model",
    )
    args = parser.parse_args()

    gen_models(args.output_dir, args.model_name, args.save_keras)
