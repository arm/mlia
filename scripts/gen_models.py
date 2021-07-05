# Copyright 2021, Arm Ltd.
"""Script for generating sample TFLite models."""
import argparse
from pathlib import Path
from typing import Any
from typing import Callable
from typing import Iterable
from typing import Optional

import numpy as np
import tensorflow as tf

tf.keras.backend.set_image_data_format("channels_last")

models = {}


def test_model(model_creator: Callable) -> Any:
    """Mark function as model creator."""
    models[model_creator.__name__] = model_creator
    return model_creator


@test_model
def simple_3_layers_model() -> Any:
    """Generate simple model with 3 layers."""
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Dense(units=1, input_shape=[1]),
            tf.keras.layers.Dense(units=16, activation="relu"),
            tf.keras.layers.Dense(units=1),
        ]
    )

    model.compile(optimizer="sgd", loss="mean_squared_error")
    return model


@test_model
def simple_conv_model() -> Any:
    """Generate simple model with Conv2d operator."""
    model = tf.keras.Sequential(
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

    model.compile(optimizer="sgd", loss="mean_squared_error")
    return model


def get_model_path(model_name: str, output_dir: Optional[str]) -> Path:
    """Get model path."""
    tflite_file_path = Path(f"{model_name}.tflite")
    if not output_dir:
        return tflite_file_path

    output_dir_path = Path(output_dir)
    if not output_dir_path.exists():
        output_dir_path.mkdir()

    return output_dir_path / tflite_file_path


def save_model(tflite_model: Any, model_name: str, output_dir: Optional[str]) -> None:
    """Save model."""
    tflite_file_path = get_model_path(model_name, output_dir)
    with open(tflite_file_path, "wb") as tflite_file:
        print(f"Model {model_name} saved to {tflite_file_path}")
        tflite_file.write(tflite_model)


def representative_dataset(model: Any) -> Callable:
    """Sample dataset used for quantization."""

    def dataset() -> Iterable:
        for _ in range(100):
            data = np.random.rand(1, *model.input_shape[1:])
            yield [data.astype(np.float32)]

    return dataset


def gen_models(output_dir: Optional[str], specific_model: Optional[str]) -> None:
    """Generate test models."""
    for model_name, model_creator in models.items():
        if specific_model and model_name != specific_model:
            continue

        print(f"==> Generate {model_name} ...")
        model = model_creator()

        converter = tf.lite.TFLiteConverter.from_keras_model(model)

        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset(model)
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8

        tflite_model = converter.convert()
        save_model(tflite_model, model_name, output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir", help="Path to the output directory where models will be saved"
    )
    parser.add_argument(
        "--model_name",
        help="Name of the particular model to generate",
        choices=models.keys(),
    )
    args = parser.parse_args()

    gen_models(args.output_dir, args.model_name)
