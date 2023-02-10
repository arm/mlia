# SPDX-FileCopyrightText: Copyright 2022-2023, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Pytest conf module."""
import shutil
from pathlib import Path
from typing import Generator

import pytest
import tensorflow as tf

from mlia.backend.vela.compiler import optimize_model
from mlia.core.context import ExecutionContext
from mlia.nn.tensorflow.utils import convert_to_tflite
from mlia.nn.tensorflow.utils import save_keras_model
from mlia.nn.tensorflow.utils import save_tflite_model
from mlia.target.ethos_u.config import EthosUConfiguration


@pytest.fixture(scope="session", name="test_resources_path")
def fixture_test_resources_path() -> Path:
    """Return test resources path."""
    return Path(__file__).parent / "test_resources"


@pytest.fixture(name="sample_context")
def fixture_sample_context(tmpdir: str) -> ExecutionContext:
    """Return sample context fixture."""
    return ExecutionContext(output_dir=tmpdir)


@pytest.fixture(scope="session")
def non_optimised_input_model_file(test_tflite_model: Path) -> Path:
    """Provide the path to a quantized test model file."""
    return test_tflite_model


@pytest.fixture(scope="session")
def optimised_input_model_file(test_tflite_vela_model: Path) -> Path:
    """Provide path to Vela-optimised test model file."""
    return test_tflite_vela_model


@pytest.fixture(scope="session")
def invalid_input_model_file(test_tflite_invalid_model: Path) -> Path:
    """Provide the path to an invalid test model file."""
    return test_tflite_invalid_model


def get_test_keras_model() -> tf.keras.Model:
    """Return test Keras model."""
    model = tf.keras.Sequential(
        [
            tf.keras.Input(shape=(28, 28, 1), batch_size=1, name="input"),
            tf.keras.layers.Reshape((28, 28, 1)),
            tf.keras.layers.Conv2D(
                filters=12, kernel_size=(3, 3), activation="relu", name="conv1"
            ),
            tf.keras.layers.Conv2D(
                filters=12, kernel_size=(3, 3), activation="relu", name="conv2"
            ),
            tf.keras.layers.MaxPool2D(2, 2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(10, name="output"),
        ]
    )

    model.compile(optimizer="sgd", loss="mean_squared_error")
    return model


@pytest.fixture(scope="session", name="test_models_path")
def fixture_test_models_path(
    tmp_path_factory: pytest.TempPathFactory,
) -> Generator[Path, None, None]:
    """Provide path to the test models."""
    tmp_path = tmp_path_factory.mktemp("models")

    keras_model = get_test_keras_model()
    save_keras_model(keras_model, tmp_path / "test_model.h5")

    tflite_model = convert_to_tflite(keras_model, quantized=True)
    tflite_model_path = tmp_path / "test_model.tflite"
    save_tflite_model(tflite_model, tflite_model_path)

    tflite_vela_model = tmp_path / "test_model_vela.tflite"

    target_config = EthosUConfiguration.load_profile("ethos-u55-256")
    optimize_model(
        tflite_model_path,
        target_config.compiler_options,
        tflite_vela_model,
    )

    tf.saved_model.save(keras_model, str(tmp_path / "tf_model_test_model"))

    invalid_tflite_model = tmp_path / "invalid.tflite"
    invalid_tflite_model.touch()

    yield tmp_path

    shutil.rmtree(tmp_path)


@pytest.fixture(scope="session", name="test_keras_model")
def fixture_test_keras_model(test_models_path: Path) -> Path:
    """Return test Keras model."""
    return test_models_path / "test_model.h5"


@pytest.fixture(scope="session", name="test_tflite_model")
def fixture_test_tflite_model(test_models_path: Path) -> Path:
    """Return test TensorFlow Lite model."""
    return test_models_path / "test_model.tflite"


@pytest.fixture(scope="session", name="test_tflite_vela_model")
def fixture_test_tflite_vela_model(test_models_path: Path) -> Path:
    """Return test Vela-optimized TensorFlow Lite model."""
    return test_models_path / "test_model_vela.tflite"


@pytest.fixture(scope="session", name="test_tf_model")
def fixture_test_tf_model(test_models_path: Path) -> Path:
    """Return test TensorFlow Lite model."""
    return test_models_path / "tf_model_test_model"


@pytest.fixture(scope="session", name="test_tflite_invalid_model")
def fixture_test_tflite_invalid_model(test_models_path: Path) -> Path:
    """Return test invalid TensorFlow Lite model."""
    return test_models_path / "invalid.tflite"
