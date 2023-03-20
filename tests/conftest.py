# SPDX-FileCopyrightText: Copyright 2022-2023, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Pytest conf module."""
import shutil
from pathlib import Path
from typing import Callable
from typing import Generator

import numpy as np
import pytest
import tensorflow as tf

from mlia.backend.vela.compiler import optimize_model
from mlia.core.context import ExecutionContext
from mlia.nn.rewrite.core.utils.numpy_tfrecord import NumpyTFWriter
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


TEST_MODEL_KERAS_FILE = "test_model.h5"
TEST_MODEL_TFLITE_FP32_FILE = "test_model_fp32.tflite"
TEST_MODEL_TFLITE_INT8_FILE = "test_model_int8.tflite"
TEST_MODEL_TFLITE_VELA_FILE = "test_model_vela.tflite"
TEST_MODEL_TF_SAVED_MODEL_FILE = "tf_model_test_model"
TEST_MODEL_INVALID_FILE = "invalid.tflite"


@pytest.fixture(scope="session", name="test_models_path")
def fixture_test_models_path(
    tmp_path_factory: pytest.TempPathFactory,
) -> Generator[Path, None, None]:
    """Provide path to the test models."""
    tmp_path = tmp_path_factory.mktemp("models")

    # Keras Model
    keras_model = get_test_keras_model()
    save_keras_model(keras_model, tmp_path / TEST_MODEL_KERAS_FILE)

    # Un-quantized TensorFlow Lite model (fp32)
    save_tflite_model(
        convert_to_tflite(keras_model, quantized=False),
        tmp_path / TEST_MODEL_TFLITE_FP32_FILE,
    )

    # Quantized TensorFlow Lite model (int8)
    tflite_model = convert_to_tflite(keras_model, quantized=True)
    tflite_model_path = tmp_path / TEST_MODEL_TFLITE_INT8_FILE
    save_tflite_model(tflite_model, tflite_model_path)

    # Vela-optimized TensorFlow Lite model (int8)
    tflite_vela_model = tmp_path / TEST_MODEL_TFLITE_VELA_FILE
    target_config = EthosUConfiguration.load_profile("ethos-u55-256")
    optimize_model(
        tflite_model_path,
        target_config.compiler_options,
        tflite_vela_model,
    )

    tf.saved_model.save(keras_model, str(tmp_path / TEST_MODEL_TF_SAVED_MODEL_FILE))

    invalid_tflite_model = tmp_path / TEST_MODEL_INVALID_FILE
    invalid_tflite_model.touch()

    yield tmp_path

    shutil.rmtree(tmp_path)


@pytest.fixture(scope="session", name="test_keras_model")
def fixture_test_keras_model(test_models_path: Path) -> Path:
    """Return test Keras model."""
    return test_models_path / TEST_MODEL_KERAS_FILE


@pytest.fixture(scope="session", name="test_tflite_model")
def fixture_test_tflite_model(test_models_path: Path) -> Path:
    """Return test TensorFlow Lite model."""
    return test_models_path / TEST_MODEL_TFLITE_INT8_FILE


@pytest.fixture(scope="session", name="test_tflite_model_fp32")
def fixture_test_tflite_model_fp32(test_models_path: Path) -> Path:
    """Return test TensorFlow Lite model."""
    return test_models_path / TEST_MODEL_TFLITE_FP32_FILE


@pytest.fixture(scope="session", name="test_tflite_vela_model")
def fixture_test_tflite_vela_model(test_models_path: Path) -> Path:
    """Return test Vela-optimized TensorFlow Lite model."""
    return test_models_path / TEST_MODEL_TFLITE_VELA_FILE


@pytest.fixture(scope="session", name="test_tf_model")
def fixture_test_tf_model(test_models_path: Path) -> Path:
    """Return test TensorFlow Lite model."""
    return test_models_path / TEST_MODEL_TF_SAVED_MODEL_FILE


@pytest.fixture(scope="session", name="test_tflite_invalid_model")
def fixture_test_tflite_invalid_model(test_models_path: Path) -> Path:
    """Return test invalid TensorFlow Lite model."""
    return test_models_path / TEST_MODEL_INVALID_FILE


def _write_tfrecord(
    tfrecord_file: Path,
    data_generator: Callable,
    input_name: str = "serving_default_input:0",
    num_records: int = 3,
) -> None:
    """Write data to a tfrecord."""
    with NumpyTFWriter(str(tfrecord_file)) as writer:
        for _ in range(num_records):
            writer.write({input_name: data_generator()})


@pytest.fixture(scope="session", name="test_tfrecord")
def fixture_test_tfrecord(
    tmp_path_factory: pytest.TempPathFactory,
) -> Generator[Path, None, None]:
    """Create a tfrecord with random data matching fixture 'test_tflite_model'."""
    tmp_path = tmp_path_factory.mktemp("tfrecords")
    tfrecord_file = tmp_path / "test_int8.tfrecord"

    def random_data() -> np.ndarray:
        return np.random.randint(low=-127, high=128, size=(1, 28, 28, 1), dtype=np.int8)

    _write_tfrecord(tfrecord_file, random_data)

    yield tfrecord_file

    shutil.rmtree(tmp_path)


@pytest.fixture(scope="session", name="test_tfrecord_fp32")
def fixture_test_tfrecord_fp32(
    tmp_path_factory: pytest.TempPathFactory,
) -> Generator[Path, None, None]:
    """Create tfrecord with random data matching fixture 'test_tflite_model_fp32'."""
    tmp_path = tmp_path_factory.mktemp("tfrecords")
    tfrecord_file = tmp_path / "test_fp32.tfrecord"

    def random_data() -> np.ndarray:
        return np.random.rand(1, 28, 28, 1).astype(np.float32)

    _write_tfrecord(tfrecord_file, random_data)

    yield tfrecord_file

    shutil.rmtree(tmp_path)
