# SPDX-FileCopyrightText: Copyright 2022, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Pytest conf module."""
import shutil
import tarfile
from pathlib import Path
from typing import Any
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
    return ExecutionContext(working_dir=tmpdir)


@pytest.fixture(scope="session")
def test_systems_path(test_resources_path: Path) -> Path:
    """Return test systems path in a pytest fixture."""
    return test_resources_path / "backends" / "systems"


@pytest.fixture(scope="session")
def test_applications_path(test_resources_path: Path) -> Path:
    """Return test applications path in a pytest fixture."""
    return test_resources_path / "backends" / "applications"


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


@pytest.fixture(autouse=True)
def test_resources(monkeypatch: pytest.MonkeyPatch, test_resources_path: Path) -> Any:
    """Force using test resources as middleware's repository."""

    def get_test_resources() -> Path:
        """Return path to the test resources."""
        return test_resources_path / "backends"

    monkeypatch.setattr(
        "mlia.backend.executor.fs.get_backend_resources", get_test_resources
    )
    yield


def create_archive(
    archive_name: str, source: Path, destination: Path, with_root_folder: bool = False
) -> None:
    """Create archive from directory source."""
    with tarfile.open(destination / archive_name, mode="w:gz") as tar:
        for item in source.iterdir():
            item_name = item.name
            if with_root_folder:
                item_name = f"{source.name}/{item_name}"
            tar.add(item, item_name)


def process_directory(source: Path, destination: Path) -> None:
    """Process resource directory."""
    destination.mkdir()

    for item in source.iterdir():
        if item.is_dir():
            create_archive(f"{item.name}.tar.gz", item, destination)
            create_archive(f"{item.name}_dir.tar.gz", item, destination, True)


@pytest.fixture(scope="session", autouse=True)
def add_archives(
    test_resources_path: Path, tmp_path_factory: pytest.TempPathFactory
) -> Any:
    """Generate archives of the test resources."""
    tmp_path = tmp_path_factory.mktemp("archives")

    archives_path = tmp_path / "archives"
    archives_path.mkdir()

    if (archives_path_link := test_resources_path / "archives").is_symlink():
        archives_path_link.unlink()

    archives_path_link.symlink_to(archives_path, target_is_directory=True)

    for item in ["applications", "systems"]:
        process_directory(test_resources_path / "backends" / item, archives_path / item)

    yield

    archives_path_link.unlink()
    shutil.rmtree(tmp_path)


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
    device = EthosUConfiguration("ethos-u55-256")
    optimize_model(tflite_model_path, device.compiler_options, tflite_vela_model)

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
