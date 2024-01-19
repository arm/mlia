# SPDX-FileCopyrightText: Copyright 2022-2024, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for config module."""
from contextlib import ExitStack as does_not_raise
from pathlib import Path
from typing import Any
from typing import Generator

import numpy as np
import pytest

from mlia.nn.rewrite.core.utils.numpy_tfrecord import numpytf_read
from mlia.nn.tensorflow.config import get_model
from mlia.nn.tensorflow.config import KerasModel
from mlia.nn.tensorflow.config import ModelConfiguration
from mlia.nn.tensorflow.config import TFLiteModel
from mlia.nn.tensorflow.config import TfModel
from tests.conftest import create_tfrecord


def test_model_configuration(test_keras_model: Path) -> None:
    """Test ModelConfiguration class."""
    model = ModelConfiguration(model_path=test_keras_model)
    assert test_keras_model.match(model.model_path)
    with pytest.raises(NotImplementedError):
        model.convert_to_keras("keras_model.h5")
    with pytest.raises(NotImplementedError):
        model.convert_to_tflite("model.tflite")


def test_convert_keras_to_tflite(tmp_path: Path, test_keras_model: Path) -> None:
    """Test Keras to TensorFlow Lite conversion."""
    keras_model = KerasModel(test_keras_model)

    tflite_model_path = tmp_path / "test.tflite"
    keras_model.convert_to_tflite(tflite_model_path)

    assert tflite_model_path.is_file()
    assert tflite_model_path.stat().st_size > 0


def test_convert_tf_to_tflite(tmp_path: Path, test_tf_model: Path) -> None:
    """Test TensorFlow saved model to TensorFlow Lite conversion."""
    tf_model = TfModel(test_tf_model)

    tflite_model_path = tmp_path / "test.tflite"
    tf_model.convert_to_tflite(tflite_model_path)

    assert tflite_model_path.is_file()
    assert tflite_model_path.stat().st_size > 0


def test_invalid_tflite_model(tmp_path: Path) -> None:
    """Check that a RuntimeError is raised when a TFLite file is invalid."""
    model_path = tmp_path / "test.tflite"
    model_path.write_text("Not a TFLite file!")

    with pytest.raises(RuntimeError):
        TFLiteModel(model_path=model_path)


@pytest.mark.parametrize(
    "model_path, expected_type, expected_error",
    [
        ("test.tflite", TFLiteModel, pytest.raises(RuntimeError)),
        ("test.h5", KerasModel, does_not_raise()),
        ("test.hdf5", KerasModel, does_not_raise()),
        (
            "test.model",
            None,
            pytest.raises(
                ValueError,
                match=(
                    "The input model format is not supported "
                    r"\(supported formats: TensorFlow Lite, Keras, "
                    r"TensorFlow saved model\)!"
                ),
            ),
        ),
    ],
)
def test_get_model_file(
    model_path: str, expected_type: type, expected_error: Any
) -> None:
    """Test TensorFlow Lite model type."""
    with expected_error:
        model = get_model(model_path)
        assert isinstance(model, expected_type)


@pytest.mark.parametrize(
    "model_path, expected_type", [("tf_model_test_model", TfModel)]
)
def test_get_model_dir(
    test_models_path: Path, model_path: str, expected_type: type
) -> None:
    """Test TensorFlow Lite model type."""
    model = get_model(str(test_models_path / model_path))
    assert isinstance(model, expected_type)


@pytest.fixture(scope="session", name="test_tfrecord_fp32_batch_3")
def fixture_test_tfrecord_fp32_batch_3(
    tmp_path_factory: pytest.TempPathFactory,
) -> Generator[Path, None, None]:
    """Create tfrecord (same as test_tfrecord_fp32) but with batch size 3."""

    def random_data() -> np.ndarray:
        return np.random.rand(3, 28, 28, 1).astype(np.float32)

    yield from create_tfrecord(tmp_path_factory, random_data)


def test_tflite_model_call(
    test_tflite_model_fp32: Path, test_tfrecord_fp32_batch_3: Path
) -> None:
    """Test inference function of class TFLiteModel."""
    model = TFLiteModel(test_tflite_model_fp32, batch_size=2)
    data = numpytf_read(test_tfrecord_fp32_batch_3)
    for named_input in data.as_numpy_iterator():
        res = model(named_input)
        assert res


def test_tflite_model_is_tensor_quantized(test_tflite_model: Path) -> None:
    """Test function TFLiteModel.is_tensor_quantized()."""
    model = TFLiteModel(test_tflite_model)
    input_details = model.input_details[0]
    assert model.is_tensor_quantized(name=input_details["name"])
    assert model.is_tensor_quantized(idx=input_details["index"])
    with pytest.raises(ValueError):
        assert model.is_tensor_quantized()
    with pytest.raises(NameError):
        assert model.is_tensor_quantized(name="NAME_DOES_NOT_EXIST")
