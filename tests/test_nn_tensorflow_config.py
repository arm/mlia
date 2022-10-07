# SPDX-FileCopyrightText: Copyright 2022, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for config module."""
from contextlib import ExitStack as does_not_raise
from pathlib import Path
from typing import Any

import pytest

from mlia.nn.tensorflow.config import get_model
from mlia.nn.tensorflow.config import KerasModel
from mlia.nn.tensorflow.config import TFLiteModel
from mlia.nn.tensorflow.config import TfModel


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


@pytest.mark.parametrize(
    "model_path, expected_type, expected_error",
    [
        ("test.tflite", TFLiteModel, does_not_raise()),
        ("test.h5", KerasModel, does_not_raise()),
        ("test.hdf5", KerasModel, does_not_raise()),
        (
            "test.model",
            None,
            pytest.raises(
                Exception,
                match="The input model format is not supported"
                r"\(supported formats: TensorFlow Lite, Keras, "
                r"TensorFlow saved model\)!",
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
