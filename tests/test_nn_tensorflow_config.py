# Copyright 2021, Arm Ltd.
"""Tests for config module."""
from pathlib import Path

import pytest
from mlia.nn.tensorflow.config import get_model
from mlia.nn.tensorflow.config import KerasModel
from mlia.nn.tensorflow.config import TFLiteModel
from mlia.nn.tensorflow.config import TfModel


def test_convert_keras_to_tflite(test_models_path: Path, tmp_path: Path) -> None:
    """Test Keras to TFLite conversion."""
    model = test_models_path / "simple_model.h5"
    keras_model = KerasModel(str(model))

    tflite_model_path = tmp_path / "test.tflite"
    keras_model.convert_to_tflite(tflite_model_path)

    assert tflite_model_path.is_file()
    assert tflite_model_path.stat().st_size > 0


def test_convert_tf_to_tflite(test_models_path: Path, tmp_path: Path) -> None:
    """Test TF saved model to TFLite conversion."""
    model = test_models_path / "tf_model_simple_3_layers_model"
    tf_model = TfModel(model)

    tflite_model_path = tmp_path / "test.tflite"
    tf_model.convert_to_tflite(tflite_model_path)

    assert tflite_model_path.is_file()
    assert tflite_model_path.stat().st_size > 0


@pytest.mark.parametrize(
    "model_path, expected_type",
    [
        ("test.tflite", TFLiteModel),
        ("test.h5", KerasModel),
        ("test.hdf5", KerasModel),
    ],
)
def test_get_model_file(model_path: str, expected_type: type) -> None:
    """Test TFLite model type."""
    model = get_model(model_path)
    assert isinstance(model, expected_type)


@pytest.mark.parametrize(
    "model_path, expected_type", [("tf_model_simple_3_layers_model", TfModel)]
)
def test_get_model_dir(
    test_models_path: Path, model_path: str, expected_type: type
) -> None:
    """Test TFLite model type."""
    model = get_model(str(test_models_path / model_path))
    assert isinstance(model, expected_type)
