# SPDX-FileCopyrightText: Copyright 2022-2023, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Test for module utils/test_utils."""
import os
from pathlib import Path
from pathlib import PosixPath
from unittest.mock import MagicMock

import numpy as np
import pytest
import tensorflow as tf

from mlia.nn.tensorflow import tflite_convert
from mlia.nn.tensorflow.tflite_convert import convert_to_tflite
from mlia.nn.tensorflow.tflite_convert import convert_to_tflite_bytes
from mlia.nn.tensorflow.tflite_convert import main
from mlia.nn.tensorflow.tflite_convert import representative_dataset


def test_generate_representative_dataset() -> None:
    """Test function for generating representative dataset."""
    dataset = representative_dataset([1, 3, 3], 5)
    data = list(dataset())

    assert len(data) == 5
    for elem in data:
        assert isinstance(elem, list)
        assert len(elem) == 1

        ndarray = elem[0]
        assert ndarray.dtype == np.float32
        assert isinstance(ndarray, np.ndarray)


def test_convert_saved_model_to_tflite(test_tf_model: Path) -> None:
    """Test converting SavedModel to TensorFlow Lite."""
    result = convert_to_tflite_bytes(test_tf_model.as_posix())
    assert isinstance(result, bytes)


def test_convert_keras_to_tflite(test_keras_model: Path) -> None:
    """Test converting Keras model to TensorFlow Lite."""
    keras_model = tf.keras.models.load_model(str(test_keras_model))
    result = convert_to_tflite_bytes(keras_model)
    assert isinstance(result, bytes)


def test_save_tflite_model(tmp_path: Path, test_keras_model: Path) -> None:
    """Test saving TensorFlow Lite model."""
    keras_model = tf.keras.models.load_model(str(test_keras_model))

    temp_file = tmp_path / "test_model_saving.tflite"
    convert_to_tflite(keras_model, output_path=temp_file)

    interpreter = tf.lite.Interpreter(model_path=str(temp_file))
    assert interpreter


def test_convert_unknown_model_to_tflite() -> None:
    """Test that unknown model type cannot be converted to TensorFlow Lite."""
    with pytest.raises(
        ValueError, match="Unable to create TensorFlow Lite converter for 123"
    ):
        convert_to_tflite(123)


@pytest.mark.parametrize(
    "convert_options,expected_args,error",
    [
        [
            {
                "input_path": PosixPath("/in"),
                "output_path": PosixPath("/out"),
                "quantized": True,
                "subprocess": True,
            },
            ["/in", "--output", "/out", "--quantize"],
            None,
        ],
        [
            {
                "input_path": None,
                "output_path": None,
                "quantized": True,
                "subprocess": False,
            },
            [True, None],
            None,
        ],
        [
            {
                "input_path": None,
                "output_path": PosixPath("/out"),
                "quantized": False,
                "subprocess": True,
                "model": None,
            },
            ["/in", "/out"],
            "Input path is required",
        ],
        [
            {
                "input_path": PosixPath("/in"),
                "output_path": PosixPath("/out"),
                "quantized": False,
                "subprocess": False,
            },
            [False, PosixPath("/out")],
            None,
        ],
        [
            {
                "input_path": PosixPath("/in"),
                "output_path": PosixPath("/out"),
                "quantized": True,
                "subprocess": False,
            },
            [True, PosixPath("/out")],
            None,
        ],
        [
            {
                "input_path": PosixPath("/in"),
                "output_path": None,
                "quantized": False,
                "subprocess": True,
            },
            ["/in"],
            None,
        ],
        [
            {
                "input_path": PosixPath("/in"),
                "output_path": PosixPath("/out"),
                "quantized": False,
                "subprocess": True,
            },
            ["/in", "--output", "/out"],
            None,
        ],
        [
            {
                "input_path": PosixPath("/in"),
                "output_path": PosixPath("/out"),
                "quantized": True,
                "subprocess": True,
            },
            ["/in", "--output", "/out", "--quantize"],
            None,
        ],
        [
            {
                "output_path": PosixPath("/out"),
                "quantized": True,
                "subprocess": True,
            },
            ["/model_path", "--output", "/out", "--quantize"],
            None,
        ],
    ],
)
def test_convert_to_tflite_subprocess(
    convert_options: dict,
    expected_args: str,
    error: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test if convert_to_tflite calls the subprocess with the correct args."""
    command_mock = MagicMock()
    function_mock = MagicMock()
    model_path_str = "/model_path"
    monkeypatch.setattr(
        "mlia.nn.tensorflow.tflite_convert.command_output", command_mock
    )

    monkeypatch.setattr(
        "mlia.nn.tensorflow.tflite_convert._convert_to_tflite", function_mock
    )

    opts = {"model": model_path_str, **convert_options}

    if error:
        with pytest.raises(Exception) as exc_info:
            convert_to_tflite(**opts)

        assert error in str(exc_info.value)
        command_mock.assert_not_called()
        function_mock.assert_not_called()
        return

    convert_to_tflite(**opts)

    if convert_options["subprocess"]:
        command_mock.assert_called_once()
        function_mock.assert_not_called()
        pyfile = os.path.abspath(tflite_convert.__file__)
        assert command_mock.mock_calls[0].args[0].cmd == [
            "python",
            pyfile,
            *expected_args,
        ]
    else:
        command_mock.assert_not_called()
        function_mock.assert_called_once()
        args = function_mock.mock_calls[0].args
        assert args == (model_path_str, *expected_args)


@pytest.mark.parametrize(
    "args,expected_convert_args",
    [
        ["{}", "{},False,None"],
        ["{} --quantize", "{},True,None"],
        ["{} --output {}", "{},False,{}"],
        ["{} --output {} --quantize", "{},True,{}"],
    ],
)
def test_main(
    args: str,
    expected_convert_args: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test main function, the entry point to subprocess mode."""
    mock = MagicMock()
    monkeypatch.setattr("mlia.nn.tensorflow.tflite_convert._convert_to_tflite", mock)

    input_path = tmp_path
    output_path = tmp_path / "out"
    argv = args.format(input_path, output_path).split()
    main(argv)

    mock.assert_called_once()
    convert_args = mock.mock_calls[0].args
    actual = ",".join(str(arg) for arg in convert_args)
    expected = expected_convert_args.format(input_path, output_path)
    assert actual == expected


def test_main_nonexistent_input() -> None:
    """Test main with missing input model."""
    with pytest.raises(ValueError) as excinfo:
        main(["/missing"])
    assert "Input file doesn't exist: [/missing]" in str(excinfo.value)
