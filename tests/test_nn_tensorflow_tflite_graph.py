# SPDX-FileCopyrightText: Copyright 2022-2023, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the tflite_graph module."""
import json
from pathlib import Path

import pytest
import tensorflow as tf
from tensorflow.lite.python.schema_py_generated import ModelT

from mlia.nn.tensorflow.tflite_graph import load_fb
from mlia.nn.tensorflow.tflite_graph import Op
from mlia.nn.tensorflow.tflite_graph import parse_subgraphs
from mlia.nn.tensorflow.tflite_graph import save_fb
from mlia.nn.tensorflow.tflite_graph import TensorInfo
from mlia.nn.tensorflow.tflite_graph import TFL_ACTIVATION_FUNCTION
from mlia.nn.tensorflow.tflite_graph import TFL_OP
from mlia.nn.tensorflow.tflite_graph import TFL_TYPE
from tests.utils.rewrite import models_are_equal


def test_tensor_info() -> None:
    """Test class 'TensorInfo'."""
    expected = {
        "name": "Test",
        "type": TFL_TYPE.INT8.name,
        "shape": (1,),
        "is_variable": False,
    }
    info = TensorInfo(**expected)
    assert vars(info) == expected

    expected = {
        "name": "Test2",
        "type": TFL_TYPE.FLOAT32.name,
        "shape": [2, 3],
        "is_variable": True,
    }
    tensor_dict = {
        "name": [ord(c) for c in expected["name"]],
        "type": TFL_TYPE[expected["type"]],
        "shape": expected["shape"],
        "is_variable": expected["is_variable"],
    }
    info = TensorInfo.from_dict(tensor_dict)
    assert vars(info) == expected

    json_repr = json.loads(repr(info))
    assert vars(info) == json_repr

    assert str(info)


def test_op() -> None:
    """Test class 'Op'."""
    expected = {
        "type": TFL_OP.CONV_2D.name,
        "builtin_options": {},
        "inputs": [],
        "outputs": [],
        "custom_type": None,
    }
    oper = Op(**expected)
    assert vars(oper) == expected

    expected["builtin_options"] = {"some_random_option": 3.14}
    oper = Op(**expected)
    assert vars(oper) == expected

    activation_func = TFL_ACTIVATION_FUNCTION.RELU
    expected["builtin_options"] = {"fused_activation_function": activation_func.value}
    oper = Op(**expected)
    assert oper.builtin_options
    assert oper.builtin_options["fused_activation_function"] == activation_func.name

    assert str(oper)
    assert repr(oper)


def test_parse_subgraphs(test_tflite_model: Path) -> None:
    """Test function 'parse_subgraphs'."""
    model = parse_subgraphs(test_tflite_model)
    assert len(model) == 1
    assert len(model[0]) == 5
    for oper in model[0]:
        assert TFL_OP[oper.type] in TFL_OP
        assert len(oper.inputs) > 0
        assert len(oper.outputs) > 0


def test_load_save(test_tflite_model: Path, tmp_path: Path) -> None:
    """Test the load/save functions for TensorFlow Lite models."""
    with pytest.raises(FileNotFoundError):
        load_fb("THIS_IS_NOT_A_REAL_FILE")

    model = load_fb(test_tflite_model)
    assert isinstance(model, ModelT)
    assert model.subgraphs

    output_file = tmp_path / "test.tflite"
    assert not output_file.is_file()
    save_fb(model, output_file)
    assert output_file.is_file()

    model_copy = load_fb(str(output_file))
    assert models_are_equal(model, model_copy)

    # Double check that the TensorFlow Lite Interpreter can still load the file.
    tf.lite.Interpreter(model_path=str(output_file))
