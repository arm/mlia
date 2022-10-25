# SPDX-FileCopyrightText: Copyright 2022, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for Cortex-A operator compatibility."""
from pathlib import Path

import pytest
import tensorflow as tf

from mlia.devices.cortexa import operator_compatibility as op_compat
from mlia.devices.cortexa.operators import CortexACompatibilityInfo
from mlia.devices.cortexa.operators import get_cortex_a_compatibility_info
from mlia.devices.cortexa.operators import Operator
from mlia.nn.tensorflow.tflite_graph import TFL_OP
from mlia.nn.tensorflow.utils import convert_to_tflite


def test_op_compat_data() -> None:
    """Make sure all data contains the necessary items."""
    builtin_tfl_ops = {op.name for op in TFL_OP}
    for data in [op_compat.ARMNN_TFLITE_DELEGATE]:
        assert "metadata" in data
        assert "backend" in data["metadata"]
        assert "version" in data["metadata"]
        assert "builtin_ops" in data
        for comp in data["builtin_ops"]:
            assert comp in builtin_tfl_ops
        assert "custom_ops" in data


def check_get_cortex_a_compatibility_info(
    model_path: Path,
    expected_success: bool,
) -> None:
    """Check the function 'get_cortex_a_compatibility_info'."""
    compat_info = get_cortex_a_compatibility_info(model_path)
    assert isinstance(compat_info, CortexACompatibilityInfo)
    assert expected_success == compat_info.cortex_a_compatible
    assert compat_info.operators
    for oper in compat_info.operators:
        assert oper.name
        assert oper.location
        assert oper.support_type in Operator.SupportType


def test_get_cortex_a_compatibility_info_compatible(
    test_tflite_model: Path,
) -> None:
    """Test a fully compatible TensorFlow Lite model."""
    check_get_cortex_a_compatibility_info(test_tflite_model, expected_success=True)


def test_get_cortex_a_compatibility_info_not_compatible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Construct and test a NOT fully compatible TensorFlow Lite model."""
    keras_model = tf.keras.Sequential(
        [
            tf.keras.Input(shape=(28, 28, 1), batch_size=1, name="input"),
            tf.keras.layers.Conv2D(
                filters=12, kernel_size=(3, 3), activation="softmax", name="conv1"
            ),
            tf.keras.layers.LeakyReLU(),
        ]
    )
    keras_model.compile(optimizer="sgd", loss="mean_squared_error")
    tflite_model = convert_to_tflite(keras_model, quantized=False)

    monkeypatch.setattr(
        "mlia.nn.tensorflow.tflite_graph.load_tflite", lambda _p: tflite_model
    )
    check_get_cortex_a_compatibility_info(
        Path("NOT_USED_BECAUSE_OF_MOCKING"), expected_success=False
    )
