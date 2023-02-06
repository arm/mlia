# SPDX-FileCopyrightText: Copyright 2022-2023, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for Cortex-A operator compatibility."""
from pathlib import Path
from typing import cast

import pytest
import tensorflow as tf

from mlia.backend.armnn_tflite_delegate import compat
from mlia.nn.tensorflow.tflite_graph import TFL_OP
from mlia.nn.tensorflow.utils import convert_to_tflite
from mlia.target.cortex_a.config import CortexAConfiguration
from mlia.target.cortex_a.operators import CortexACompatibilityInfo
from mlia.target.cortex_a.operators import get_cortex_a_compatibility_info


def test_compat_data() -> None:
    """Make sure all data contains the necessary items."""
    builtin_tfl_ops = {op.name for op in TFL_OP}
    assert "backend" in compat.ARMNN_TFLITE_DELEGATE
    assert "ops" in compat.ARMNN_TFLITE_DELEGATE

    ops = cast(dict, compat.ARMNN_TFLITE_DELEGATE["ops"])
    for data in ops.values():
        assert "builtin_ops" in data
        for comp in data["builtin_ops"]:
            assert comp in builtin_tfl_ops
        assert "custom_ops" in data


def check_get_cortex_a_compatibility_info(
    model_path: Path,
    expected_success: bool,
) -> None:
    """Check the function 'get_cortex_a_compatibility_info'."""
    compat_info = get_cortex_a_compatibility_info(
        model_path, CortexAConfiguration.load_profile("cortex-a")
    )
    assert isinstance(compat_info, CortexACompatibilityInfo)
    assert expected_success == compat_info.is_cortex_a_compatible
    assert compat_info.operators
    for oper in compat_info.operators:
        assert oper.name
        assert oper.location
        assert (
            compat_info.get_support_type(oper) in CortexACompatibilityInfo.SupportType
        )


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
