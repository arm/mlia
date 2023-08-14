# SPDX-FileCopyrightText: Copyright 2022-2023, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for ArmNN TensorFlow Lite Delegate backend."""
from typing import cast

from mlia.backend.armnn_tflite_delegate import compat
from mlia.nn.tensorflow.tflite_graph import TFL_OP


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
