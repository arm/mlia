# SPDX-FileCopyrightText: Copyright 2022-2023, 2025, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for ArmNN TensorFlow Lite Delegate backend."""
import importlib
import sys
import warnings
from typing import cast

from mlia.backend.armnn_tflite_delegate.compat import (
    ARMNN_TFLITE_DELEGATE,
)
from mlia.nn.tensorflow.tflite_graph import TFL_OP


def test_compat_data() -> None:
    """Make sure all data contains the necessary items."""
    builtin_tfl_ops = {op.name for op in TFL_OP}
    assert "backend" in ARMNN_TFLITE_DELEGATE
    assert "ops" in ARMNN_TFLITE_DELEGATE

    ops = cast(dict, ARMNN_TFLITE_DELEGATE["ops"])
    for data in ops.values():
        assert "builtin_ops" in data
        for comp in data["builtin_ops"]:
            assert comp in builtin_tfl_ops
        assert "custom_ops" in data


def test_backend_module_deprecation_warning() -> None:
    """Test that importing the backend module triggers deprecation warning."""
    with warnings.catch_warnings(record=True) as warning_list:
        warnings.simplefilter("always")

        # Re-import the backend module to trigger the warning
        # Use importlib to force a fresh import

        # Remove module from cache if it exists to force re-import
        module_name = "mlia.backend.armnn_tflite_delegate"
        if module_name in sys.modules:
            del sys.modules[module_name]

        # Import the module
        importlib.import_module(module_name)

        # Check that a deprecation warning was issued
        deprecation_warnings = [
            w for w in warning_list if issubclass(w.category, DeprecationWarning)
        ]
        assert (
            any(deprecation_warnings) > 0
        ), "No DeprecationWarning was issued when importing backend module"

        # Check the warning message content
        warning_message = str(deprecation_warnings[0].message)
        assert "ArmNN TensorFlow Lite Delegate backend is deprecated" in warning_message
        assert "will be removed in the next major release" in warning_message
        assert "unmaintained project" in warning_message
