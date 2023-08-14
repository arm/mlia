# SPDX-FileCopyrightText: Copyright 2022-2023, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Collection of Cortex-A operator compatibility information."""
from __future__ import annotations

from mlia.backend.armnn_tflite_delegate.ops.v22_08 import OPERATORS as ops_v22_08
from mlia.backend.armnn_tflite_delegate.ops.v22_08 import VERSION as v22_08
from mlia.backend.armnn_tflite_delegate.ops.v22_11 import OPERATORS as ops_v22_11
from mlia.backend.armnn_tflite_delegate.ops.v22_11 import VERSION as v22_11
from mlia.backend.armnn_tflite_delegate.ops.v23_05 import OPERATORS as ops_v23_05
from mlia.backend.armnn_tflite_delegate.ops.v23_05 import VERSION as v23_05

ARMNN_TFLITE_DELEGATE: dict = {
    "backend": "Arm NN TensorFlow Lite Delegate",
    "ops": {
        v22_08: ops_v22_08,
        v22_11: ops_v22_11,
        v23_05: ops_v23_05,
    },
}
