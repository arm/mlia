# SPDX-FileCopyrightText: Copyright 2022, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for Cortex-A data analysis module."""
from __future__ import annotations

import pytest

from mlia.core.common import DataItem
from mlia.core.data_analysis import Fact
from mlia.devices.cortexa.data_analysis import CortexADataAnalyzer
from mlia.devices.cortexa.data_analysis import ModelIsCortexACompatible
from mlia.devices.cortexa.data_analysis import ModelIsNotCortexACompatible
from mlia.devices.cortexa.data_analysis import ModelIsNotTFLiteCompatible
from mlia.devices.cortexa.operators import CortexACompatibilityInfo
from mlia.nn.tensorflow.tflite_compat import TFLiteCompatibilityInfo
from mlia.nn.tensorflow.tflite_compat import TFLiteConversionError
from mlia.nn.tensorflow.tflite_compat import TFLiteConversionErrorCode


@pytest.mark.parametrize(
    "input_data, expected_facts",
    [
        [
            CortexACompatibilityInfo(True, []),
            [ModelIsCortexACompatible()],
        ],
        [
            CortexACompatibilityInfo(False, []),
            [ModelIsNotCortexACompatible()],
        ],
        [
            TFLiteCompatibilityInfo(compatible=True),
            [],
        ],
        [
            TFLiteCompatibilityInfo(compatible=False),
            [ModelIsNotTFLiteCompatible(custom_ops=[], flex_ops=[])],
        ],
        [
            TFLiteCompatibilityInfo(
                compatible=False,
                conversion_errors=[
                    TFLiteConversionError(
                        "error",
                        TFLiteConversionErrorCode.NEEDS_CUSTOM_OPS,
                        "custom_op1",
                        [],
                    ),
                    TFLiteConversionError(
                        "error",
                        TFLiteConversionErrorCode.NEEDS_FLEX_OPS,
                        "flex_op1",
                        [],
                    ),
                ],
            ),
            [
                ModelIsNotTFLiteCompatible(
                    custom_ops=["custom_op1"],
                    flex_ops=["flex_op1"],
                )
            ],
        ],
    ],
)
def test_cortex_a_data_analyzer(
    input_data: DataItem, expected_facts: list[Fact]
) -> None:
    """Test Cortex-A data analyzer."""
    analyzer = CortexADataAnalyzer()
    analyzer.analyze_data(input_data)
    assert analyzer.get_analyzed_data() == expected_facts
