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
from mlia.devices.cortexa.operators import CortexACompatibilityInfo


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
    ],
)
def test_cortex_a_data_analyzer(
    input_data: DataItem, expected_facts: list[Fact]
) -> None:
    """Test Cortex-A data analyzer."""
    analyzer = CortexADataAnalyzer()
    analyzer.analyze_data(input_data)
    assert analyzer.get_analyzed_data() == expected_facts
