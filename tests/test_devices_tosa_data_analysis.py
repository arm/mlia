# SPDX-FileCopyrightText: Copyright 2022, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for TOSA data analysis module."""
from typing import List

import pytest

from mlia.core.common import DataItem
from mlia.core.data_analysis import Fact
from mlia.devices.tosa.data_analysis import ModelIsNotTOSACompatible
from mlia.devices.tosa.data_analysis import ModelIsTOSACompatible
from mlia.devices.tosa.data_analysis import TOSADataAnalyzer
from mlia.devices.tosa.operators import TOSACompatibilityInfo


@pytest.mark.parametrize(
    "input_data, expected_facts",
    [
        [
            TOSACompatibilityInfo(True, []),
            [ModelIsTOSACompatible()],
        ],
        [
            TOSACompatibilityInfo(False, []),
            [ModelIsNotTOSACompatible()],
        ],
    ],
)
def test_tosa_data_analyzer(input_data: DataItem, expected_facts: List[Fact]) -> None:
    """Test TOSA data analyzer."""
    analyzer = TOSADataAnalyzer()
    analyzer.analyze_data(input_data)
    assert analyzer.get_analyzed_data() == expected_facts
