# SPDX-FileCopyrightText: Copyright 2022, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for advice generation."""
from __future__ import annotations

import pytest

from mlia.core.advice_generation import Advice
from mlia.core.common import AdviceCategory
from mlia.core.common import DataItem
from mlia.core.context import ExecutionContext
from mlia.devices.tosa.advice_generation import TOSAAdviceProducer
from mlia.devices.tosa.data_analysis import ModelIsNotTOSACompatible
from mlia.devices.tosa.data_analysis import ModelIsTOSACompatible


@pytest.mark.parametrize(
    "input_data, advice_category, expected_advice",
    [
        [
            ModelIsNotTOSACompatible(),
            AdviceCategory.OPERATORS,
            [
                Advice(
                    [
                        "Some operators in the model are not TOSA compatible. "
                        "Please, refer to the operators table for more information."
                    ]
                )
            ],
        ],
        [
            ModelIsTOSACompatible(),
            AdviceCategory.OPERATORS,
            [Advice(["Model is fully TOSA compatible."])],
        ],
    ],
)
def test_tosa_advice_producer(
    tmpdir: str,
    input_data: DataItem,
    advice_category: AdviceCategory,
    expected_advice: list[Advice],
) -> None:
    """Test TOSA advice producer."""
    producer = TOSAAdviceProducer()

    context = ExecutionContext(
        advice_category=advice_category,
        working_dir=tmpdir,
    )

    producer.set_context(context)
    producer.produce_advice(input_data)

    assert producer.get_advice() == expected_advice
