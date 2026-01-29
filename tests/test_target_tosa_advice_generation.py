# SPDX-FileCopyrightText: Copyright 2022-2023, 2025-2026, Arm Limited and/or
# its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for advice generation."""
from __future__ import annotations

import pytest

from mlia.core.advice_generation import Advice
from mlia.core.common import AdviceCategory
from mlia.core.common import DataItem
from mlia.core.context import ExecutionContext
from mlia.core.output_schema import AdviceCategory as SchemaAdviceCategory
from mlia.core.output_schema import AdviceSeverity
from mlia.target.common.reporters import ModelIsNotTFLiteCompatible
from mlia.target.common.reporters import TFLiteCompatibilityCheckFailed
from mlia.target.tosa.advice_generation import TOSAAdviceProducer
from mlia.target.tosa.data_analysis import ModelIsNotTOSACompatible
from mlia.target.tosa.data_analysis import ModelIsTOSACompatible


@pytest.mark.parametrize(
    "input_data, advice_category, expected_advice",
    [
        [
            ModelIsNotTOSACompatible(),
            {AdviceCategory.COMPATIBILITY},
            [
                Advice(
                    id="0",
                    category=SchemaAdviceCategory.COMPATIBILITY,
                    severity=AdviceSeverity.WARNING,
                    message=(
                        "Some operators in the model are not TOSA compatible. "
                        "Please, refer to the operators table for more information."
                    ),
                )
            ],
        ],
        [
            ModelIsTOSACompatible(),
            {AdviceCategory.COMPATIBILITY},
            [
                Advice(
                    id="0",
                    category=SchemaAdviceCategory.COMPATIBILITY,
                    severity=AdviceSeverity.INFO,
                    message="Model is fully TOSA compatible.",
                )
            ],
        ],
        [
            ModelIsNotTFLiteCompatible(),
            {AdviceCategory.COMPATIBILITY},
            [
                Advice(
                    id="0",
                    category=SchemaAdviceCategory.COMPATIBILITY,
                    severity=AdviceSeverity.ERROR,
                    message=(
                        "Model could not be converted into TensorFlow Lite format. "
                        "Please refer to the table for more details."
                    ),
                )
            ],
        ],
        [
            TFLiteCompatibilityCheckFailed(),
            {AdviceCategory.COMPATIBILITY},
            [
                Advice(
                    id="0",
                    category=SchemaAdviceCategory.COMPATIBILITY,
                    severity=AdviceSeverity.ERROR,
                    message=(
                        "Model could not be converted into TensorFlow Lite format. "
                        "Please refer to the table for more details."
                    ),
                )
            ],
        ],
    ],
)
def test_tosa_advice_producer(
    tmpdir: str,
    input_data: DataItem,
    advice_category: set[AdviceCategory],
    expected_advice: list[Advice],
) -> None:
    """Test TOSA advice producer."""
    producer = TOSAAdviceProducer()

    context = ExecutionContext(
        advice_category=advice_category,
        output_dir=tmpdir,
    )

    producer.set_context(context)
    producer.produce_advice(input_data)

    advice = producer.get_advice()
    assert isinstance(advice, list)
    assert len(advice) == len(expected_advice)
    for actual, expected in zip(advice, expected_advice):
        assert actual.message == expected.message
