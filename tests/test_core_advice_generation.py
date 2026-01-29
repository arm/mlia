# SPDX-FileCopyrightText: Copyright 2022-2023, 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for module advice_generation."""
from __future__ import annotations

import pytest

from mlia.core.advice_generation import advice_category
from mlia.core.advice_generation import FactBasedAdviceProducer
from mlia.core.common import AdviceCategory
from mlia.core.common import DataItem
from mlia.core.context import Context
from mlia.core.output_schema import AdviceCategory as SchemaAdviceCategory
from mlia.core.output_schema import AdviceSeverity


def test_advice_generation() -> None:
    """Test advice generation."""

    class SampleProducer(FactBasedAdviceProducer):
        """Sample producer."""

        def produce_advice(self, data_item: DataItem) -> None:
            """Process data."""
            self.add_advice(
                message=f"Advice for {data_item}",
                category=SchemaAdviceCategory.COMPATIBILITY,
                severity=AdviceSeverity.INFO,
            )

    producer = SampleProducer()
    producer.produce_advice(123)
    producer.produce_advice("hello")

    advice = producer.get_advice()
    assert isinstance(advice, list)
    assert len(advice) == 2
    assert advice[0].message == "Advice for 123"
    assert advice[1].message == "Advice for hello"


@pytest.mark.parametrize(
    "category, expected_count",
    [
        [{AdviceCategory.COMPATIBILITY}, 1],
        [{AdviceCategory.PERFORMANCE}, 0],
    ],
)
def test_advice_category_decorator(
    category: set[AdviceCategory],
    expected_count: int,
    sample_context: Context,
) -> None:
    """Test for advice_category decorator."""

    class SampleAdviceProducer(FactBasedAdviceProducer):
        """Sample advice producer."""

        @advice_category(AdviceCategory.COMPATIBILITY)
        def produce_advice(self, data_item: DataItem) -> None:
            """Produce the advice."""
            self.add_advice(
                message="Good advice!",
                category=SchemaAdviceCategory.COMPATIBILITY,
                severity=AdviceSeverity.INFO,
            )

    producer = SampleAdviceProducer()
    sample_context.update(
        advice_category=category, event_handlers=[], config_parameters={}
    )
    producer.set_context(sample_context)

    producer.produce_advice("some_data")
    advice = producer.get_advice()

    assert isinstance(advice, list)
    assert len(advice) == expected_count
