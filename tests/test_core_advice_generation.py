# Copyright (C) 2021-2022, Arm Ltd.
"""Tests for module advice_generation."""
from typing import List

import pytest
from mlia.core.advice_generation import Advice
from mlia.core.advice_generation import advice_category
from mlia.core.advice_generation import FactBasedAdviceProducer
from mlia.core.common import AdviceCategory
from mlia.core.common import DataItem
from mlia.core.context import Context


def test_advice_generation() -> None:
    """Test advice generation."""

    class SampleProducer(FactBasedAdviceProducer):
        """Sample producer."""

        def produce_advice(self, data_item: DataItem) -> None:
            """Process data."""
            self.add_advice([f"Advice for {data_item}"])

    producer = SampleProducer()
    producer.produce_advice(123)
    producer.produce_advice("hello")

    advice = producer.get_advice()
    assert advice == [Advice(["Advice for 123"]), Advice(["Advice for hello"])]


@pytest.mark.parametrize(
    "category, expected_advice",
    [
        [
            AdviceCategory.OPERATORS,
            [Advice(["Good advice!"])],
        ],
        [
            AdviceCategory.PERFORMANCE,
            [],
        ],
    ],
)
def test_advice_category_decorator(
    category: AdviceCategory,
    expected_advice: List[Advice],
    dummy_context: Context,
) -> None:
    """Test for advice_category decorator."""

    class SampleAdviceProducer(FactBasedAdviceProducer):
        """Sample advice producer."""

        @advice_category(AdviceCategory.OPERATORS)
        def produce_advice(self, data_item: DataItem) -> None:
            """Produce the advice."""
            self.add_advice(["Good advice!"])

    producer = SampleAdviceProducer()
    dummy_context.update(
        advice_category=category, event_handlers=[], config_parameters={}
    )
    producer.set_context(dummy_context)

    producer.produce_advice("some_data")
    advice = producer.get_advice()

    assert advice == expected_advice
