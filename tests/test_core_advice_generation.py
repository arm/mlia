# Copyright 2021, Arm Ltd.
"""Tests for module advice_generation."""
from mlia.core.advice_generation import Advice
from mlia.core.advice_generation import FactBasedAdviceProducer
from mlia.core.common import DataItem


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
