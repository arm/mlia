# Copyright 2021, Arm Ltd.
"""Tests for module data_analysis."""
from dataclasses import dataclass

from mlia.core.common import DataItem
from mlia.core.data_analysis import Fact
from mlia.core.data_analysis import FactExtractor


def test_fact_extractor() -> None:
    """Test fact extractor."""

    @dataclass
    class SampleFact(Fact):
        """Sample fact."""

        msg: str

    class SampleExtractor(FactExtractor):
        """Sample extractor."""

        def analyze_data(self, data_item: DataItem) -> None:
            return self.add_fact(SampleFact(f"Fact for {data_item}"))

    extractor = SampleExtractor()
    extractor.analyze_data(42)
    extractor.analyze_data("some data")

    facts = extractor.get_analyzed_data()
    assert facts == [SampleFact("Fact for 42"), SampleFact("Fact for some data")]
