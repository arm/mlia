# Copyright 2021, Arm Ltd.
"""Tests for metrics module."""
from mlia.config import EthosU55
from mlia.metrics import MemoryUsage
from mlia.metrics import NPUCycles
from mlia.metrics import PerformanceMetrics


def test_performance_metrics_to_dict() -> None:
    """Test if converting to dictionary works as expected."""
    perf_metrics_from_default_init = PerformanceMetrics(
        EthosU55(), NPUCycles(0, 0, 0, 0, 0, 0), MemoryUsage(0, 0, 0, 0, 0)
    )
    perf_metrics_dict = dict(perf_metrics_from_default_init)
    assert isinstance(perf_metrics_dict, dict)
