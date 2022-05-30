# SPDX-FileCopyrightText: Copyright 2022, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Performance estimation tests."""
from unittest.mock import MagicMock

import pytest

from mlia.devices.ethosu.performance import MemorySizeType
from mlia.devices.ethosu.performance import MemoryUsage


def test_memory_usage_conversion() -> None:
    """Test MemoryUsage objects conversion."""
    memory_usage_in_kb = MemoryUsage(1, 2, 3, 4, 5, MemorySizeType.KILOBYTES)
    assert memory_usage_in_kb.in_kilobytes() == memory_usage_in_kb

    memory_usage_in_bytes = MemoryUsage(
        1 * 1024, 2 * 1024, 3 * 1024, 4 * 1024, 5 * 1024
    )
    assert memory_usage_in_bytes.in_kilobytes() == memory_usage_in_kb


def mock_performance_estimation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mock performance estimation."""
    monkeypatch.setattr(
        "mlia.tools.aiet_wrapper.estimate_performance",
        MagicMock(return_value=MagicMock()),
    )
