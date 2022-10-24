# SPDX-FileCopyrightText: Copyright 2022, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for Cortex-A data collection module."""
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from mlia.core.context import ExecutionContext
from mlia.devices.cortexa.data_collection import CortexAOperatorCompatibility
from mlia.devices.cortexa.operators import CortexACompatibilityInfo


def test_cortex_a_data_collection(
    monkeypatch: pytest.MonkeyPatch, test_tflite_model: Path, tmpdir: str
) -> None:
    """Test Cortex-A data collection."""
    monkeypatch.setattr(
        "mlia.devices.cortexa.data_collection.get_cortex_a_compatibility_info",
        MagicMock(return_value=CortexACompatibilityInfo(True, [])),
    )
    context = ExecutionContext(working_dir=tmpdir)
    collector = CortexAOperatorCompatibility(test_tflite_model)
    collector.set_context(context)

    data_item = collector.collect_data()

    assert isinstance(data_item, CortexACompatibilityInfo)
