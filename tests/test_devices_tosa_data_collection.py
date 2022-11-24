# SPDX-FileCopyrightText: Copyright 2022, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for TOSA data collection module."""
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from mlia.backend.tosa_checker.compat import TOSACompatibilityInfo
from mlia.core.context import ExecutionContext
from mlia.devices.tosa.data_collection import TOSAOperatorCompatibility


def test_tosa_data_collection(
    monkeypatch: pytest.MonkeyPatch, test_tflite_model: Path, tmpdir: str
) -> None:
    """Test TOSA data collection."""
    monkeypatch.setattr(
        "mlia.devices.tosa.data_collection.get_tosa_compatibility_info",
        MagicMock(return_value=TOSACompatibilityInfo(True, [])),
    )
    context = ExecutionContext(working_dir=tmpdir)
    collector = TOSAOperatorCompatibility(test_tflite_model)
    collector.set_context(context)

    data_item = collector.collect_data()

    assert isinstance(data_item, TOSACompatibilityInfo)
