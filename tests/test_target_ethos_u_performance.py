# SPDX-FileCopyrightText: Copyright 2022-2024, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Performance estimation tests."""
from unittest.mock import MagicMock

import pytest


def mock_performance_estimation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mock performance estimation."""
    monkeypatch.setattr(
        "mlia.backend.corstone.performance.estimate_performance",
        MagicMock(return_value=MagicMock()),
    )
