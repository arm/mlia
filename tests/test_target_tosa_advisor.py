# SPDX-FileCopyrightText: Copyright 2022-2023, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for TOSA advisor."""
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from mlia.core.context import ExecutionContext
from mlia.core.workflow import DefaultWorkflowExecutor
from mlia.target.tosa.advisor import configure_and_get_tosa_advisor
from mlia.target.tosa.advisor import TOSAInferenceAdvisor


def test_configure_and_get_tosa_advisor(
    monkeypatch: pytest.MonkeyPatch, test_tflite_model: Path
) -> None:
    """Test TOSA advisor configuration."""
    ctx = ExecutionContext()
    get_events_mock = MagicMock()
    monkeypatch.setattr(
        "mlia.target.tosa.advisor.TOSAInferenceAdvisor.get_events",
        MagicMock(return_value=get_events_mock),
    )

    advisor = configure_and_get_tosa_advisor(ctx, "tosa", test_tflite_model)
    workflow = advisor.configure(ctx)

    assert isinstance(advisor, TOSAInferenceAdvisor)

    assert advisor.get_events(ctx) == get_events_mock
    assert ctx.event_handlers is not None
    assert ctx.config_parameters == {
        "tosa_inference_advisor": {
            "model": str(test_tflite_model),
            "target_profile": "tosa",
        }
    }

    assert isinstance(workflow, DefaultWorkflowExecutor)
