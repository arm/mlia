# SPDX-FileCopyrightText: Copyright 2025-2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the various event handlers."""
from pathlib import Path

import pytest

from mlia.backend.tosa_checker.compat import TOSACompatibilityInfo
from mlia.core.context import ExecutionContext
from mlia.core.events import CollectedDataEvent
from mlia.core.events import ExecutionStartedEvent
from mlia.core.handlers import WorkflowEventsHandler
from mlia.nn.tensorflow.tflite_compat import TFLiteCompatibilityInfo
from mlia.nn.tensorflow.tflite_compat import TFLiteCompatibilityStatus
from mlia.target.tosa.config import TOSAConfiguration
from mlia.target.tosa.events import TOSAAdvisorStartedEvent
from mlia.target.tosa.handlers import TOSAEventHandler


@pytest.mark.parametrize(
    "handler,event",
    [
        (
            TOSAEventHandler(),
            CollectedDataEvent(
                TOSACompatibilityInfo(tosa_compatible=True, operators=[])
            ),
        ),
        (
            TOSAEventHandler(),
            CollectedDataEvent(
                TFLiteCompatibilityInfo(
                    status=TFLiteCompatibilityStatus.TFLITE_CONVERSION_ERROR
                )
            ),
        ),
    ],
)
def test_data_collection_event(
    handler: WorkflowEventsHandler, event: CollectedDataEvent
) -> None:
    """Coverage for the on_collected_data function."""
    handler.set_context(ExecutionContext())
    handler.on_execution_started(ExecutionStartedEvent())
    handler.on_collected_data(event)


def test_tosa_advisor_started(test_tflite_model: Path) -> None:
    """Coverage for the on_tosa_advisor_started function."""
    advisor_event = TOSAAdvisorStartedEvent(
        model=test_tflite_model,
        target=TOSAConfiguration(target="tosa"),
        tosa_metadata=None,
    )
    handler = TOSAEventHandler()
    handler.set_context(ExecutionContext())
    handler.on_execution_started(ExecutionStartedEvent())
    with pytest.raises(RuntimeError):
        handler.on_tosa_advisor_started(advisor_event)
