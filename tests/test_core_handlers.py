# SPDX-FileCopyrightText: Copyright 2025, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Test event handlers module."""
from __future__ import annotations

from logging import Logger
from typing import Any
from typing import Callable
from unittest.mock import MagicMock

import pytest

from mlia.core.advice_generation import AdviceEvent
from mlia.core.context import Context
from mlia.core.events import AdviceStageFinishedEvent
from mlia.core.events import AdviceStageStartedEvent
from mlia.core.events import DataAnalysisStageFinishedEvent
from mlia.core.events import DataCollectionStageStartedEvent
from mlia.core.events import DataCollectorSkippedEvent
from mlia.core.events import ExecutionFailedEvent
from mlia.core.events import ExecutionStartedEvent
from mlia.core.handlers import WorkflowEventsHandler
from mlia.core.reporters import Report
from mlia.core.reporting import JSONReporter
from mlia.core.reporting import Reporter
from mlia.core.reporting import TextReporter


def _get_workflow_events_handler(output_format: str) -> WorkflowEventsHandler:
    class TestReport(Report):
        """Test report class"""

        def to_json(self, **_kwargs: Any) -> Any:
            """to_json override"""
            return {"test_key": "test_value"}

        def to_plain_text(self, **_kwargs: Any) -> str:
            """to_plain_text override"""
            return "test_key: test_value"

    def test_formatter_resolver(_outer_arg: Any) -> Callable[[Any], Report]:
        def formatter(_inner_arg: Any) -> Report:
            return TestReport()

        return formatter

    events_handler = WorkflowEventsHandler(test_formatter_resolver)

    mock_context = MagicMock(spec=Context)
    mock_context.output_format = output_format
    events_handler.context = mock_context

    mock_reporter = MagicMock(spec=Reporter)
    events_handler.reporter = mock_reporter

    return events_handler


@pytest.fixture(name="workflow_events_handler")
def events_handler_fixture() -> WorkflowEventsHandler:
    """Plain text workflow events handler"""
    return _get_workflow_events_handler("plain_text")


@pytest.fixture(name="handlers_logger")
def mock_handlers_logger_fixture(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Mock logger from the mlia.core.handlers module"""
    logger_mock = MagicMock(spec=Logger)
    monkeypatch.setattr("mlia.core.handlers.logger", logger_mock)
    return logger_mock


@pytest.mark.parametrize(
    "output_format, expected_reporter_type",
    [
        ("json", JSONReporter),
        ("plain_text", TextReporter),
    ],
)
def test_workflow_events_handler_on_execution_started(
    output_format: str, expected_reporter_type: Any, handlers_logger: MagicMock
) -> None:
    """Test on_execution_started method"""
    events_handler = _get_workflow_events_handler(output_format)

    mock_event = MagicMock(spec=ExecutionStartedEvent)
    events_handler.on_execution_started(mock_event)
    assert isinstance(events_handler.reporter, expected_reporter_type)
    handlers_logger.info.assert_called_once()


def test_workflow_events_handler_on_execution_failed(
    workflow_events_handler: WorkflowEventsHandler,
) -> None:
    """Test on_execution_failed method"""

    mock_event = MagicMock(spec=ExecutionFailedEvent)
    mock_event.err = ValueError("Test exception")

    with pytest.raises(ValueError, match="Test exception"):
        workflow_events_handler.on_execution_failed(mock_event)


def test_workflow_events_handler_on_data_collection_stage_started(
    workflow_events_handler: WorkflowEventsHandler,
    handlers_logger: MagicMock,
) -> None:
    """Test on_data_collection_stage_started method"""
    mock_event = MagicMock(spec=DataCollectionStageStartedEvent)
    workflow_events_handler.on_data_collection_stage_started(mock_event)
    handlers_logger.info.assert_called_once()


def test_workflow_events_handler_on_advice_stage_started(
    workflow_events_handler: WorkflowEventsHandler,
    handlers_logger: MagicMock,
) -> None:
    """Test on_advice_stage_started method"""
    mock_event = MagicMock(spec=AdviceStageStartedEvent)
    workflow_events_handler.on_advice_stage_started(mock_event)
    handlers_logger.info.assert_called_once()


def test_workflow_events_handler_on_data_collector_skipped(
    workflow_events_handler: WorkflowEventsHandler,
    handlers_logger: MagicMock,
) -> None:
    """Test on_data_collector_skipped method"""
    mock_event = MagicMock(spec=DataCollectorSkippedEvent)
    mock_event.reason = "Test skip reason"
    workflow_events_handler.on_data_collector_skipped(mock_event)
    handlers_logger.info.assert_called_once_with("Skipped: %s", "Test skip reason")


def test_workflow_events_handler_on_data_analysis_stage_finished(
    workflow_events_handler: WorkflowEventsHandler,
    handlers_logger: MagicMock,
) -> None:
    """Test on_data_analysis_stage_finished method"""
    mock_event = MagicMock(spec=DataAnalysisStageFinishedEvent)
    workflow_events_handler.on_data_analysis_stage_finished(mock_event)
    handlers_logger.info.assert_called_once()
    # fmt: off
    workflow_events_handler. \
        reporter.print_delayed.assert_called_once()  # type: ignore[attr-defined]
    # fmt: on


def test_workflow_events_handler_on_advice_event(
    workflow_events_handler: WorkflowEventsHandler,
) -> None:
    """Test on_advice_event method"""
    mock_event = MagicMock(spec=AdviceEvent)
    mock_event.advice = "Test advice"
    workflow_events_handler.on_advice_event(mock_event)
    assert len(workflow_events_handler.advice) == 1
    assert workflow_events_handler.advice[0] == "Test advice"


def test_workflow_events_handler_on_advice_stage_finished(
    workflow_events_handler: WorkflowEventsHandler,
) -> None:
    """Test on_advice_stage_finished method"""
    mock_event = MagicMock(spec=AdviceStageFinishedEvent)
    workflow_events_handler.on_advice_stage_finished(mock_event)
    # fmt: off
    workflow_events_handler. \
        reporter.submit.assert_called_once()  # type: ignore[attr-defined]
    workflow_events_handler. \
        reporter.generate_report.assert_called_once()  # type: ignore[attr-defined]
    # fmt: on
