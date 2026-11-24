# SPDX-FileCopyrightText: Copyright 2022-2023, 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for module workflow."""

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from unittest.mock import MagicMock

import pytest

from mlia.core.common import AdviceCategory
from mlia.core.context import ExecutionContext
from mlia.core.data_analysis import ContextAwareDataAnalyzer
from mlia.core.data_collection import ContextAwareDataCollector
from mlia.core.errors import FunctionalityNotSupportedError
from mlia.core.workflow import DefaultWorkflowExecutor


def test_workflow_executor(tmpdir: str) -> None:
    """Test workflow executor."""
    data_collector_mock = MagicMock(spec=ContextAwareDataCollector)
    data_collector_mock.collect_data.return_value = 42

    data_collector_mock_no_value = MagicMock(spec=ContextAwareDataCollector)
    data_collector_mock_no_value.collect_data.return_value = None

    data_collector_mock_skipped = MagicMock(spec=ContextAwareDataCollector)
    data_collector_mock_skipped.collect_data.side_effect = (
        FunctionalityNotSupportedError("Error!", "Error!")
    )

    data_analyzer_mock = MagicMock(spec=ContextAwareDataAnalyzer)
    data_analyzer_mock.get_analyzed_data.return_value = ["Really good number!"]

    context = ExecutionContext(output_dir=tmpdir)

    executor = DefaultWorkflowExecutor(
        context,
        [
            data_collector_mock,
            data_collector_mock_no_value,
            data_collector_mock_skipped,
        ],
        [data_analyzer_mock],
    )

    executor.run()

    data_collector_mock.collect_data.assert_called_once()
    data_collector_mock_no_value.collect_data.assert_called_once()
    data_collector_mock_skipped.collect_data.assert_called_once()
    data_analyzer_mock.analyze_data.assert_called_once_with(42)


def test_workflow_executor_failed(tmpdir: str) -> None:
    """Workflow failures should propagate to callers."""
    context = ExecutionContext(output_dir=tmpdir)
    data_collector_mock = MagicMock(spec=ContextAwareDataCollector)
    data_collector_mock.collect_data.side_effect = RuntimeError("Collection failed")

    executor = DefaultWorkflowExecutor(context, [data_collector_mock], [])

    with pytest.raises(RuntimeError, match="Collection failed"):
        executor.run()


def test_workflow_reports_main_check_stages(
    tmpdir: str, caplog: pytest.LogCaptureFixture
) -> None:
    """The user sees each requested check begin and finish in execution order."""
    performance = MagicMock(spec=ContextAwareDataCollector)
    performance.name.return_value = "renamed_collector"
    performance.kind = AdviceCategory.PERFORMANCE
    performance.collect_data.return_value = None
    compatibility = MagicMock(spec=ContextAwareDataCollector)
    compatibility.name.return_value = "another_name"
    compatibility.kind = AdviceCategory.COMPATIBILITY
    compatibility.collect_data.return_value = None
    executor = DefaultWorkflowExecutor(
        ExecutionContext(output_dir=tmpdir), [performance, compatibility], []
    )

    with caplog.at_level(logging.INFO, logger="mlia.core.workflow"):
        executor.collect_data()

    messages = [
        record.message
        for record in caplog.records
        if record.name == "mlia.core.workflow"
    ]
    assert messages == [
        "Checking performance...",
        "Performance check complete.",
        "Checking compatibility...",
        "Compatibility check complete.",
    ]


def test_workflow_keeps_progress_visible_during_collection(
    tmpdir: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A backend without its own heartbeat still shows active work."""
    collector = MagicMock(spec=ContextAwareDataCollector)
    collector.kind = AdviceCategory.PERFORMANCE
    collector.progress_label = "Running Ethos-U analysis"
    events: list[str] = []

    @contextmanager
    def display(_logger: logging.Logger, label: str) -> Iterator[None]:
        events.append(f"start {label}")
        try:
            yield
        finally:
            events.append(f"stop {label}")

    collector.collect_data.side_effect = lambda: events.append("collect")
    monkeypatch.setattr("mlia.core.workflow.progress_heartbeat", display)
    executor = DefaultWorkflowExecutor(
        ExecutionContext(output_dir=tmpdir), [collector], []
    )

    executor.collect_data()

    assert events == [
        "start Running Ethos-U analysis",
        "collect",
        "stop Running Ethos-U analysis",
    ]


def test_workflow_uses_declared_progress_label_without_kind(
    tmpdir: str, caplog: pytest.LogCaptureFixture
) -> None:
    """A combined collector can name its work without claiming one category."""
    collector = MagicMock(spec=ContextAwareDataCollector)
    collector.kind = None
    collector.progress_label = "Ethos-U analysis"
    collector.collect_data.return_value = None
    executor = DefaultWorkflowExecutor(
        ExecutionContext(output_dir=tmpdir), [collector], []
    )

    with caplog.at_level(logging.INFO, logger="mlia.core.workflow"):
        executor.collect_data()

    assert [
        record.message
        for record in caplog.records
        if record.name == "mlia.core.workflow"
    ] == [
        "Ethos-U analysis...",
        "Ethos-U analysis complete.",
    ]


def test_workflow_executor_preserves_result_local_advice(tmpdir: str) -> None:
    """Complete backend results should retain only their own advice."""
    first_item = MagicMock()
    first_item.standardized_output = {
        "schema_version": "1.1.0",
        "results": [
            {
                "kind": "compatibility",
                "advice": [{"id": "compatibility", "message": "Compatibility"}],
            }
        ],
        "backends": [],
    }
    second_item = MagicMock()
    second_item.standardized_output = {
        "schema_version": "1.1.0",
        "results": [
            {
                "kind": "performance",
                "advice": [{"id": "performance", "message": "Performance"}],
            }
        ],
        "backends": [],
    }
    first_collector = MagicMock(spec=ContextAwareDataCollector)
    first_collector.collect_data.return_value = first_item
    second_collector = MagicMock(spec=ContextAwareDataCollector)
    second_collector.collect_data.return_value = second_item
    context = ExecutionContext(output_dir=tmpdir)
    executor = DefaultWorkflowExecutor(context, [first_collector, second_collector], [])

    output = executor.run()

    assert output is not None
    assert output["results"] == [
        first_item.standardized_output["results"][0],
        second_item.standardized_output["results"][0],
    ]
