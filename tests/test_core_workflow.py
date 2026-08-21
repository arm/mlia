# SPDX-FileCopyrightText: Copyright 2022-2023, 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for module workflow."""

from unittest.mock import MagicMock

import pytest

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
