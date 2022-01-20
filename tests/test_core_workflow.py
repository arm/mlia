# Copyright 2021, Arm Ltd.
"""Tests for module workflow."""
from dataclasses import dataclass
from unittest.mock import call
from unittest.mock import MagicMock

from mlia.core.advice_generation import Advice
from mlia.core.advice_generation import AdviceEvent
from mlia.core.advice_generation import ContextAwareAdviceProducer
from mlia.core.context import ExecutionContext
from mlia.core.data_analysis import ContextAwareDataAnalyzer
from mlia.core.data_collection import ContextAwareDataCollector
from mlia.core.errors import FunctionalityNotSupportedError
from mlia.core.events import AdviceStageFinishedEvent
from mlia.core.events import AdviceStageStartedEvent
from mlia.core.events import AnalyzedDataEvent
from mlia.core.events import CollectedDataEvent
from mlia.core.events import DataAnalysisStageFinishedEvent
from mlia.core.events import DataAnalysisStageStartedEvent
from mlia.core.events import DataCollectionStageFinishedEvent
from mlia.core.events import DataCollectionStageStartedEvent
from mlia.core.events import DataCollectorSkippedEvent
from mlia.core.events import DefaultEventPublisher
from mlia.core.events import Event
from mlia.core.events import EventHandler
from mlia.core.events import ExecutionFailedEvent
from mlia.core.events import ExecutionFinishedEvent
from mlia.core.events import ExecutionStartedEvent
from mlia.core.workflow import DefaultWorkflowExecutor


def test_workflow_executor(tmpdir: str) -> None:  # pylint: disable=too-many-locals
    """Test workflow executor."""
    handler_mock = MagicMock(spec=EventHandler)
    data_collector_mock = MagicMock(spec=ContextAwareDataCollector)
    data_collector_mock.collect_data.return_value = 42

    data_collector_mock_no_value = MagicMock(spec=ContextAwareDataCollector)
    data_collector_mock_no_value.collect_data.return_value = None

    data_collector_mock_skipped = MagicMock(spec=ContextAwareDataCollector)
    data_collector_mock_skipped.name.return_value = "skipped_collector"
    data_collector_mock_skipped.collect_data.side_effect = (
        FunctionalityNotSupportedError("Error!", "Error!")
    )

    data_analyzer_mock = MagicMock(spec=ContextAwareDataAnalyzer)
    data_analyzer_mock.get_analyzed_data.return_value = ["Really good number!"]

    advice_producer_mock1 = MagicMock(spec=ContextAwareAdviceProducer)
    advice_producer_mock1.get_advice.return_value = Advice(["All good!"])

    advice_producer_mock2 = MagicMock(spec=ContextAwareAdviceProducer)
    advice_producer_mock2.get_advice.return_value = [Advice(["Good advice!"])]

    context = ExecutionContext(
        working_dir=tmpdir,
        event_handlers=[handler_mock],
        event_publisher=DefaultEventPublisher(),
    )

    @dataclass
    class SampleEvent(Event):
        """Sample event."""

        msg: str

    executor = DefaultWorkflowExecutor(
        context,
        [
            data_collector_mock,
            data_collector_mock_no_value,
            data_collector_mock_skipped,
        ],
        [data_analyzer_mock],
        [
            advice_producer_mock1,
            advice_producer_mock2,
        ],
        [SampleEvent("Hello from advisor!")],
    )

    executor.run()

    data_collector_mock.collect_data.assert_called_once()
    data_collector_mock_no_value.collect_data.assert_called_once()
    data_collector_mock_skipped.collect_data.assert_called_once()

    data_analyzer_mock.analyze_data.assert_called_once_with(42)

    advice_producer_mock1.produce_advice.assert_called_once_with("Really good number!")
    advice_producer_mock1.get_advice.assert_called_once()

    advice_producer_mock2.produce_advice.called_once_with("Really good number!")
    advice_producer_mock2.get_advice.assert_called_once()

    expected_mock_calls = [
        call(ExecutionStartedEvent()),
        call(SampleEvent("Hello from advisor!")),
        call(DataCollectionStageStartedEvent()),
        call(CollectedDataEvent(data_item=42)),
        call(DataCollectorSkippedEvent("skipped_collector", "Error!: Error!")),
        call(DataCollectionStageFinishedEvent()),
        call(DataAnalysisStageStartedEvent()),
        call(AnalyzedDataEvent(data_item="Really good number!")),
        call(DataAnalysisStageFinishedEvent()),
        call(AdviceStageStartedEvent()),
        call(AdviceEvent(advice=Advice(messages=["All good!"]))),
        call(AdviceEvent(advice=Advice(messages=["Good advice!"]))),
        call(AdviceStageFinishedEvent()),
        call(ExecutionFinishedEvent()),
    ]

    for expected_call, actual_call in zip(
        expected_mock_calls, handler_mock.handle_event.mock_calls
    ):
        expected_event = expected_call.args[0]
        actual_event = actual_call.args[0]

        assert actual_event.compare_without_id(expected_event)


def test_workflow_executor_failed(tmpdir: str) -> None:
    """Test scenario when one of the components raises exception."""
    handler_mock = MagicMock(spec=EventHandler)

    context = ExecutionContext(
        working_dir=tmpdir,
        event_handlers=[handler_mock],
        event_publisher=DefaultEventPublisher(),
    )

    collection_exception = Exception("Collection failed")

    data_collector_mock = MagicMock(spec=ContextAwareDataCollector)
    data_collector_mock.collect_data.side_effect = collection_exception

    executor = DefaultWorkflowExecutor(context, [data_collector_mock], [], [])
    executor.run()

    expected_mock_calls = [
        call(ExecutionStartedEvent()),
        call(DataCollectionStageStartedEvent()),
        call(ExecutionFailedEvent(collection_exception)),
    ]

    for expected_call, actual_call in zip(
        expected_mock_calls, handler_mock.handle_event.mock_calls
    ):
        expected_event = expected_call.args[0]
        actual_event = actual_call.args[0]

        if isinstance(actual_event, ExecutionFailedEvent):
            # seems that dataclass comparison doesn't work well
            # for the exceptions
            actual_exception = actual_event.err
            expected_exception = expected_event.err

            assert actual_exception == expected_exception
            continue

        assert actual_event.compare_without_id(expected_event)
