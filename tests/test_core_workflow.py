# Copyright 2021, Arm Ltd.
"""Tests for module workflow."""
from unittest.mock import call
from unittest.mock import MagicMock

from mlia.core.advice_generation import Advice
from mlia.core.advice_generation import AdviceEvent
from mlia.core.advice_generation import ContextAwareAdviceProducer
from mlia.core.context import ExecutionContext
from mlia.core.data_analysis import ContextAwareDataAnalyzer
from mlia.core.data_collection import ContextAwareDataCollector
from mlia.core.events import AdviceStageFinishedEvent
from mlia.core.events import AdviceStageStartedEvent
from mlia.core.events import AnalyzedDataEvent
from mlia.core.events import CollectedDataEvent
from mlia.core.events import DataAnalysisStageFinishedEvent
from mlia.core.events import DataAnalysisStageStartedEvent
from mlia.core.events import DataCollectionStageFinishedEvent
from mlia.core.events import DataCollectionStageStartedEvent
from mlia.core.events import DefaultEventPublisher
from mlia.core.events import EventHandler
from mlia.core.workflow import DefaultWorkflowExecutor


def test_workflow_executor(tmpdir: str) -> None:
    """Test workflow executor."""
    handler_mock = MagicMock(spec=EventHandler)
    data_collector_mock = MagicMock(spec=ContextAwareDataCollector)
    data_collector_mock.collect_data.return_value = 42

    data_analyzer_mock = MagicMock(spec=ContextAwareDataAnalyzer)
    data_analyzer_mock.get_analyzed_data.return_value = ["Really good number!"]

    advice_producer_mock = MagicMock(spec=ContextAwareAdviceProducer)
    advice_producer_mock.get_advice.return_value = Advice(["All good!"])

    context = ExecutionContext(
        advice_categories=[],
        config_parameters={},
        working_dir=tmpdir,
        event_handlers=[handler_mock],
        event_publisher=DefaultEventPublisher(),
    )

    executor = DefaultWorkflowExecutor(
        context,
        [data_collector_mock],
        [data_analyzer_mock],
        [advice_producer_mock],
        [],
    )

    executor.run()

    data_collector_mock.collect_data.assert_called_once()
    data_analyzer_mock.analyze_data.assert_called_once_with(42)
    advice_producer_mock.produce_advice.assert_called_once_with("Really good number!")

    expected_mock_calls = [
        call(DataCollectionStageStartedEvent()),
        call(CollectedDataEvent(data_item=42)),
        call(DataCollectionStageFinishedEvent()),
        call(DataAnalysisStageStartedEvent()),
        call(AnalyzedDataEvent(data_item="Really good number!")),
        call(DataAnalysisStageFinishedEvent()),
        call(AdviceStageStartedEvent()),
        call(AdviceEvent(advice=Advice(msgs=["All good!"]))),
        call(AdviceStageFinishedEvent()),
    ]
    assert handler_mock.handle_event.mock_calls == expected_mock_calls
