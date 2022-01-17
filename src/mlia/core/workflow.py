# Copyright 2021, Arm Ltd.
"""Module for executors.

This module contains implementation of workflow
executors.
"""
import itertools
from abc import ABC
from abc import abstractmethod
from functools import wraps
from typing import Any
from typing import Callable
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple

from mlia.core.advice_generation import Advice
from mlia.core.advice_generation import AdviceEvent
from mlia.core.advice_generation import AdviceProducer
from mlia.core.common import DataItem
from mlia.core.context import Context
from mlia.core.data_analysis import DataAnalyzer
from mlia.core.data_collection import DataCollector
from mlia.core.events import AdviceStageFinishedEvent
from mlia.core.events import AdviceStageStartedEvent
from mlia.core.events import AnalyzedDataEvent
from mlia.core.events import CollectedDataEvent
from mlia.core.events import DataAnalysisStageFinishedEvent
from mlia.core.events import DataAnalysisStageStartedEvent
from mlia.core.events import DataCollectionStageFinishedEvent
from mlia.core.events import DataCollectionStageStartedEvent
from mlia.core.events import Event
from mlia.core.events import ExecutionFailedEvent
from mlia.core.events import ExecutionFinishedEvent
from mlia.core.events import ExecutionStartedEvent
from mlia.core.events import stage
from mlia.core.mixins import ContextMixin


class WorkflowExecutor(ABC):
    """Base workflow executor."""

    @abstractmethod
    def run(self) -> None:
        """Run the module."""


STAGE_COLLECTION = (
    DataCollectionStageStartedEvent(),
    DataCollectionStageFinishedEvent(),
)
STAGE_ANALYSIS = (DataAnalysisStageStartedEvent(), DataAnalysisStageFinishedEvent())
STAGE_ADVICE = (AdviceStageStartedEvent(), AdviceStageFinishedEvent())


def on_stage(stage_events: Tuple[Event, Event]) -> Callable:
    """Mark start/finish of the stage with appropriate events."""

    def wrapper(method: Callable) -> Callable:
        """Wrap method."""

        @wraps(method)
        def publish_events(self: Any, *args: Any, **kwargs: Any) -> Any:
            """Publish events before and after execution."""
            with stage(self.context.event_publisher, stage_events):
                return method(self, *args, **kwargs)

        return publish_events

    return wrapper


class DefaultWorkflowExecutor(WorkflowExecutor):
    """Default module executor.

    This is a default implementation of the workflow executor.
    All components are launched sequentually in the same process.
    """

    def __init__(
        self,
        context: Context,
        collectors: Sequence[DataCollector],
        analyzers: Sequence[DataAnalyzer],
        producers: Sequence[AdviceProducer],
        before_start_events: Optional[Sequence[Event]] = None,
    ):
        """Init default workflow executor.

        :param context: Context instance
        :param collectors: List of the data collectors
        :param analyzers: List of the data analyzers
        :param producers: List of the advice producers
        :param before_start_events: Optional list of the custom events that
               should be published before start of the worfkow execution.
        """
        self.context = context
        self.collectors = collectors
        self.analyzers = analyzers
        self.producers = producers
        self.before_start_events = before_start_events

    def run(self) -> None:
        """Run the workflow."""
        self.inject_context()
        self.context.register_event_handlers()

        try:
            self.publish(ExecutionStartedEvent())

            self.before_start()

            collected_data = self.collect_data()
            analyzed_data = self.analyze_data(collected_data)

            self.produce_advice(analyzed_data)
        except Exception as err:  # pylint: disable=broad-except
            self.publish(ExecutionFailedEvent(err))
        else:
            self.publish(ExecutionFinishedEvent())

    def before_start(self) -> None:
        """Run actions before start of the workflow execution."""
        events = self.before_start_events or []
        for event in events:
            self.publish(event)

    @on_stage(STAGE_COLLECTION)
    def collect_data(self) -> List[DataItem]:
        """Collect data.

        Run each of data collector components and return list of
        the collected data items.
        """
        collected_data = []
        for collector in self.collectors:
            data_item = collector.collect_data()
            if data_item is not None:
                collected_data.append(data_item)
                self.publish(CollectedDataEvent(data_item))

        return collected_data

    @on_stage(STAGE_ANALYSIS)
    def analyze_data(self, collected_data: List[DataItem]) -> List[DataItem]:
        """Analyze data.

        Pass each collected data item into each data analyzer and
        return analyzed data.

        :param collected_data: list of collected data items
        """
        analyzed_data = []
        for analyzer in self.analyzers:
            for item in collected_data:
                analyzer.analyze_data(item)

            for data_item in analyzer.get_analyzed_data():
                analyzed_data.append(data_item)

                self.publish(AnalyzedDataEvent(data_item))
        return analyzed_data

    @on_stage(STAGE_ADVICE)
    def produce_advice(self, analyzed_data: List[DataItem]) -> None:
        """Produce advice.

        Pass each analyzed data item into each advice producer and
        publish generated advice.

        :param analyzed_data: list of analyzed data items
        """
        for producer in self.producers:
            for data_item in analyzed_data:
                producer.produce_advice(data_item)

            advice = producer.get_advice()
            if isinstance(advice, Advice):
                advice = [advice]

            for item in advice:
                self.publish(AdviceEvent(item))

    def inject_context(self) -> None:
        """Inject context object into components.

        Inject context object into components that supports context
        injection.
        """
        context_aware_components = (
            comp
            for comp in itertools.chain(
                self.collectors,
                self.analyzers,
                self.producers,
            )
            if isinstance(comp, ContextMixin)
        )

        for component in context_aware_components:
            component.set_context(self.context)

    def publish(self, event: Event) -> None:
        """Publish event.

        Helper method for event publising.

        :param event: event instance
        """
        self.context.event_publisher.publish_event(event)
