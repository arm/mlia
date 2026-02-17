# SPDX-FileCopyrightText: Copyright 2022-2024, 2026, Arm Limited
# and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Module for executors.

This module contains implementation of the workflow
executors.
"""

from __future__ import annotations

import itertools
from abc import ABC, abstractmethod
from functools import wraps
from typing import Any, Callable, Sequence

from mlia.core.advice_generation import Advice, AdviceEvent, AdviceProducer
from mlia.core.common import DataItem
from mlia.core.context import Context
from mlia.core.data_analysis import DataAnalyzer, PatternAnalyzer
from mlia.core.data_collection import DataCollector
from mlia.core.errors import FunctionalityNotSupportedError
from mlia.core.events import (
    AdviceStageFinishedEvent,
    AdviceStageStartedEvent,
    AnalyzedDataEvent,
    CollectedDataEvent,
    DataAnalysisStageFinishedEvent,
    DataAnalysisStageStartedEvent,
    DataCollectionStageFinishedEvent,
    DataCollectionStageStartedEvent,
    DataCollectorSkippedEvent,
    DetectedPatternEvent,
    Event,
    ExecutionFailedEvent,
    ExecutionFinishedEvent,
    ExecutionStartedEvent,
    PatternDetectionPassEvent,
    PatternDetectionStageFinishedEvent,
    PatternDetectionStageStartedEvent,
    stage,
)
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
STAGE_PATTERN_DETECTION = (
    PatternDetectionStageStartedEvent(),
    PatternDetectionStageFinishedEvent(),
)
STAGE_ADVICE = (AdviceStageStartedEvent(), AdviceStageFinishedEvent())


def on_stage(stage_events: tuple[Event, Event]) -> Callable:
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
        startup_events: Sequence[Event] | None = None,
        pattern_analyzers: Sequence[PatternAnalyzer] | None = None,
        max_pattern_passes: int = 5,
    ):
        """Init default workflow executor.

        :param context: Context instance
        :param collectors: List of the data collectors
        :param analyzers: List of the data analyzers
        :param producers: List of the advice producers
        :param startup_events: Optional list of the custom events that
               should be published before start of the worfkow execution.
        :param pattern_analyzers: Optional list of pattern analyzers
        :param max_pattern_passes: Maximum number of pattern detection passes
        """
        self.context = context
        self.collectors = collectors
        self.analyzers = analyzers
        self.producers = producers
        self.startup_events = startup_events
        self.pattern_analyzers = pattern_analyzers or []
        self.max_pattern_passes = max_pattern_passes

    def run(self) -> None:
        """Run the workflow."""
        self.inject_context()
        self.context.register_event_handlers()

        try:
            self.publish(ExecutionStartedEvent())

            self.before_start()

            collected_data = self.collect_data()

            analyzed_data = self.analyze_data(collected_data)

            # Run pattern detection if pattern analyzers are configured
            if self.pattern_analyzers:
                analyzed_data = self.detect_patterns(analyzed_data)

            self.produce_advice(analyzed_data)
        except Exception as err:  # pylint: disable=broad-except
            self.publish(ExecutionFailedEvent(err))
        else:
            self.publish(ExecutionFinishedEvent())

    def before_start(self) -> None:
        """Run actions before start of the workflow execution."""
        events = self.startup_events or []
        for event in events:
            self.publish(event)

    @on_stage(STAGE_COLLECTION)
    def collect_data(self) -> list[DataItem]:
        """Collect data.

        Run each of data collector components and return list of
        the collected data items.
        """
        collected_data = []
        for collector in self.collectors:
            try:
                if (data_item := collector.collect_data()) is not None:
                    collected_data.append(data_item)
                    self.publish(CollectedDataEvent(data_item))
            except FunctionalityNotSupportedError as err:
                self.publish(DataCollectorSkippedEvent(collector.name(), str(err)))

        return collected_data

    @on_stage(STAGE_ANALYSIS)
    def analyze_data(self, collected_data: list[DataItem]) -> list[DataItem]:
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

    @on_stage(STAGE_PATTERN_DETECTION)
    def detect_patterns(self, analyzed_data: list[DataItem]) -> list[DataItem]:
        """Detect patterns in analyzed facts.

        Run pattern analyzers in multiple passes until convergence or
        max passes reached. Each analyzer processes all facts and can
        generate new composite facts.

        :param analyzed_data: list of analyzed data items (facts)
        :return: extended list including detected pattern facts
        """
        all_facts = list(analyzed_data)
        pass_number = 0

        while pass_number < self.max_pattern_passes:
            pass_number += 1
            new_facts_this_pass = 0

            for analyzer in self.pattern_analyzers:
                # Clear cache at the start of each pass
                if hasattr(analyzer, "clear_cache"):
                    analyzer.clear_cache()

                # Let analyzer process all current facts
                new_patterns = analyzer.analyze_patterns(all_facts)

                # Add newly detected patterns to the fact list
                for pattern_fact in new_patterns:
                    all_facts.append(pattern_fact)
                    new_facts_this_pass += 1
                    self.publish(DetectedPatternEvent(pattern_fact))

                # Update analyzer's internal state
                if hasattr(analyzer, "detected_patterns"):
                    analyzer.detected_patterns = new_patterns

            # Publish pass event
            self.publish(
                PatternDetectionPassEvent(
                    pass_number=pass_number, new_facts_count=new_facts_this_pass
                )
            )

            # Converged - no new facts detected
            if new_facts_this_pass == 0:
                break

        return all_facts

    @on_stage(STAGE_ADVICE)
    def produce_advice(self, analyzed_data: list[DataItem]) -> None:
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
                self.pattern_analyzers,
                self.producers,
                self.context.event_handlers or [],
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
