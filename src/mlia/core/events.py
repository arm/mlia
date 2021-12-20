# Copyright 2021, Arm Ltd.
"""Module for the events and related functionality.

This module represents one of the main component of the workflow -
events publishing and provides a way for delivering results to the
calling application.

Each component of the workflow can generate events of specific type.
Application can subscribe and react to those events.
"""
from abc import ABC
from abc import abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Generator
from typing import List
from typing import Tuple

from mlia.core.common import DataItem


@dataclass
class Event:
    """Base class for the events.

    This class is used as a root node of the events class hierarchy.
    """


@dataclass
class SystemEvent(Event):
    """System event.

    System event class represents events that published by components
    of the core module. Most common example is an workflow executor
    that publishes number of system events for starting/completion
    of different stages/workflow.

    Events that published by components outside of core module should not
    use this class as base class.
    """


@dataclass
class ExecutionStartedEvent(SystemEvent):
    """Execution started event.

    This event is published when workflow execution started.
    """


@dataclass
class ExecutionFinishedEvent(SystemEvent):
    """Execution finished event.

    This event is published when workflow execution finished.
    """


@dataclass
class DataCollectionStageStartedEvent(SystemEvent):
    """Data collection stage started.

    This event is published when data collection stage started.
    """


@dataclass
class DataCollectionStageFinishedEvent(SystemEvent):
    """Data collection stage finished.

    This event is published when data collection stage finished.
    """


@dataclass
class DataAnalysisStageStartedEvent(SystemEvent):
    """Data analysis stage started.

    This event is published when data analysis stage started.
    """


@dataclass
class DataAnalysisStageFinishedEvent(SystemEvent):
    """Data analysis stage finished.

    This event is published when data analysis stage finished.
    """


@dataclass
class AdviceStageStartedEvent(SystemEvent):
    """Advace producing stage started.

    This event is published when advice generation stage started.
    """


@dataclass
class AdviceStageFinishedEvent(SystemEvent):
    """Advace producing stage finished.

    This event is published when advice generation stage finished.
    """


@dataclass
class CollectedDataEvent(SystemEvent):
    """Collected data event.

    This event is published for every collected data item.

    :param data_item: collected data item
    """

    data_item: DataItem


@dataclass
class AnalyzedDataEvent(SystemEvent):
    """Analyzed data event.

    This event is published for every analyzed data item.

    :param data_item: analyzed data item
    """

    data_item: DataItem


class EventHandler(ABC):
    """Base class for the event handlers.

    Each event handler should derive from this base class.
    """

    @abstractmethod
    def handle_event(self, event: Event) -> None:
        """Handle the event.

        By default all published events are being passed to each
        registered event handler. It is handler's responsibility
        to filter events that it interested in.
        """


class EventPublisher(ABC):
    """Base class for the event publisher.

    Event publisher is a intermidiate component between event emitter
    and event consumer.
    """

    @abstractmethod
    def register_event_handler(self, event_handler: EventHandler) -> None:
        """Register event handler.

        :param event_handler: instance of the event handler
        """

    def register_event_handlers(self, event_handlers: List[EventHandler]) -> None:
        """Register event handlers.

        Can be used for batch registration of the event handlers:

        :param event_handlers: list of the event handler instances
        """
        for handler in event_handlers:
            self.register_event_handler(handler)

    @abstractmethod
    def publish_event(self, event: Event) -> None:
        """Publish the event.

        Deliver the event to the all registered event handlers.

        :param event: event instance
        """


class DefaultEventPublisher(EventPublisher):
    """Default event publishing implementation.

    Simple implementation that maintains list of the registered event
    handlers.
    """

    def __init__(self) -> None:
        """Init the event publisher."""
        self.handlers: List[EventHandler] = []

    def register_event_handler(self, event_handler: EventHandler) -> None:
        """Register the event handler.

        :param event_handler: instance of the event handler
        """
        self.handlers.append(event_handler)

    def publish_event(self, event: Event) -> None:
        """Publish the event.

        Publisher does not catch exceptions that could be raised by event handlers.
        """
        for handler in self.handlers:
            handler.handle_event(event)


@contextmanager
def stage(
    publisher: EventPublisher, events: Tuple[Event, Event]
) -> Generator[None, None, None]:
    """Generate events before and after stage.

    This context manager could be used to mark start/finish
    execution of a particular logical part of the workflow.
    """
    started, finished = events

    publisher.publish_event(started)
    yield
    publisher.publish_event(finished)
