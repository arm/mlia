# SPDX-FileCopyrightText: Copyright 2022, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Module for the events and related functionality.

This module represents one of the main component of the workflow -
events publishing and provides a way for delivering results to the
calling application.

Each component of the workflow can generate events of specific type.
Application can subscribe and react to those events.
"""
from __future__ import annotations

import traceback
import uuid
from abc import ABC
from abc import abstractmethod
from contextlib import contextmanager
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field
from functools import singledispatchmethod
from typing import Any
from typing import Generator

from mlia.core.common import DataItem


@dataclass
class Event:
    """Base class for the events.

    This class is used as a root node of the events class hierarchy.
    """

    event_id: str = field(init=False)

    def __post_init__(self) -> None:
        """Generate unique ID for the event."""
        self.event_id = str(uuid.uuid4())

    def compare_without_id(self, other: Event) -> bool:
        """Compare two events without event_id field."""
        if not isinstance(other, Event) or self.__class__ != other.__class__:
            return False

        self_as_dict = asdict(self)
        self_as_dict.pop("event_id")

        other_as_dict = asdict(other)
        other_as_dict.pop("event_id")

        return self_as_dict == other_as_dict


@dataclass
class ChildEvent(Event):
    """Child event.

    This class could be used to link event with the parent event.
    """

    parent_event_id: str


@dataclass
class ActionStartedEvent(Event):
    """Action started event.

    This event is published when some action has been started.
    """

    action_type: str
    params: dict | None = None


@dataclass
class SubActionEvent(ChildEvent):
    """SubAction event.

    This event could be used to represent some action during parent action.
    """

    action_type: str
    params: dict | None = None


@dataclass
class ActionFinishedEvent(ChildEvent):
    """Action finished event.

    This event is published when some action has been finished.
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
class ExecutionFailedEvent(SystemEvent):
    """Execution failed event."""

    err: Exception


@dataclass
class DataCollectionStageStartedEvent(SystemEvent):
    """Data collection stage started.

    This event is published when data collection stage started.
    """


@dataclass
class DataCollectorSkippedEvent(SystemEvent):
    """Data collector skipped event.

    This event is published when particular data collector can
    not provide data for the provided parameters.
    """

    data_collector: str
    reason: str


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


class EventHandler:
    """Base class for the event handlers.

    Each event handler should derive from this base class.
    """

    def handle_event(self, event: Event) -> None:
        """Handle the event.

        By default all published events are being passed to each
        registered event handler. It is handler's responsibility
        to filter events that it interested in.
        """


class DebugEventHandler(EventHandler):
    """Event handler for debugging purposes.

    This handler could print every published event to the
    standard output.
    """

    def __init__(self, with_stacktrace: bool = False) -> None:
        """Init event handler.

        :param with_stacktrace: enable printing stacktrace of the
               place where event publishing occurred.
        """
        self.with_stacktrace = with_stacktrace

    def handle_event(self, event: Event) -> None:
        """Handle event."""
        print(f"Got event {event}")

        if self.with_stacktrace:
            traceback.print_stack()


class EventDispatcherMetaclass(type):
    """Metaclass for event dispatching.

    It could be tedious to check type of the published event
    inside event handler. Instead the following convention could be
    established: if method name of the class starts with some
    prefix then it is considered to be event handler of particular
    type.

    This metaclass goes through the list of class methods and
    links all methods with the prefix "on_" to the common dispatcher
    method.
    """

    def __new__(
        cls,
        clsname: str,
        bases: tuple[type, ...],
        namespace: dict[str, Any],
        event_handler_method_prefix: str = "on_",
    ) -> Any:
        """Create event dispatcher and link event handlers."""
        new_class = super().__new__(cls, clsname, bases, namespace)

        @singledispatchmethod
        def dispatcher(_self: Any, _event: Event) -> Any:
            """Event dispatcher."""

        # get all class methods which starts with particular prefix
        event_handler_methods = (
            (item_name, item)
            for item_name in dir(new_class)
            if callable((item := getattr(new_class, item_name)))
            and item_name.startswith(event_handler_method_prefix)
        )

        # link all collected event handlers to one dispatcher method
        for method_name, method in event_handler_methods:
            event_handler = dispatcher.register(method)
            setattr(new_class, method_name, event_handler)

        # override default handle_event method, replace it with the
        # dispatcher
        setattr(new_class, "handle_event", dispatcher)

        return new_class


class EventDispatcher(EventHandler, metaclass=EventDispatcherMetaclass):
    """Event dispatcher."""


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

    def register_event_handlers(
        self, event_handlers: list[EventHandler] | None
    ) -> None:
        """Register event handlers.

        Can be used for batch registration of the event handlers:

        :param event_handlers: list of the event handler instances
        """
        if not event_handlers:
            return

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
        self.handlers: list[EventHandler] = []

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
    publisher: EventPublisher, events: tuple[Event, Event]
) -> Generator[None, None, None]:
    """Generate events before and after stage.

    This context manager could be used to mark start/finish
    execution of a particular logical part of the workflow.
    """
    started, finished = events

    publisher.publish_event(started)
    yield
    publisher.publish_event(finished)


@contextmanager
def action(
    publisher: EventPublisher, action_type: str, params: dict | None = None
) -> Generator[None, None, None]:
    """Generate events before and after action."""
    action_started = ActionStartedEvent(action_type, params)
    action_finished = ActionFinishedEvent(action_started.event_id)

    publisher.publish_event(action_started)
    yield
    publisher.publish_event(action_finished)
