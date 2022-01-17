# Copyright 2021, Arm Ltd.
"""Tests for the module events."""
from dataclasses import dataclass
from typing import Any
from unittest.mock import call
from unittest.mock import MagicMock

from mlia.core.events import action
from mlia.core.events import ActionFinishedEvent
from mlia.core.events import ActionStartedEvent
from mlia.core.events import DebugEventHandler
from mlia.core.events import DefaultEventPublisher
from mlia.core.events import Event
from mlia.core.events import EventDispatcher
from mlia.core.events import EventHandler
from mlia.core.events import ExecutionFinishedEvent
from mlia.core.events import ExecutionStartedEvent
from mlia.core.events import stage
from mlia.core.events import SystemEventsHandler


@dataclass
class SampleEvent(Event):
    """Sample event."""

    msg: str


def test_event_publisher() -> None:
    """Test event publishing."""
    publisher = DefaultEventPublisher()
    handler_mock1 = MagicMock(spec=EventHandler)
    handler_mock2 = MagicMock(spec=EventHandler)

    publisher.register_event_handlers([handler_mock1, handler_mock2])

    event = SampleEvent("hello, event!")
    publisher.publish_event(event)

    handler_mock1.handle_event.assert_called_once_with(event)
    handler_mock2.handle_event.assert_called_once_with(event)


def test_stage_context_manager() -> None:
    """Test stage context manager."""
    publisher = DefaultEventPublisher()

    handler_mock = MagicMock(spec=EventHandler)
    publisher.register_event_handler(handler_mock)

    events = (SampleEvent("hello"), SampleEvent("goodbye"))
    with stage(publisher, events):
        print("perform actions")

    assert handler_mock.handle_event.call_count == 2
    calls = [call(event) for event in events]
    handler_mock.handle_event.assert_has_calls(calls)


def test_action_context_manager() -> None:
    """Test action stage context manager."""
    publisher = DefaultEventPublisher()

    handler_mock = MagicMock(spec=EventHandler)
    publisher.register_event_handler(handler_mock)

    with action(publisher, "Sample action"):
        print("perform actions")

    assert handler_mock.handle_event.call_count == 2
    calls = handler_mock.handle_event.mock_calls

    action_started = calls[0].args[0]
    action_finished = calls[1].args[0]

    assert isinstance(action_started, ActionStartedEvent)
    assert isinstance(action_finished, ActionFinishedEvent)

    assert action_finished.parent_event_id == action_started.event_id


def test_debug_event_handler(capsys: Any) -> None:
    """Test debugging event handler."""
    publisher = DefaultEventPublisher()

    publisher.register_event_handler(DebugEventHandler())
    publisher.register_event_handler(DebugEventHandler(with_stacktrace=True))

    msgs = ["Sample event 1", "Sample event 2"]
    for msg in msgs:
        publisher.publish_event(SampleEvent(msg))

    captured = capsys.readouterr()
    for msg in msgs:
        assert msg in captured.out

    assert "traceback.print_stack" in captured.err


def test_event_dispatcher(capsys: Any) -> None:
    """Test event dispatcher."""

    class SampleEventHandler(EventDispatcher):
        """Sample event handler."""

        def on_sample_event(  # pylint: disable=no-self-use
            self, _event: SampleEvent
        ) -> None:
            """Event handler for SampleEvent."""
            print("Got sample event")

    publisher = DefaultEventPublisher()
    publisher.register_event_handler(SampleEventHandler())
    publisher.publish_event(SampleEvent("Sample event"))

    captured = capsys.readouterr()
    assert captured.out.strip() == "Got sample event"


def test_system_events_handler(capsys: Any) -> None:
    """Test system events handler."""

    class CustomSystemEventHandler(SystemEventsHandler):
        """Custom system event handler."""

        def on_execution_started(self, event: ExecutionStartedEvent) -> None:
            """Handle ExecutionStarted event."""
            print("Execution started")

        def on_execution_finished(self, event: ExecutionFinishedEvent) -> None:
            """Handle ExecutionFinished event."""
            print("Execution finished")

    publisher = DefaultEventPublisher()
    publisher.register_event_handler(CustomSystemEventHandler())

    publisher.publish_event(ExecutionStartedEvent())
    publisher.publish_event(SampleEvent("Hello world!"))
    publisher.publish_event(ExecutionFinishedEvent())

    captured = capsys.readouterr()
    assert captured.out.strip() == "Execution started\nExecution finished"


def test_compare_without_id() -> None:
    """Test event comparison without event_id."""
    event1 = SampleEvent("message")
    event2 = SampleEvent("message")

    assert event1 != event2
    assert event1.compare_without_id(event2)

    assert not event1.compare_without_id("message")  # type: ignore
