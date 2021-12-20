# Copyright 2021, Arm Ltd.
"""Tests for the module events."""
from dataclasses import dataclass
from unittest.mock import call
from unittest.mock import MagicMock

from mlia.core.events import DefaultEventPublisher
from mlia.core.events import Event
from mlia.core.events import EventHandler
from mlia.core.events import stage


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
