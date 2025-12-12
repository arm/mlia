# SPDX-FileCopyrightText: Copyright 2022, 2025, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Module for advice generation."""
from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from functools import wraps
from typing import Any
from typing import Callable

from mlia.core.common import AdviceCategory
from mlia.core.common import DataItem
from mlia.core.events import SystemEvent
from mlia.core.mixins import ContextMixin


@dataclass
class Advice:
    """Base class for the advice."""

    messages: list[str]
    metadata: list[dict[str, Any]] | None = None

    def to_extension_dict(self) -> dict[str, Any]:
        """Convert advice to extension dictionary format.

        Returns:
            Dictionary with id, message, and optional metadata
        """
        result: dict[str, Any] = {
            "id": id(self),
            "message": " ".join(self.messages) if self.messages else "",
        }
        if self.metadata:
            result["metadata"] = self.metadata
        return result


@dataclass
class AdviceEvent(SystemEvent):
    """Advice event.

    This event is published for every produced advice.

    :param advice: Advice instance
    """

    advice: Advice


class AdviceProducer(ABC):
    """Base class for the advice producer.

    Producer has two methods for advice generation:
      - produce_advice - used to generate advice based on provided
        data (analyzed data item from analyze stage)
      - get_advice - used for getting generated advice

    Advice producers that have predefined advice could skip
    implementation of produce_advice method.
    """

    @abstractmethod
    def produce_advice(self, data_item: DataItem) -> None:
        """Process data item and produce advice.

        :param data_item: piece of data that could be used
               for advice generation
        """

    @abstractmethod
    def get_advice(self) -> Advice | list[Advice]:
        """Get produced advice."""


class ContextAwareAdviceProducer(AdviceProducer, ContextMixin):
    """Context aware advice producer.

    This class makes easier access to the Context object. Context object could
    be automatically injected during workflow configuration.
    """


class FactBasedAdviceProducer(ContextAwareAdviceProducer):
    """Advice producer based on provided facts.

    This is an utility class that maintain list of generated Advice instances.
    """

    def __init__(self) -> None:
        """Init advice producer."""
        self.advice: list[Advice] = []

    def get_advice(self) -> Advice | list[Advice]:
        """Get produced advice."""
        return self.advice

    def add_advice(self, messages: list[str]) -> None:
        """Add advice."""
        self.advice.append(Advice(messages))


def advice_category(*categories: AdviceCategory) -> Callable:
    """Filter advice generation handler by advice category."""

    def wrapper(handler: Callable) -> Callable:
        """Wrap data handler."""

        @wraps(handler)
        def check_category(self: Any, *args: Any, **kwargs: Any) -> Any:
            """Check if handler can produce advice for the requested category."""
            if not self.context.any_category_enabled(*categories):
                return

            handler(self, *args, **kwargs)

        return check_category

    return wrapper
