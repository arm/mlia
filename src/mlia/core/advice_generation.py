# SPDX-FileCopyrightText: Copyright 2022, 2025-2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Module for advice generation."""
from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from dataclasses import field
from functools import wraps
from typing import Any
from typing import Callable

from mlia.core.common import AdviceCategory
from mlia.core.common import DataItem
from mlia.core.events import SystemEvent
from mlia.core.mixins import ContextMixin
from mlia.core.output_schema import Advice as SchemaAdvice
from mlia.core.output_schema import AdviceCategory as SchemaAdviceCategory
from mlia.core.output_schema import AdviceSeverity
from mlia.core.output_schema import OperatorIdentifier


@dataclass
class Advice:
    """Base class for the advice."""

    id: str  # pylint: disable=invalid-name
    category: SchemaAdviceCategory
    severity: AdviceSeverity
    message: str
    affected_entities: list[OperatorIdentifier] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)

    def to_extension_dict(self) -> dict[str, Any]:
        """Convert advice to extension dictionary format.

        Returns:
            Dictionary with id, message, category, severity, and optional details
        """
        result: dict[str, Any] = {
            "id": self.id,
            "category": self.category.value.upper(),
            "severity": self.severity.value.upper(),
            "message": self.message,
        }
        if self.affected_entities:
            result["affected_entities"] = [e.to_dict() for e in self.affected_entities]
        if self.details:
            result["details"] = self.details
        return result

    def to_schema(self) -> SchemaAdvice:
        """Convert to schema Advice object.

        Returns:
            SchemaAdvice object
        """
        return SchemaAdvice(
            id=self.id,
            category=self.category,
            severity=self.severity,
            message=self.message,
            affected_entities=self.affected_entities,
            details=self.details,
        )


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

    def add_advice(
        self,
        message: str,
        category: SchemaAdviceCategory,
        severity: AdviceSeverity = AdviceSeverity.INFO,
        affected_entities: list[OperatorIdentifier] | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Add advice.

        Args:
            message: Advice message
            category: Advice category
            severity: Advice severity (default: INFO)
            affected_entities: List of affected entities
            details: Additional details
        """
        advice = Advice(
            id=str(len(self.advice)),
            category=category,
            severity=severity,
            message=message,
            affected_entities=affected_entities or [],
            details=details or {},
        )
        self.advice.append(advice)


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
