# SPDX-FileCopyrightText: Copyright 2022, 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Module for data analysis."""
from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import Callable
from typing import ClassVar

from mlia.core.common import DataItem
from mlia.core.mixins import ContextMixin

# Global registry for fact types
FACT_TYPE_REGISTRY: dict[str, FactType] = {}


@dataclass
class FactType:
    """Fact type metadata for schema generation and registration."""

    name: str
    category: str
    description: str
    schema_fields: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON Schema format."""
        return {
            "type": "object",
            "properties": self.schema_fields,
            "description": self.description,
            "category": self.category,
        }


def register_fact_type(
    name: str, category: str, description: str
) -> Callable[[type], type]:
    """Decorate fact types with metadata.

    :param name: Unique name for the fact type
    :param category: Category of the fact (e.g., 'layer', 'network', 'pattern')
    :param description: Human-readable description
    :return: Decorator function
    """

    def decorator(cls: type) -> type:
        """Register the fact class."""
        # Extract schema fields from dataclass
        if hasattr(cls, "__dataclass_fields__"):
            schema_fields = {
                field_name: {"type": str(field_info.type)}
                for field_name, field_info in cls.__dataclass_fields__.items()
                if field_name != "fact_type"
            }
        else:
            schema_fields = {}

        fact_type = FactType(
            name=name,
            category=category,
            description=description,
            schema_fields=schema_fields,
        )

        # Register in global registry
        if name in FACT_TYPE_REGISTRY:
            raise ValueError(
                f"Fact type '{name}' is already registered. "
                "Use a different name or create a subclass."
            )

        FACT_TYPE_REGISTRY[name] = fact_type

        # Attach fact_type to the class
        cls.fact_type = fact_type  # type: ignore[attr-defined]

        return cls

    return decorator


class DataAnalyzer(ABC):
    """Base class for the data analysis.

    Purpose of this class is to extract valuable data out of
    collected data which could be used for advice generation.

    This process consists of two steps:
      - analyze every item of the collected data
      - get analyzed data
    """

    @abstractmethod
    def analyze_data(self, data_item: DataItem) -> None:
        """Analyze data.

        :param data_item: item of the collected data
        """

    @abstractmethod
    def get_analyzed_data(self) -> list[DataItem]:
        """Get analyzed data."""


class ContextAwareDataAnalyzer(DataAnalyzer, ContextMixin):
    """Context aware data analyzer.

    This class makes easier access to the Context object. Context object could
    be automatically injected during workflow configuration.
    """


@dataclass
class Fact:
    """Base class for the facts.

    Fact represents some piece of knowledge about collected
    data. Each fact type must have a ClassVar fact_type attribute
    set via the @register_fact_type decorator.
    """

    fact_type: ClassVar[FactType]

    def to_dict(self) -> dict[str, Any]:
        """Convert fact to dictionary for JSON serialization.

        Subclasses can override to customize serialization.
        """
        result: dict[str, Any] = {"fact_type": self.fact_type.name}

        # Serialize all dataclass fields
        if hasattr(self, "__dataclass_fields__"):
            for field_name in self.__dataclass_fields__:
                if field_name != "fact_type":
                    value = getattr(self, field_name)
                    result[field_name] = value

        return result


@register_fact_type(
    "network_fact",
    "network",
    "Base class for facts about the entire network",
)
@dataclass
class NetworkFact(Fact):
    """Base class for network-level facts.

    Network facts represent aggregate information about the entire model.
    """


@register_fact_type(
    "layer_fact",
    "layer",
    "Base class for facts about individual layers/operators",
)
@dataclass
class LayerFact(Fact):
    """Base class for layer-level facts.

    Layer facts represent information about specific operators or layers.
    """

    operator_name: str
    location: str


@register_fact_type(
    "layer_compatibility_issue",
    "layer",
    "A layer has compatibility issues with the target",
)
@dataclass
class LayerCompatibilityIssue(LayerFact):
    """Fact indicating a layer has compatibility issues."""

    operator_type: str
    is_supported: bool
    reasons: list[tuple[str, str]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary with proper list serialization."""
        result = super().to_dict()
        # Convert reasons tuples to dicts for JSON compatibility
        result["reasons"] = [
            {"category": cat, "detail": detail} for cat, detail in self.reasons
        ]
        return result


@register_fact_type(
    "layer_uses_operator",
    "layer",
    "A layer uses a specific operator type",
)
@dataclass
class LayerUsesOperator(LayerFact):
    """Fact indicating a layer uses a specific operator type."""

    operator_type: str


class FactExtractor(ContextAwareDataAnalyzer):
    """Data analyzer based on extracting facts.

    Utility class that makes fact extraction easier.
    Class maintains list of the extracted facts.
    """

    def __init__(self) -> None:
        """Init fact extractor."""
        self.facts: list[Fact] = []

    def get_analyzed_data(self) -> list[DataItem]:
        """Return list of the collected facts."""
        return self.facts

    def add_fact(self, fact: Fact) -> None:
        """Add fact."""
        self.facts.append(fact)
