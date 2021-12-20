# Copyright 2021, Arm Ltd.
"""Common module.

This module contains common interfaces/classess shared across
core module.
"""
from abc import ABC
from abc import abstractmethod
from enum import Enum
from typing import Any

# This type is used as type alias for the items which are being passed around
# in advisor workflow. There are no restrictions on the type of the
# object. This alias used only to emphasize the nature of the input/output
# arguments.
DataItem = Any


class AdviceCategory(Enum):
    """Advice category.

    Enumeration of advice categories supported by inference advisor.
    """

    OPERATORS_COMPATIBILITY = 1
    PERFORMANCE = 2
    OPTIMIZATION = 3
    COMMON = 4


class NamedEntity(ABC):
    """Entity with a name and description."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return name of the entity."""

    @property
    @abstractmethod
    def description(self) -> str:
        """Return description of the entity."""
