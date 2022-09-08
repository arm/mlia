# SPDX-FileCopyrightText: Copyright 2022, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Common module.

This module contains common interfaces/classess shared across
core module.
"""
from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from enum import auto
from enum import Flag
from typing import Any

# This type is used as type alias for the items which are being passed around
# in advisor workflow. There are no restrictions on the type of the
# object. This alias used only to emphasize the nature of the input/output
# arguments.
DataItem = Any


class AdviceCategory(Flag):
    """Advice category.

    Enumeration of advice categories supported by ML Inference Advisor.
    """

    OPERATORS = auto()
    PERFORMANCE = auto()
    OPTIMIZATION = auto()
    ALL = OPERATORS | PERFORMANCE | OPTIMIZATION

    @classmethod
    def from_string(cls, value: str) -> AdviceCategory:
        """Resolve enum value from string value."""
        category_names = [item.name for item in AdviceCategory]
        if not value or value.upper() not in category_names:
            raise Exception(f"Invalid advice category {value}")

        return AdviceCategory[value.upper()]


class NamedEntity(ABC):
    """Entity with a name and description."""

    @classmethod
    @abstractmethod
    def name(cls) -> str:
        """Return name of the entity."""
