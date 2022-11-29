# SPDX-FileCopyrightText: Copyright 2022-2023, Arm Limited and/or its affiliates.
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

from mlia.core.typing import OutputFormat
from mlia.core.typing import PathOrFileLike

# This type is used as type alias for the items which are being passed around
# in advisor workflow. There are no restrictions on the type of the
# object. This alias used only to emphasize the nature of the input/output
# arguments.
DataItem = Any


class FormattedFilePath:
    """Class used to keep track of the format that a path points to."""

    def __init__(self, path: PathOrFileLike, fmt: OutputFormat = "plain_text") -> None:
        """Init FormattedFilePath."""
        self._path = path
        self._fmt = fmt

    @property
    def fmt(self) -> OutputFormat:
        """Return file format."""
        return self._fmt

    @property
    def path(self) -> PathOrFileLike:
        """Return file path."""
        return self._path

    def __eq__(self, other: object) -> bool:
        """Check for equality with other objects."""
        if isinstance(other, FormattedFilePath):
            return other.fmt == self.fmt and other.path == self.path

        return False

    def __repr__(self) -> str:
        """Represent object."""
        return f"FormattedFilePath {self.path=}, {self.fmt=}"


class AdviceCategory(Flag):
    """Advice category.

    Enumeration of advice categories supported by ML Inference Advisor.
    """

    COMPATIBILITY = auto()
    PERFORMANCE = auto()
    OPTIMIZATION = auto()

    @classmethod
    def from_string(cls, values: set[str]) -> set[AdviceCategory]:
        """Resolve enum value from string value."""
        category_names = [item.name for item in AdviceCategory]
        for advice_value in values:
            if advice_value.upper() not in category_names:
                raise Exception(f"Invalid advice category {advice_value}")

        return {AdviceCategory[value.upper()] for value in values}


class NamedEntity(ABC):
    """Entity with a name and description."""

    @classmethod
    @abstractmethod
    def name(cls) -> str:
        """Return name of the entity."""
