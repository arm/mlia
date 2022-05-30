# SPDX-FileCopyrightText: Copyright 2022, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Module for data analysis."""
from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from typing import List

from mlia.core.common import DataItem
from mlia.core.mixins import ContextMixin


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
    def get_analyzed_data(self) -> List[DataItem]:
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
    data.
    """


class FactExtractor(ContextAwareDataAnalyzer):
    """Data analyzer based on extracting facts.

    Utility class that makes fact extraction easier.
    Class maintains list of the extracted facts.
    """

    def __init__(self) -> None:
        """Init fact extractor."""
        self.facts: List[Fact] = []

    def get_analyzed_data(self) -> List[DataItem]:
        """Return list of the collected facts."""
        return self.facts

    def add_fact(self, fact: Fact) -> None:
        """Add fact."""
        self.facts.append(fact)
