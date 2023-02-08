# SPDX-FileCopyrightText: Copyright 2022-2023, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Module for data collection.

This module contains base classes for the first stage
of the ML Inference Advisor workflow - data collection.
"""
from abc import abstractmethod

from mlia.core.common import DataItem
from mlia.core.common import NamedEntity
from mlia.core.mixins import ContextMixin
from mlia.core.mixins import ParameterResolverMixin


class DataCollector(NamedEntity):
    """Base class for the data collection.

    Data collection is the first step in the process of the advice
    generation.

    Different implementations of this class can provide various
    information about model or target. This information is being used
    at later stages.
    """

    @abstractmethod
    def collect_data(self) -> DataItem:
        """Collect data."""


class ContextAwareDataCollector(DataCollector, ContextMixin, ParameterResolverMixin):
    """Context aware data collector.

    This class makes easier access to the Context object. Context object could
    be automatically injected during workflow configuration.
    """
