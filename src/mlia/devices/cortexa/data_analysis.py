# SPDX-FileCopyrightText: Copyright 2022, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Cortex-A data analysis module."""
from dataclasses import dataclass
from functools import singledispatchmethod

from mlia.core.common import DataItem
from mlia.core.data_analysis import Fact
from mlia.core.data_analysis import FactExtractor
from mlia.devices.cortexa.operators import CortexACompatibilityInfo


class CortexADataAnalyzer(FactExtractor):
    """Cortex-A data analyzer."""

    @singledispatchmethod
    def analyze_data(self, data_item: DataItem) -> None:
        """Analyse the data."""

    @analyze_data.register
    def analyze_operator_compatibility(
        self, data_item: CortexACompatibilityInfo
    ) -> None:
        """Analyse operator compatibility information."""
        if data_item.cortex_a_compatible:
            self.add_fact(ModelIsCortexACompatible())
        else:
            self.add_fact(ModelIsNotCortexACompatible())


@dataclass
class ModelIsCortexACompatible(Fact):
    """Model is completely compatible with Cortex-A."""


@dataclass
class ModelIsNotCortexACompatible(Fact):
    """Model is not compatible with Cortex-A."""
