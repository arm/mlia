# SPDX-FileCopyrightText: Copyright 2022, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""TOSA data analysis module."""
from dataclasses import dataclass
from functools import singledispatchmethod

from mlia.core.common import DataItem
from mlia.core.data_analysis import Fact
from mlia.core.data_analysis import FactExtractor
from mlia.devices.tosa.operators import TOSACompatibilityInfo


@dataclass
class ModelIsTOSACompatible(Fact):
    """Model is completely TOSA compatible."""


@dataclass
class ModelIsNotTOSACompatible(Fact):
    """Model is not TOSA compatible."""


class TOSADataAnalyzer(FactExtractor):
    """TOSA data analyzer."""

    @singledispatchmethod
    def analyze_data(self, data_item: DataItem) -> None:
        """Analyse the data."""

    @analyze_data.register
    def analyze_tosa_compatibility(self, data_item: TOSACompatibilityInfo) -> None:
        """Analyse TOSA compatibility information."""
        if data_item.tosa_compatible:
            self.add_fact(ModelIsTOSACompatible())
        else:
            self.add_fact(ModelIsNotTOSACompatible())
