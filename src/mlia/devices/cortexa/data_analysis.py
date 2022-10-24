# SPDX-FileCopyrightText: Copyright 2022, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Cortex-A data analysis module."""
from __future__ import annotations

from dataclasses import dataclass
from functools import singledispatchmethod

from mlia.core.common import DataItem
from mlia.core.data_analysis import Fact
from mlia.core.data_analysis import FactExtractor
from mlia.devices.cortexa.operators import CortexACompatibilityInfo
from mlia.nn.tensorflow.tflite_compat import TFLiteCompatibilityInfo
from mlia.nn.tensorflow.tflite_compat import TFLiteConversionErrorCode


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

    @analyze_data.register
    def analyze_tflite_compatibility(self, data_item: TFLiteCompatibilityInfo) -> None:
        """Analyze TensorFlow Lite compatibility information."""
        if data_item.compatible:
            return

        custom_ops, flex_ops = [], []
        if data_item.conversion_errors:
            custom_ops = data_item.unsupported_ops_by_code(
                TFLiteConversionErrorCode.NEEDS_CUSTOM_OPS
            )
            flex_ops = data_item.unsupported_ops_by_code(
                TFLiteConversionErrorCode.NEEDS_FLEX_OPS
            )

        self.add_fact(
            ModelIsNotTFLiteCompatible(custom_ops=custom_ops, flex_ops=flex_ops)
        )


@dataclass
class ModelIsCortexACompatible(Fact):
    """Model is completely compatible with Cortex-A."""


@dataclass
class ModelIsNotCortexACompatible(Fact):
    """Model is not compatible with Cortex-A."""


@dataclass
class ModelIsNotTFLiteCompatible(Fact):
    """Model could not be converted into TensorFlow Lite format."""

    custom_ops: list[str] | None = None
    flex_ops: list[str] | None = None
