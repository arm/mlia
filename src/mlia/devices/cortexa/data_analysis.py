# SPDX-FileCopyrightText: Copyright 2022, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Cortex-A data analysis module."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from dataclasses import field
from functools import singledispatchmethod

from mlia.core.common import DataItem
from mlia.core.data_analysis import Fact
from mlia.core.data_analysis import FactExtractor
from mlia.devices.cortexa.operators import CortexACompatibilityInfo
from mlia.devices.cortexa.operators import Operator
from mlia.nn.tensorflow.tflite_compat import TFLiteCompatibilityInfo


class CortexADataAnalyzer(FactExtractor):
    """Cortex-A data analyzer."""

    @singledispatchmethod
    def analyze_data(self, data_item: DataItem) -> None:  # type: ignore
        """Analyse the data."""

    @analyze_data.register
    def analyze_operator_compatibility(
        self, data_item: CortexACompatibilityInfo
    ) -> None:
        """Analyse operator compatibility information."""
        if data_item.cortex_a_compatible:
            self.add_fact(ModelIsCortexACompatible(data_item.backend_info))
        else:
            unsupported_ops = set()
            activation_func_support: defaultdict[
                str, ModelIsNotCortexACompatible.ActivationFunctionSupport
            ] = defaultdict(ModelIsNotCortexACompatible.ActivationFunctionSupport)
            for oper in data_item.operators:
                if oper.support_type == Operator.SupportType.OP_NOT_SUPPORTED:
                    unsupported_ops.add(oper.full_name)

                if oper.support_type == Operator.SupportType.ACTIVATION_NOT_SUPPORTED:
                    # Add used but unsupported actication functions
                    activation_func_support[oper.full_name].used_unsupported.add(
                        oper.activation_func.name
                    )
                    # Add supported activation functions
                    activation_func_support[oper.full_name].supported.update(
                        oper.supported_activation_functions
                    )

            assert (
                unsupported_ops or activation_func_support or not data_item.operators
            ), (
                "The model is marked as not compatible with Cortex-A but there "
                "are no unsupported ops activation functions listed."
            )

            self.add_fact(
                ModelIsNotCortexACompatible(
                    data_item.backend_info, unsupported_ops, activation_func_support
                )
            )

    @analyze_data.register
    def analyze_tflite_compatibility(self, data_item: TFLiteCompatibilityInfo) -> None:
        """Analyze TensorFlow Lite compatibility information."""
        if data_item.compatible:
            return

        if data_item.conversion_failed_with_errors:
            self.add_fact(
                ModelIsNotTFLiteCompatible(
                    custom_ops=data_item.required_custom_ops,
                    flex_ops=data_item.required_flex_ops,
                )
            )

        if data_item.check_failed_with_unknown_error:
            self.add_fact(TFLiteCompatibilityCheckFailed())

        if data_item.conversion_failed_for_model_with_custom_ops:
            self.add_fact(ModelHasCustomOperators())


@dataclass
class CortexACompatibility(Fact):
    """Base class for Cortex-A compatibility providing backend info."""

    backend_info: str


@dataclass
class ModelIsCortexACompatible(CortexACompatibility):
    """Model is completely compatible with Cortex-A."""


@dataclass
class ModelIsNotCortexACompatible(CortexACompatibility):
    """Model is not compatible with Cortex-A."""

    @dataclass
    class ActivationFunctionSupport:
        """Activation function support per operator."""

        used_unsupported: set[str] = field(default_factory=set)
        supported: set[str] = field(default_factory=set)

    unsupported_ops: set[str]
    activation_func_support: dict[str, ActivationFunctionSupport]


@dataclass
class ModelIsNotTFLiteCompatible(Fact):
    """Model could not be converted into TensorFlow Lite format."""

    custom_ops: list[str] | None = None
    flex_ops: list[str] | None = None


@dataclass
class TFLiteCompatibilityCheckFailed(Fact):
    """TensorFlow Lite compatibility check failed by unknown reason."""


@dataclass
class ModelHasCustomOperators(Fact):
    """Model could not be loaded because it contains custom ops."""
