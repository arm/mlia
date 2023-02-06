# SPDX-FileCopyrightText: Copyright 2022-2023, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Cortex-A tools module."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any
from typing import cast

from mlia.backend.armnn_tflite_delegate.compat import (
    ARMNN_TFLITE_DELEGATE as TFLITE_DELEGATE_COMPAT,
)
from mlia.nn.tensorflow.tflite_graph import Op
from mlia.nn.tensorflow.tflite_graph import parse_subgraphs
from mlia.nn.tensorflow.tflite_graph import TFL_ACTIVATION_FUNCTION
from mlia.target.cortex_a.config import CortexAConfiguration


@dataclass
class Operator:
    """Cortex-A compatibility information of the operator."""

    name: str
    location: str
    activation_func: TFL_ACTIVATION_FUNCTION
    custom_name: str | None = None

    @property
    def full_name(self) -> str:
        """Returun the full name including the custom name if applicable."""
        return self.name + (f" - '{self.custom_name}'" if self.custom_name else "")

    @property
    def is_custom(self) -> bool:
        """Check if this is a custom operator."""
        return bool(self.custom_name)

    @classmethod
    def from_tflite_op(cls, tfl_op: Op, location: str) -> Operator:
        """Create a new instance from TensorFlow Lite operator and location."""
        activation_func = (
            TFL_ACTIVATION_FUNCTION[tfl_op.builtin_options["fused_activation_function"]]
            if (
                tfl_op.builtin_options
                and "fused_activation_function" in tfl_op.builtin_options
            )
            else TFL_ACTIVATION_FUNCTION.NONE
        )
        return Operator(
            tfl_op.type,
            location,
            activation_func=activation_func,
            custom_name=(tfl_op.custom_type if tfl_op.is_custom else None),
        )


class CortexACompatibilityInfo:
    """Model's operators."""

    class SupportType(Enum):
        """Type of operator support."""

        COMPATIBLE = "Compatible"
        OP_NOT_SUPPORTED = "Operator not supported"
        ACTIVATION_NOT_SUPPORTED = "Activation not supported"

    def __init__(self, ops: list[Operator], armnn_tfl_delegate_version: str) -> None:
        """Create a new collection of op compatibility information."""
        compat_data = TFLITE_DELEGATE_COMPAT["ops"][armnn_tfl_delegate_version]
        self._builtin_compatibility = compat_data["builtin_ops"]
        self._custom_compatibility = compat_data["custom_ops"]

        self.backend_info = (
            f"{TFLITE_DELEGATE_COMPAT['backend']} {armnn_tfl_delegate_version}"
        )

        self.operators = ops

    @property
    def is_cortex_a_compatible(self) -> bool:
        """Check if all operators are compatible."""
        return all(self.is_op_compatible(oper) for oper in self.operators)

    def is_op_compatible(self, operator: Operator) -> bool:
        """Check if the given operator is compatible."""
        return self.get_support_type(operator) == self.SupportType.COMPATIBLE

    def compatibility_data(self, operator: Operator) -> dict[str, dict[str, Any]]:
        """Get the compatibility data (builtin or custom ops)."""
        return (
            cast(dict, self._custom_compatibility)
            if operator.is_custom
            else cast(dict, self._builtin_compatibility)
        )

    def supported_activation_functions(self, operator: Operator) -> list[str]:
        """Return a list of fused activation functions supported by this op."""
        op_name = operator.custom_name if operator.custom_name else operator.name
        return self.compatibility_data(operator)[op_name].get(
            "supported_fused_activation", []
        )

    def get_support_type(
        self, operator: Operator
    ) -> CortexACompatibilityInfo.SupportType:
        """Get the support type from the TensorFlow Lite operator."""
        compat_data = self.compatibility_data(operator)
        op_name = operator.custom_name if operator.is_custom else operator.name

        if op_name not in compat_data:
            return CortexACompatibilityInfo.SupportType.OP_NOT_SUPPORTED

        compat_op = compat_data[op_name]
        if "supported_fused_activation" in compat_op:
            if (
                operator.activation_func.name
                not in compat_op["supported_fused_activation"]
            ):
                return CortexACompatibilityInfo.SupportType.ACTIVATION_NOT_SUPPORTED

        return CortexACompatibilityInfo.SupportType.COMPATIBLE


def get_cortex_a_compatibility_info(
    model_path: Path, target_config: CortexAConfiguration
) -> CortexACompatibilityInfo:
    """Return list of model's operators."""
    model = parse_subgraphs(model_path)

    op_list = [
        Operator.from_tflite_op(oper, f"subgraph:{g_idx},oper:{op_idx}")
        for g_idx, g in enumerate(model)
        for op_idx, oper in enumerate(g)
    ]
    compat_info = CortexACompatibilityInfo(
        op_list, target_config.armnn_tflite_delegate_version
    )

    return compat_info


def report() -> None:
    """Generate supported operators report."""
    raise Exception(
        "Generating a supported operators report is not "
        "currently supported with Cortex-A target profile."
    )
