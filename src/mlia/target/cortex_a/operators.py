# SPDX-FileCopyrightText: Copyright 2022, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Cortex-A tools module."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any
from typing import ClassVar

from mlia.backend.armnn_tflite_delegate.compat import (
    ARMNN_TFLITE_DELEGATE as TFLITE_DELEGATE_COMPAT,
)
from mlia.nn.tensorflow.tflite_graph import Op
from mlia.nn.tensorflow.tflite_graph import parse_subgraphs
from mlia.nn.tensorflow.tflite_graph import TFL_ACTIVATION_FUNCTION


@dataclass
class Operator:
    """Cortex-A compatibility information of the operator."""

    BUILTIN_COMPATIBILITY = TFLITE_DELEGATE_COMPAT["builtin_ops"]
    CUSTOM_COMPATIBILITY = TFLITE_DELEGATE_COMPAT["custom_ops"]

    class SupportType(Enum):
        """Type of operator support."""

        COMPATIBLE = "Compatible"
        OP_NOT_SUPPORTED = "Operator not supported"
        ACTIVATION_NOT_SUPPORTED = "Activation not supported"

    name: str
    location: str
    support_type: SupportType
    activation_func: TFL_ACTIVATION_FUNCTION
    custom_name: str | None = None

    @property
    def is_cortex_a_compatible(self) -> bool:
        """Check if this operator is compatible."""
        return self.support_type == Operator.SupportType.COMPATIBLE

    @property
    def full_name(self) -> str:
        """Returun the full name including the custom name if applicable."""
        return self.name + (f" - '{self.custom_name}'" if self.custom_name else "")

    @property
    def is_custom(self) -> bool:
        """Check if this is a custom operator."""
        return bool(self.custom_name)

    @property
    def compatibility_data(self) -> dict[str, dict[str, Any]]:
        """Get the compatibility data (builtin or custom ops)."""
        return (
            Operator.CUSTOM_COMPATIBILITY
            if self.is_custom
            else Operator.BUILTIN_COMPATIBILITY
        )

    @property
    def supported_activation_functions(self) -> list[str]:
        """Return a list of fused activation functions supported by this op."""
        op_name = self.custom_name if self.custom_name else self.name
        return self.compatibility_data[op_name].get("supported_fused_activation", [])

    @classmethod
    def from_tflite_op(cls, tfl_op: Op, location: str) -> Operator:
        """Create a new instance from TensorFlow Lite operator and location."""
        support_type = cls._get_support_type(tfl_op)
        activation_func = (
            tfl_op.builtin_options["fused_activation_function"]
            if (
                tfl_op.builtin_options
                and "fused_activation_function" in tfl_op.builtin_options
            )
            else TFL_ACTIVATION_FUNCTION.NONE
        )
        return Operator(
            tfl_op.type,
            location,
            support_type,
            activation_func=activation_func,
            custom_name=(tfl_op.custom_type if tfl_op.is_custom else None),
        )

    @staticmethod
    def _get_support_type(tfl_op: Op) -> Operator.SupportType:
        """Get the support type from the TensorFlow Lite operator."""
        compat_data = (
            Operator.CUSTOM_COMPATIBILITY
            if tfl_op.is_custom
            else Operator.BUILTIN_COMPATIBILITY
        )
        op_type = tfl_op.custom_type if tfl_op.is_custom else tfl_op.type

        if op_type not in compat_data:
            return Operator.SupportType.OP_NOT_SUPPORTED

        compat_op = compat_data[op_type]
        if "supported_fused_activation" in compat_op:
            assert tfl_op.builtin_options
            assert "fused_activation_function" in tfl_op.builtin_options
            if (
                tfl_op.builtin_options["fused_activation_function"]
                not in compat_op["supported_fused_activation"]
            ):
                return Operator.SupportType.ACTIVATION_NOT_SUPPORTED

        return Operator.SupportType.COMPATIBLE


@dataclass
class CortexACompatibilityInfo:
    """Model's operators."""

    cortex_a_compatible: bool
    operators: list[Operator]
    backend_info: ClassVar[str] = (
        f"{TFLITE_DELEGATE_COMPAT['metadata']['backend']} "
        f"{TFLITE_DELEGATE_COMPAT['metadata']['version']}"
    )


def get_cortex_a_compatibility_info(model_path: Path) -> CortexACompatibilityInfo:
    """Return list of model's operators."""
    model = parse_subgraphs(model_path)

    op_list = [
        Operator.from_tflite_op(oper, f"subgraph:{g_idx},oper:{op_idx}")
        for g_idx, g in enumerate(model)
        for op_idx, oper in enumerate(g)
    ]
    all_compatible = all(oper.is_cortex_a_compatible for oper in op_list)
    compat_info = CortexACompatibilityInfo(all_compatible, op_list)

    return compat_info


def report() -> None:
    """Generate supported operators report."""
    raise Exception(
        "Generating a supported operators report is not "
        "currently supported with Cortex-A target profile."
    )
