# SPDX-FileCopyrightText: Copyright 2022-2023, 2025, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Cortex-A tools module."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any
from typing import cast

import mlia
import mlia.core.output_schema as schema
from mlia.backend.armnn_tflite_delegate.compat import (
    ARMNN_TFLITE_DELEGATE as TFLITE_DELEGATE_COMPAT,
)
from mlia.nn.tensorflow.tflite_graph import Op
from mlia.nn.tensorflow.tflite_graph import parse_subgraphs
from mlia.nn.tensorflow.tflite_graph import TFL_ACTIVATION_FUNCTION
from mlia.target.cortex_a.config import CortexAConfiguration
from mlia.utils.filesystem import sha256

logger = logging.getLogger(__name__)


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

    def to_standardized_output(  # pylint: disable=too-many-locals
        self,
        model_path: Path,
        run_id: str | None = None,
        timestamp: str | None = None,
        cli_arguments: list[str] | None = None,
        target_config: dict[str, Any] | None = None,
        backend_config: dict[str, Any] | None = None,
    ) -> Any:  # Returns StandardizedOutput but avoid circular import
        """Convert to standardized output format.

        Args:
            model_path: Path to the model file
            run_id: Optional run ID (will be generated if not provided)
            timestamp: Optional ISO 8601 timestamp (will be generated if not provided)
            cli_arguments: Optional CLI arguments used for the run
            target_config: Optional target configuration parameters
            backend_config: Optional backend configuration parameters

        Returns:
            StandardizedOutput object
        """
        # Generate run_id and timestamp if not provided
        if run_id is None:
            run_id = schema.StandardizedOutput.create_run_id()
        if timestamp is None:
            timestamp = schema.StandardizedOutput.create_timestamp()

        # Create tool info
        tool = schema.Tool(name="mlia", version=mlia.__version__)

        # Create backend with version from self.backend_info
        # Extract version from backend_info string like
        # "Arm NN TensorFlow Lite Delegate 23.05"
        backend_parts = self.backend_info.rsplit(" ", 1)
        backend_version = backend_parts[1] if len(backend_parts) > 1 else "unknown"

        backend = schema.Backend(
            id="armnn-tflite-delegate",
            name="Arm NN TensorFlow Lite Delegate",
            version=backend_version,
            configuration=backend_config or {},
        )

        # Create target with CPU component
        cpu_type = (target_config or {}).get("cpu", "cortex-a")

        cpu_component = schema.Component(
            type=schema.ComponentType.CPU,
            family=cpu_type,
            model=None,
            variant=None,
        )

        target = schema.Target(
            profile_name=cpu_type,
            target_type="cpu",
            components=[cpu_component],
            configuration=target_config or {},
        )

        # Create model
        model_hash = sha256(model_path)
        model = schema.Model(
            name=str(model_path),
            format="tflite",
            hash=model_hash,
        )

        # Create context
        context = schema.Context(
            cli_arguments=cli_arguments or [],
        )

        # Create checks for each operator
        checks: list[schema.Check] = []
        entities: list[schema.Entity] = []

        for idx, operator in enumerate(self.operators):
            entity_id = f"op_{idx}"

            # Determine support type and placement
            support_type = self.get_support_type(operator)
            is_compatible = support_type == self.SupportType.COMPATIBLE

            # Create entity for this operator
            entity = schema.Entity(
                scope=schema.OperatorScope.OPERATOR,
                name=operator.full_name,
                location=operator.location,
                placement="cpu" if is_compatible else "unsupported",
                id=entity_id,
                attributes={
                    "operator_name": operator.name,
                    "is_custom": operator.is_custom,
                    "activation_function": operator.activation_func.name,
                    "index": idx,
                },
            )
            entities.append(entity)

            # Create check for compatibility
            if is_compatible:
                status = schema.CheckStatus.PASS
                details = {}
            else:
                status = schema.CheckStatus.FAIL
                reasons = []
                if support_type == self.SupportType.OP_NOT_SUPPORTED:
                    reasons.append(
                        {
                            "description": "Operator not supported",
                            "detail": (
                                f"Operator '{operator.full_name}' is not "
                                f"supported by Arm NN TFLite Delegate"
                            ),
                        }
                    )
                elif support_type == self.SupportType.ACTIVATION_NOT_SUPPORTED:
                    supported_funcs = self.supported_activation_functions(operator)
                    reasons.append(
                        {
                            "description": "Activation function not supported",
                            "detail": (
                                f"Activation '{operator.activation_func.name}' "
                                f"not in supported list: {supported_funcs}"
                            ),
                        }
                    )
                details = {"reasons": reasons}

            check = schema.Check(
                id=f"armnn_support_{entity_id}",
                status=status,
                details=details,
            )
            checks.append(check)

        # Determine overall result status
        if self.is_cortex_a_compatible:
            result_status = schema.ResultStatus.OK
        elif any(self.is_op_compatible(op) for op in self.operators):
            result_status = schema.ResultStatus.PARTIAL
        else:
            result_status = schema.ResultStatus.INCOMPATIBLE

        # Create result
        result = schema.Result(
            kind=schema.ResultKind.COMPATIBILITY,
            status=result_status,
            producer=backend.id,
            warnings=[],
            errors=[],
            checks=checks,
            entities=entities,
        )

        return schema.StandardizedOutput(
            schema_version="1.0.0",
            run_id=run_id,
            timestamp=timestamp,
            tool=tool,
            target=target,
            model=model,
            context=context,
            backends=[backend],
            results=[result],
            extensions={},
        )


@dataclass
class CortexACompatibilityResult:
    """Wrapper for Cortex-A compatibility with both legacy and standardized output."""

    legacy_info: CortexACompatibilityInfo
    standardized_output: Any | None = (
        None  # StandardizedOutput object, Any to avoid circular import
    )


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
    raise NotImplementedError(
        "Generating a supported operators report is not "
        "currently supported with Cortex-A target profile."
    )
