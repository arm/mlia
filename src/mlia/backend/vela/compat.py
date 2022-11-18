# SPDX-FileCopyrightText: Copyright 2022, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Vela operator compatibility module."""
from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass
from pathlib import Path

from ethosu.vela.operation import Op
from ethosu.vela.tflite_mapping import optype_to_builtintype
from ethosu.vela.tflite_model_semantic import TFLiteSemantic
from ethosu.vela.tflite_supported_operators import TFLiteSupportedOperators
from ethosu.vela.vela import generate_supported_ops

from mlia.backend.vela.compiler import VelaCompiler
from mlia.backend.vela.compiler import VelaCompilerOptions
from mlia.utils.logging import redirect_output


logger = logging.getLogger(__name__)

VELA_INTERNAL_OPS = (Op.Placeholder, Op.SubgraphInput, Op.Const)


@dataclass
class NpuSupported:
    """Operator's npu supported attribute."""

    supported: bool
    reasons: list[tuple[str, str]]


@dataclass
class Operator:
    """Model operator."""

    name: str
    op_type: str
    run_on_npu: NpuSupported

    @property
    def cpu_only(self) -> bool:
        """Return true if operator is CPU only."""
        cpu_only_reasons = [("CPU only operator", "")]
        return (
            not self.run_on_npu.supported
            and self.run_on_npu.reasons == cpu_only_reasons
        )


@dataclass
class Operators:
    """Model's operators."""

    ops: list[Operator]

    @property
    def npu_supported_ratio(self) -> float:
        """Return NPU supported ratio."""
        total = self.total_number
        npu_supported = self.npu_supported_number

        if total == 0 or npu_supported == 0:
            return 0

        return npu_supported / total

    @property
    def npu_unsupported_ratio(self) -> float:
        """Return NPU unsupported ratio."""
        return 1 - self.npu_supported_ratio

    @property
    def total_number(self) -> int:
        """Return total number of operators."""
        return len(self.ops)

    @property
    def npu_supported_number(self) -> int:
        """Return number of npu supported operators."""
        return sum(op.run_on_npu.supported for op in self.ops)


def supported_operators(
    model_path: Path, compiler_options: VelaCompilerOptions
) -> Operators:
    """Return list of model's operators."""
    logger.debug("Check supported operators for the model %s", model_path)

    vela_compiler = VelaCompiler(compiler_options)
    initial_model = vela_compiler.read_model(model_path)

    return Operators(
        [
            Operator(op.name, optype_to_builtintype(op.type), run_on_npu(op))
            for sg in initial_model.nng.subgraphs
            for op in sg.get_all_ops()
            if op.type not in VELA_INTERNAL_OPS
        ]
    )


def run_on_npu(operator: Op) -> NpuSupported:
    """Return information if operator can run on NPU.

    Vela does a number of checks that can help establish whether
    a particular operator is supported to run on NPU.

    There are two groups of checks:
      - general TensorFlow Lite constraints
      - operator specific constraints

    If an operator is not supported on NPU then this function
    will return the reason of that.

    The reason is split in two parts:
      - general description of why the operator cannot be placed on NPU
      - details on the particular operator
    """
    semantic_checker = TFLiteSemantic()
    semantic_constraints = itertools.chain(
        semantic_checker.generic_constraints,
        semantic_checker.specific_constraints[operator.type],
    )

    for constraint in semantic_constraints:
        op_valid, op_reason = constraint(operator)
        if not op_valid:
            return NpuSupported(False, [(constraint.__doc__, op_reason)])

    if operator.type not in TFLiteSupportedOperators.supported_operators:
        reasons = (
            [("CPU only operator", "")]
            if operator.type not in VELA_INTERNAL_OPS
            else []
        )

        return NpuSupported(False, reasons)

    tflite_supported_operators = TFLiteSupportedOperators()
    operation_constraints = itertools.chain(
        tflite_supported_operators.generic_constraints,
        tflite_supported_operators.specific_constraints[operator.type],
    )
    for constraint in operation_constraints:
        op_valid, op_reason = constraint(operator)
        if not op_valid:
            return NpuSupported(False, [(constraint.__doc__, op_reason)])

    return NpuSupported(True, [])


def generate_supported_operators_report() -> None:
    """Generate supported operators report in current working directory."""
    with redirect_output(logger):
        generate_supported_ops()
