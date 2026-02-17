# SPDX-FileCopyrightText: Copyright 2022-2024, 2026, Arm Limited
# and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Ethos-U data analysis module."""

from __future__ import annotations

from dataclasses import dataclass
from functools import singledispatchmethod

from mlia.backend.vela.compat import Operators, VelaCompatibilityResult
from mlia.core.common import DataItem
from mlia.core.data_analysis import (
    Fact,
    FactExtractor,
    LayerCompatibilityIssue,
    register_fact_type,
)
from mlia.nn.select import OptimizationSettings
from mlia.nn.tensorflow.tflite_compat import TFLiteCompatibilityInfo
from mlia.target.common.reporters import analyze_tflite_compatibility_common
from mlia.target.ethos_u.performance import OptimizationPerformanceMetrics


@dataclass
class HasCPUOnlyOperators(Fact):
    """Model has CPU only operators."""

    cpu_only_ops: list[str]


@dataclass
class HasUnsupportedOnNPUOperators(Fact):
    """Model has unsupported on NPU operators."""

    npu_unsupported_ratio: float


@dataclass
class AllOperatorsSupportedOnNPU(Fact):
    """All model's operators supported on NPU."""


@register_fact_type(
    "ethos_u_layer_compatibility",
    "layer",
    "Ethos-U specific layer compatibility information",
)
@dataclass
class EthosULayerCompatibilityIssue(LayerCompatibilityIssue):
    """Ethos-U specific layer compatibility fact.

    Extends base LayerCompatibilityIssue with NPU placement information.
    """

    npu_placement: str = "unknown"  # 'npu', 'cpu', or 'unknown'

    def to_dict(self) -> dict:
        """Convert to dictionary with proper serialization."""
        result = super().to_dict()
        result["npu_placement"] = self.npu_placement
        return result


@register_fact_type(
    "ethos_u_layer_suboptimal_activation",
    "layer",
    "Layer uses suboptimal activation function on Ethos-U",
)
@dataclass
class EthosULayerSuboptimalActivation(LayerCompatibilityIssue):
    """Fact indicating a layer uses suboptimal activation function."""

    activation_type: str = "unknown"

    def to_dict(self) -> dict:
        """Convert to dictionary with proper serialization."""
        result = super().to_dict()
        result["activation_type"] = self.activation_type
        return result


@dataclass
class PerfMetricDiff:
    """Performance metric difference."""

    original_value: int | float
    optimized_value: int | float

    @property
    def diff(self) -> float:
        """Difference between metrics."""
        if self.original_value == 0:
            return 0

        return 100 - ((self.optimized_value / self.original_value) * 100)

    @property
    def improved(self) -> bool:
        """Return true if metric improved."""
        return self.diff > 0

    @property
    def degraded(self) -> bool:
        """Return true if metric degraded."""
        return self.diff < 0

    @property
    def same(self) -> bool:
        """Return true if metric stays the same."""
        return self.diff == 0


@dataclass
class OptimizationDiff:
    """Optimization performance impact."""

    opt_type: list[OptimizationSettings]
    opt_diffs: dict[str, PerfMetricDiff]


@dataclass
class OptimizationResults(Fact):
    """Optimization results."""

    diffs: list[OptimizationDiff]


@dataclass
class LutPatternRule:
    """Activation funciton operator patterns."""

    op_seq: list[str]
    op_label: str


PATTERN_RULES = [
    LutPatternRule(
        ["greater", "fully_connected", "exp", "sub", "mul", "select"], "SELU"
    ),
    LutPatternRule(["greater", "mul", "exp", "sub", "mul", "select"], "SELU"),
    LutPatternRule(["exp", "add", "log", "tanh", "mul"], "MISH"),
    LutPatternRule(["exp", "add", "log"], "SOFTPLUS"),
    LutPatternRule(["logistic", "mul"], "LOGISTIC BASED OPERATION"),
    LutPatternRule(["exp"], "EXPONENTIAL"),
    LutPatternRule(["gelu"], "GELU"),
    LutPatternRule(["elu"], "ELU"),
    LutPatternRule(["tanh"], "TANH"),
    LutPatternRule(["logistic"], "SIGMOID"),
    LutPatternRule(["softmax"], "SOFTMAX"),
]


class EthosUDataAnalyzer(FactExtractor):
    """Ethos-U data analyzer."""

    @singledispatchmethod
    def analyze_data(self, data_item: DataItem) -> None:  # type: ignore
        """Analyse the data."""
        print(
            f"DEBUG: Unhandled data_item type: {type(data_item)} - "
            f"{data_item.__class__.__module__}.{data_item.__class__.__name__}"
        )

    @analyze_data.register
    def analyze_vela_compatibility(self, vela_result: VelaCompatibilityResult) -> None:
        """Analyse Vela compatibility result and extract operator information."""
        # Extract the Operators object from VelaCompatibilityResult
        self.analyze_operator_compatibility(vela_result.legacy_info)
        # Analyze activation usage patterns
        self._analyze_activation_function(vela_result.legacy_info)

    def _sequence_matches(self, ops: list, idx: int, function_ops: list[str]) -> bool:
        """Check operators sequence matches activation function operator pattern."""
        if idx + len(function_ops) > len(ops):
            return False
        for i, function_op in enumerate(function_ops):
            # Check compatibility operators sequentially against activation function
            if not ops[idx + i].op_type.lower() == function_op:
                return False
        return True

    def _analyze_activation_function(self, operators: Operators) -> None:
        """Analyze operators from compatibility check for activation functions."""
        ops = operators.ops
        idx = 0
        while idx < len(ops):
            # Check if activation function pattern occers
            for rule in PATTERN_RULES:
                if self._sequence_matches(ops, idx, rule.op_seq):
                    operator = ops[idx]
                    lut_fact = EthosULayerSuboptimalActivation(
                        operator_name=operator.name,
                        location=f"operator/{idx}",
                        operator_type=operator.op_type,
                        is_supported=operator.run_on_npu.supported,
                        reasons=operator.run_on_npu.reasons,
                        activation_type=rule.op_label,
                    )
                    self.add_fact(lut_fact)
                    # Skip to next operation not in detected pattern
                    idx += len(rule.op_seq)
                    break
            else:
                idx += 1

    @analyze_data.register
    def analyze_operator_compatibility(self, operators: Operators) -> None:
        """Analyse operator compatibility information."""
        for idx, operator in enumerate(operators.ops):
            # Determine NPU placement
            if operator.cpu_only:
                npu_placement = "cpu"
            elif operator.run_on_npu.supported:
                npu_placement = "npu"
            else:
                npu_placement = "unknown"

            # Create layer compatibility fact
            layer_fact = EthosULayerCompatibilityIssue(
                operator_name=operator.name,
                location=f"operator/{idx}",
                operator_type=operator.op_type,
                is_supported=operator.run_on_npu.supported,
                reasons=operator.run_on_npu.reasons,
                npu_placement=npu_placement,
            )
            self.add_fact(layer_fact)

        # Keep network-level facts for backward compatibility
        cpu_only = [operator.op_type for operator in operators.ops if operator.cpu_only]
        if cpu_only:
            self.add_fact(HasCPUOnlyOperators(cpu_only))

        if operators.npu_unsupported_ratio != 0:
            self.add_fact(HasUnsupportedOnNPUOperators(operators.npu_unsupported_ratio))

        if operators.npu_unsupported_ratio == 0:
            self.add_fact(AllOperatorsSupportedOnNPU())

    @analyze_data.register
    def analyze_optimization_results(
        self, optimization_results: OptimizationPerformanceMetrics
    ) -> None:
        """Analyse optimization performance metrics."""
        optimizations = optimization_results.optimizations_perf_metrics
        if not optimizations:
            return

        orig = optimization_results.original_perf_metrics
        orig_memory = orig.memory_usage
        orig_cycles = orig.npu_cycles

        diffs: list[OptimizationDiff] = []
        for opt_type, opt_perf_metrics in optimizations:
            opt = opt_perf_metrics
            opt_memory = opt.memory_usage
            opt_cycles = opt.npu_cycles

            opt_diffs: dict[str, PerfMetricDiff] = {}

            if orig_memory and opt_memory:
                opt_diffs.update(
                    {
                        "sram": PerfMetricDiff(
                            orig_memory.sram_memory_area_size,
                            opt_memory.sram_memory_area_size,
                        ),
                        "dram": PerfMetricDiff(
                            orig_memory.dram_memory_area_size,
                            opt_memory.dram_memory_area_size,
                        ),
                        "on_chip_flash": PerfMetricDiff(
                            orig_memory.on_chip_flash_memory_area_size,
                            opt_memory.on_chip_flash_memory_area_size,
                        ),
                        "off_chip_flash": PerfMetricDiff(
                            orig_memory.off_chip_flash_memory_area_size,
                            opt_memory.off_chip_flash_memory_area_size,
                        ),
                    }
                )
            if orig_cycles and opt_cycles:
                opt_diffs["npu_total_cycles"] = PerfMetricDiff(
                    orig_cycles.npu_total_cycles,
                    opt_cycles.npu_total_cycles,
                )

            diff = OptimizationDiff(opt_type=opt_type, opt_diffs=opt_diffs)
            diffs.append(diff)

        self.add_fact(OptimizationResults(diffs))

    @analyze_data.register
    def analyze_tflite_compatibility(self, data_item: TFLiteCompatibilityInfo) -> None:
        """Analyze TensorFlow Lite compatibility information."""
        analyze_tflite_compatibility_common(self, data_item)
