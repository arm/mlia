# SPDX-FileCopyrightText: Copyright 2022-2024, 2026, Arm Limited
# and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Ethos-U data analysis module."""
from __future__ import annotations

from dataclasses import dataclass
from functools import singledispatchmethod

from mlia.backend.vela.compat import Operators
from mlia.core.common import DataItem
from mlia.core.data_analysis import Fact
from mlia.core.data_analysis import FactExtractor
from mlia.core.data_analysis import LayerCompatibilityIssue
from mlia.core.data_analysis import register_fact_type
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
    "ethos_u_layer_uses_lut",
    "layer",
    "Layer uses lookup table (LUT) operation on Ethos-U",
)
@dataclass
class EthosULayerUsesLUT(LayerCompatibilityIssue):
    """Fact indicating a layer uses LUT operation.

    LUT operations typically run on CPU and may indicate
    activation functions that could be optimized.
    """

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


class EthosUDataAnalyzer(FactExtractor):
    """Ethos-U data analyzer."""

    @singledispatchmethod
    def analyze_data(self, data_item: DataItem) -> None:  # type: ignore
        """Analyse the data."""

    @analyze_data.register
    def analyze_operator_compatibility(self, operators: Operators) -> None:
        """Analyse operator compatibility information."""
        for idx, op in enumerate(operators.ops):  # pylint: disable=invalid-name
            # Determine NPU placement
            if op.cpu_only:
                npu_placement = "cpu"
            elif op.run_on_npu.supported:
                npu_placement = "npu"
            else:
                npu_placement = "unknown"

            # Create layer compatibility fact
            layer_fact = EthosULayerCompatibilityIssue(
                operator_name=op.name,
                location=f"operator/{idx}",
                operator_type=op.op_type,
                is_supported=op.run_on_npu.supported,
                reasons=op.run_on_npu.reasons,
                npu_placement=npu_placement,
            )
            self.add_fact(layer_fact)

            # Check for LUT usage (common CPU-only patterns)
            if not op.run_on_npu.supported:
                # Check if this is a LUT-based operation
                for reason_category, reason_detail in op.run_on_npu.reasons:
                    if "LUT" in reason_category or "lookup" in reason_detail.lower():
                        # This layer uses LUT
                        lut_fact = EthosULayerUsesLUT(
                            operator_name=op.name,
                            location=f"operator/{idx}",
                            operator_type=op.op_type,
                            is_supported=False,
                            reasons=op.run_on_npu.reasons,
                            activation_type=op.op_type,
                        )
                        self.add_fact(lut_fact)
                        break

        # Keep network-level facts for backward compatibility
        cpu_only = [op.op_type for op in operators.ops if op.cpu_only]
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
