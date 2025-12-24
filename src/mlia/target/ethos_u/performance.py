# SPDX-FileCopyrightText: Copyright 2022-2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Performance estimation."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any
from typing import Union

import mlia.backend.vela.compiler as vela_comp
import mlia.backend.vela.performance as vela_perf
from mlia.backend.corstone import is_corstone_backend
from mlia.backend.corstone.performance import CorstonePerformanceMetrics
from mlia.backend.corstone.performance import estimate_performance
from mlia.backend.errors import BackendUnavailableError
from mlia.backend.vela.performance import LayerwisePerfInfo
from mlia.core.context import Context
from mlia.core.context import ExecutionContext
from mlia.core.performance import PerformanceEstimator
from mlia.nn.select import OptimizationSettings
from mlia.nn.tensorflow.config import get_tflite_model
from mlia.nn.tensorflow.config import ModelConfiguration
from mlia.target.ethos_u.config import EthosUConfiguration
from mlia.target.registry import supported_backends
from mlia.utils.logging import log_action

logger = logging.getLogger(__name__)


@dataclass
class NPUCycles:
    """NPU cycles metrics."""

    npu_active_cycles: int
    npu_idle_cycles: int
    npu_total_cycles: int
    npu_axi0_rd_data_beat_received: int
    npu_axi0_wr_data_beat_written: int
    npu_axi1_rd_data_beat_received: int
    npu_axi1_wr_data_beat_written: int | None = None


BYTES_PER_KILOBYTE = 1024


class MemorySizeType(Enum):
    """Memory size type enumeration."""

    BYTES = 0
    KILOBYTES = 1


@dataclass
class MemoryUsage:
    """Memory usage metrics."""

    sram_memory_area_size: int | float
    dram_memory_area_size: int | float
    on_chip_flash_memory_area_size: int | float
    off_chip_flash_memory_area_size: int | float
    memory_size_type: MemorySizeType = MemorySizeType.BYTES

    _default_columns = [
        "SRAM used",
        "DRAM used",
        "Unknown memory used",
        "On chip flash used",
        "Off chip flash used",
    ]


@dataclass
class PerformanceMetrics:
    """Performance metrics."""

    target_config: EthosUConfiguration
    npu_cycles: NPUCycles | None
    memory_usage: MemoryUsage | None
    layerwise_perf_info: LayerwisePerfInfo | None
    corstone_metrics: Any = None  # Backend PerformanceMetrics for standardized output

    def to_standardized_output(
        self,
        model_path: Path,
        backend_name: str | None = None,
        cli_arguments: list[str] | None = None,
        backend_config: dict[str, Any] | None = None,
    ) -> Any:  # Returns StandardizedOutput but avoid circular import
        """Convert to standardized output format.

        Args:
            model_path: Path to the model file
            backend_name: Name of the backend used (e.g., 'corstone-300')
            cli_arguments: Optional CLI arguments used for the run
            backend_config: Optional backend configuration parameters

        Returns:
            StandardizedOutput object or None if no corstone metrics available
        """
        if self.corstone_metrics is None:
            return None

        # Build target config dict from EthosUConfiguration
        target_config: dict[str, Any] = {
            "target": self.target_config.target,
            "mac": self.target_config.mac,
        }

        # Use backend_name from stored metrics or parameter
        if backend_name is None:
            backend_name = "corstone-300"  # Default

        result_dict = self.corstone_metrics.to_standardized_output(
            model_path=model_path,
            backend_name=backend_name,
            target_config=target_config,
            cli_arguments=cli_arguments,
            backend_config=backend_config,
        )
        return result_dict


@dataclass
class CorstonePerformanceResult:
    """Wrapper for performance metrics with both legacy and standardized output."""

    legacy_info: PerformanceMetrics
    standardized_output: dict[str, Any] | None = None


@dataclass
class VelaPerformanceResult:
    """Wrapper for Vela performance metrics with both legacy and standardized output."""

    legacy_info: PerformanceMetrics
    standardized_output: dict[str, Any] | None = None


@dataclass
class CombinedPerformanceResult:
    """Wrapper for combined multi-backend performance metrics."""

    legacy_info: PerformanceMetrics
    standardized_output: dict[str, Any] | None = None


def merge_performance_outputs(
    vela_output: dict[str, Any] | None,
    corstone_output: dict[str, Any] | None,
) -> dict[str, Any]:
    """Merge Vela and Corstone standardized outputs into a single output.

    Args:
        vela_output: Vela standardized output dict
        corstone_output: Corstone standardized output dict

    Returns:
        Combined standardized output with both backends and results
    """
    if not vela_output and not corstone_output:
        raise ValueError("At least one output must be provided")

    # Use the first available output as base
    base = vela_output or corstone_output
    if not base:
        raise ValueError("No valid output provided")

    # Start with base output
    merged = base.copy()

    # If we have both outputs, merge backends and results
    if vela_output and corstone_output:
        # Combine backends from both
        vela_backends = vela_output.get("backends", [])
        corstone_backends = corstone_output.get("backends", [])
        merged["backends"] = vela_backends + corstone_backends

        # Combine results from both
        vela_results = vela_output.get("results", [])
        corstone_results = corstone_output.get("results", [])
        merged["results"] = vela_results + corstone_results

    return merged


@dataclass
class OptimizationPerformanceMetrics:
    """Optimization performance metrics."""

    original_perf_metrics: PerformanceMetrics
    optimizations_perf_metrics: list[
        tuple[list[OptimizationSettings], PerformanceMetrics]
    ]


class VelaPerformanceEstimator(
    PerformanceEstimator[
        Union[Path, ModelConfiguration], tuple[MemoryUsage, LayerwisePerfInfo]
    ]
):
    """Vela based performance estimator."""

    def __init__(self, context: Context, target_config: EthosUConfiguration) -> None:
        """Init Vela based performance estimator."""
        self.context = context
        self.target = target_config

    def estimate(  # pylint: disable=attribute-defined-outside-init
        self, model: Path | ModelConfiguration
    ) -> tuple[MemoryUsage, LayerwisePerfInfo]:
        """Estimate performance."""
        with log_action("Getting the memory usage metrics ..."):
            model_path = (
                Path(model.model_path)
                if isinstance(model, ModelConfiguration)
                else model
            )

            if self.target.compiler_options is None:
                raise BackendUnavailableError("Backend vela is not available", "vela")

            vela_perf_metrics = vela_perf.estimate_performance(
                model_path, self.target.compiler_options
            )

            # Store the raw backend metrics and compiler options for standardized output
            self.vela_perf_metrics = (
                vela_perf_metrics  # pylint: disable=attribute-defined-outside-init
            )
            self.vela_compiler_options = (
                self.target.compiler_options
            )  # pylint: disable=attribute-defined-outside-init

            return (
                MemoryUsage(
                    vela_perf_metrics.sram_memory_area_size,
                    vela_perf_metrics.dram_memory_area_size,
                    vela_perf_metrics.on_chip_flash_memory_area_size,
                    vela_perf_metrics.off_chip_flash_memory_area_size,
                ),
                vela_perf_metrics.layerwise_performance_info,
            )


class CorstonePerformanceEstimator(
    PerformanceEstimator[Union[Path, ModelConfiguration], NPUCycles]
):
    """Corstone-based performance estimator."""

    def __init__(
        self,
        context: ExecutionContext,
        target_config: EthosUConfiguration,
        backend: str,
    ) -> None:
        """Init Corstone-based performance estimator."""
        self.context = context
        self.target_config = target_config
        self.backend = backend
        self.backend_metrics: CorstonePerformanceMetrics | None = None

    def estimate(self, model: Path | ModelConfiguration) -> NPUCycles:
        """Estimate performance."""
        with log_action(f"Getting the performance metrics for '{self.backend}' ..."):
            logger.info(
                "WARNING: This task may require several minutes "
                "(press ctrl-c to interrupt)"
            )

            model_path = (
                Path(model.model_path)
                if isinstance(model, ModelConfiguration)
                else model
            )

            if self.target_config.compiler_options is None:
                raise BackendUnavailableError("Backend vela is not available", "vela")

            optimized_model_path = vela_comp.compile_model(
                model_path, self.target_config.compiler_options
            )

            corstone_perf_metrics = estimate_performance(
                self.target_config.target,
                self.target_config.mac,
                optimized_model_path,
                self.backend,
                self.context.output_dir,
            )

            # Store the raw backend metrics for standardized output generation
            self.backend_metrics = corstone_perf_metrics

            return NPUCycles(
                corstone_perf_metrics.npu_model_stats.npu_active_cycles,
                corstone_perf_metrics.npu_model_stats.npu_idle_cycles,
                corstone_perf_metrics.npu_model_stats.npu_total_cycles,
                corstone_perf_metrics.npu_model_stats.npu_axi0_rd_data_beat_received,
                corstone_perf_metrics.npu_model_stats.npu_axi0_wr_data_beat_written,
                corstone_perf_metrics.npu_model_stats.npu_axi1_rd_data_beat_received,
                corstone_perf_metrics.npu_model_stats.npu_axi1_wr_data_beat_written,
            )


class EthosUPerformanceEstimator(
    PerformanceEstimator[Union[Path, ModelConfiguration], PerformanceMetrics]
):
    """Ethos-U performance estimator."""

    def __init__(
        self,
        context: ExecutionContext,
        target_config: EthosUConfiguration,
        backends: list[str] | None = None,
    ) -> None:
        """Init performance estimator."""
        self.context = context
        self.target_config = target_config
        if backends is None:
            backends = ["vela"]  # Only Vela is always available as default
        ethos_u_backends = supported_backends(target_config.target)
        for backend in backends:
            if backend != "vela" and backend not in ethos_u_backends:
                raise ValueError(
                    f"Unsupported backend '{backend}'. "
                    f"Only 'Vela' and {ethos_u_backends} "
                    "are supported."
                )
        self.backends = set(backends)

    def estimate(  # pylint: disable=attribute-defined-outside-init
        self, model: Path | ModelConfiguration
    ) -> PerformanceMetrics:
        """Estimate performance."""
        model_path = (
            Path(model.model_path) if isinstance(model, ModelConfiguration) else model
        )

        tflite_model = get_tflite_model(model_path, self.context)

        memory_usage = None
        npu_cycles = None
        layerwise_perf_info = None
        corstone_metrics = None
        vela_perf_metrics = None
        for backend in self.backends:
            if backend == "vela":
                vela_estimator = VelaPerformanceEstimator(
                    self.context, self.target_config
                )
                memory_usage, layerwise_perf_info = vela_estimator.estimate(
                    tflite_model
                )
                # Store the raw vela metrics for standardized output
                # VelaPerformanceEstimator.estimate() stores vela_perf_metrics on self
                if hasattr(vela_estimator, "vela_perf_metrics"):
                    vela_perf_metrics = vela_estimator.vela_perf_metrics
                    # Also copy compiler options for backend configuration
                    if hasattr(vela_estimator, "vela_compiler_options"):
                        self.vela_compiler_options = (
                            vela_estimator.vela_compiler_options
                        )  # pylint: disable=attribute-defined-outside-init
            elif is_corstone_backend(backend):
                corstone_estimator = CorstonePerformanceEstimator(
                    self.context, self.target_config, backend
                )
                # Get NPUCycles for legacy display and save the raw backend metrics
                npu_cycles = corstone_estimator.estimate(tflite_model)
                # Store the original corstone backend metrics for standardized output
                corstone_metrics = corstone_estimator.backend_metrics
            else:
                logger.warning(
                    "Backend '%s' is not supported for Ethos-U performance "
                    "estimation.",
                    backend,
                )
        perf = PerformanceMetrics(
            self.target_config, npu_cycles, memory_usage, layerwise_perf_info
        )

        # Attach vela raw metrics object so callers can build standardized output
        # if available. We set attribute only when vela metrics were produced.
        if vela_perf_metrics is not None:
            self.vela_perf_metrics = (
                vela_perf_metrics  # pylint: disable=attribute-defined-outside-init
            )

        # Attach corstone raw metrics object so callers can build standardized output
        # if available. We set attribute only when corstone metrics were produced.
        if corstone_metrics is not None:
            perf.corstone_metrics = corstone_metrics

        return perf
