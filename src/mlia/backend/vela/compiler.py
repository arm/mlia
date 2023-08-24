# SPDX-FileCopyrightText: Copyright 2022-2023, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Vela compiler wrapper module."""
from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing import Literal

from ethosu.vela.architecture_features import ArchitectureFeatures
from ethosu.vela.compiler_driver import compiler_driver
from ethosu.vela.compiler_driver import CompilerOptions
from ethosu.vela.compiler_driver import TensorAllocator
from ethosu.vela.model_reader import ModelReaderOptions
from ethosu.vela.model_reader import read_model
from ethosu.vela.nn_graph import Graph
from ethosu.vela.nn_graph import NetworkType
from ethosu.vela.operation import CustomType
from ethosu.vela.scheduler import OptimizationStrategy
from ethosu.vela.scheduler import SchedulerOptions
from ethosu.vela.tensor import BandwidthDirection
from ethosu.vela.tensor import MemArea
from ethosu.vela.tensor import Tensor
from ethosu.vela.tflite_writer import write_tflite

from mlia.utils.logging import redirect_output

logger = logging.getLogger(__name__)


@dataclass
class Model:
    """Model metadata."""

    nng: Graph
    network_type: NetworkType

    @property
    def optimized(self) -> bool:
        """Return true if model is already optimized."""
        return any(
            op.attrs.get("custom_type") == CustomType.ExistingNpuOp
            for sg in self.nng.subgraphs
            for op in sg.get_all_ops()
        )


@dataclass
class OptimizedModel:
    """Instance of the Vela optimized model."""

    nng: Graph
    arch: ArchitectureFeatures
    compiler_options: CompilerOptions
    scheduler_options: SchedulerOptions

    def save(self, output_filename: str | Path) -> None:
        """Save instance of the optimized model to the file."""
        write_tflite(self.nng, output_filename)


AcceleratorConfigType = Literal[
    "ethos-u55-32",
    "ethos-u55-64",
    "ethos-u55-128",
    "ethos-u55-256",
    "ethos-u65-256",
    "ethos-u65-512",
]

TensorAllocatorType = Literal["LinearAlloc", "Greedy", "HillClimb"]

OptimizationStrategyType = Literal["Performance", "Size"]


@dataclass
class VelaCompilerOptions:  # pylint: disable=too-many-instance-attributes
    """Vela compiler options."""

    config_files: str | list[str] | None = None
    system_config: str = ArchitectureFeatures.DEFAULT_CONFIG
    memory_mode: str = ArchitectureFeatures.DEFAULT_CONFIG
    accelerator_config: AcceleratorConfigType | None = None
    max_block_dependency: int = ArchitectureFeatures.MAX_BLOCKDEP
    arena_cache_size: int | None = None
    tensor_allocator: TensorAllocatorType = "HillClimb"
    cpu_tensor_alignment: int = Tensor.AllocationQuantum
    optimization_strategy: OptimizationStrategyType = "Performance"
    output_dir: str = "output"
    recursion_limit: int = 1000


class VelaCompiler:  # pylint: disable=too-many-instance-attributes
    """Vela compiler wrapper."""

    def __init__(self, compiler_options: VelaCompilerOptions):
        """Init Vela wrapper instance."""
        self.config_files = compiler_options.config_files
        self.system_config = compiler_options.system_config
        self.memory_mode = compiler_options.memory_mode
        self.accelerator_config = compiler_options.accelerator_config
        self.max_block_dependency = compiler_options.max_block_dependency
        self.arena_cache_size = compiler_options.arena_cache_size
        self.tensor_allocator = TensorAllocator[compiler_options.tensor_allocator]
        self.cpu_tensor_alignment = compiler_options.cpu_tensor_alignment
        self.optimization_strategy = OptimizationStrategy[
            compiler_options.optimization_strategy
        ]
        self.output_dir = compiler_options.output_dir
        self.recursion_limit = compiler_options.recursion_limit

        sys.setrecursionlimit(self.recursion_limit)

    def read_model(self, model: str | Path) -> Model:
        """Read model."""
        logger.debug("Read model %s", model)

        nng, network_type = self._read_model(model)
        return Model(nng, network_type)

    def compile_model(self, model: str | Path | Model) -> OptimizedModel:
        """Compile the model."""
        if isinstance(model, (str, Path)):
            nng, network_type = self._read_model(model)
        else:
            nng, network_type = model.nng, NetworkType.TFLite

        if not nng:
            raise ValueError("Unable to read model: model.nng is not available")

        output_basename = f"{self.output_dir}/{nng.name}"

        try:
            arch = self._architecture_features()
            compiler_options = self._compiler_options()
            scheduler_options = self._scheduler_options()

            with redirect_output(
                logger, stdout_level=logging.DEBUG, stderr_level=logging.DEBUG
            ):
                compiler_driver(
                    nng,
                    arch,
                    compiler_options,
                    scheduler_options,
                    network_type,
                    output_basename,
                )

            return OptimizedModel(nng, arch, compiler_options, scheduler_options)
        except (SystemExit, Exception) as err:
            raise RuntimeError(
                "Model could not be optimized with Vela compiler."
            ) from err

    def get_config(self) -> dict[str, Any]:
        """Get compiler configuration."""
        arch = self._architecture_features()

        memory_area = {
            mem.name: {
                "clock_scales": arch.memory_clock_scales[mem],
                "burst_length": arch.memory_burst_length[mem],
                "read_latency": arch.memory_latency[mem][BandwidthDirection.Read],
                "write_latency": arch.memory_latency[mem][BandwidthDirection.Write],
            }
            for mem in (
                MemArea.Sram,
                MemArea.Dram,
                MemArea.OnChipFlash,
                MemArea.OffChipFlash,
            )
        }

        return {
            "accelerator_config": arch.accelerator_config.value,
            "system_config": arch.system_config,
            "core_clock": arch.core_clock,
            "axi0_port": arch.axi0_port.name,
            "axi1_port": arch.axi1_port.name,
            "memory_mode": arch.memory_mode,
            "const_mem_area": arch.const_mem_area.name,
            "arena_mem_area": arch.arena_mem_area.name,
            "cache_mem_area": arch.cache_mem_area.name,
            "arena_cache_size": arch.arena_cache_size,
            "permanent_storage_mem_area": arch.permanent_storage_mem_area.name,
            "feature_map_storage_mem_area": arch.feature_map_storage_mem_area.name,
            "fast_storage_mem_area": arch.fast_storage_mem_area.name,
            "memory_area": memory_area,
        }

    @staticmethod
    def _read_model(model: str | Path) -> tuple[Graph, NetworkType]:
        """Read TensorFlow Lite model."""
        model_path = str(model) if isinstance(model, Path) else model
        try:
            with redirect_output(
                logger, stdout_level=logging.DEBUG, stderr_level=logging.DEBUG
            ):
                return read_model(model_path, ModelReaderOptions())  # type: ignore
        except (SystemExit, Exception) as err:
            raise RuntimeError(f"Unable to read model {model_path}.") from err

    def _architecture_features(self) -> ArchitectureFeatures:
        """Return ArchitectureFeatures instance."""
        return ArchitectureFeatures(
            vela_config_files=self.config_files,
            accelerator_config=self.accelerator_config,
            system_config=self.system_config,
            memory_mode=self.memory_mode,
            max_blockdep=self.max_block_dependency,
            verbose_config=False,
            arena_cache_size=self.arena_cache_size,
        )

    def _scheduler_options(self) -> SchedulerOptions:
        """Return SchedulerOptions instance."""
        arch = self._architecture_features()

        return SchedulerOptions(
            optimization_strategy=self.optimization_strategy,
            sram_target=arch.arena_cache_size,
            verbose_schedule=False,
        )

    def _compiler_options(self) -> CompilerOptions:
        """Return CompilerOptions instance."""
        return CompilerOptions(
            verbose_graph=False,
            verbose_quantization=False,
            verbose_packing=False,
            verbose_tensor_purpose=False,
            verbose_tensor_format=False,
            verbose_allocation=False,
            verbose_high_level_command_stream=False,
            verbose_register_command_stream=False,
            verbose_operators=False,
            verbose_weights=False,
            show_cpu_operations=False,
            tensor_allocator=self.tensor_allocator,
            timing=False,
            output_dir=self.output_dir,
            cpu_tensor_alignment=self.cpu_tensor_alignment,
        )


def resolve_compiler_config(
    vela_compiler_options: VelaCompilerOptions,
) -> dict[str, Any]:
    """Resolve passed compiler options.

    Vela has number of configuration parameters that being
    resolved during passing compiler options. E.g. Vela
    reads configuration parameters from vela.ini and fills
    it's internal structures with resolved values (memory mode,
    system mode, etc.).

    In order to get this information we need to create
    instance of the Vela compiler first.
    """
    vela_compiler = VelaCompiler(vela_compiler_options)
    return vela_compiler.get_config()


def optimize_model(
    model_path: Path, compiler_options: VelaCompilerOptions, output_model_path: Path
) -> None:
    """Optimize model and return it's path after optimization."""
    logger.debug(
        "Optimize model %s for target %s",
        model_path,
        compiler_options.accelerator_config,
    )

    vela_compiler = VelaCompiler(compiler_options)
    optimized_model = vela_compiler.compile_model(model_path)

    logger.debug("Save optimized model into %s", output_model_path)
    optimized_model.save(output_model_path)
