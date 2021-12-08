# Copyright 2021, Arm Ltd.
"""Vela wrapper module."""
import itertools
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Literal
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
from ethosu.vela.architecture_features import ArchitectureFeatures
from ethosu.vela.compiler_driver import compiler_driver
from ethosu.vela.compiler_driver import CompilerOptions
from ethosu.vela.compiler_driver import TensorAllocator
from ethosu.vela.model_reader import ModelReaderOptions
from ethosu.vela.model_reader import read_model
from ethosu.vela.nn_graph import Graph
from ethosu.vela.nn_graph import NetworkType
from ethosu.vela.npu_performance import PassCycles
from ethosu.vela.operation import CustomType
from ethosu.vela.operation import Op
from ethosu.vela.scheduler import OptimizationStrategy
from ethosu.vela.scheduler import SchedulerOptions
from ethosu.vela.tensor import BandwidthDirection
from ethosu.vela.tensor import MemArea
from ethosu.vela.tensor import Tensor
from ethosu.vela.tflite_mapping import optype_to_builtintype
from ethosu.vela.tflite_model_semantic import TFLiteSemantic
from ethosu.vela.tflite_supported_operators import TFLiteSupportedOperators
from ethosu.vela.tflite_writer import write_tflite
from ethosu.vela.vela import generate_supported_ops
from mlia.utils.logging import redirect_output


logger = logging.getLogger(__name__)

VELA_INTERNAL_OPS = (Op.Placeholder, Op.SubgraphInput, Op.Const)


@dataclass
class PerformanceMetrics:  # pylint: disable=too-many-instance-attributes
    """Contains all the performance metrics Vela generates in a run."""

    npu_cycles: int
    sram_access_cycles: int
    dram_access_cycles: int
    on_chip_flash_access_cycles: int
    off_chip_flash_access_cycles: int
    total_cycles: int
    batch_inference_time: float
    inferences_per_second: float
    batch_size: int
    unknown_memory_area_size: int
    sram_memory_area_size: int
    dram_memory_area_size: int
    on_chip_flash_memory_area_size: int
    off_chip_flash_memory_area_size: int


@dataclass
class ModelOperator:
    """Model operator."""

    name: str
    op_type: str
    supported_on_npu: Tuple[bool, List[Tuple[str, str]]]


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
    """Instance of the vela optimized model."""

    nng: Graph
    arch: ArchitectureFeatures
    compiler_options: CompilerOptions
    scheduler_options: SchedulerOptions

    def save(self, output_filename: Union[str, Path]) -> None:
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

    config_files: Optional[Union[str, List[str]]] = None
    system_config: str = ArchitectureFeatures.DEFAULT_CONFIG
    memory_mode: str = ArchitectureFeatures.DEFAULT_CONFIG
    accelerator_config: AcceleratorConfigType = "ethos-u55-256"
    max_block_dependency: int = ArchitectureFeatures.MAX_BLOCKDEP
    arena_cache_size: Optional[int] = None
    tensor_allocator: TensorAllocatorType = "HillClimb"
    cpu_tensor_alignment: int = Tensor.AllocationQuantum
    optimization_strategy: OptimizationStrategyType = "Performance"
    output_dir: Optional[str] = None
    recursion_limit: int = 1000


class VelaCompiler:  # pylint: disable=too-many-instance-attributes
    """Vela compiler wrapper."""

    def __init__(  # pylint: disable=too-many-arguments
        self, compiler_options: VelaCompilerOptions
    ):
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

    def read_model(self, model: Union[str, Path]) -> Model:
        """Read model."""
        logger.debug("Read model %s", model)

        nng, network_type = self._read_model(model)
        return Model(nng, network_type)

    def compile_model(self, model: Union[str, Path, Model]) -> OptimizedModel:
        """Compile the model."""
        if isinstance(model, (str, Path)):
            nng, network_type = self._read_model(model)
        else:
            nng, network_type = model.nng, NetworkType.TFLite

        if not nng:
            raise Exception("Unable to read model")

        arch = self._architecture_features()
        compiler_options = self._compiler_options()
        scheduler_options = self._scheduler_options()

        with redirect_output(logger):
            compiler_driver(
                nng, arch, compiler_options, scheduler_options, network_type
            )

        return OptimizedModel(nng, arch, compiler_options, scheduler_options)

    def get_config(self) -> Dict[str, Any]:
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
    def _read_model(model: Union[str, Path]) -> Tuple[Graph, NetworkType]:
        """Read tflite model."""
        model_path = str(model) if isinstance(model, Path) else model

        with redirect_output(logger, stdout_level=logging.DEBUG):
            return read_model(model_path, ModelReaderOptions())  # type: ignore

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
) -> Dict[str, Any]:
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


def estimate_performance(
    model_path: Path, compiler_options: VelaCompilerOptions
) -> PerformanceMetrics:
    """Return performance estimations for the model/device.

    Logic for this function comes from vela module stats_writer.py
    """
    logger.debug(
        "Estimate performance for the model %s on %s",
        model_path,
        compiler_options.accelerator_config,
    )

    vela_compiler = VelaCompiler(compiler_options)

    initial_model = vela_compiler.read_model(model_path)
    if initial_model.optimized:
        raise Exception("Unable to estimate performance for the given optimized model")

    optimized_model = vela_compiler.compile_model(initial_model)

    return _performance_metrics(optimized_model)


def optimize_model(
    model_path: Path, compiler_options: VelaCompilerOptions, output_model_path: Path
) -> None:
    """Optimize model and return it's path after optimization."""
    logger.debug(
        "Optimize model %s for device %s",
        model_path,
        compiler_options.accelerator_config,
    )

    vela_compiler = VelaCompiler(compiler_options)
    optimized_model = vela_compiler.compile_model(model_path)

    logger.debug("Save optimized model into %s", output_model_path)
    optimized_model.save(output_model_path)


def _performance_metrics(optimized_model: OptimizedModel) -> PerformanceMetrics:
    """Return performance metrics for optimized model."""
    cycles = optimized_model.nng.cycles

    def memory_usage(mem_area: MemArea) -> int:
        """Get memory usage for the proviced memory area type."""
        memory_used: Dict[MemArea, int] = optimized_model.nng.memory_used
        bandwidths = optimized_model.nng.bandwidths

        return memory_used.get(mem_area, 0) if np.sum(bandwidths[mem_area]) > 0 else 0

    midpoint_fps = np.nan
    midpoint_inference_time = cycles[PassCycles.Total] / optimized_model.arch.core_clock
    if midpoint_inference_time > 0:
        midpoint_fps = 1 / midpoint_inference_time

    return PerformanceMetrics(
        npu_cycles=int(cycles[PassCycles.Npu]),
        sram_access_cycles=int(cycles[PassCycles.SramAccess]),
        dram_access_cycles=int(cycles[PassCycles.DramAccess]),
        on_chip_flash_access_cycles=int(cycles[PassCycles.OnChipFlashAccess]),
        off_chip_flash_access_cycles=int(cycles[PassCycles.OffChipFlashAccess]),
        total_cycles=int(cycles[PassCycles.Total]),
        batch_inference_time=midpoint_inference_time * 1000,
        inferences_per_second=midpoint_fps,
        batch_size=optimized_model.nng.batch_size,
        unknown_memory_area_size=memory_usage(MemArea.Unknown),
        sram_memory_area_size=memory_usage(MemArea.Sram),
        dram_memory_area_size=memory_usage(MemArea.Dram),
        on_chip_flash_memory_area_size=memory_usage(MemArea.OnChipFlash),
        off_chip_flash_memory_area_size=memory_usage(MemArea.OffChipFlash),
    )


def supported_operators(
    model_path: Path, compiler_options: VelaCompilerOptions
) -> List[ModelOperator]:
    """Return list of model's operators."""
    logger.debug("Check supported operators for the model %s", model_path)

    vela_compiler = VelaCompiler(compiler_options)
    initial_model = vela_compiler.read_model(model_path)

    return [
        ModelOperator(op.name, optype_to_builtintype(op.type), run_on_npu(op))
        for sg in initial_model.nng.subgraphs
        for op in sg.get_all_ops()
        if op.type not in VELA_INTERNAL_OPS
    ]


def run_on_npu(operator: Op) -> Tuple[bool, List[Tuple[str, str]]]:
    """Return information if operator can run on NPU.

    Vela does a number of checks that can help establish whether
    a particular operator is supported to run on NPU.

    There are two groups of checks:
      - general TFLite constraints
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
            return (False, [(constraint.__doc__, op_reason)])

    if operator.type not in TFLiteSupportedOperators.supported_operators:
        reasons = (
            [("CPU only operator", "")]
            if operator.type not in VELA_INTERNAL_OPS
            else []
        )

        return (False, reasons)

    tflite_supported_operators = TFLiteSupportedOperators()
    operation_constraints = itertools.chain(
        tflite_supported_operators.generic_constraints,
        tflite_supported_operators.specific_constraints[operator.type],
    )
    for constraint in operation_constraints:
        op_valid, op_reason = constraint(operator)
        if not op_valid:
            return (False, [(constraint.__doc__, op_reason)])

    return (True, [])


def generate_supported_operators_report() -> None:
    """Generate supported operators report in current working directory."""
    with redirect_output(logger):
        generate_supported_ops()
