# Copyright 2021, Arm Ltd.
"""Vela wrapper module."""
import itertools
import logging
from pathlib import Path
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

import numpy as np
from ethosu.vela.architecture_features import ArchitectureFeatures
from ethosu.vela.compiler_driver import compiler_driver
from ethosu.vela.compiler_driver import CompilerOptions
from ethosu.vela.compiler_driver import TensorAllocator
from ethosu.vela.model_reader import ModelReaderOptions
from ethosu.vela.model_reader import read_model
from ethosu.vela.nn_graph import Graph
from ethosu.vela.npu_performance import PassCycles
from ethosu.vela.operation import CustomType
from ethosu.vela.operation import Op
from ethosu.vela.scheduler import OptimizationStrategy
from ethosu.vela.scheduler import SchedulerOptions
from ethosu.vela.supported_operators import SupportedOperators
from ethosu.vela.tensor import MemArea
from ethosu.vela.tensor import Tensor
from ethosu.vela.tflite_mapping import optype_to_builtintype
from ethosu.vela.tflite_writer import write_tflite
from ethosu.vela.vela import generate_supported_ops
from mlia.config import EthosUConfiguration
from mlia.config import TFLiteModel
from mlia.metadata import NpuSupported
from mlia.metadata import Operator
from mlia.metadata import Operators
from mlia.utils.general import redirect_output
from typing_extensions import Literal


LOGGER = logging.getLogger("mlia.tools.vela")

VELA_INTERNAL_OPS = (Op.Placeholder, Op.SubgraphInput, Op.Const)


class PerformanceMetrics:
    """Contains all the performance metrics Vela generates in a run."""

    def __init__(
        self,
        npu_cycles: int,
        sram_access_cycles: int,
        dram_access_cycles: int,
        on_chip_flash_access_cycles: int,
        off_chip_flash_access_cycles: int,
        total_cycles: int,
        batch_inference_time: float,
        inferences_per_second: float,
        batch_size: int,
        unknown_memory_area_size: int,
        sram_memory_area_size: int,
        dram_memory_area_size: int,
        on_chip_flash_memory_area_size: int,
        off_chip_flash_memory_area_size: int,
    ) -> None:
        """Initialize the performance metrics instance."""
        self.npu_cycles = npu_cycles
        self.sram_access_cycles = sram_access_cycles
        self.dram_access_cycles = dram_access_cycles
        self.on_chip_flash_access_cycles = on_chip_flash_access_cycles
        self.off_chip_flash_access_cycles = off_chip_flash_access_cycles
        self.total_cycles = total_cycles
        self.batch_inference_time = batch_inference_time
        self.inferences_per_second = inferences_per_second
        self.batch_size = batch_size

        self.cycles_per_batch_unit = "cycles/batch"
        self.inference_time_unit = "ms"
        self.inferences_per_second_unit = "inf/s"

        self.unknown_memory_area_size = unknown_memory_area_size
        self.sram_memory_area_size = sram_memory_area_size
        self.dram_memory_area_size = dram_memory_area_size
        self.on_chip_flash_memory_area_size = on_chip_flash_memory_area_size
        self.off_chip_flash_memory_area_size = off_chip_flash_memory_area_size


class Model:
    """Model metadata."""

    def __init__(self, nng: Graph) -> None:
        """Instance of the model metadata."""
        self.nng = nng

    @property
    def optimized(self) -> bool:
        """Return true if model is already optimized."""
        return any(
            op.attrs.get("custom_type") == CustomType.ExistingNpuOp
            for sg in self.nng.subgraphs
            for op in sg.get_all_ops()
        )


class OptimizedModel:
    """Instance of the vela optimized model."""

    def __init__(
        self,
        nng: Graph,
        arch: ArchitectureFeatures,
        compiler_options: CompilerOptions,
        scheduler_options: SchedulerOptions,
    ) -> None:
        """Vela optimized model instance."""
        self.nng = nng
        self.arch = arch
        self.compiler_options = compiler_options
        self.scheduler_options = scheduler_options

    def save(self, output_filename: str) -> None:
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


class VelaCompiler:
    """Vela compiler wrapper."""

    def __init__(
        self,
        config_files: Optional[Union[str, List[str]]] = None,
        system_config: str = ArchitectureFeatures.DEFAULT_CONFIG,
        memory_mode: str = ArchitectureFeatures.DEFAULT_CONFIG,
        accelerator_config: AcceleratorConfigType = "ethos-u55-256",
        max_block_dependency: int = ArchitectureFeatures.MAX_BLOCKDEP,
        arena_cache_size: Optional[int] = None,
        tensor_allocator: TensorAllocatorType = "HillClimb",
        cpu_tensor_alignment: int = Tensor.AllocationQuantum,
        optimization_strategy: OptimizationStrategyType = "Performance",
        output_dir: Optional[str] = None,
    ):
        """Init Vela wrapper instance."""
        self.config_files = config_files
        self.system_config = system_config
        self.memory_mode = memory_mode
        self.accelerator_config = accelerator_config
        self.max_block_dependency = max_block_dependency
        self.arena_cache_size = arena_cache_size
        self.tensor_allocator = TensorAllocator[tensor_allocator]
        self.cpu_tensor_alignment = cpu_tensor_alignment
        self.optimization_strategy = OptimizationStrategy[optimization_strategy]
        self.output_dir = output_dir

    def read_model(self, model: Union[str, Path]) -> Model:
        """Read model."""
        LOGGER.debug(f"Read model {model}")

        nng = self._read_model(model)
        return Model(nng)

    def compile_model(self, model: Union[str, Path, Model]) -> OptimizedModel:
        """Compile the model."""
        if isinstance(model, (str, Path)):
            nng = self._read_model(model)
        else:
            nng = model.nng

        if not nng:
            raise Exception("Unable to read model")

        arch = self._architecture_features()
        compiler_options = self._compiler_options()
        scheduler_options = self._scheduler_options()

        with redirect_output(LOGGER):
            compiler_driver(nng, arch, compiler_options, scheduler_options)

        return OptimizedModel(nng, arch, compiler_options, scheduler_options)

    @staticmethod
    def _read_model(model: Union[str, Path]) -> Graph:
        """Read tflite model."""
        model_path = str(model) if isinstance(model, Path) else model

        with redirect_output(LOGGER):
            return read_model(model_path, ModelReaderOptions())

    def _architecture_features(self) -> ArchitectureFeatures:
        """Return ArchitectureFeatures instance."""
        return ArchitectureFeatures(
            vela_config_files=self.config_files,
            system_config=self.system_config,
            memory_mode=self.memory_mode,
            accelerator_config=self.accelerator_config,
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


def get_vela_compiler(device: EthosUConfiguration) -> VelaCompiler:
    """Get Vela compiler instance for provided device configuration."""
    compiler_options = device.compiler_options

    return VelaCompiler(
        config_files=compiler_options.config_files,
        system_config=compiler_options.system_config,
        memory_mode=compiler_options.memory_mode,
        accelerator_config=compiler_options.accelerator_config,
        max_block_dependency=compiler_options.max_block_dependency,
        arena_cache_size=compiler_options.arena_cache_size,
        tensor_allocator=compiler_options.tensor_allocator,
        cpu_tensor_alignment=compiler_options.cpu_tensor_alignment,
        optimization_strategy=compiler_options.optimization_strategy,
        output_dir=compiler_options.output_dir,
    )


def estimate_performance(
    model: TFLiteModel, device: EthosUConfiguration
) -> PerformanceMetrics:
    """Return performance estimations for the model/device.

    Logic for this function comes from vela module stats_writer.py
    """
    LOGGER.debug(
        f"Estimate performance for the model {model.model_path} "
        f"on device {device.ip_class}"
    )

    vela_compiler = get_vela_compiler(device)

    initial_model = vela_compiler.read_model(model.model_path)
    if initial_model.optimized:
        raise Exception("Unable to estimate performance for the given optimized model")

    optimized_model = vela_compiler.compile_model(initial_model)

    return _performance_metrics(optimized_model)


def optimize_model(
    model: TFLiteModel, device: EthosUConfiguration, output_filename: str
) -> None:
    """Optimize model and return it's path after optimization."""
    LOGGER.debug(f"Optimize model {model.model_path} for device {device.ip_class}")

    vela_compiler = get_vela_compiler(device)
    optimized_model = vela_compiler.compile_model(model.model_path)

    LOGGER.debug(f"Save optimized model into {output_filename}")
    optimized_model.save(output_filename)


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


def supported_operators(model: TFLiteModel, device: EthosUConfiguration) -> Operators:
    """Return list of model's operations."""
    LOGGER.debug(f"Check supported operators for the model {model.model_path}")

    vela_compiler = get_vela_compiler(device)
    initial_model = vela_compiler.read_model(model.model_path)

    return Operators(
        [
            Operator(op.name, optype_to_builtintype(op.type), run_on_npu(op))
            for sg in initial_model.nng.subgraphs
            for op in sg.get_all_ops()
            if op.type not in VELA_INTERNAL_OPS
        ]
    )


def run_on_npu(op: Op) -> NpuSupported:
    """Return true if operation can run on NPU."""
    supported_operators = SupportedOperators()
    if op.type not in SupportedOperators.supported_operators:
        reasons = (
            [("CPU only operator", "")] if op.type not in VELA_INTERNAL_OPS else []
        )

        return NpuSupported(False, reasons)

    operation_constraints = itertools.chain(
        supported_operators.generic_constraints,
        supported_operators.specific_constraints[op.type],
    )
    for constraint in operation_constraints:
        op_valid, op_reason = constraint(op)
        if not op_valid:
            return NpuSupported(False, [(constraint.__doc__, op_reason)])

    return NpuSupported(True, [])


def generate_supported_operators_report() -> None:
    """Generate supported operators report in current working directory."""
    with redirect_output(LOGGER):
        generate_supported_ops()
