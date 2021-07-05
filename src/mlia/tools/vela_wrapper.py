# Copyright 2021, Arm Ltd.
"""Vela wrapper module."""
import itertools
from pathlib import Path
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
from ethosu.vela.operation import Op
from ethosu.vela.scheduler import OptimizationStrategy
from ethosu.vela.scheduler import SchedulerOptions
from ethosu.vela.supported_operators import SupportedOperators
from ethosu.vela.tensor import Tensor
from mlia.config import EthosUConfiguration
from mlia.config import TFLiteModel
from mlia.metadata import NpuSupported
from mlia.metadata import Operation
from mlia.metrics import PerformanceMetrics
from typing_extensions import Literal


class Model:
    """Model metadata."""

    def __init__(self, nng: Graph) -> None:
        """Instance of the model metadata."""
        self.nng = nng


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
        nng = self._read_model(model)

        return Model(nng)

    def compile_model(self, model: Union[str, Path]) -> OptimizedModel:
        """Compile the model."""
        nng = self._read_model(model)

        if not nng:
            raise Exception("Unable to read model")

        arch = self._architecture_features()
        compiler_options = self._compiler_options()
        scheduler_options = self._scheduler_options()

        compiler_driver(nng, arch, compiler_options, scheduler_options)

        return OptimizedModel(nng, arch, compiler_options, scheduler_options)

    @staticmethod
    def _read_model(model: Union[str, Path]) -> Graph:
        """Read tflite model."""
        model_path = str(model) if isinstance(model, Path) else model

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
    """Return performance estimations for the model/device."""
    vela_compiler = get_vela_compiler(device)
    optimized_model = vela_compiler.compile_model(model.model_path)

    arch = optimized_model.arch
    cycles = optimized_model.nng.cycles
    batch_size = optimized_model.nng.batch_size

    # this logic comes from Vela's module stats_writer.py
    midpoint_fps = np.nan
    midpoint_inference_time = cycles[PassCycles.Total] / arch.core_clock
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
        batch_size=batch_size,
    )


def supported_operators(
    model: TFLiteModel, device: EthosUConfiguration
) -> List[Operation]:
    """Return list of model's operations."""
    vela_compiler = get_vela_compiler(device)
    initial_model = vela_compiler.read_model(model.model_path)

    return [
        Operation(op.name, str(op.type), run_on_npu(op))
        for sg in initial_model.nng.subgraphs
        for op in sg.get_all_ops()
    ]


def run_on_npu(op: Op) -> NpuSupported:
    """Return true if operation can run on NPU."""
    supported_operators = SupportedOperators()
    if op.type not in SupportedOperators.supported_operators:
        return NpuSupported(False, [])

    operation_constraints = itertools.chain(
        supported_operators.generic_constraints,
        supported_operators.specific_constraints[op.type],
    )
    for constraint in operation_constraints:
        op_valid, op_reason = constraint(op)
        if not op_valid:
            return NpuSupported(False, [(constraint.__doc__, op_reason)])

    return NpuSupported(True, [])
