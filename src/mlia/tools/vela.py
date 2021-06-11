"""Vela wrapper module."""
import itertools
from pathlib import Path
from typing import cast
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

from ethosu.vela.architecture_features import ArchitectureFeatures
from ethosu.vela.compiler_driver import compiler_driver
from ethosu.vela.compiler_driver import CompilerOptions
from ethosu.vela.compiler_driver import TensorAllocator
from ethosu.vela.model_reader import ModelReaderOptions
from ethosu.vela.model_reader import read_model
from ethosu.vela.nn_graph import Graph
from ethosu.vela.operation import Op
from ethosu.vela.scheduler import SchedulerOptions
from ethosu.vela.supported_operators import SupportedOperators
from typing_extensions import Literal


class Operation:
    """Operation details class."""

    def __init__(self, op: Op) -> None:
        """Init operation details instance."""
        self.op = op

    def name(self) -> str:
        """Return operation name."""
        return cast(str, self.op.name)

    def type(self) -> str:
        """Return operation type."""
        return str(self.op.type)

    def run_on_npu(self) -> Tuple[bool, List[Tuple[str, str]]]:
        """Return true if operation can run on NPU."""
        supported_operators = SupportedOperators()
        if self.op.type not in SupportedOperators.supported_operators:
            return False, []

        operation_constraints = itertools.chain(
            supported_operators.generic_constraints,
            supported_operators.specific_constraints[self.op.type],
        )
        for constraint in operation_constraints:
            op_valid, op_reason = constraint(self.op)
            if not op_valid:
                return False, [(constraint.__doc__, op_reason)]

        return True, []


class Model:
    """Model metadata."""

    def __init__(self, nng: Graph) -> None:
        """Instance of the model metadata."""
        self.nng = nng

    def operations(self) -> List[Operation]:
        """Return operation details."""
        op_details = [
            Operation(op) for sg in self.nng.subgraphs for op in sg.get_all_ops()
        ]

        return op_details


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

OptimizationStrategyType = Literal["Performance", "Size"]

TensorAllocatorType = Literal["LinearAlloc", "Greedy", "HillClimb"]


class VelaCompiler:
    """Vela compiler wrapper."""

    def __init__(
        self,
        config_files: Optional[Union[str, List[str]]] = None,
        system_config: str = "internal-default",
        memory_mode: str = "internal-default",
        accelerator_config: AcceleratorConfigType = "ethos-u55-256",
        max_block_dependency: int = 3,
        arena_cache_size: Optional[int] = None,
        tensor_allocator: TensorAllocatorType = "HillClimb",
        cpu_tensor_alignment: int = 16,
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
        self.optimization_strategy = optimization_strategy
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
        return SchedulerOptions(
            optimization_strategy=self.optimization_strategy,
            sram_target=self.arena_cache_size,
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


def operations(model: Union[str, Path]) -> List[Operation]:
    """Return list of model operations."""
    compiler = VelaCompiler()
    model_metadata = compiler.read_model(model)

    return model_metadata.operations()
