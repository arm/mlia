"""Tests for module tools/vela."""
from pathlib import Path

from ethosu.vela.compiler_driver import TensorAllocator
from mlia.tools.vela import operations
from mlia.tools.vela import OptimizedModel
from mlia.tools.vela import VelaCompiler


def test_default_vela_compiler() -> None:
    """Test default Vela compiler instance."""
    default_compiler = VelaCompiler()

    assert default_compiler.config_files is None
    assert default_compiler.system_config == "internal-default"
    assert default_compiler.memory_mode == "internal-default"
    assert default_compiler.accelerator_config == "ethos-u55-256"
    assert default_compiler.max_block_dependency == 3
    assert default_compiler.arena_cache_size is None
    assert default_compiler.tensor_allocator == TensorAllocator.HillClimb
    assert default_compiler.cpu_tensor_alignment == 16
    assert default_compiler.optimization_strategy == "Performance"
    assert default_compiler.output_dir is None


def test_vela_compiler_with_parameters() -> None:
    """Test creation of Vela compiler instance with non-default params."""
    compiler = VelaCompiler(
        config_files=["vela.ini"],
        system_config="test_system_config",
        memory_mode="test_memory_mode",
        accelerator_config="ethos-u65-256",
        max_block_dependency=1,
        arena_cache_size=10,
        tensor_allocator="Greedy",
        cpu_tensor_alignment=4,
        optimization_strategy="Size",
        output_dir="output",
    )

    assert compiler.config_files == ["vela.ini"]
    assert compiler.system_config == "test_system_config"
    assert compiler.memory_mode == "test_memory_mode"
    assert compiler.accelerator_config == "ethos-u65-256"
    assert compiler.max_block_dependency == 1
    assert compiler.arena_cache_size == 10
    assert compiler.tensor_allocator == TensorAllocator.Greedy
    assert compiler.cpu_tensor_alignment == 4
    assert compiler.optimization_strategy == "Size"
    assert compiler.output_dir == "output"


def test_model_optimization(test_models_path: Path) -> None:
    """Test model optimization."""
    model = test_models_path / "simple_3_layer_model.tflite"

    compiler = VelaCompiler()
    optimized_model = compiler.compile_model(model)
    assert isinstance(optimized_model, OptimizedModel)


def test_operations(test_models_path: Path) -> None:
    """Test operations function."""
    model = test_models_path / "simple_3_layer_model.tflite"
    ops = operations(model)

    assert len(ops) == 10

    op = ops[0]
    assert op.name() == "dense_input"
    assert op.type() == "Placeholder"
    assert op.run_on_npu() == (False, [])
