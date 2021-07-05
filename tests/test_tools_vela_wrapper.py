# Copyright 2021, Arm Ltd.
"""Tests for module tools/vela."""
from pathlib import Path
from typing import List
from typing import Tuple

import pytest
from ethosu.vela.compiler_driver import TensorAllocator
from ethosu.vela.scheduler import OptimizationStrategy
from mlia.config import EthosU55
from mlia.config import TFLiteModel
from mlia.tools.vela_wrapper import OptimizedModel
from mlia.tools.vela_wrapper import supported_operators
from mlia.tools.vela_wrapper import VelaCompiler


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
    assert default_compiler.optimization_strategy == OptimizationStrategy.Performance
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
    assert compiler.optimization_strategy == OptimizationStrategy.Size
    assert compiler.output_dir == "output"


def test_model_optimization(test_models_path: Path) -> None:
    """Test model optimization."""
    model = test_models_path / "simple_3_layers_model.tflite"

    compiler = VelaCompiler()
    optimized_model = compiler.compile_model(model)
    assert isinstance(optimized_model, OptimizedModel)


@pytest.mark.parametrize(
    "model, expected_ops",
    [
        (
            "simple_3_layers_model.tflite",
            [
                ("dense_input", "Placeholder", (False, [])),
                ("sequential/dense/MatMul_reshape", "Const", (False, [])),
                (
                    "sequential/dense/MatMul;sequential/dense/BiasAdd",
                    "FullyConnected",
                    (
                        True,
                        [],
                    ),
                ),
                ("sequential/dense_1/MatMul_reshape", "Const", (False, [])),
                (
                    "sequential/dense_1/MatMul;sequential/dense_1/BiasAdd;sequential/"
                    "dense_1/Relu",
                    "FullyConnected",
                    (
                        True,
                        [],
                    ),
                ),
                ("sequential/dense_2/MatMul_reshape", "Const", (False, [])),
                (
                    "Identity",
                    "FullyConnected",
                    (
                        True,
                        [],
                    ),
                ),
            ],
        )
    ],
)
def test_operations(
    test_models_path: Path, model: str, expected_ops: List[Tuple]
) -> None:
    """Test operations function."""
    tflite_model = TFLiteModel(str(test_models_path / model))
    device = EthosU55()

    ops = supported_operators(tflite_model, device)

    assert len(ops) == len(expected_ops)
    for i, op in enumerate(ops):
        (
            expected_name,
            expected_type,
            (expected_run_on_npu, expected_reasons),
        ) = expected_ops[i]
        assert op.name == expected_name
        assert op.op_type == expected_type
        assert op.run_on_npu == (expected_run_on_npu, expected_reasons)
