# Copyright 2021, Arm Ltd.
"""Tests for module tools/vela_wrapper."""
from pathlib import Path
from typing import Any
from typing import List
from typing import Tuple

import pytest
from ethosu.vela.compiler_driver import TensorAllocator
from ethosu.vela.scheduler import OptimizationStrategy
from mlia.config import EthosU55
from mlia.config import TFLiteModel
from mlia.tools.vela_wrapper import estimate_performance
from mlia.tools.vela_wrapper import generate_supported_operators_report
from mlia.tools.vela_wrapper import get_vela_compiler
from mlia.tools.vela_wrapper import optimize_model
from mlia.tools.vela_wrapper import OptimizedModel
from mlia.tools.vela_wrapper import PerformanceMetrics
from mlia.tools.vela_wrapper import supported_operators
from mlia.tools.vela_wrapper import VelaCompiler
from mlia.utils.proc import working_directory


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


def test_compile_model(test_models_path: Path) -> None:
    """Test model optimization."""
    model = test_models_path / "simple_3_layers_model.tflite"
    compiler = get_vela_compiler(EthosU55())

    optimized_model = compiler.compile_model(model)
    assert isinstance(optimized_model, OptimizedModel)


def test_optimize_model(test_models_path: Path, tmpdir: Any) -> None:
    """Test model optimization and saving into file."""
    model = test_models_path / "simple_3_layers_model.tflite"
    tmp_file = Path(tmpdir) / "temp.tflite"

    optimize_model(TFLiteModel(str(model)), EthosU55(), str(tmp_file.absolute()))

    assert tmp_file.is_file()
    assert tmp_file.stat().st_size > 0


def test_get_compiler_for_device() -> None:
    """Test getting Vela compiler for the device."""
    device = EthosU55(
        mac=32,
        config_files="test_vela.ini",
        system_config="test_system_config",
        memory_mode="test_memory_mode",
        max_block_dependency=1,
        arena_cache_size=123,
        tensor_allocator="Greedy",
        cpu_tensor_alignment=1,
        optimization_strategy="Size",
        output_dir="output",
    )

    compile = get_vela_compiler(device)

    assert compile.config_files == "test_vela.ini"
    assert compile.system_config == "test_system_config"
    assert compile.memory_mode == "test_memory_mode"
    assert compile.accelerator_config == "ethos-u55-32"
    assert compile.arena_cache_size == 123
    assert compile.tensor_allocator == TensorAllocator.Greedy
    assert compile.cpu_tensor_alignment == 1
    assert compile.output_dir == "output"
    assert compile.optimization_strategy == OptimizationStrategy.Size


@pytest.mark.parametrize(
    "model, expected_ops",
    [
        (
            "simple_3_layers_model.tflite",
            [
                (
                    "sequential/dense/MatMul1",
                    "FULLY_CONNECTED",
                    (
                        True,
                        [],
                    ),
                ),
                (
                    "sequential/dense/BiasAdd;sequential/dense_1/MatMul;"
                    "sequential/dense_1/Relu;sequential/dense_1/BiasAdd",
                    "FULLY_CONNECTED",
                    (
                        True,
                        [],
                    ),
                ),
                (
                    "Identity",
                    "FULLY_CONNECTED",
                    (
                        True,
                        [],
                    ),
                ),
            ],
        )
    ],
)
def test_operators(
    test_models_path: Path, model: str, expected_ops: List[Tuple]
) -> None:
    """Test operators function."""
    tflite_model = TFLiteModel(str(test_models_path / model))
    device = EthosU55()

    operators = supported_operators(tflite_model, device)

    assert operators.total_number == len(expected_ops)
    assert operators.npu_supported_number == operators.total_number
    assert operators.npu_supported_ratio == 1.0
    assert operators.npu_unsupported_ratio == 0.0

    for i, op in enumerate(operators.ops):
        (
            expected_name,
            expected_type,
            (expected_run_on_npu, expected_reasons),
        ) = expected_ops[i]
        assert op.name == expected_name
        assert op.op_type == expected_type
        assert op.run_on_npu == (expected_run_on_npu, expected_reasons)


def test_estimate_performance(test_models_path: Path) -> None:
    """Test getting performance estimations."""
    model = TFLiteModel(str(test_models_path / "simple_3_layers_model.tflite"))
    perf_metrics = estimate_performance(model, EthosU55())

    assert isinstance(perf_metrics, PerformanceMetrics)


@pytest.mark.skip("Failed with Vela 3.1, further investigation is needed")
def test_estimate_performance_already_optimized(
    test_models_path: Path, tmpdir: Any
) -> None:
    """Test that performance estimation should fail for already optimized model."""
    device = EthosU55()
    model = TFLiteModel(str(test_models_path / "simple_3_layers_model.tflite"))
    optimized_model_path = str(Path(tmpdir) / "optimized_model.tflite")

    optimize_model(model, device, optimized_model_path)

    with pytest.raises(
        Exception, match="Unable to estimate performance for the given optimized model"
    ):
        estimate_performance(TFLiteModel(optimized_model_path), device)


def test_generate_supported_operators_report(tmp_path: Path) -> None:
    """Test generating supported operators report."""
    with working_directory(tmp_path):
        generate_supported_operators_report()

        md_file = tmp_path / "SUPPORTED_OPS.md"
        assert md_file.is_file()
        assert md_file.stat().st_size > 0
