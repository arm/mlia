# SPDX-FileCopyrightText: Copyright 2022, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for module tools/vela_wrapper."""
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from ethosu.vela.compiler_driver import TensorAllocator
from ethosu.vela.scheduler import OptimizationStrategy

from mlia.devices.ethosu.config import EthosUConfiguration
from mlia.tools.vela_wrapper import estimate_performance
from mlia.tools.vela_wrapper import generate_supported_operators_report
from mlia.tools.vela_wrapper import NpuSupported
from mlia.tools.vela_wrapper import Operator
from mlia.tools.vela_wrapper import Operators
from mlia.tools.vela_wrapper import optimize_model
from mlia.tools.vela_wrapper import OptimizedModel
from mlia.tools.vela_wrapper import PerformanceMetrics
from mlia.tools.vela_wrapper import supported_operators
from mlia.tools.vela_wrapper import VelaCompiler
from mlia.tools.vela_wrapper import VelaCompilerOptions
from mlia.utils.proc import working_directory


def test_default_vela_compiler() -> None:
    """Test default Vela compiler instance."""
    default_compiler_options = VelaCompilerOptions(accelerator_config="ethos-u55-256")
    default_compiler = VelaCompiler(default_compiler_options)

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

    assert default_compiler.get_config() == {
        "accelerator_config": "ethos-u55-256",
        "system_config": "internal-default",
        "core_clock": 500000000.0,
        "axi0_port": "Sram",
        "axi1_port": "OffChipFlash",
        "memory_mode": "internal-default",
        "const_mem_area": "Axi1",
        "arena_mem_area": "Axi0",
        "cache_mem_area": "Axi0",
        "arena_cache_size": 4294967296,
        "permanent_storage_mem_area": "OffChipFlash",
        "feature_map_storage_mem_area": "Sram",
        "fast_storage_mem_area": "Sram",
        "memory_area": {
            "Sram": {
                "clock_scales": 1.0,
                "burst_length": 32,
                "read_latency": 32,
                "write_latency": 32,
            },
            "Dram": {
                "clock_scales": 1.0,
                "burst_length": 1,
                "read_latency": 0,
                "write_latency": 0,
            },
            "OnChipFlash": {
                "clock_scales": 1.0,
                "burst_length": 1,
                "read_latency": 0,
                "write_latency": 0,
            },
            "OffChipFlash": {
                "clock_scales": 0.125,
                "burst_length": 128,
                "read_latency": 64,
                "write_latency": 64,
            },
        },
    }


def test_vela_compiler_with_parameters(test_resources_path: Path) -> None:
    """Test creation of Vela compiler instance with non-default params."""
    vela_ini_path = str(test_resources_path / "vela/sample_vela.ini")

    compiler_options = VelaCompilerOptions(
        config_files=vela_ini_path,
        system_config="Ethos_U65_High_End",
        memory_mode="Shared_Sram",
        accelerator_config="ethos-u65-256",
        max_block_dependency=1,
        arena_cache_size=10,
        tensor_allocator="Greedy",
        cpu_tensor_alignment=4,
        optimization_strategy="Size",
        output_dir="output",
    )
    compiler = VelaCompiler(compiler_options)

    assert compiler.config_files == vela_ini_path
    assert compiler.system_config == "Ethos_U65_High_End"
    assert compiler.memory_mode == "Shared_Sram"
    assert compiler.accelerator_config == "ethos-u65-256"
    assert compiler.max_block_dependency == 1
    assert compiler.arena_cache_size == 10
    assert compiler.tensor_allocator == TensorAllocator.Greedy
    assert compiler.cpu_tensor_alignment == 4
    assert compiler.optimization_strategy == OptimizationStrategy.Size
    assert compiler.output_dir == "output"

    assert compiler.get_config() == {
        "accelerator_config": "ethos-u65-256",
        "system_config": "Ethos_U65_High_End",
        "core_clock": 1000000000.0,
        "axi0_port": "Sram",
        "axi1_port": "Dram",
        "memory_mode": "Shared_Sram",
        "const_mem_area": "Axi1",
        "arena_mem_area": "Axi0",
        "cache_mem_area": "Axi0",
        "arena_cache_size": 10,
        "permanent_storage_mem_area": "Dram",
        "feature_map_storage_mem_area": "Sram",
        "fast_storage_mem_area": "Sram",
        "memory_area": {
            "Sram": {
                "clock_scales": 1.0,
                "burst_length": 32,
                "read_latency": 32,
                "write_latency": 32,
            },
            "Dram": {
                "clock_scales": 0.234375,
                "burst_length": 128,
                "read_latency": 500,
                "write_latency": 250,
            },
            "OnChipFlash": {
                "clock_scales": 1.0,
                "burst_length": 1,
                "read_latency": 0,
                "write_latency": 0,
            },
            "OffChipFlash": {
                "clock_scales": 1.0,
                "burst_length": 1,
                "read_latency": 0,
                "write_latency": 0,
            },
        },
    }


def test_compile_model(test_tflite_model: Path) -> None:
    """Test model optimization."""
    compiler = VelaCompiler(EthosUConfiguration("ethos-u55-256").compiler_options)

    optimized_model = compiler.compile_model(test_tflite_model)
    assert isinstance(optimized_model, OptimizedModel)


def test_optimize_model(tmp_path: Path, test_tflite_model: Path) -> None:
    """Test model optimization and saving into file."""
    tmp_file = tmp_path / "temp.tflite"

    device = EthosUConfiguration("ethos-u55-256")
    optimize_model(test_tflite_model, device.compiler_options, tmp_file.absolute())

    assert tmp_file.is_file()
    assert tmp_file.stat().st_size > 0


@pytest.mark.parametrize(
    "model, expected_ops",
    [
        (
            "test_model.tflite",
            Operators(
                ops=[
                    Operator(
                        name="sequential/conv1/Relu;sequential/conv1/BiasAdd;"
                        "sequential/conv2/Conv2D;sequential/conv1/Conv2D",
                        op_type="CONV_2D",
                        run_on_npu=NpuSupported(supported=True, reasons=[]),
                    ),
                    Operator(
                        name="sequential/conv2/Relu;sequential/conv2/BiasAdd;"
                        "sequential/conv2/Conv2D",
                        op_type="CONV_2D",
                        run_on_npu=NpuSupported(supported=True, reasons=[]),
                    ),
                    Operator(
                        name="sequential/max_pooling2d/MaxPool",
                        op_type="MAX_POOL_2D",
                        run_on_npu=NpuSupported(supported=True, reasons=[]),
                    ),
                    Operator(
                        name="sequential/flatten/Reshape",
                        op_type="RESHAPE",
                        run_on_npu=NpuSupported(supported=True, reasons=[]),
                    ),
                    Operator(
                        name="Identity",
                        op_type="FULLY_CONNECTED",
                        run_on_npu=NpuSupported(supported=True, reasons=[]),
                    ),
                ]
            ),
        )
    ],
)
def test_operators(test_models_path: Path, model: str, expected_ops: Operators) -> None:
    """Test operators function."""
    device = EthosUConfiguration("ethos-u55-256")

    operators = supported_operators(test_models_path / model, device.compiler_options)
    for expected, actual in zip(expected_ops.ops, operators.ops):
        # do not compare names as they could be different on each model generation
        assert expected.op_type == actual.op_type
        assert expected.run_on_npu == actual.run_on_npu


def test_estimate_performance(test_tflite_model: Path) -> None:
    """Test getting performance estimations."""
    device = EthosUConfiguration("ethos-u55-256")
    perf_metrics = estimate_performance(test_tflite_model, device.compiler_options)

    assert isinstance(perf_metrics, PerformanceMetrics)


def test_estimate_performance_already_optimized(
    tmp_path: Path, test_tflite_model: Path
) -> None:
    """Test that performance estimation should fail for already optimized model."""
    device = EthosUConfiguration("ethos-u55-256")

    optimized_model_path = tmp_path / "optimized_model.tflite"

    optimize_model(test_tflite_model, device.compiler_options, optimized_model_path)

    with pytest.raises(
        Exception, match="Unable to estimate performance for the given optimized model"
    ):
        estimate_performance(optimized_model_path, device.compiler_options)


def test_generate_supported_operators_report(tmp_path: Path) -> None:
    """Test generating supported operators report."""
    with working_directory(tmp_path):
        generate_supported_operators_report()

        md_file = tmp_path / "SUPPORTED_OPS.md"
        assert md_file.is_file()
        assert md_file.stat().st_size > 0


def test_read_invalid_model(test_tflite_invalid_model: Path) -> None:
    """Test that reading invalid model should fail with exception."""
    with pytest.raises(
        Exception, match=f"Unable to read model {test_tflite_invalid_model}"
    ):
        device = EthosUConfiguration("ethos-u55-256")
        estimate_performance(test_tflite_invalid_model, device.compiler_options)


def test_compile_invalid_model(
    test_tflite_model: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test that if model could not be compiled then correct exception raised."""
    mock_compiler = MagicMock()
    mock_compiler.side_effect = Exception("Bad model!")

    monkeypatch.setattr("mlia.tools.vela_wrapper.compiler_driver", mock_compiler)

    model_path = tmp_path / "optimized_model.tflite"
    with pytest.raises(
        Exception, match="Model could not be optimized with Vela compiler"
    ):
        device = EthosUConfiguration("ethos-u55-256")
        optimize_model(test_tflite_model, device.compiler_options, model_path)

    assert not model_path.exists()
