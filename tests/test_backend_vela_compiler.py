# SPDX-FileCopyrightText: Copyright 2022-2023, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for module vela/compiler."""
from pathlib import Path
from typing import Any

import pytest
from ethosu.vela.compiler_driver import TensorAllocator
from ethosu.vela.scheduler import OptimizationStrategy

from mlia.backend.vela.compiler import optimize_model
from mlia.backend.vela.compiler import OptimizedModel
from mlia.backend.vela.compiler import VelaCompiler
from mlia.backend.vela.compiler import VelaCompilerOptions
from mlia.target.ethos_u.config import EthosUConfiguration


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
    assert default_compiler.output_dir == "output"

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
        output_dir="custom_output",
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
    assert compiler.output_dir == "custom_output"

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
    compiler = VelaCompiler(
        EthosUConfiguration.load_profile("ethos-u55-256").compiler_options
    )

    optimized_model = compiler.compile_model(test_tflite_model)
    assert isinstance(optimized_model, OptimizedModel)


def test_compile_model_fail_sram_exceeded(
    test_tflite_model: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test model optimization."""
    compiler = VelaCompiler(
        EthosUConfiguration.load_profile("ethos-u55-256").compiler_options
    )

    def fake_compiler(*_: Any) -> None:
        print("Warning: SRAM target for arena memory area exceeded.")

    monkeypatch.setattr("mlia.backend.vela.compiler.compiler_driver", fake_compiler)
    with pytest.raises(Exception) as exc_info:
        compiler.compile_model(test_tflite_model)

    assert str(exc_info.value) == "Model is too large and uses too much RAM"


def test_optimize_model(tmp_path: Path, test_tflite_model: Path) -> None:
    """Test model optimization and saving into file."""
    tmp_file = tmp_path / "temp.tflite"

    target_config = EthosUConfiguration.load_profile("ethos-u55-256")
    optimize_model(
        test_tflite_model, target_config.compiler_options, tmp_file.absolute()
    )

    assert tmp_file.is_file()
    assert tmp_file.stat().st_size > 0
