# Copyright 2021, Arm Ltd.
"""Tests for config module."""
from mlia.devices.ethosu.config import EthosUConfiguration
from mlia.tools.vela_wrapper import VelaCompilerOptions


def test_compiler_options_default_init() -> None:
    """Test compiler options default init."""
    opts = VelaCompilerOptions()

    assert opts.config_files is None
    assert opts.system_config == "internal-default"
    assert opts.memory_mode == "internal-default"
    assert opts.accelerator_config is None
    assert opts.max_block_dependency == 3
    assert opts.arena_cache_size is None
    assert opts.tensor_allocator == "HillClimb"
    assert opts.cpu_tensor_alignment == 16
    assert opts.optimization_strategy == "Performance"
    assert opts.output_dir is None


def test_ethosu_target() -> None:
    """Test ethosu target configuration init."""
    default_config = EthosUConfiguration(target="U55-256")

    assert default_config.ip_class == "ethos-u55"
    assert default_config.mac == 256
    assert default_config.compiler_options is not None
