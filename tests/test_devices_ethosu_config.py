# Copyright 2021, Arm Ltd.
"""Tests for config module."""
import pytest
from mlia.devices.ethosu.config import EthosU55
from mlia.devices.ethosu.config import EthosU65
from mlia.tools.vela_wrapper import VelaCompilerOptions


def test_compiler_options_default_init() -> None:
    """Test compiler options default init."""
    opts = VelaCompilerOptions()

    assert opts.config_files is None
    assert opts.system_config == "internal-default"
    assert opts.memory_mode == "internal-default"
    assert opts.accelerator_config == "ethos-u55-256"
    assert opts.max_block_dependency == 3
    assert opts.arena_cache_size is None
    assert opts.tensor_allocator == "HillClimb"
    assert opts.cpu_tensor_alignment == 16
    assert opts.optimization_strategy == "Performance"
    assert opts.output_dir is None


def test_ethosu55_init_configuration() -> None:
    """Test ethos u55 configuration init."""
    default_config = EthosU55()

    assert default_config.ip_class == "ethos-u55"
    assert default_config.mac == 256
    assert default_config.compiler_options is not None

    with pytest.raises(Exception, match="Wrong or empty MAC value"):
        EthosU55(mac=1024)  # type: ignore

    config_mac_32 = EthosU55(mac=32)
    assert config_mac_32.mac == 32
    assert config_mac_32.compiler_options.accelerator_config == "ethos-u55-32"


def test_ethosu65_init_configuration() -> None:
    """Test ethos u56 configuration init."""
    default_config = EthosU65()

    assert default_config.ip_class == "ethos-u65"
    assert default_config.mac == 256
    assert default_config.compiler_options is not None

    with pytest.raises(Exception, match="Wrong or empty MAC value"):
        EthosU65(mac=None)  # type: ignore

    config_mac_512 = EthosU65(mac=512)
    assert config_mac_512.mac == 512
    assert config_mac_512.compiler_options.accelerator_config == "ethos-u65-512"
