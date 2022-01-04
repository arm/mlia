# Copyright 2021, Arm Ltd.
"""Tests for config module."""
from pathlib import Path

import pytest
from mlia.cli.common import ExecutionContext
from mlia.config import CompilerOptions
from mlia.config import EthosU55
from mlia.config import EthosU65
from mlia.config import get_model
from mlia.config import KerasModel
from mlia.config import TFLiteModel
from mlia.config import TfModel


def test_compiler_options_default_init() -> None:
    """Test compiler options default init."""
    opts = CompilerOptions()

    assert opts.config_files is None
    assert opts.system_config == "internal-default"
    assert opts.memory_mode == "internal-default"
    assert opts.accelerator_config == "ethos-u55-256"
    assert opts.max_block_dependency == 3
    assert opts.arena_cache_size is None
    assert opts.tensor_allocator == "HillClimb"
    assert opts.cpu_tensor_alignment == 16
    assert opts.recursion_limit == 1000
    assert opts.optimization_strategy == "Performance"
    assert opts.output_dir is None

    assert str(opts) == (
        "Compiler options "
        "config_files: None, system_config: internal-default, "
        "memory_mode: internal-default, "
        "accelerator_config: ethos-u55-256, max_block_dependency: 3, "
        "arena_cache_size: None, "
        "tensor_allocator: HillClimb, cpu_tensor_alignment: 16, "
        "recursion_limit: 1000, optimization_strategy: Performance, "
        "output_dir: None"
    )


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
    assert str(config_mac_512) == (
        "EthosU ip_class=ethos-u65 mac=512 "
        "compiler_options= Compiler options config_files: None, "
        "system_config: internal-default, memory_mode: internal-default, "
        "accelerator_config: ethos-u65-512, max_block_dependency: 3, "
        "arena_cache_size: None, "
        "tensor_allocator: HillClimb, cpu_tensor_alignment: 16, "
        "recursion_limit: 1000, optimization_strategy: Performance, "
        "output_dir: None"
    )


def test_convert_keras_to_tflite(test_models_path: Path, tmp_path: Path) -> None:
    """Test Keras to TFLite conversion."""
    model = test_models_path / "simple_model.h5"
    keras_model = KerasModel(str(model))

    tflite_model_path = tmp_path / "test.tflite"
    keras_model.convert_to_tflite(tflite_model_path)

    assert tflite_model_path.is_file()
    assert tflite_model_path.stat().st_size > 0


def test_convert_tf_to_tflite(test_models_path: Path, tmp_path: Path) -> None:
    """Test TF saved model to TFLite conversion."""
    model = test_models_path / "tf_model_simple_3_layers_model"
    tf_model = TfModel(model)

    tflite_model_path = tmp_path / "test.tflite"
    tf_model.convert_to_tflite(tflite_model_path)

    assert tflite_model_path.is_file()
    assert tflite_model_path.stat().st_size > 0


@pytest.mark.parametrize(
    "model_path, expected_type",
    [
        ("test.tflite", TFLiteModel),
        ("test.h5", KerasModel),
        ("test.hdf5", KerasModel),
    ],
)
def test_get_model_file(
    model_path: str, expected_type: type, dummy_context: ExecutionContext
) -> None:
    """Test TFLite model type."""
    model = get_model(model_path, ctx=dummy_context)
    assert isinstance(model, expected_type)


@pytest.mark.parametrize(
    "model_path, expected_type", [("tf_model_simple_3_layers_model", TfModel)]
)
def test_get_model_dir(
    test_models_path: Path,
    model_path: str,
    expected_type: type,
    dummy_context: ExecutionContext,
) -> None:
    """Test TFLite model type."""
    model = get_model(str(test_models_path / model_path), ctx=dummy_context)
    assert isinstance(model, expected_type)
