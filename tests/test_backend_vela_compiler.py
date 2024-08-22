# SPDX-FileCopyrightText: Copyright 2022-2024, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for module vela/compiler."""
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from ethosu.vela.vela import main

from mlia.backend.vela.compiler import compile_model
from mlia.backend.vela.compiler import parse_summary_csv_file
from mlia.backend.vela.compiler import parse_vela_initialisation_file
from mlia.backend.vela.compiler import resolve_compiler_config
from mlia.backend.vela.compiler import VelaCompiler
from mlia.backend.vela.compiler import VelaCompilerOptions
from mlia.backend.vela.compiler import VelaInitData
from mlia.backend.vela.compiler import VelaInitMemoryData
from mlia.backend.vela.compiler import VelaSummary
from mlia.target.ethos_u.config import EthosUConfiguration
from mlia.utils.filesystem import recreate_directory


def test_default_vela_compiler() -> None:
    """Test default Vela compiler instance."""
    default_compiler_options = VelaCompilerOptions(accelerator_config="ethos-u55-256")
    default_compiler = VelaCompiler(default_compiler_options)

    assert default_compiler.config_file is None
    assert default_compiler.system_config == "internal-default"
    assert default_compiler.memory_mode == "internal-default"
    assert default_compiler.accelerator_config == "ethos-u55-256"
    assert default_compiler.max_block_dependency == 3
    assert default_compiler.arena_cache_size is None
    assert default_compiler.tensor_allocator == "HillClimb"
    assert default_compiler.cpu_tensor_alignment == 16
    assert default_compiler.optimization_strategy == "Performance"
    assert default_compiler.output_dir == Path("output")

    with pytest.raises(
        ValueError, match="System Config: internal-default not present in vela.ini file"
    ):
        resolve_compiler_config(vela_compiler_options=default_compiler_options)


def test_vela_compiler_with_parameters(test_resources_path: Path) -> None:
    """Test creation of Vela compiler instance with non-default params."""
    vela_ini_path = str(test_resources_path / "vela/sample_vela.ini")

    compiler_options = VelaCompilerOptions(
        config_file=vela_ini_path,
        system_config="Ethos_U65_High_End",
        memory_mode="Shared_Sram",
        accelerator_config="ethos-u65-256",
        max_block_dependency=1,
        arena_cache_size=10,
        tensor_allocator="Greedy",
        cpu_tensor_alignment=4,
        optimization_strategy="Size",
        output_dir=Path("custom_output"),
    )
    compiler = VelaCompiler(compiler_options)

    assert compiler.config_file == vela_ini_path
    assert compiler.system_config == "Ethos_U65_High_End"
    assert compiler.memory_mode == "Shared_Sram"
    assert compiler.accelerator_config == "ethos-u65-256"
    assert compiler.max_block_dependency == 1
    assert compiler.arena_cache_size == 10
    assert compiler.tensor_allocator == "Greedy"
    assert compiler.cpu_tensor_alignment == 4
    assert compiler.optimization_strategy == "Size"
    assert compiler.output_dir == Path("custom_output")

    assert resolve_compiler_config(
        vela_compiler_options=compiler_options
    ) == VelaInitData(
        system_config="Ethos_U65_High_End",
        core_clock=1000000000.0,
        axi0_port="Sram",
        axi1_port="Dram",
        memory_mode="Shared_Sram",
        const_mem_area="Axi1",
        arena_mem_area="Axi0",
        cache_mem_area="Axi0",
        arena_cache_size=None,
        sram_memory_data=VelaInitMemoryData(
            clock_scale=1.0,
            burst_length=32,
            read_latency=32,
            write_latency=32,
        ),
        dram_memory_data=VelaInitMemoryData(
            clock_scale=0.234375,
            burst_length=128,
            read_latency=500,
            write_latency=250,
        ),
        on_chip_flash_memory_data=VelaInitMemoryData(
            clock_scale=None,
            burst_length=None,
            read_latency=None,
            write_latency=None,
        ),
        off_chip_flash_memory_data=VelaInitMemoryData(
            clock_scale=None,
            burst_length=None,
            read_latency=None,
            write_latency=None,
        ),
    )


def test_vela_compiler_with_parameters_inherit_memory_mode(
    test_resources_path: Path,
) -> None:
    """Test creation of Vela compiler instance with non-default params
    that inherits a memory mode.
    """
    vela_ini_path = str(test_resources_path / "vela/sample_vela.ini")

    compiler_options = VelaCompilerOptions(
        config_file=vela_ini_path,
        system_config="Ethos_U65_High_End",
        memory_mode="Dedicated_Sram_512KB_custom",
        accelerator_config="ethos-u65-256",
        max_block_dependency=1,
        arena_cache_size=10,
        tensor_allocator="Greedy",
        cpu_tensor_alignment=4,
        optimization_strategy="Size",
        output_dir=Path("custom_output"),
    )
    compiler = VelaCompiler(compiler_options)

    assert compiler.config_file == vela_ini_path
    assert compiler.system_config == "Ethos_U65_High_End"
    assert compiler.memory_mode == "Dedicated_Sram_512KB_custom"
    assert compiler.accelerator_config == "ethos-u65-256"
    assert compiler.max_block_dependency == 1
    assert compiler.arena_cache_size == 10
    assert compiler.tensor_allocator == "Greedy"
    assert compiler.cpu_tensor_alignment == 4
    assert compiler.optimization_strategy == "Size"
    assert compiler.output_dir == Path("custom_output")

    assert resolve_compiler_config(
        vela_compiler_options=compiler_options
    ) == VelaInitData(
        system_config="Ethos_U65_High_End",
        core_clock=1000000000.0,
        axi0_port="Sram",
        axi1_port="Dram",
        memory_mode="Dedicated_Sram_512KB_custom",
        const_mem_area="Axi1",
        arena_mem_area="Axi1",
        cache_mem_area="Axi0",
        arena_cache_size=524288,
        sram_memory_data=VelaInitMemoryData(
            clock_scale=1.0,
            burst_length=32,
            read_latency=32,
            write_latency=32,
        ),
        dram_memory_data=VelaInitMemoryData(
            clock_scale=0.234375,
            burst_length=128,
            read_latency=500,
            write_latency=250,
        ),
        on_chip_flash_memory_data=VelaInitMemoryData(
            clock_scale=None,
            burst_length=None,
            read_latency=None,
            write_latency=None,
        ),
        off_chip_flash_memory_data=VelaInitMemoryData(
            clock_scale=None,
            burst_length=None,
            read_latency=None,
            write_latency=None,
        ),
    )


def test_compile_model(test_tflite_model: Path) -> None:
    """Test model optimization."""
    compiler = VelaCompiler(
        EthosUConfiguration.load_profile("ethos-u55-256").compiler_options
    )

    expected_model_path = Path(
        compiler.output_dir.as_posix()
        + "/"
        + test_tflite_model.stem
        + "_vela"
        + test_tflite_model.suffix
    )
    vela_summary_data, optimized_model_path = compiler.compile_model(test_tflite_model)
    assert isinstance(vela_summary_data, VelaSummary)
    assert isinstance(optimized_model_path, Path)
    assert expected_model_path == optimized_model_path


def test_csv_file_created(test_tflite_model: Path) -> None:
    """Test that a csv file is created by the vela backend"""
    compiler = VelaCompiler(
        EthosUConfiguration.load_profile("ethos-u55-256").compiler_options
    )
    csv_file_name = test_tflite_model.stem + "_per-layer.csv"
    compiler.compile_model(test_tflite_model)
    assert (compiler.output_dir / csv_file_name).is_file()


# Test to see if the new flag is passed to Vela
def test_verbose_flag_passed() -> None:
    """Test that the verbose_performance flag is passed to vela backend"""
    compiler = VelaCompiler(
        EthosUConfiguration.load_profile("ethos-u55-256").compiler_options
    )
    assert compiler.verbose_performance


def test_compile_model_fail_sram_exceeded(
    test_tflite_model: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test model optimization."""
    compiler = VelaCompiler(
        EthosUConfiguration.load_profile("ethos-u55-256").compiler_options
    )

    def fake_compiler(*_: Any) -> None:
        print("Warning: SRAM target for arena memory area exceeded.")

    monkeypatch.setattr("mlia.backend.vela.compiler.main", fake_compiler)
    with pytest.raises(Exception) as exc_info:
        compiler.compile_model(test_tflite_model)

    assert str(exc_info.value) == "Model is too large and uses too much RAM"


def test_optimize_model(tmp_path: Path, test_tflite_model: Path) -> None:
    """Test model optimization and saving into file."""
    tmp_file = tmp_path / "test_model_int8_vela.tflite"
    target_config = EthosUConfiguration.load_profile("ethos-u55-256")
    target_config.compiler_options.output_dir = tmp_path
    compile_model(test_tflite_model, target_config.compiler_options)

    assert tmp_file.is_file()
    assert tmp_file.stat().st_size > 0


SUMMARY_TMP_DATA = """
experiment,network,accelerator_configuration,system_config,memory_mode,core_clock,arena_cache_size,sram_bandwidth,dram_bandwidth,on_chip_flash_bandwidth,off_chip_flash_bandwidth,weights_storage_area,feature_map_storage_area,inferences_per_second,batch_size,inference_time,passes_before_fusing,passes_after_fusing,sram_memory_used,dram_memory_used,on_chip_flash_memory_used,off_chip_flash_memory_used,total_original_weights,total_npu_encoded_weights,sram_feature_map_read_bytes,sram_feature_map_write_bytes,sram_weight_read_bytes,sram_weight_write_bytes,sram_total_bytes,dram_feature_map_read_bytes,dram_feature_map_write_bytes,dram_weight_read_bytes,dram_weight_write_bytes,dram_total_bytes,on_chip_flash_feature_map_read_bytes,on_chip_flash_feature_map_write_bytes,on_chip_flash_weight_read_bytes,on_chip_flash_weight_write_bytes,on_chip_flash_total_bytes,off_chip_flash_feature_map_read_bytes,off_chip_flash_feature_map_write_bytes,off_chip_flash_weight_read_bytes,off_chip_flash_weight_write_bytes,off_chip_flash_total_bytes,nn_macs,nn_tops,cycles_npu,cycles_sram_access,cycles_dram_access,cycles_on_chip_flash_access,cycles_off_chip_flash_access,cycles_total
default,test_model_fp32,Ethos_U55_256,Ethos_U55_High_End_Embedded,Shared_Sram,0.0,0.9,4.0,4.0,4.0,0.5,Off-chip Flash,SRAM,0.0,1,12.1e-05,7,2.0,1.5,0.0,0.0,1.4,7,8,6.0,5.0,7552.0,5.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,4.0,0.0,1.0,2,0.1,23297.0,1.5,0.0,0.0,1.0,2
""".strip()

SUMMARY_TMP_DATA_MISSING_HEADER = """
experiment,network,accelerator_configuration,system_config,memory_mode,core_clock,arena_cache_size,sram_bandwidth,dram_bandwidth,on_chip_flash_bandwidth,off_chip_flash_bandwidth,weights_storage_area,feature_map_storage_area,inferences_per_second,batch_size,inference_time,passes_before_fusing,passes_after_fusing,sram_memory_used,dram_memory_used,on_chip_flash_memory_used,off_chip_flash_memory_used,total_original_weights,total_npu_encoded_weights,sram_feature_map_read_bytes,sram_feature_map_write_bytes,sram_weight_read_bytes,sram_weight_write_bytes,sram_total_bytes,dram_feature_map_read_bytes,dram_feature_map_write_bytes,dram_weight_read_bytes,dram_weight_write_bytes,dram_total_bytes,on_chip_flash_feature_map_read_bytes,on_chip_flash_feature_map_write_bytes,on_chip_flash_weight_read_bytes,on_chip_flash_weight_write_bytes,on_chip_flash_total_bytes,off_chip_flash_feature_map_read_bytes,off_chip_flash_feature_map_write_bytes,off_chip_flash_weight_read_bytes,off_chip_flash_weight_write_bytes,off_chip_flash_total_bytes,nn_macs,nn_tops,cycles_npu,cycles_sram_access,cycles_dram_access,cycles_on_chip_flash_access,cycles_off_chip_flash_access
default,test_model_fp32,Ethos_U55_256,Ethos_U55_High_End_Embedded,Shared_Sram,0.0,0.9,4.0,4.0,4.0,0.5,Off-chip Flash,SRAM,0.0,1,12.1e-05,7,2.0,1.5,0.0,0.0,1.4,7,8,6.0,5.0,7552.0,5.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,4.0,0.0,1.0,2,0.1,23297.0,1.5,0.0,0.0,1.0
""".strip()

TMP_DATA_EXPECTED_STRING = "\
cycles_total: 2.0, \
cycles_npu: 23297.0, \
cycles_sram_access: 1.5, \
cycles_dram_access: 0.0, \
cycles_on_chip_flash_access: 0.0, \
cycles_off_chip_flash_access: 1.0, \
core_clock: 0.0, \
dram_memory_used: 0.0, \
sram_memory_used: 1.5, \
on_chip_flash_memory_used: 0.0, \
off_chip_flash_memory_used: 1.4, \
batch_size: 1, \
memory_mode: Shared_Sram, \
system_config: Ethos_U55_High_End_Embedded, \
accelerator_configuration: Ethos_U55_256, \
arena_cache_size: 0.9, \
"


def test_backend_compiler_parse_summary_csv_file(test_csv_file: Path) -> None:
    """Test that parsing a csv file produces a LayerwisePerfInfo object."""
    with open(test_csv_file, "w", encoding="utf8") as csv_file:
        csv_file.write(SUMMARY_TMP_DATA)
    summary_object = parse_summary_csv_file(test_csv_file)
    strings_to_check = repr(summary_object)
    assert isinstance(summary_object, VelaSummary)
    assert TMP_DATA_EXPECTED_STRING == strings_to_check


def test_backend_compiler_summary_csv_parsed_empty(empty_test_csv_file: Path) -> None:
    """Test that ensures when we have an empty
    CSV file we get None as backend data.
    """
    empty_test_csv_file.touch()
    with pytest.raises(RuntimeError, match="Generated Vela Summary CSV is empty"):
        parse_summary_csv_file(empty_test_csv_file)


def test_backend_compiler_summary_csv_parsed_missing_headers(
    test_csv_file: Path,
) -> None:
    """Test that ensures a KeyError
    is raised when a csv with missing
    expected headers is parsed.
    """
    with open(test_csv_file, "w", encoding="utf8") as csv_file:
        csv_file.write(SUMMARY_TMP_DATA_MISSING_HEADER)
    with pytest.raises(
        KeyError,
        match="Generated Vela Summary CSV missing expected header: cycles_total.",  # pylint: disable=line-too-long
    ):
        parse_summary_csv_file(test_csv_file)


def test_backend_compiler_summary_csv_parsed_missing_file() -> None:
    """Test that ensures a FileNotFoundError
    is raised when a non-existent csv file is parsed.
    """
    with pytest.raises(
        FileNotFoundError, match="CSV File not found at missing_file.csv"
    ):
        parse_summary_csv_file(Path("missing_file.csv"))


def test_backend_compiler_parsing_vela_ini_file_missing_init_file() -> None:
    """Test that ensures a FileNotFoundError
    is raised when a non-existent ini file is parsed.
    """
    with pytest.raises(
        FileNotFoundError,
        match="Vela Initialisation File not found at missing_init_file.ini",
    ):
        parse_vela_initialisation_file(
            Path("missing_init_file.ini"), "internal-default", "internal-default"
        )


def test_backend_compiler_parsing_vela_ini_file_empty_init_file(
    empty_vela_ini_file: Path,
) -> None:
    """Test that ensures a OSError
    is raised when an empty vela.ini file is parsed.
    """
    empty_vela_ini_file.touch()
    with pytest.raises(OSError, match="vela.ini File Is Empty"):
        parse_vela_initialisation_file(
            empty_vela_ini_file, "internal-default", "internal-default"
        )


@pytest.mark.parametrize(
    "input_str",
    [
        """
; SPDX-FileCopyrightText: Copyright 2022, Arm Limited and/or its affiliates.
; SPDX-License-Identifier: Apache-2.0
; Ethos-U55 High-End Embedded: SRAM (4 GB/s) and Flash (0.5 GB/s)
[System_Config.Ethos_U55_High_End_Embedded]
core_clock=500e6
axi0_port=Sram
axi1_port=OffChipFlash
Sram_clock_scale=1.0
Sram_burst_length=32
Sram_read_latency=32
Sram_write_latency=32
OffChipFlash_clock_scale=0.125
OffChipFlash_burst_length=128
OffChipFlash_read_latency=64
OffChipFlash_write_latency=64

; Ethos-U65 High-End: SRAM (16 GB/s) and DRAM (3.75 GB/s)
[System_Config.Ethos_U65_High_End]
core_clock=1e9
axi0_port=Sram
axi1_port=Dram
Sram_clock_scale=1.0
Sram_burst_length=32
Sram_read_latency=32
Sram_write_latency=32
Dram_clock_scale=0.234375
Dram_burst_length=128
Dram_read_latency=500
Dram_write_latency=250
"""
    ],
)
def test_backend_compiler_parsing_vela_ini_file_missing_memory_modes(
    vela_ini_file: Path,
    input_str: str,
) -> None:
    """Test that ensures a IndexError
    is raised when a vela.ini file with no memory modes
    is parsed.
    """
    with open(vela_ini_file, "w", encoding="utf8") as vela_file:
        vela_file.write(input_str)
    with pytest.raises(
        IndexError, match="No memory modes are present in vela.ini file."
    ):
        parse_vela_initialisation_file(
            vela_ini_file, "Ethos_U65_High_End", "Shared_Sram"
        )


@pytest.mark.parametrize(
    "input_str",
    [
        """
; -----------------------------------------------------------------------------
; Memory Mode

; Shared SRAM: the SRAM is shared between the Ethos-U and the Cortex-M software
; The non-SRAM memory is assumed to be read-only
[Memory_Mode.Shared_Sram]
const_mem_area=Axi1
arena_mem_area=Axi0
cache_mem_area=Axi0

; The SRAM (384KB) is only for use by the Ethos-U
; The non-SRAM memory is assumed to be read-writeable
[Memory_Mode.Dedicated_Sram]
const_mem_area=Axi1
arena_mem_area=Axi1
cache_mem_area=Axi0
arena_cache_size=393216

"""
    ],
)
def test_backend_compiler_parsing_vela_ini_file_missing_system_configs(
    vela_ini_file: Path,
    input_str: str,
) -> None:
    """Test that ensures a IndexError
    is raised when a vela.ini file with no system configs
    is parsed.
    """
    with open(vela_ini_file, "w", encoding="utf8") as vela_file:
        vela_file.write(input_str)
    with pytest.raises(
        IndexError, match="No system configs are present in vela.ini file."
    ):
        parse_vela_initialisation_file(
            vela_ini_file, "Ethos_U65_High_End", "Shared_Sram"
        )


@pytest.mark.parametrize(
    "input_str",
    [
        """
; SPDX-FileCopyrightText: Copyright 2022, Arm Limited and/or its affiliates.
; SPDX-License-Identifier: Apache-2.0
; Ethos-U55 High-End Embedded: SRAM (4 GB/s) and Flash (0.5 GB/s)
[System_Config.Ethos_U55_High_End_Embedded]
core_clock=500e6
axi0_port=Sram
axi1_port=OffChipFlash
Sram_clock_scale=1.0
Sram_burst_length=32
Sram_read_latency=32
Sram_write_latency=32
OffChipFlash_clock_scale=0.125
OffChipFlash_burst_length=128
OffChipFlash_read_latency=64
OffChipFlash_write_latency=64

; Ethos-U65 High-End: SRAM (16 GB/s) and DRAM (3.75 GB/s)
[System_Config.Ethos_U65_High_End]
core_clock=1e9
axi0_port=Sram
axi1_port=Dram
Sram_clock_scale=1.0
Sram_burst_length=32
Sram_read_latency=32
Sram_write_latency=32
Dram_clock_scale=0.234375
Dram_burst_length=128
Dram_read_latency=500
Dram_write_latency=250

; -----------------------------------------------------------------------------
; Memory Mode

; Shared SRAM: the SRAM is shared between the Ethos-U and the Cortex-M software
; The non-SRAM memory is assumed to be read-only
[Memory_Mode.Shared_Sram]
const_mem_area=Axi1
arena_mem_area=Axi0
cache_mem_area=Axi0

"""
    ],
)
def test_backend_compiler_parsing_vela_ini_file_missing_specific_memory_mode(
    vela_ini_file: Path,
    input_str: str,
) -> None:
    """Test that ensures a ValueError
    is raised when a vela.ini file with specific missing memory mode
    is parsed.
    """
    with open(vela_ini_file, "w", encoding="utf8") as vela_file:
        vela_file.write(input_str)
    with pytest.raises(
        ValueError, match="Memory Mode: Dedicated_Sram not present in vela.ini file."
    ):
        parse_vela_initialisation_file(
            vela_ini_file, "Ethos_U65_High_End", "Dedicated_Sram"
        )


@pytest.mark.parametrize(
    "input_str",
    [
        """
; SPDX-FileCopyrightText: Copyright 2022, Arm Limited and/or its affiliates.
; SPDX-License-Identifier: Apache-2.0
; Ethos-U55 High-End Embedded: SRAM (4 GB/s) and Flash (0.5 GB/s)
[System_Config.Ethos_U55_High_End_Embedded]
core_clock=500e6
axi0_port=Sram
axi1_port=OffChipFlash
Sram_clock_scale=1.0
Sram_burst_length=32
Sram_read_latency=32
Sram_write_latency=32
OffChipFlash_clock_scale=0.125
OffChipFlash_burst_length=128
OffChipFlash_read_latency=64
OffChipFlash_write_latency=64

; -----------------------------------------------------------------------------
; Memory Mode

; Shared SRAM: the SRAM is shared between the Ethos-U and the Cortex-M software
; The non-SRAM memory is assumed to be read-only
[Memory_Mode.Shared_Sram]
const_mem_area=Axi1
arena_mem_area=Axi0
cache_mem_area=Axi0

; The SRAM (384KB) is only for use by the Ethos-U
; The non-SRAM memory is assumed to be read-writeable
[Memory_Mode.Dedicated_Sram]
const_mem_area=Axi1
arena_mem_area=Axi1
cache_mem_area=Axi0
arena_cache_size=393216

"""
    ],
)
def test_backend_compiler_parsing_vela_ini_file_missing_specific_system_config(
    vela_ini_file: Path,
    input_str: str,
) -> None:
    """Test that ensures a ValueError
    is raised when a vela.ini file with specific missing system config
    is parsed.
    """
    with open(vela_ini_file, "w", encoding="utf8") as vela_file:
        vela_file.write(input_str)
    with pytest.raises(
        ValueError,
        match="System Config: Ethos_U65_High_End not present in vela.ini file.",
    ):
        parse_vela_initialisation_file(
            vela_ini_file, "Ethos_U65_High_End", "Shared_Sram"
        )


@pytest.mark.parametrize(
    "input_str",
    [
        """
; SPDX-FileCopyrightText: Copyright 2022, Arm Limited and/or its affiliates.
; SPDX-License-Identifier: Apache-2.0
; Ethos-U55 High-End Embedded: SRAM (4 GB/s) and Flash (0.5 GB/s)
[System_Config.Ethos_U55_High_End_Embedded]
axi0_port=Sram
axi1_port=OffChipFlash
Sram_clock_scale=1.0
Sram_burst_length=32
Sram_read_latency=32
Sram_write_latency=32
OffChipFlash_clock_scale=0.125
OffChipFlash_burst_length=128
OffChipFlash_read_latency=64
OffChipFlash_write_latency=64

; -----------------------------------------------------------------------------
; Memory Mode

; Shared SRAM: the SRAM is shared between the Ethos-U and the Cortex-M software
; The non-SRAM memory is assumed to be read-only
[Memory_Mode.Shared_Sram]
const_mem_area=Axi1
arena_mem_area=Axi0
cache_mem_area=Axi0

; The SRAM (384KB) is only for use by the Ethos-U
; The non-SRAM memory is assumed to be read-writeable
[Memory_Mode.Dedicated_Sram]
const_mem_area=Axi1
arena_mem_area=Axi1
cache_mem_area=Axi0
arena_cache_size=393216

"""
    ],
)
def test_backend_compiler_parsing_vela_ini_file_missing_header(
    vela_ini_file: Path,
    input_str: str,
) -> None:
    """Test that ensures a KeyError
    is raised when a vela.ini file with a missing header
    is parsed.
    """
    with open(vela_ini_file, "w", encoding="utf8") as vela_file:
        vela_file.write(input_str)
    with pytest.raises(
        KeyError, match="Vela.ini file missing expected header: core_clock"
    ):
        parse_vela_initialisation_file(
            vela_ini_file, "Ethos_U55_High_End_Embedded", "Shared_Sram"
        )


def test_backend_compiler_model_already_compiled(
    test_tflite_model: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test that if we try compile a model twice,
    the correct flag is passed and that main is called only once.
    """
    target_config = EthosUConfiguration.load_profile("ethos-u55-256")
    recreate_directory(Path(target_config.compiler_options.output_dir))

    main_mock = MagicMock(side_effect=main)
    monkeypatch.setattr("mlia.backend.vela.compiler.main", main_mock)
    compile_model(test_tflite_model, target_config.compiler_options)

    def vela_compiler_compile_model_mock(
        model_path: Path, *_: Any
    ) -> tuple[None, Path]:
        return None, Path(
            Path(target_config.compiler_options.output_dir).as_posix()
            + "/"
            + model_path.stem
            + "_vela"
            + model_path.suffix
        )

    compiler_mock = MagicMock(side_effect=vela_compiler_compile_model_mock)
    monkeypatch.setattr(
        "mlia.backend.vela.compiler.VelaCompiler.compile_model", compiler_mock
    )
    compile_model(test_tflite_model, target_config.compiler_options)
    main_mock.assert_called_once()
    compiler_mock.assert_called_once_with(test_tflite_model, True)
