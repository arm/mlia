# SPDX-FileCopyrightText: Copyright 2022-2024, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Vela compiler wrapper module."""
from __future__ import annotations

import csv
import logging
import re
import sys
from dataclasses import dataclass
from dataclasses import fields
from io import StringIO
from pathlib import Path
from typing import Literal

from ethosu.vela.model_reader import ModelReaderOptions
from ethosu.vela.model_reader import read_model
from ethosu.vela.nn_graph import Graph
from ethosu.vela.nn_graph import NetworkType
from ethosu.vela.operation import CustomType
from ethosu.vela.vela import main

from mlia.utils.filesystem import get_vela_config
from mlia.utils.logging import redirect_output
from mlia.utils.logging import redirect_raw_output

logger = logging.getLogger(__name__)


@dataclass
class VelaInitMemoryData:
    """Memory Data from vela.ini."""

    clock_scale: float | None
    burst_length: int | None
    read_latency: int | None
    write_latency: int | None


@dataclass
class VelaInitData:  # pylint: disable=too-many-instance-attributes
    """Data gathered from the vela.ini file we provide to vela."""

    system_config: str
    core_clock: float
    axi0_port: str
    axi1_port: str
    sram_memory_data: VelaInitMemoryData
    dram_memory_data: VelaInitMemoryData
    off_chip_flash_memory_data: VelaInitMemoryData
    on_chip_flash_memory_data: VelaInitMemoryData
    memory_mode: str
    const_mem_area: str
    arena_mem_area: str
    cache_mem_area: str
    arena_cache_size: int | None


@dataclass
class VelaSummary:  # pylint: disable=too-many-instance-attributes
    """Data gathered from the summary CSV file that Vela produces."""

    cycles_total: float
    cycles_npu: float
    cycles_sram_access: float
    cycles_dram_access: float
    cycles_on_chip_flash_access: float
    cycles_off_chip_flash_access: float
    core_clock: float
    dram_memory_used: float
    sram_memory_used: float
    on_chip_flash_memory_used: float
    off_chip_flash_memory_used: float
    batch_size: int
    memory_mode: str
    system_config: str
    accelerator_configuration: str
    arena_cache_size: float

    def __repr__(self) -> str:
        """Return String Representation of VelaSummary object."""
        header_values = dict(summary_metrics)
        string_to_check = ""
        for field in fields(self):
            string_to_check += (
                f"{header_values[field.name]}: {getattr(self, field.name)}, "
            )
        return string_to_check


complete_summary_metrics = [
    ("experiment", "experiment"),
    ("network", "network"),
    ("accelerator_configuration", "accelerator_configuration"),
    ("system_config", "system_config"),
    ("memory_mode", "memory_mode"),
    ("core_clock", "core_clock"),
    ("arena_cache_size", "arena_cache_size"),
    ("sram_bandwidth", "sram_bandwidth"),
    ("dram_bandwidth", "dram_bandwidth"),
    ("on_chip_flash_bandwidth", "on_chip_flash_bandwidth"),
    ("off_chip_flash_bandwidth", "off_chip_flash_bandwidth"),
    ("weights_storage_area", "weights_storage_area"),
    ("feature_map_storage_area", "feature_map_storage_area"),
    ("inferences_per_second", "inferences_per_second"),
    ("batch_size", "batch_size"),
    ("inference_time", "inference_time"),
    ("passes_before_fusing", "passes_before_fusing"),
    ("sram_memory_used", "sram_memory_used"),
    ("dram_memory_used", "dram_memory_used"),
    (
        "on_chip_flash_memory_used",
        "on_chip_flash_memory_used",
    ),
    ("off_chip_flash_memory_used", "off_chip_flash_memory_used"),
    ("total_original_weights", "total_original_weights"),
    ("total_npu_encoded_weights", "total_npu_encoded_weights"),
    ("dram_total_bytes", "dram_total_bytes"),
    (
        "on_chip_flash_feature_map_read_bytes",
        "on_chip_flash_feature_map_read_bytes",
    ),
    ("on_chip_flash_feature_map_write_bytes", "on_chip_flash_feature_map_write_bytes"),
    ("on_chip_flash_weight_read_bytes", "on_chip_flash_weight_read_bytes"),
    ("on_chip_flash_weight_write_bytes", "on_chip_flash_weight_write_bytes"),
    ("on_chip_flash_total_bytes", "on_chip_flash_total_bytes"),
    ("off_chip_flash_feature_map_read_bytes", "off_chip_flash_feature_map_read_bytes"),
    (
        "off_chip_flash_feature_map_write_bytes",
        "off_chip_flash_feature_map_write_bytes",
    ),
    ("off_chip_flash_weight_read_bytes", "off_chip_flash_weight_read_bytes"),
    ("off_chip_flash_weight_write_bytes", "off_chip_flash_weight_write_bytes"),
    ("off_chip_flash_total_bytes", "off_chip_flash_total_bytes"),
    ("nn_macs", "nn_macs"),
    ("nn_tops", "nn_tops"),
    ("cycles_npu", "cycles_npu"),
    ("cycles_sram_access", "cycles_sram_access"),
    ("cycles_dram_access", "cycles_dram_access"),
    ("cycles_on_chip_flash_access", "cycles_on_chip_flash_access"),
    ("cycles_off_chip_flash_access", "cycles_off_chip_flash_access"),
    ("cycles_total", "cycles_total"),
]

OUTPUT_METRICS = [field.name for field in fields(VelaSummary)]

summary_metrics = [
    summary_metric
    for summary_metric in complete_summary_metrics
    if summary_metric[0] in OUTPUT_METRICS
]
summary_metrics.sort(key=lambda e: OUTPUT_METRICS.index(e[0]))


@dataclass
class Model:
    """Model metadata."""

    nng: Graph
    network_type: NetworkType

    @property
    def optimized(self) -> bool:
        """Return true if model is already optimized."""
        return any(
            op.attrs.get("custom_type") == CustomType.ExistingNpuOp
            for sg in self.nng.subgraphs
            for op in sg.get_all_ops()
        )


AcceleratorConfigType = Literal[
    "ethos-u55-32",
    "ethos-u55-64",
    "ethos-u55-128",
    "ethos-u55-256",
    "ethos-u65-256",
    "ethos-u65-512",
]

TensorAllocatorType = Literal["LinearAlloc", "Greedy", "HillClimb"]

OptimizationStrategyType = Literal["Performance", "Size"]


@dataclass
class VelaCompilerOptions:  # pylint: disable=too-many-instance-attributes
    """Vela compiler options."""

    config_file: str | None = None
    system_config: str = "internal-default"
    memory_mode: str = "internal-default"
    accelerator_config: AcceleratorConfigType | None = None
    max_block_dependency: int = 3
    arena_cache_size: int | None = None
    tensor_allocator: TensorAllocatorType = "HillClimb"
    cpu_tensor_alignment: int = 16
    optimization_strategy: OptimizationStrategyType = "Performance"
    output_dir: Path = Path("output")
    recursion_limit: int = 1000
    verbose_performance: bool = True


class VelaCompiler:  # pylint: disable=too-many-instance-attributes
    """Vela compiler wrapper."""

    def __init__(self, compiler_options: VelaCompilerOptions):
        """Init Vela wrapper instance."""
        self.config_file = compiler_options.config_file
        self.system_config = compiler_options.system_config
        self.memory_mode = compiler_options.memory_mode
        self.accelerator_config = compiler_options.accelerator_config
        self.max_block_dependency = compiler_options.max_block_dependency
        self.arena_cache_size = compiler_options.arena_cache_size
        self.tensor_allocator = compiler_options.tensor_allocator
        self.cpu_tensor_alignment = compiler_options.cpu_tensor_alignment
        self.optimization_strategy = compiler_options.optimization_strategy
        self.output_dir = Path(compiler_options.output_dir)
        self.recursion_limit = compiler_options.recursion_limit
        self.verbose_performance = compiler_options.verbose_performance

        sys.setrecursionlimit(self.recursion_limit)

    def read_model(self, model: str | Path) -> Model:
        """Read model."""
        logger.debug("Read model %s", model)

        nng, network_type = self._read_model(model)
        return Model(nng, network_type)

    def compile_model(
        self, model_path: Path, already_compiled: bool = False
    ) -> tuple[VelaSummary, Path]:
        """Compile the model."""
        try:
            with redirect_raw_output(
                logger, stdout_level=logging.DEBUG, stderr_level=logging.DEBUG
            ):
                tmp = sys.stdout
                output_message = StringIO()
                sys.stdout = output_message
                main_args = [
                    "--output-dir",
                    str(self.output_dir.as_posix()),
                    "--tensor-allocator",
                    str(self.tensor_allocator),
                    "--cpu-tensor-alignment",
                    str(self.cpu_tensor_alignment),
                    "--accelerator-config",
                    str(self.accelerator_config),
                    "--system-config",
                    str(self.system_config),
                    "--memory-mode",
                    str(self.memory_mode),
                    "--max-block-dependency",
                    str(self.max_block_dependency),
                    "--optimise",
                    str(self.optimization_strategy),
                    model_path.as_posix(),
                    "--config",
                    str(self.config_file),
                ]
                if self.verbose_performance:
                    main_args.append("--verbose-performance")
                if not already_compiled:
                    main(main_args)
                optimized_model_path = Path(
                    self.output_dir.as_posix()
                    + "/"
                    + model_path.stem
                    + "_vela"
                    + model_path.suffix
                )
                sys.stdout = tmp
                if (
                    "Warning: SRAM target for arena memory area exceeded."
                    in output_message.getvalue()
                ):
                    raise MemoryError("Model is too large and uses too much RAM")
            summary_data = parse_summary_csv_file(
                Path(
                    self.output_dir.as_posix()
                    + "/"
                    + model_path.stem
                    + "_summary_"
                    + self.system_config
                    + ".csv"
                )
            )
            return summary_data, optimized_model_path
        except MemoryError as err:
            raise err
        except (SystemExit, Exception) as err:
            if (
                "Error: Invalid tflite file." in output_message.getvalue()
                and isinstance(err, SystemExit)
            ):
                raise RuntimeError(f"Unable to read model {model_path}") from err
            raise RuntimeError(
                "Model could not be optimized with Vela compiler."
            ) from err

    @staticmethod
    def _read_model(model: str | Path) -> tuple[Graph, NetworkType]:
        """Read TensorFlow Lite model."""
        model_path = str(model) if isinstance(model, Path) else model
        try:
            with redirect_output(
                logger, stdout_level=logging.DEBUG, stderr_level=logging.DEBUG
            ):
                return read_model(model_path, ModelReaderOptions())  # type: ignore
        except (SystemExit, Exception) as err:
            raise RuntimeError(f"Unable to read model {model_path}.") from err


def resolve_compiler_config(
    vela_compiler_options: VelaCompilerOptions,
) -> VelaInitData:
    """Resolve passed compiler options.

    Vela has number of configuration parameters that being
    resolved during passing compiler options. E.g. Vela
    reads configuration parameters from vela.ini and fills
    it's internal structures with resolved values (memory mode,
    system mode, etc.).

    In order to get this information we need to create
    instance of the Vela compiler first.
    """
    config_file = vela_compiler_options.config_file or get_vela_config()
    return parse_vela_initialisation_file(
        Path(config_file),
        vela_compiler_options.system_config,
        vela_compiler_options.memory_mode,
    )


def compile_model(model_path: Path, compiler_options: VelaCompilerOptions) -> Path:
    """Compile model."""
    vela_compiler = VelaCompiler(compiler_options)
    # output dir could be a path or str, cast to Path object
    output_dir = Path(compiler_options.output_dir)
    if Path(
        output_dir.as_posix()
        + "/"
        + model_path.stem
        + "_summary_"
        + compiler_options.system_config
        + ".csv"
    ).is_file():
        _, optimized_model_path = vela_compiler.compile_model(model_path, True)
    else:
        _, optimized_model_path = vela_compiler.compile_model(model_path)
    return optimized_model_path


def parse_summary_csv_file(vela_summary_csv_file: Path) -> VelaSummary:
    """Parse the summary csv file from Vela."""
    if not vela_summary_csv_file.is_file():
        raise FileNotFoundError(f"CSV File not found at {vela_summary_csv_file}")

    with open(vela_summary_csv_file, encoding="UTF-8") as csv_file:
        summary_reader = csv.DictReader(csv_file, delimiter=",")
        try:
            row = next(summary_reader)
        except StopIteration as err:
            raise RuntimeError("Generated Vela Summary CSV is empty") from err
        try:
            # pylint: disable=eval-used
            key_types = {
                field.name: eval(field.type)  # type: ignore # nosec
                for field in fields(VelaSummary)
            }
            # pylint: enable=eval-used
            summary_data = VelaSummary(
                **{key: key_types[key](row[title]) for key, title in summary_metrics}
            )
        except KeyError as err:
            raise KeyError(
                f"Generated Vela Summary CSV missing expected header: {err.args[0]}."
            ) from err
    return summary_data


def parse_vela_initialisation_file(  # pylint: disable=too-many-locals
    vela_init_file: Path, system_config: str, memory_mode: str
) -> VelaInitData:
    """Parse the vela.ini to retrieve data for the target information table."""
    if not vela_init_file.is_file():
        raise FileNotFoundError(
            f"Vela Initialisation File not found at {vela_init_file}"
        )

    lines = []
    with open(vela_init_file, encoding="UTF-8") as init_file:
        lines = init_file.readlines()

    if len(lines) == 0:
        raise OSError("vela.ini File Is Empty")

    lines = [line.strip("\n][ ") for line in lines]

    idxs_memory_mode = [
        idx for idx, item in enumerate(lines) if re.search("^Memory_Mode.*", item)
    ]

    if len(idxs_memory_mode) == 0:
        raise IndexError("No memory modes are present in vela.ini file.")

    idxs_system_config = [
        idx for idx, item in enumerate(lines) if re.search("^System_Config.*", item)
    ] + [idxs_memory_mode[0]]

    if len(idxs_system_config) <= 1:
        raise IndexError("No system configs are present in vela.ini file.")

    try:
        idx_config = lines.index("System_Config." + system_config)
    except ValueError as err:
        raise ValueError(
            f"System Config: {system_config} not present in vela.ini file."
        ) from err

    lines_to_probe = lines[
        idx_config : idxs_system_config[  # noqa: E203
            idxs_system_config.index(idx_config) + 1
        ]
    ]

    def collect_memory_mode_lines(memory_mode: str) -> list[str]:
        try:
            idx_memory_mode = lines.index("Memory_Mode." + memory_mode)
        except ValueError as err:
            raise ValueError(
                f"Memory Mode: {memory_mode} not present in vela.ini file."
            ) from err
        if idxs_memory_mode.index(idx_memory_mode) == len(idxs_memory_mode) - 1:
            lines_to_probe = lines[idx_memory_mode:]
        else:
            lines_to_probe = lines[
                idx_memory_mode : idxs_memory_mode[  # noqa: E203
                    idxs_memory_mode.index(idx_memory_mode) + 1
                ]
            ]
        return lines_to_probe

    lines_to_probe_memory_mode = collect_memory_mode_lines(memory_mode)
    extra_memory_mode_lines = []
    for line in lines_to_probe_memory_mode:
        if "inherit=Memory_Mode." in line:
            extra_memory_mode = line[line.rindex(".") + 1 :]  # noqa: E203
            extra_memory_mode_lines = collect_memory_mode_lines(extra_memory_mode)

    lines_to_probe += extra_memory_mode_lines + lines_to_probe_memory_mode

    init_dict = {}
    for line in lines_to_probe:
        if "=" in line:
            init_dict[line[: line.index("=")]] = line[
                line.index("=") + 1 :  # noqa: E203
            ]
    try:
        init_data = VelaInitData(
            system_config=system_config,
            core_clock=float(init_dict["core_clock"]),
            axi0_port=str(init_dict["axi0_port"]),
            axi1_port=str(init_dict["axi1_port"]),
            memory_mode=memory_mode,
            sram_memory_data=VelaInitMemoryData(
                clock_scale=float(init_dict["Sram_clock_scale"])
                if "Sram_clock_scale" in init_dict
                else None,
                burst_length=int(init_dict["Sram_burst_length"])
                if "Sram_burst_length" in init_dict
                else None,
                read_latency=int(init_dict["Sram_read_latency"])
                if "Sram_read_latency" in init_dict
                else None,
                write_latency=int(init_dict["Sram_write_latency"])
                if "Sram_write_latency" in init_dict
                else None,
            ),
            dram_memory_data=VelaInitMemoryData(
                clock_scale=float(init_dict["Dram_clock_scale"])
                if "Dram_clock_scale" in init_dict
                else None,
                burst_length=int(init_dict["Dram_burst_length"])
                if "Dram_burst_length" in init_dict
                else None,
                read_latency=int(init_dict["Dram_read_latency"])
                if "Dram_read_latency" in init_dict
                else None,
                write_latency=int(init_dict["Dram_write_latency"])
                if "Dram_write_latency" in init_dict
                else None,
            ),
            off_chip_flash_memory_data=VelaInitMemoryData(
                clock_scale=float(init_dict["OffChipFlash_clock_scale"])
                if "OffChipFlash_clock_scale" in init_dict
                else None,
                burst_length=int(init_dict["OffChipFlash_burst_length"])
                if "OffChipFlash_burst_length" in init_dict
                else None,
                read_latency=int(init_dict["OffChipFlash_read_latency"])
                if "OffChipFlash_read_latency" in init_dict
                else None,
                write_latency=int(init_dict["OffChipFlash_write_latency"])
                if "OffChipFlash_write_latency" in init_dict
                else None,
            ),
            on_chip_flash_memory_data=VelaInitMemoryData(
                clock_scale=float(init_dict["OnChipFlash_clock_scale"])
                if "OnChipFlash_clock_scale" in init_dict
                else None,
                burst_length=int(init_dict["OnChipFlash_burst_length"])
                if "OnChipFlash_burst_length" in init_dict
                else None,
                read_latency=int(init_dict["OnChipFlash_read_latency"])
                if "OnChipFlash_read_latency" in init_dict
                else None,
                write_latency=int(init_dict["OnChipFlash_write_latency"])
                if "OnChipFlash_write_latency" in init_dict
                else None,
            ),
            const_mem_area=str(init_dict["const_mem_area"]),
            arena_mem_area=str(init_dict["arena_mem_area"]),
            cache_mem_area=str(init_dict["cache_mem_area"]),
            arena_cache_size=int(init_dict["arena_cache_size"])
            if "arena_cache_size" in init_dict
            else None,
        )

    except KeyError as err:
        raise KeyError(f"Vela.ini file missing expected header: {err.args[0]}") from err

    return init_data
