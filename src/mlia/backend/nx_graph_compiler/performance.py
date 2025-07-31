# SPDX-FileCopyrightText: Copyright 2023-2025, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: LicenseRef-LICENSE
"""Backend module for Neural Accelerator Graph Compiler performance estimation."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Union

from mlia.backend.nx_graph_compiler.config import NXGraphCompilerConfig
from mlia.backend.nx_graph_compiler.output_parsing import NXDebugDatabaseParser
from mlia.backend.nx_graph_compiler.output_parsing import NXPerformanceDatabaseParser
from mlia.backend.nx_graph_compiler.statistics import NXOperatorPerformanceStats
from mlia.backend.nx_graph_compiler.statistics import NXPerformanceStats
from mlia.backend.repo import get_backend_repository
from mlia.backend.vulkan_model_converter.conversion import VulkanModelConverter
from mlia.core.performance import PerformanceEstimator
from mlia.nn.tensorflow.config import ModelConfiguration
from mlia.utils.filesystem import get_mlia_resources
from mlia.utils.logging import log_action
from mlia.utils.proc import Command
from mlia.utils.proc import OutputLogger
from mlia.utils.proc import process_command_output

logger = logging.getLogger(__name__)

GC_OUTPUT_CONTROL_PARAMS = [  # #               | Corresponding option in the .ini file
    "--enable-irrigation-output",  # #          | <no equivalent .ini option>
    "--enable-command-list-summary-dump",  # #  | commandListSummary
    "--enable-debug-database-dump",  # #        | debugDatabase
    "--enable-performance-database-dump",  # #  | performanceDatabase
    "--enable-performance-summary-dump",  # #   | networkPerformanceEstimation
    "--enable-dot-dump",  # #                   | enableDotDump
]


@dataclass
class NXGraphCompilerOutputFiles:
    """Collection of output files of the Neural Accelerator Graph Compiler."""

    cmd_stream_summary: Path
    debug_database: Path
    dot_file: Path
    irrigation_bin: Path
    irrigation_json: Path
    network_performance_summary: Path
    performance_database: Path

    @classmethod
    def from_output_dir(
        cls, output_dir: Path, output_name: str
    ) -> NXGraphCompilerOutputFiles:
        """Create files in the Neural Accelerator Graph Compiler output dir."""
        name_to_suffix = {
            "cmd_stream_summary": "_command_stream_summary.dat",
            "debug_database": "_debug_database.dat",
            "dot_file": ".dot",
            "irrigation_bin": "_irrigation.bin",
            "irrigation_json": "_irrigation.json",
            "network_performance_summary": "_network_performance_summary.json",
            "performance_database": "_performance_database.dat",
        }
        args = {
            name: output_dir / f"{output_name}{suffix}"
            for name, suffix in name_to_suffix.items()
        }
        return cls(**args)

    def check_exists(self) -> None:
        """Raise a FileNotFoundError if one of the files does not exist."""
        for path in vars(self).values():
            if isinstance(path, Path) and not path.is_file():
                raise FileNotFoundError(
                    f"Expected output file '{path}' of the Neural Accelerator Graph "
                    "compiler does not exist."
                )


@dataclass
class NXGraphCompilerPerformanceMetrics:
    """Neural Accelerator Graph Compiler configuration and performance metrics."""

    backend_config: NXGraphCompilerConfig
    output_files: NXGraphCompilerOutputFiles
    performance_db_parser: NXPerformanceDatabaseParser
    performance_metrics: dict[str, NXOperatorPerformanceStats]


class NXGraphCompilerPerformanceEstimator(
    PerformanceEstimator[
        Union[Path, ModelConfiguration], NXGraphCompilerPerformanceMetrics
    ]
):
    """Performance estimator for the Neural Accelerator Graph Compiler."""

    resource_dir = get_mlia_resources() / "nx-graph-compiler"

    def __init__(
        self, output_dir: Path, backend_config: dict, operator_types_mapping: dict
    ) -> None:
        """Init performance estimator."""
        self.backend_config = NXGraphCompilerConfig(
            **backend_config.get("nx-graph-compiler", {})
        )
        self.backend_config.set_config_dir(self.resource_dir)
        self.output_dir = output_dir
        self.operator_types_mapping = operator_types_mapping

    def estimate(
        self,
        model: Path | ModelConfiguration,
    ) -> NXGraphCompilerPerformanceMetrics:
        """Estimate performance."""
        with log_action("Getting the performance data..."):
            model_path = (
                Path(model.model_path)
                if isinstance(model, ModelConfiguration)
                else model
            )

            vgf_file = self._run_vulkan_model_converter(model_path)
            output = self._run_nx_graph_compiler(vgf_file, model_path.stem)

            perf_db_parser = NXPerformanceDatabaseParser(
                db_path=Path(output.performance_database)
            )
            performance_db = perf_db_parser.parse_performance_database()

            ddb_parser = NXDebugDatabaseParser(Path(output.debug_database))
            debug_db = ddb_parser.parse_debug_database()

            perf_stats = NXPerformanceStats(
                debug_db=debug_db,
                performance_db=performance_db,
                operator_types_mapping=self.operator_types_mapping,
            )
            stats_per_chain = perf_stats.process_stats_per_chain()
            output_file_path = self.output_dir / "nx_performance_statistics.json"
            self.json_dump(stats_per_chain, output_file_path)

            return NXGraphCompilerPerformanceMetrics(
                self.backend_config,
                output,
                perf_db_parser,
                stats_per_chain,
            )

    def _run_vulkan_model_converter(self, model_path: Path) -> Path:
        """Run the Vulkan Model Converter and return the path to the SPIR-V file."""
        backend_repo = get_backend_repository()
        vmc_path, _ = backend_repo.get_backend_settings("vulkan-model-converter")
        output_dir = self.output_dir / "vulkan-model-converter"
        output_dir.mkdir()

        model_converter = VulkanModelConverter(vmc_path)
        vgf_file = model_converter(model_path, output_dir)
        return vgf_file

    def _run_nx_graph_compiler(
        self, vgf_file: Path, output_name: str
    ) -> NXGraphCompilerOutputFiles:
        """Run the Neural Accelerator Graph Compiler and return the output files."""
        backend_repo = get_backend_repository()
        gc_path, _ = backend_repo.get_backend_settings("nx-graph-compiler")
        output_dir = self.output_dir / "nx-graph-compiler"
        output_dir.mkdir()
        # We need to specify the stem of the output files here, i.e. neither
        # the output directory or the specific output file.
        output = output_dir / output_name
        system_config = self.backend_config.system_config
        compiler_config = self.backend_config.compiler_config

        output.mkdir()

        system_config_args = (
            []
            if system_config == NXGraphCompilerConfig.DEFAULT
            else ["--system_config", str(system_config)]
        )

        compiler_config_args = (
            []
            if compiler_config == NXGraphCompilerConfig.DEFAULT
            else ["--compiler_config", str(compiler_config)]
        )

        cmd = Command(
            cmd=[
                str(gc_path / "regorc-0.1.0"),
                "-i",
                str(vgf_file),
                "-o",
                str(output.stem),
                "--enable-config-file-dump",
                *system_config_args,
                *compiler_config_args,
                *GC_OUTPUT_CONTROL_PARAMS,
            ],
            cwd=output_dir,
        )

        process_command_output(cmd, [OutputLogger(logger, logging.INFO)])

        output_files = NXGraphCompilerOutputFiles.from_output_dir(
            output_dir, output_name
        )
        output_files.check_exists()
        return output_files

    def json_dump(self, stats_per_chain: dict, output_file_path: Path) -> None:
        """Make a json dump of the stats_per_chain dict."""
        json_serializable_stats = {
            key: obj.to_dict() for key, obj in stats_per_chain.items()
        }
        with output_file_path.open("w") as json_file:
            json.dump(json_serializable_stats, json_file, indent=4)
