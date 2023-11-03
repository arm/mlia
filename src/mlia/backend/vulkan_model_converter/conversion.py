# SPDX-FileCopyrightText: Copyright 2023, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: LicenseRef-LICENSE
"""Convert TensorFlow Lite models with the Vulkan Model Converter."""
from __future__ import annotations

import logging
from pathlib import Path

from mlia.utils.logging import log_action
from mlia.utils.proc import Command
from mlia.utils.proc import OutputConsumer
from mlia.utils.proc import OutputLogger
from mlia.utils.proc import process_command_output

logger = logging.getLogger(__name__)


class VulkanModelConverter:
    """Wrapper class to run the Vulkan Model Converter."""

    FRONT_END_DIR = "front-ends/tflite"
    FRONT_END_EXE = f"{FRONT_END_DIR}/vulkan-converter-tflite-front-end"
    BACK_END_EXE = "back-end/vulkan-converter-back-end"

    def __init__(self, converter_path: Path) -> None:
        """Set up some paths to run the Vulkan Model Converter."""
        self.converter_path = converter_path.resolve()
        self.frontend_library_paths = self._library_paths()
        self.output_consumers: list[OutputConsumer] = [
            OutputLogger(logger, logging.INFO)
        ]

    def __call__(self, tflite_file: Path, output_dir: Path) -> Path:
        """
        Run the Vulkan Model Converter with the given TensorFlow Lite file.

        Returns the path of the SPIR-V output archive created in the output dir.
        """
        if not output_dir.is_dir():
            raise NotADirectoryError(
                f"Path '{output_dir}' is not a directory. Unable to run "
                "Vulkan Model Converter."
            )
        with log_action("Running Vulkan Model Converter..."):
            logger.debug("Vulkan Model Converter path: %s", self.converter_path)

            tosa_file = self._run_front_end(tflite_file, output_dir)
            spirv_file = self._run_back_end(tosa_file, output_dir)

            logger.debug("Output file: %s", spirv_file)

        return spirv_file

    def _create_front_end_command(self, tflite_file: Path, tosa_file: Path) -> Command:
        """Create the command to run the front end."""
        env = {
            "LD_LIBRARY_PATH": ":".join(
                str(path) for path in self.frontend_library_paths
            )
        }
        cmd = Command(
            cmd=[self.FRONT_END_EXE, "-i", str(tflite_file), "-o", str(tosa_file)],
            cwd=self.converter_path,
            env=env,
        )
        return cmd

    def _run_front_end(self, tflite_file: Path, output_dir: Path) -> Path:
        """Run the frontend and return the TOSA MLIR output file."""
        tosa_file = output_dir / f"{tflite_file.stem}.tosamlir"
        cmd = self._create_front_end_command(tflite_file, tosa_file)
        process_command_output(cmd, self.output_consumers)

        if not tosa_file.is_file():
            raise FileNotFoundError(
                "No output from the Vulkan Model Converter frontend found. "
                f"File {tosa_file} does not exist."
            )
        logger.debug(
            "Frontend of Vulkan Model Converter run successfully. See output: %s",
            tosa_file,
        )

        return tosa_file

    def _create_back_end_command(self, tosa_file: Path, spirv_file: Path) -> Command:
        """Create the command to run the front end."""
        cmd = Command(
            cmd=[
                self.BACK_END_EXE,
                "-i",
                str(tosa_file),
                "-o",
                str(spirv_file),
                "--package-spv",
            ],
            cwd=self.converter_path,
        )
        return cmd

    def _run_back_end(self, tosa_file: Path, output_dir: Path) -> Path:
        """Run the backend and return the SPIR-V output archive."""
        spirv_file = output_dir / f"{tosa_file.stem}_spirv.zip"
        cmd = self._create_back_end_command(tosa_file, spirv_file)
        process_command_output(cmd, self.output_consumers)

        if not spirv_file.is_file():
            raise FileNotFoundError(
                "No output from the Vulkan Model Converter backend found. "
                f"File {spirv_file} does not exist."
            )
        logger.debug(
            "Back end of Vulkan Model Converter run successfully. See output: %s",
            spirv_file,
        )

        return spirv_file

    def _library_paths(self) -> list[Path]:
        paths = [
            self.converter_path / path
            for path in (
                self.FRONT_END_DIR,
                "tensorflow/lite",
                "tensorflow/compiler/mlir/tosa",
                "tensorflow/compiler/mlir/lite",
                "tensorflow/compiler/mlir/tensorflow",
                "tensorflow/compiler/mlir/lite/quantization",
                "tensorflow/compiler/mlir/lite/quantization/ir",
                "tensorflow/compiler/mlir/lite/quantization/lite",
                "tensorflow/lite/kernels/internal",
                "tensorflow/lite/tools/optimize",
                "tensorflow/lite/core/api",
                "tensorflow/lite/schema",
                "tensorflow/lite/c",
                "tensorflow/lite/kernels",
                "external/cpuinfo",
                "tensorflow/core/ir/types",
                "external/ruy/ruy",
                "tensorflow",
                "tensorflow/lite/experimental/remat",
                "tensorflow/lite/core/c",
                "tensorflow/lite/core",
                "tensorflow/core/ir/importexport",
                "tensorflow/compiler/xla/mlir/utils",
                "external/pthreadpool",
            )
        ]
        return paths
