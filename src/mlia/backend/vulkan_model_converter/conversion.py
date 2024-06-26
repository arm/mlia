# SPDX-FileCopyrightText: Copyright 2023-2024, Arm Limited and/or its affiliates.
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


class VulkanModelConverterBase:
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
            cmd=[
                str(self.converter_path / self.FRONT_END_EXE),
                "-i",
                str(tflite_file),
                "-o",
                str(tosa_file),
                *self._extra_front_end_arguments(),
            ],
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
                str(self.converter_path / self.BACK_END_EXE),
                "-i",
                str(tosa_file),
                "-o",
                str(spirv_file),
                *self._extra_back_end_arguments(),
            ],
        )
        return cmd

    def _run_back_end(self, tosa_file: Path, output_dir: Path) -> Path:
        """Run the backend and return the SPIR-V output archive."""
        spirv_file = output_dir / f"{tosa_file.stem}_spirv.zip"
        cmd = self._create_back_end_command(tosa_file, spirv_file)
        process_command_output(cmd, self.output_consumers)

        return spirv_file

    def _library_paths(self) -> list[Path]:
        paths = [self.converter_path / path for path in (self.FRONT_END_DIR,)]
        return paths

    def _extra_front_end_arguments(self) -> list[str]:
        """Return any extra arguments to be used with the VMC front-end."""
        return ["--emit-byte-code"]

    def _extra_back_end_arguments(self) -> list[str]:
        """Return any extra arguments to be used with the VMC back-end."""
        return []


# pylint: disable=too-few-public-methods
class VulkanModelConverter(VulkanModelConverterBase):
    """Run the Vulkan Model Converter to produce a SPIR-v file."""

    def _run_back_end(self, tosa_file: Path, output_dir: Path) -> Path:
        """Run the backend and return the SPIR-V output archive."""
        spirv_file = super()._run_back_end(tosa_file, output_dir)

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

    def _extra_back_end_arguments(self) -> list[str]:
        """Return any extra arguments to be used with the VMC back-end."""
        return ["--package-spv", "--emit-debug-info"]
