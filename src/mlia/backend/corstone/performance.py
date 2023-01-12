# SPDX-FileCopyrightText: Copyright 2022-2023, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Module for backend integration."""
from __future__ import annotations

import logging
from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from mlia.backend.executor.output_consumer import Base64OutputConsumer
from mlia.backend.executor.output_consumer import OutputConsumer
from mlia.backend.executor.runner import BackendRunner
from mlia.backend.executor.runner import ExecutionParams
from mlia.backend.install import get_application_name
from mlia.backend.install import get_system_name


logger = logging.getLogger(__name__)


@dataclass
class DeviceInfo:
    """Device information."""

    device_type: Literal["Ethos-U55", "Ethos-U65", "ethos-u55", "ethos-u65"]
    mac: int


@dataclass
class ModelInfo:
    """Model info."""

    model_path: Path


@dataclass
class PerformanceMetrics:
    """Performance metrics parsed from generic inference output."""

    npu_active_cycles: int
    npu_idle_cycles: int
    npu_total_cycles: int
    npu_axi0_rd_data_beat_received: int
    npu_axi0_wr_data_beat_written: int
    npu_axi1_rd_data_beat_received: int


class LogWriter(OutputConsumer):
    """Redirect output to the logger."""

    def feed(self, line: str) -> bool:
        """Process line from the output."""
        logger.debug(line.strip())
        return False


class GenericInferenceOutputParser(Base64OutputConsumer):
    """Generic inference app output parser."""

    def __init__(self) -> None:
        """Init generic inference output parser instance."""
        super().__init__()
        self._map = {
            "NPU ACTIVE": "npu_active_cycles",
            "NPU IDLE": "npu_idle_cycles",
            "NPU TOTAL": "npu_total_cycles",
            "NPU AXI0_RD_DATA_BEAT_RECEIVED": "npu_axi0_rd_data_beat_received",
            "NPU AXI0_WR_DATA_BEAT_WRITTEN": "npu_axi0_wr_data_beat_written",
            "NPU AXI1_RD_DATA_BEAT_RECEIVED": "npu_axi1_rd_data_beat_received",
        }

    @property
    def result(self) -> dict:
        """Merge the raw results and map the names to the right output names."""
        merged_result = {}
        for raw_result in self.parsed_output:
            for profiling_result in raw_result:
                for sample in profiling_result["samples"]:
                    name, values = (sample["name"], sample["value"])
                    if name in merged_result:
                        raise KeyError(
                            f"Duplicate key '{name}' in base64 output.",
                        )
                    new_name = self._map[name]
                    merged_result[new_name] = values[0]
        return merged_result

    def is_ready(self) -> bool:
        """Return true if all expected data has been parsed."""
        return set(self.result.keys()) == set(self._map.values())

    def missed_keys(self) -> set[str]:
        """Return a set of the keys that have not been found in the output."""
        return set(self._map.values()) - set(self.result.keys())


class GenericInferenceRunner(ABC):
    """Abstract class for generic inference runner."""

    def __init__(self, backend_runner: BackendRunner):
        """Init generic inference runner instance."""
        self.backend_runner = backend_runner

    def run(
        self, model_info: ModelInfo, output_consumers: list[OutputConsumer]
    ) -> None:
        """Run generic inference for the provided device/model."""
        execution_params = self.get_execution_params(model_info)

        ctx = self.backend_runner.run_application(execution_params)
        if ctx.stdout is not None:
            ctx.stdout = self.consume_output(ctx.stdout, output_consumers)

    @abstractmethod
    def get_execution_params(self, model_info: ModelInfo) -> ExecutionParams:
        """Get execution params for the provided model."""

    def check_system_and_application(self, system_name: str, app_name: str) -> None:
        """Check if requested system and application installed."""
        if not self.backend_runner.is_system_installed(system_name):
            raise Exception(f"System {system_name} is not installed")

        if not self.backend_runner.is_application_installed(app_name, system_name):
            raise Exception(
                f"Application {app_name} for the system {system_name} "
                "is not installed"
            )

    @staticmethod
    def consume_output(output: bytearray, consumers: list[OutputConsumer]) -> bytearray:
        """
        Pass program's output to the consumers and filter it.

        Returns the filtered output.
        """
        filtered_output = bytearray()
        for line_bytes in output.splitlines():
            line = line_bytes.decode("utf-8")
            remove_line = False
            for consumer in consumers:
                if consumer.feed(line):
                    remove_line = True
            if not remove_line:
                filtered_output.extend(line_bytes)

        return filtered_output


class GenericInferenceRunnerEthosU(GenericInferenceRunner):
    """Generic inference runner on U55/65."""

    def __init__(
        self, backend_runner: BackendRunner, device_info: DeviceInfo, backend: str
    ) -> None:
        """Init generic inference runner instance."""
        super().__init__(backend_runner)

        system_name, app_name = self.resolve_system_and_app(device_info, backend)
        self.system_name = system_name
        self.app_name = app_name
        self.device_info = device_info

    @staticmethod
    def resolve_system_and_app(
        device_info: DeviceInfo, backend: str
    ) -> tuple[str, str]:
        """Find appropriate system and application for the provided device/backend."""
        try:
            system_name = get_system_name(backend, device_info.device_type)
        except KeyError as ex:
            raise RuntimeError(
                f"Unsupported device {device_info.device_type} "
                f"for backend {backend}"
            ) from ex

        try:
            app_name = get_application_name(system_name)
        except KeyError as err:
            raise RuntimeError(f"System {system_name} is not installed") from err

        return system_name, app_name

    def get_execution_params(self, model_info: ModelInfo) -> ExecutionParams:
        """Get execution params for Ethos-U55/65."""
        self.check_system_and_application(self.system_name, self.app_name)

        system_params = [
            f"mac={self.device_info.mac}",
            f"input_file={model_info.model_path.absolute()}",
        ]

        return ExecutionParams(
            self.app_name,
            self.system_name,
            [],
            system_params,
        )


def get_generic_runner(device_info: DeviceInfo, backend: str) -> GenericInferenceRunner:
    """Get generic runner for provided device and backend."""
    backend_runner = get_backend_runner()
    return GenericInferenceRunnerEthosU(backend_runner, device_info, backend)


def estimate_performance(
    model_info: ModelInfo, device_info: DeviceInfo, backend: str
) -> PerformanceMetrics:
    """Get performance estimations."""
    output_parser = GenericInferenceOutputParser()
    output_consumers = [output_parser, LogWriter()]

    generic_runner = get_generic_runner(device_info, backend)
    generic_runner.run(model_info, output_consumers)

    if not output_parser.is_ready():
        missed_data = ",".join(output_parser.missed_keys())
        logger.debug("Unable to get performance metrics, missed data %s", missed_data)
        raise Exception("Unable to get performance metrics, insufficient data")

    return PerformanceMetrics(**output_parser.result)


def get_backend_runner() -> BackendRunner:
    """
    Return BackendRunner instance.

    Note: This is needed for the unit tests.
    """
    return BackendRunner()
