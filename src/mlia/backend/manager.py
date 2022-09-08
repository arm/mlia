# SPDX-FileCopyrightText: Copyright 2022, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Module for backend integration."""
from __future__ import annotations

import logging
from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from mlia.backend.application import get_available_applications
from mlia.backend.application import install_application
from mlia.backend.execution import ExecutionContext
from mlia.backend.execution import run_application
from mlia.backend.output_consumer import Base64OutputConsumer
from mlia.backend.output_consumer import OutputConsumer
from mlia.backend.system import get_available_systems
from mlia.backend.system import install_system


logger = logging.getLogger(__name__)

# Mapping backend -> device_type -> system_name
_SUPPORTED_SYSTEMS = {
    "Corstone-300": {
        "ethos-u55": "Corstone-300: Cortex-M55+Ethos-U55",
        "ethos-u65": "Corstone-300: Cortex-M55+Ethos-U65",
    },
    "Corstone-310": {
        "ethos-u55": "Corstone-310: Cortex-M85+Ethos-U55",
    },
}

# Mapping system_name -> memory_mode -> application
_SYSTEM_TO_APP_MAP = {
    "Corstone-300: Cortex-M55+Ethos-U55": {
        "Sram": "Generic Inference Runner: Ethos-U55 SRAM",
        "Shared_Sram": "Generic Inference Runner: Ethos-U55/65 Shared SRAM",
    },
    "Corstone-300: Cortex-M55+Ethos-U65": {
        "Shared_Sram": "Generic Inference Runner: Ethos-U55/65 Shared SRAM",
        "Dedicated_Sram": "Generic Inference Runner: Ethos-U65 Dedicated SRAM",
    },
    "Corstone-310: Cortex-M85+Ethos-U55": {
        "Sram": "Generic Inference Runner: Ethos-U55 SRAM",
        "Shared_Sram": "Generic Inference Runner: Ethos-U55/65 Shared SRAM",
    },
}


def get_system_name(backend: str, device_type: str) -> str:
    """Get the system name for the given backend and device type."""
    return _SUPPORTED_SYSTEMS[backend][device_type]


def is_supported(backend: str, device_type: str | None = None) -> bool:
    """Check if the backend (and optionally device type) is supported."""
    if device_type is None:
        return backend in _SUPPORTED_SYSTEMS

    try:
        get_system_name(backend, device_type)
        return True
    except KeyError:
        return False


def supported_backends() -> list[str]:
    """Get a list of all backends supported by the backend manager."""
    return list(_SUPPORTED_SYSTEMS.keys())


def get_all_system_names(backend: str) -> list[str]:
    """Get all systems supported by the backend."""
    return list(_SUPPORTED_SYSTEMS.get(backend, {}).values())


def get_all_application_names(backend: str) -> list[str]:
    """Get all applications supported by the backend."""
    app_set = {
        app
        for sys in get_all_system_names(backend)
        for app in _SYSTEM_TO_APP_MAP[sys].values()
    }
    return list(app_set)


@dataclass
class DeviceInfo:
    """Device information."""

    device_type: Literal["ethos-u55", "ethos-u65"]
    mac: int
    memory_mode: Literal["Sram", "Shared_Sram", "Dedicated_Sram"]


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


@dataclass
class ExecutionParams:
    """Application execution params."""

    application: str
    system: str
    application_params: list[str]
    system_params: list[str]


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


class BackendRunner:
    """Backend runner."""

    def __init__(self) -> None:
        """Init BackendRunner instance."""

    @staticmethod
    def get_installed_systems() -> list[str]:
        """Get list of the installed systems."""
        return [system.name for system in get_available_systems()]

    @staticmethod
    def get_installed_applications(system: str | None = None) -> list[str]:
        """Get list of the installed application."""
        return [
            app.name
            for app in get_available_applications()
            if system is None or app.can_run_on(system)
        ]

    def is_application_installed(self, application: str, system: str) -> bool:
        """Return true if requested application installed."""
        return application in self.get_installed_applications(system)

    def is_system_installed(self, system: str) -> bool:
        """Return true if requested system installed."""
        return system in self.get_installed_systems()

    def systems_installed(self, systems: list[str]) -> bool:
        """Check if all provided systems are installed."""
        if not systems:
            return False

        installed_systems = self.get_installed_systems()
        return all(system in installed_systems for system in systems)

    def applications_installed(self, applications: list[str]) -> bool:
        """Check if all provided applications are installed."""
        if not applications:
            return False

        installed_apps = self.get_installed_applications()
        return all(app in installed_apps for app in applications)

    def all_installed(self, systems: list[str], apps: list[str]) -> bool:
        """Check if all provided artifacts are installed."""
        return self.systems_installed(systems) and self.applications_installed(apps)

    @staticmethod
    def install_system(system_path: Path) -> None:
        """Install system."""
        install_system(system_path)

    @staticmethod
    def install_application(app_path: Path) -> None:
        """Install application."""
        install_application(app_path)

    @staticmethod
    def run_application(execution_params: ExecutionParams) -> ExecutionContext:
        """Run requested application."""
        ctx = run_application(
            execution_params.application,
            execution_params.application_params,
            execution_params.system,
            execution_params.system_params,
        )
        return ctx

    @staticmethod
    def _params(name: str, params: list[str]) -> list[str]:
        return [p for item in [(name, param) for param in params] for p in item]


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

        if system_name not in _SYSTEM_TO_APP_MAP:
            raise RuntimeError(f"System {system_name} is not installed")

        try:
            app_name = _SYSTEM_TO_APP_MAP[system_name][device_info.memory_mode]
        except KeyError as err:
            raise RuntimeError(
                f"Unsupported memory mode {device_info.memory_mode}"
            ) from err

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
