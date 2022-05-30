# SPDX-FileCopyrightText: Copyright 2022, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Module for AIET integration."""
import logging
import re
from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Literal
from typing import Optional
from typing import Tuple

from aiet.backend.application import get_available_applications
from aiet.backend.application import install_application
from aiet.backend.system import get_available_systems
from aiet.backend.system import install_system
from mlia.utils.proc import CommandExecutor
from mlia.utils.proc import OutputConsumer
from mlia.utils.proc import RunningCommand


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
    """Get the AIET system name for the given backend and device type."""
    return _SUPPORTED_SYSTEMS[backend][device_type]


def is_supported(backend: str, device_type: Optional[str] = None) -> bool:
    """Check if the backend (and optionally device type) is supported."""
    if device_type is None:
        return backend in _SUPPORTED_SYSTEMS

    try:
        get_system_name(backend, device_type)
        return True
    except KeyError:
        return False


def supported_backends() -> List[str]:
    """Get a list of all backends supported by the AIET wrapper."""
    return list(_SUPPORTED_SYSTEMS.keys())


def get_all_system_names(backend: str) -> List[str]:
    """Get all systems supported by the backend."""
    return list(_SUPPORTED_SYSTEMS.get(backend, {}).values())


def get_all_application_names(backend: str) -> List[str]:
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
    application_params: List[str]
    system_params: List[str]
    deploy_params: List[str]


class AIETLogWriter(OutputConsumer):
    """Redirect AIET command output to the logger."""

    def feed(self, line: str) -> None:
        """Process line from the output."""
        logger.debug(line.strip())


class GenericInferenceOutputParser(OutputConsumer):
    """Generic inference app output parser."""

    PATTERNS = {
        name: tuple(re.compile(pattern, re.IGNORECASE) for pattern in patterns)
        for name, patterns in (
            (
                "npu_active_cycles",
                (
                    r"NPU ACTIVE cycles: (?P<value>\d+)",
                    r"NPU ACTIVE: (?P<value>\d+) cycles",
                ),
            ),
            (
                "npu_idle_cycles",
                (
                    r"NPU IDLE cycles: (?P<value>\d+)",
                    r"NPU IDLE: (?P<value>\d+) cycles",
                ),
            ),
            (
                "npu_total_cycles",
                (
                    r"NPU TOTAL cycles: (?P<value>\d+)",
                    r"NPU TOTAL: (?P<value>\d+) cycles",
                ),
            ),
            (
                "npu_axi0_rd_data_beat_received",
                (
                    r"NPU AXI0_RD_DATA_BEAT_RECEIVED beats: (?P<value>\d+)",
                    r"NPU AXI0_RD_DATA_BEAT_RECEIVED: (?P<value>\d+) beats",
                ),
            ),
            (
                "npu_axi0_wr_data_beat_written",
                (
                    r"NPU AXI0_WR_DATA_BEAT_WRITTEN beats: (?P<value>\d+)",
                    r"NPU AXI0_WR_DATA_BEAT_WRITTEN: (?P<value>\d+) beats",
                ),
            ),
            (
                "npu_axi1_rd_data_beat_received",
                (
                    r"NPU AXI1_RD_DATA_BEAT_RECEIVED beats: (?P<value>\d+)",
                    r"NPU AXI1_RD_DATA_BEAT_RECEIVED: (?P<value>\d+) beats",
                ),
            ),
        )
    }

    def __init__(self) -> None:
        """Init generic inference output parser instance."""
        self.result: Dict = {}

    def feed(self, line: str) -> None:
        """Feed new line to the parser."""
        for name, patterns in self.PATTERNS.items():
            for pattern in patterns:
                match = pattern.search(line)

                if match:
                    self.result[name] = int(match["value"])
                    return

    def is_ready(self) -> bool:
        """Return true if all expected data has been parsed."""
        return self.result.keys() == self.PATTERNS.keys()

    def missed_keys(self) -> List[str]:
        """Return list of the keys that have not been found in the output."""
        return sorted(self.PATTERNS.keys() - self.result.keys())


class AIETRunner:
    """AIET runner."""

    def __init__(self, executor: CommandExecutor) -> None:
        """Init AIET runner instance."""
        self.executor = executor

    @staticmethod
    def get_installed_systems() -> List[str]:
        """Get list of the installed systems."""
        return [system.name for system in get_available_systems()]

    @staticmethod
    def get_installed_applications(system: Optional[str] = None) -> List[str]:
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

    def systems_installed(self, systems: List[str]) -> bool:
        """Check if all provided systems are installed."""
        if not systems:
            return False

        installed_systems = self.get_installed_systems()
        return all(system in installed_systems for system in systems)

    def applications_installed(self, applications: List[str]) -> bool:
        """Check if all provided applications are installed."""
        if not applications:
            return False

        installed_apps = self.get_installed_applications()
        return all(app in installed_apps for app in applications)

    def all_installed(self, systems: List[str], apps: List[str]) -> bool:
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

    def run_application(self, execution_params: ExecutionParams) -> RunningCommand:
        """Run requested application."""
        command = [
            "aiet",
            "application",
            "run",
            "-n",
            execution_params.application,
            "-s",
            execution_params.system,
            *self._params("-p", execution_params.application_params),
            *self._params("--system-param", execution_params.system_params),
            *self._params("--deploy", execution_params.deploy_params),
        ]

        return self._submit(command)

    @staticmethod
    def _params(name: str, params: List[str]) -> List[str]:
        return [p for item in [(name, param) for param in params] for p in item]

    def _submit(self, command: List[str]) -> RunningCommand:
        """Submit command for the execution."""
        logger.debug("Submit command %s", " ".join(command))
        return self.executor.submit(command)


class GenericInferenceRunner(ABC):
    """Abstract class for generic inference runner."""

    def __init__(self, aiet_runner: AIETRunner):
        """Init generic inference runner instance."""
        self.aiet_runner = aiet_runner
        self.running_inference: Optional[RunningCommand] = None

    def run(
        self, model_info: ModelInfo, output_consumers: List[OutputConsumer]
    ) -> None:
        """Run generic inference for the provided device/model."""
        execution_params = self.get_execution_params(model_info)

        self.running_inference = self.aiet_runner.run_application(execution_params)
        self.running_inference.output_consumers = output_consumers
        self.running_inference.consume_output()

    def stop(self) -> None:
        """Stop running inference."""
        if self.running_inference is None:
            return

        self.running_inference.stop()

    @abstractmethod
    def get_execution_params(self, model_info: ModelInfo) -> ExecutionParams:
        """Get execution params for the provided model."""

    def __enter__(self) -> "GenericInferenceRunner":
        """Enter context."""
        return self

    def __exit__(self, *_args: Any) -> None:
        """Exit context."""
        self.stop()

    def check_system_and_application(self, system_name: str, app_name: str) -> None:
        """Check if requested system and application installed."""
        if not self.aiet_runner.is_system_installed(system_name):
            raise Exception(f"System {system_name} is not installed")

        if not self.aiet_runner.is_application_installed(app_name, system_name):
            raise Exception(
                f"Application {app_name} for the system {system_name} "
                "is not installed"
            )


class GenericInferenceRunnerEthosU(GenericInferenceRunner):
    """Generic inference runner on U55/65."""

    def __init__(
        self, aiet_runner: AIETRunner, device_info: DeviceInfo, backend: str
    ) -> None:
        """Init generic inference runner instance."""
        super().__init__(aiet_runner)

        system_name, app_name = self.resolve_system_and_app(device_info, backend)
        self.system_name = system_name
        self.app_name = app_name
        self.device_info = device_info

    @staticmethod
    def resolve_system_and_app(
        device_info: DeviceInfo, backend: str
    ) -> Tuple[str, str]:
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
            [],
        )


def get_generic_runner(device_info: DeviceInfo, backend: str) -> GenericInferenceRunner:
    """Get generic runner for provided device and backend."""
    aiet_runner = get_aiet_runner()
    return GenericInferenceRunnerEthosU(aiet_runner, device_info, backend)


def estimate_performance(
    model_info: ModelInfo, device_info: DeviceInfo, backend: str
) -> PerformanceMetrics:
    """Get performance estimations."""
    with get_generic_runner(device_info, backend) as generic_runner:
        output_parser = GenericInferenceOutputParser()
        output_consumers = [output_parser, AIETLogWriter()]

        generic_runner.run(model_info, output_consumers)

        if not output_parser.is_ready():
            missed_data = ",".join(output_parser.missed_keys())
            logger.debug(
                "Unable to get performance metrics, missed data %s", missed_data
            )
            raise Exception("Unable to get performance metrics, insufficient data")

        return PerformanceMetrics(**output_parser.result)


def get_aiet_runner() -> AIETRunner:
    """Return AIET runner."""
    executor = CommandExecutor()
    return AIETRunner(executor)
