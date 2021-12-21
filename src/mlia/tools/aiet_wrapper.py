# Copyright 2021, Arm Ltd.
"""Module for AIET integration."""
import json
import logging
import os
import re
from abc import ABC
from abc import abstractmethod
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing import cast
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
from mlia.config import EthosU55
from mlia.config import EthosU65
from mlia.config import EthosUConfiguration
from mlia.config import TFLiteModel
from mlia.utils.proc import CommandExecutor
from mlia.utils.proc import OutputConsumer
from mlia.utils.proc import RunningCommand


logger = logging.getLogger(__name__)


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


class AIETLogWriter(OutputConsumer):  # pylint: disable=too-few-public-methods
    """Redirect AIET command output to the logger."""

    def feed(self, line: str) -> None:
        """Process line from the output."""
        logger.debug(line.strip())


class GenericInferenceOutputParser(OutputConsumer):
    """Generic inference app output parser."""

    PATTERNS = {
        name: re.compile(pattern, re.IGNORECASE)
        for name, pattern in {
            ("npu_active_cycles", r"NPU ACTIVE cycles: (?P<value>\d+)"),
            ("npu_idle_cycles", r"NPU IDLE cycles: (?P<value>\d+)"),
            ("npu_total_cycles", r"NPU TOTAL cycles: (?P<value>\d+)"),
            (
                "npu_axi0_rd_data_beat_received",
                r"NPU AXI0_RD_DATA_BEAT_RECEIVED beats: (?P<value>\d+)",
            ),
            (
                "npu_axi0_wr_data_beat_written",
                r"NPU AXI0_WR_DATA_BEAT_WRITTEN beats: (?P<value>\d+)",
            ),
            (
                "npu_axi1_rd_data_beat_received",
                r"NPU AXI1_RD_DATA_BEAT_RECEIVED beats: (?P<value>\d+)",
            ),
        }
    }

    def __init__(self) -> None:
        """Init generic inference output parser instance."""
        self.result: Dict = {}

    def feed(self, line: str) -> None:
        """Feed new line to the parser."""
        for name, pattern in self.PATTERNS.items():
            match = pattern.search(line)

            if match:
                self.result[name] = int(match["value"])
                break

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

    def get_installed_systems(self) -> List[str]:
        """Get list of the installed systems."""
        command = self._system_command("list")

        json_data = self._execute_and_parse(command)
        if not isinstance(json_data, dict) or "available" not in json_data:
            raise Exception("Unable to get system information")

        return cast(List[str], json_data["available"])

    def get_installed_applications(self, system: Optional[str] = None) -> List[str]:
        """Get list of the installed application."""
        params = ["-s", system] if system else []
        command = self._application_command("list", *params)

        json_data = self._execute_and_parse(command)
        if not isinstance(json_data, dict) or "available" not in json_data:
            raise Exception("Unable to get application information")

        return cast(List[str], json_data["available"])

    def is_application_installed(self, application: str, system: str) -> bool:
        """Return true if requested application installed."""
        return application in self.get_installed_applications(system)

    def is_system_installed(self, system: str) -> bool:
        """Return true if requested system installed."""
        return system in self.get_installed_systems()

    def run_application(self, execution_params: ExecutionParams) -> RunningCommand:
        """Run requested application."""
        command = self._aiet_command(
            "application",
            "run",
            "-n",
            execution_params.application,
            "-s",
            execution_params.system,
            *self._params("-p", execution_params.application_params),
            *self._params("--system-param", execution_params.system_params),
            *self._params("--deploy", execution_params.deploy_params),
        )

        return self._submit(command)

    @staticmethod
    def _params(name: str, params: List[str]) -> List[str]:
        return [p for item in [(name, param) for param in params] for p in item]

    def _application_command(self, cmd: str, *params: str) -> List[str]:
        return self._aiet_command("application", "-f", "json", cmd, *params)

    def _system_command(self, cmd: str, *params: str) -> List[str]:
        return self._aiet_command("system", "-f", "json", cmd, *params)

    @staticmethod
    def _aiet_command(subcommand: str, *params: str) -> List[str]:
        return ["aiet", subcommand, *params]

    def _execute(self, command: List[str]) -> Tuple[int, bytes, bytes]:
        logger.debug("Execute command %s", " ".join(command))
        return self.executor.execute(command)

    def _submit(self, command: List[str]) -> RunningCommand:
        """Submit command for the execution."""
        logger.debug("Submit command %s", " ".join(command))
        return self.executor.submit(command)

    def _execute_and_parse(self, command: List[str]) -> Any:
        """Execute command and parse output."""
        _, stdout, _ = self._execute(command)

        return json.loads(stdout)


def save_random_input(
    input_shape: Tuple, input_dtype: Union[np.dtype, Any], file: str
) -> None:
    """Generate random input."""
    if not isinstance(input_dtype, np.dtype):
        input_dtype = np.dtype(input_dtype)

    random_input = os.urandom(input_dtype.itemsize * np.prod(input_shape))
    with open(file, "wb") as input_file:
        input_file.write(random_input)


class GenericInferenceRunner(ABC):
    """Abstract class for generic inference runner."""

    def __init__(self, aiet_runner: AIETRunner):
        """Init generic inference runner instance."""
        self.aiet_runner = aiet_runner
        self.running_inference: Optional[RunningCommand] = None
        self.context_stack = ExitStack()

    def run(
        self,
        device: EthosUConfiguration,
        model: TFLiteModel,
        output_consumers: List[OutputConsumer],
    ) -> None:
        """Run generic inference for the provided device/model."""
        self.check_system_and_application(self.system, self.application)

        with self.context_stack:
            execution_params = self.get_execution_params(device, model)

            self.running_inference = self.aiet_runner.run_application(execution_params)
            self.running_inference.output_consumers = output_consumers
            self.running_inference.consume_output()

    def stop(self) -> None:
        """Stop running inference."""
        if self.running_inference is None:
            return

        self.running_inference.stop()

    @abstractmethod
    def get_execution_params(
        self, device: EthosUConfiguration, model: TFLiteModel
    ) -> ExecutionParams:
        """Get execution params for the provided device."""

    @property
    def application(self) -> str:
        """Return AIET application name."""
        return "generic_inference"

    @property
    @abstractmethod
    def system(self) -> str:
        """Return AIET system name."""

    def __enter__(self) -> "GenericInferenceRunner":
        """Enter context."""
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        """Exit context."""
        self.stop()

    def check_system_and_application(
        self, system_name: str, application_name: str
    ) -> None:
        """Check if requested system and application installed."""
        if not self.aiet_runner.is_system_installed(system_name):
            raise Exception(f"System {system_name} is not installed")

        if not self.aiet_runner.is_application_installed(application_name, system_name):
            raise Exception(
                f"Application {application_name} for the system {system_name} "
                "is not installed"
            )


class GenericInferenceRunnerEthosU(GenericInferenceRunner):
    """Generic inference runner on U55/65."""

    def __init__(self, system_name: str, aietrunner: AIETRunner):
        """Init generic inference runner instance."""
        self.system_name = system_name
        super().__init__(aietrunner)

    @property
    def system(self) -> str:
        """Return AIET system name."""
        return self.system_name

    def get_execution_params(
        self, device: EthosUConfiguration, model: TFLiteModel
    ) -> ExecutionParams:
        """Get execution params for Ethos-U55/65."""
        application_params = [f"input_file={Path(model.model_path).absolute()}"]
        system_params = [f"mac={device.mac}"]

        return ExecutionParams(
            self.application, self.system, application_params, system_params, []
        )


def get_generic_runner(
    device: EthosUConfiguration, aiet_runner: AIETRunner
) -> GenericInferenceRunner:
    """Get generic runner for provided device."""
    if isinstance(device, EthosU55):
        return GenericInferenceRunnerEthosU("CS-300: Cortex-M55+Ethos-U55", aiet_runner)

    if isinstance(device, EthosU65):
        return GenericInferenceRunnerEthosU("CS-300: Cortex-M55+Ethos-U65", aiet_runner)

    raise Exception(f"Unsupported device {device}")


def estimate_performance(
    model: TFLiteModel, device: EthosUConfiguration
) -> PerformanceMetrics:
    """Get performance estimations."""
    with get_generic_runner(device, get_aiet_runner()) as generic_runner:
        output_parser = GenericInferenceOutputParser()
        generic_runner.run(device, model, [output_parser, AIETLogWriter()])

        if output_parser.is_ready():
            return PerformanceMetrics(**output_parser.result)

        missed_data = ",".join(output_parser.missed_keys())
        raise Exception(f"Unable to get performance metrics, missed data {missed_data}")


def get_aiet_runner() -> AIETRunner:
    """Return AIET runner."""
    executor = CommandExecutor()
    return AIETRunner(executor)
