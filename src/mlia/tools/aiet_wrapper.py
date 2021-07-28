# Copyright 2021, Arm Ltd.
"""Module for AIET integration."""
import json
import logging
import os
import re
from abc import ABC
from abc import abstractmethod
from contextlib import ExitStack
from pathlib import Path
from typing import Any
from typing import cast
from typing import Dict
from typing import List
from typing import NamedTuple
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
from mlia.config import EthosU55
from mlia.config import EthosU65
from mlia.config import EthosUConfiguration
from mlia.config import TFLiteModel
from mlia.utils.filesystem import temp_file
from mlia.utils.proc import CommandExecutor
from mlia.utils.proc import OutputConsumer
from mlia.utils.proc import RunningCommand


LOGGER = logging.getLogger("mlia.tools.aiet")


class PerformanceMetrics:
    """Performance metrics parsed from generic inference output."""

    def __init__(
        self,
        npu_axi0_rd_data_beat_received: int,
        npu_axi0_wr_data_beat_written: int,
        npu_axi1_rd_data_beat_received: int,
        npu_active_cycles: int,
        npu_idle_cycles: int,
        npu_total_cycles: int,
    ):
        """Init performance metrics instance."""
        self.npu_axi0_rd_data_beat_received = npu_axi0_rd_data_beat_received
        self.npu_axi0_wr_data_beat_written = npu_axi0_wr_data_beat_written
        self.npu_axi1_rd_data_beat_received = npu_axi1_rd_data_beat_received
        self.npu_active_cycles = npu_active_cycles
        self.npu_idle_cycles = npu_idle_cycles
        self.npu_total_cycles = npu_total_cycles


class ExecutionParams(NamedTuple):
    """Software execution params."""

    software: str
    system: str
    software_params: List[str]
    system_params: List[str]
    deploy_params: List[str]


class AIETLogWriter(OutputConsumer):
    """Redirect AIET command output to the logger."""

    def feed(self, line: str) -> None:
        """Process line from the output."""
        LOGGER.info(line.strip())


class GenericInferenceOutputParser(OutputConsumer):
    """Generic inference app output parser."""

    PATTERNS = {
        name: re.compile(pattern, re.IGNORECASE)
        for name, pattern in {
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
            ("npu_active_cycles", r"NPU ACTIVE cycles: (?P<value>\d+)"),
            ("npu_idle_cycles", r"NPU IDLE cycles: (?P<value>\d+)"),
            ("npu_total_cycles", r"NPU TOTAL cycles: (?P<value>\d+)"),
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

    def get_installed_software(self, system: Optional[str] = None) -> List[str]:
        """Get list of the installed software."""
        params = ["-s", system] if system else []
        command = self._software_command("list", *params)

        json_data = self._execute_and_parse(command)
        if not isinstance(json_data, dict) or "available" not in json_data:
            raise Exception("Unable to get software information")

        return cast(List[str], json_data["available"])

    def is_software_installed(self, software: str, system: str) -> bool:
        """Return true if requested software installed."""
        return software in self.get_installed_software(system)

    def is_system_installed(self, system: str) -> bool:
        """Return true if requested system installed."""
        return system in self.get_installed_systems()

    def run_software(self, execution_params: ExecutionParams) -> RunningCommand:
        """Run requested software."""
        command = self._aiet_command(
            "software",
            "run",
            "-n",
            execution_params.software,
            "-s",
            execution_params.system,
            *self._params("-p", execution_params.software_params),
            *self._params("--system-param", execution_params.system_params),
            *self._params("--deploy", execution_params.deploy_params),
        )

        return self._submit(command)

    @staticmethod
    def _params(name: str, params: List[str]) -> List[str]:
        return [p for item in [(name, param) for param in params] for p in item]

    def _software_command(self, cmd: str, *params: str) -> List[str]:
        return self._aiet_command("software", "-f", "json", cmd, *params)

    def _system_command(self, cmd: str, *params: str) -> List[str]:
        return self._aiet_command("system", "-f", "json", cmd, *params)

    @staticmethod
    def _aiet_command(subcommand: str, *params: str) -> List[str]:
        return ["aiet", subcommand] + [p for p in params]

    def _execute(self, command: List[str]) -> Tuple[int, bytes, bytes]:
        LOGGER.debug(f"Execute command {' '.join(command)}")
        return self.executor.execute(command)

    def _submit(self, command: List[str]) -> RunningCommand:
        """Submit command for the execution."""
        LOGGER.debug(f"Submit command {' '.join(command)}")
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
    with open(file, "wb") as f:
        f.write(random_input)


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
        self.check_system_and_software(self.system, self.software)

        with self.context_stack:
            execution_params = self.get_execution_params(device, model)

            self.running_inference = self.aiet_runner.run_software(execution_params)
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
    def software(self) -> str:
        """Return AIET software name."""
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

    def check_system_and_software(self, system_name: str, software_name: str) -> None:
        """Check if requested system and software installed."""
        if not self.aiet_runner.is_system_installed(system_name):
            raise Exception(f"System {system_name} is not installed")

        if not self.aiet_runner.is_software_installed(software_name, system_name):
            raise Exception(
                f"Software {software_name} for the system {system_name} "
                "is not installed"
            )


class GenericInferenceRunnerU65(GenericInferenceRunner):
    """Generic inference runner on U65."""

    @property
    def system(self) -> str:
        """Return AIET system name."""
        return "SGM-775"

    def get_execution_params(
        self, device: EthosUConfiguration, model: TFLiteModel
    ) -> ExecutionParams:
        """Get execution params for Ethos-U65."""
        model_file, input_file = "/tmp/model.tflite", "/tmp/input.ifm"
        software_params = [f"model_file={model_file}", f"input_file={input_file}"]

        input = model.input_details()
        if not input:
            raise Exception(
                f"Unable to get input details for the model {model.model_path}"
            )

        random_input_file_path: str = self.context_stack.enter_context(temp_file())
        LOGGER.debug(f"Save random input into {random_input_file_path}")
        save_random_input(input[0]["shape"], input[0]["dtype"], random_input_file_path)

        deploy_params = [
            f"{Path(model.model_path).absolute()}:{model_file}",
            f"{random_input_file_path}:{input_file}",
        ]
        system_params = [f"-c=Y{device.mac}"]

        return ExecutionParams(
            self.software, self.system, software_params, system_params, deploy_params
        )


class GenericInferenceRunnerU55(GenericInferenceRunner):
    """Generic inference runner on U55."""

    @property
    def system(self) -> str:
        """Return AIET system name."""
        return "CS-300: Cortex-M55+Ethos-U55"

    def get_execution_params(
        self, device: EthosUConfiguration, model: TFLiteModel
    ) -> ExecutionParams:
        """Get execution params for Ethous-U55."""
        software_params = [f"input_file={Path(model.model_path).absolute()}"]
        system_params = [f"mac={device.mac}"]

        return ExecutionParams(
            self.software, self.system, software_params, system_params, []
        )


def get_generic_runner(
    device: EthosUConfiguration, aiet_runner: AIETRunner
) -> GenericInferenceRunner:
    """Get generic runner for provided device."""
    if isinstance(device, EthosU55):
        return GenericInferenceRunnerU55(aiet_runner)

    if isinstance(device, EthosU65):
        return GenericInferenceRunnerU65(aiet_runner)

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
