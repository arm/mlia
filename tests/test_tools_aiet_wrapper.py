# Copyright 2021, Arm Ltd.
"""Tests for module tools/aiet_wrapper."""
# pylint: disable=no-self-use,too-many-arguments
from contextlib import ExitStack as does_not_raise
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Tuple
from unittest.mock import MagicMock
from unittest.mock import PropertyMock

import numpy as np
import pytest
from mlia.config import EthosU55
from mlia.config import EthosU65
from mlia.config import EthosUConfiguration
from mlia.config import TFLiteModel
from mlia.tools.aiet_wrapper import AIETRunner
from mlia.tools.aiet_wrapper import estimate_performance
from mlia.tools.aiet_wrapper import ExecutionParams
from mlia.tools.aiet_wrapper import GenericInferenceOutputParser
from mlia.tools.aiet_wrapper import get_aiet_runner
from mlia.tools.aiet_wrapper import PerformanceMetrics
from mlia.tools.aiet_wrapper import save_random_input
from mlia.utils.proc import RunningCommand


@pytest.mark.parametrize(
    "data, is_ready, result, missed_keys",
    [
        (
            [],
            False,
            {},
            [
                "npu_active_cycles",
                "npu_axi0_rd_data_beat_received",
                "npu_axi0_wr_data_beat_written",
                "npu_axi1_rd_data_beat_received",
                "npu_idle_cycles",
                "npu_total_cycles",
            ],
        ),
        (
            ["sample text"],
            False,
            {},
            [
                "npu_active_cycles",
                "npu_axi0_rd_data_beat_received",
                "npu_axi0_wr_data_beat_written",
                "npu_axi1_rd_data_beat_received",
                "npu_idle_cycles",
                "npu_total_cycles",
            ],
        ),
        (
            [
                ["NPU AXI0_RD_DATA_BEAT_RECEIVED beats: 123"],
                False,
                {"npu_axi0_rd_data_beat_received": 123},
                [
                    "npu_active_cycles",
                    "npu_axi0_wr_data_beat_written",
                    "npu_axi1_rd_data_beat_received",
                    "npu_idle_cycles",
                    "npu_total_cycles",
                ],
            ]
        ),
        (
            [
                "NPU AXI0_RD_DATA_BEAT_RECEIVED beats: 1",
                "NPU AXI0_WR_DATA_BEAT_WRITTEN beats: 2",
                "NPU AXI1_RD_DATA_BEAT_RECEIVED beats: 3",
                "NPU ACTIVE cycles: 4",
                "NPU IDLE cycles: 5",
                "NPU TOTAL cycles: 6",
            ],
            True,
            {
                "npu_axi0_rd_data_beat_received": 1,
                "npu_axi0_wr_data_beat_written": 2,
                "npu_axi1_rd_data_beat_received": 3,
                "npu_active_cycles": 4,
                "npu_idle_cycles": 5,
                "npu_total_cycles": 6,
            },
            [],
        ),
    ],
)
def test_generic_inference_output_parser(
    data: List[str], is_ready: bool, result: Dict, missed_keys: List[str]
) -> None:
    """Test generic runner output parser."""
    parser = GenericInferenceOutputParser()

    for line in data:
        parser.feed(line)

    assert parser.is_ready() == is_ready
    assert parser.result == result
    assert parser.missed_keys() == missed_keys


class TestAIETRunner:
    """Tests for  AIETRunner class."""

    @pytest.mark.parametrize(
        "json_output, system, installed, expected_error",
        [
            ('{"available":[]}', "system1", False, does_not_raise()),
            ('{"available":["system1", "system2"]}', "system1", True, does_not_raise()),
            (
                "{}",
                "system1",
                False,
                pytest.raises(Exception, match="Unable to get system information"),
            ),
        ],
    )
    def test_is_system_installed(
        self,
        json_output: str,
        system: str,
        installed: bool,
        expected_error: Any,
    ) -> None:
        """Test method is_system_installed."""
        mock_executor = MagicMock()
        mock_executor.execute.return_value = (0, bytes(json_output.encode()), bytes())

        with expected_error:
            aiet_runner = AIETRunner(mock_executor)
            assert installed == aiet_runner.is_system_installed(system)

        mock_executor.execute.assert_called_once_with(
            ["aiet", "system", "-f", "json", "list"]
        )

    @pytest.mark.parametrize(
        "json_output, systems, expected_error",
        [
            ('{"available":[]}', [], does_not_raise()),
            ('{"available":["system1"]}', ["system1"], does_not_raise()),
            (
                "{}",
                "system1",
                pytest.raises(Exception, match="Unable to get system information"),
            ),
        ],
    )
    def test_installed_systems(
        self,
        json_output: str,
        systems: str,
        expected_error: Any,
    ) -> None:
        """Test method installed_systems."""
        mock_executor = MagicMock()
        mock_executor.execute.return_value = (0, bytes(json_output.encode()), bytes())

        with expected_error:
            aiet_runner = AIETRunner(mock_executor)
            assert systems == aiet_runner.get_installed_systems()

        mock_executor.execute.assert_called_once_with(
            ["aiet", "system", "-f", "json", "list"]
        )

    @pytest.mark.parametrize(
        "json_output, applications, expected_error",
        [
            ('{"available":[]}', [], does_not_raise()),
            (
                '{"available":["application1", "application2"]}',
                ["application1", "application2"],
                does_not_raise(),
            ),
            (
                "{}",
                [],
                pytest.raises(Exception, match="Unable to get application information"),
            ),
        ],
    )
    def test_get_installed_applications(
        self, json_output: str, applications: List[str], expected_error: Any
    ) -> None:
        """Test method get_installed_applications."""
        mock_executor = MagicMock()
        mock_executor.execute.return_value = (0, bytes(json_output.encode()), bytes())

        with expected_error:
            aiet_runner = AIETRunner(mock_executor)
            assert applications == aiet_runner.get_installed_applications()

        mock_executor.execute.assert_called_once_with(
            ["aiet", "application", "-f", "json", "list"]
        )

    @pytest.mark.parametrize(
        "json_output, application, installed, expected_error",
        [
            ('{"available":[]}', "system1", False, does_not_raise()),
            (
                '{"available":["application1", "application2"]}',
                "application1",
                True,
                does_not_raise(),
            ),
            (
                "{}",
                "application1",
                False,
                pytest.raises(Exception, match="Unable to get application information"),
            ),
        ],
    )
    def test_is_application_installed(
        self, json_output: str, application: str, installed: bool, expected_error: Any
    ) -> None:
        """Test method is_application_installed."""
        mock_executor = MagicMock()
        mock_executor.execute.return_value = (0, bytes(json_output.encode()), bytes())

        with expected_error:
            aiet_runner = AIETRunner(mock_executor)
            assert installed == aiet_runner.is_application_installed(
                application, "system1"
            )

        mock_executor.execute.assert_called_once_with(
            ["aiet", "application", "-f", "json", "list", "-s", "system1"]
        )

    @pytest.mark.parametrize(
        "execution_params, expected_command",
        [
            (
                ExecutionParams("application1", "system1", [], [], []),
                ["aiet", "application", "run", "-n", "application1", "-s", "system1"],
            ),
            (
                ExecutionParams(
                    "application1",
                    "system1",
                    ["input_file=123.txt", "size=777"],
                    ["param1=456", "param2=789"],
                    ["source1.txt:dest1.txt", "source2.txt:dest2.txt"],
                ),
                [
                    "aiet",
                    "application",
                    "run",
                    "-n",
                    "application1",
                    "-s",
                    "system1",
                    "-p",
                    "input_file=123.txt",
                    "-p",
                    "size=777",
                    "--system-param",
                    "param1=456",
                    "--system-param",
                    "param2=789",
                    "--deploy",
                    "source1.txt:dest1.txt",
                    "--deploy",
                    "source2.txt:dest2.txt",
                ],
            ),
        ],
    )
    def test_run_application(
        self, execution_params: ExecutionParams, expected_command: List[str]
    ) -> None:
        """Test method run_application."""
        mock_executor = MagicMock()
        mock_running_command = MagicMock()
        mock_executor.submit.return_value = mock_running_command

        aiet_runner = AIETRunner(mock_executor)
        aiet_runner.run_application(execution_params)

        mock_executor.submit.assert_called_once_with(expected_command)


@pytest.mark.parametrize(
    "input_shape, input_dtype, expected_size",
    [
        ((1, 3), np.dtype(np.uint8), 3),
        ((1, 3), np.dtype(np.float32), 12),
        ((1, 10, 2, 2), np.int64, 320),
    ],
)
def test_save_random_input(
    input_shape: Tuple, input_dtype: np.dtype, expected_size: int, tmpdir: Any
) -> None:
    """Test getting random input."""
    tmp_file = Path(tmpdir) / "input.frm"
    save_random_input(input_shape, input_dtype, str(tmp_file))

    assert tmp_file.is_file()
    assert tmp_file.stat().st_size == expected_size


@pytest.mark.parametrize(
    "device, system, application, expected_error",
    [
        (
            EthosU55(mac=32),
            ("CS-300: Cortex-M55+Ethos-U55", True),
            ("generic_inference", True),
            does_not_raise(),
        ),
        (
            EthosU55(mac=32),
            ("CS-300: Cortex-M55+Ethos-U55", False),
            ("generic_inference", False),
            pytest.raises(
                Exception,
                match=r"System CS-300: Cortex-M55\+Ethos-U55 is not installed",
            ),
        ),
        (
            EthosU55(mac=32),
            ("CS-300: Cortex-M55+Ethos-U55", True),
            ("generic_inference", False),
            pytest.raises(
                Exception,
                match=r"Application generic_inference for the system "
                r"CS-300: Cortex-M55\+Ethos-U55 is not installed",
            ),
        ),
        (
            EthosU65(mac=512),
            ("SGM-775", True),
            ("generic_inference", True),
            does_not_raise(),
        ),
        (
            EthosU65(mac=512),
            ("SGM-775", False),
            ("generic_inference", False),
            pytest.raises(Exception, match="System SGM-775 is not installed"),
        ),
        (
            EthosU65(mac=512),
            ("SGM-775", True),
            ("generic_inference", False),
            pytest.raises(
                Exception,
                match="Application generic_inference for the system SGM-775"
                " is not installed",
            ),
        ),
        (
            "unknown_device",
            ("some_system", False),
            ("some_application", False),
            pytest.raises(Exception, match="Unsupported device unknown_device"),
        ),
    ],
)
def test_estimate_performance(
    device: EthosUConfiguration,
    system: Tuple[str, bool],
    application: Tuple[str, bool],
    test_models_path: Path,
    expected_error: Any,
    monkeypatch: Any,
) -> None:
    """Test getting performance estimations."""
    system_name, system_installed = system
    application_name, application_installed = application

    mock_aiet_runner = MagicMock()
    mock_aiet_runner.is_system_installed.return_value = system_installed
    mock_aiet_runner.is_application_installed.return_value = application_installed

    mock_process = create_mock_process(
        [
            "NPU AXI0_RD_DATA_BEAT_RECEIVED beats: 1",
            "NPU AXI0_WR_DATA_BEAT_WRITTEN beats: 2",
            "NPU AXI1_RD_DATA_BEAT_RECEIVED beats: 3",
            "NPU ACTIVE cycles: 4",
            "NPU IDLE cycles: 5",
            "NPU TOTAL cycles: 6",
        ],
        [],
    )

    mock_generic_inference_run = RunningCommand(mock_process)
    mock_aiet_runner.run_application.return_value = mock_generic_inference_run

    monkeypatch.setattr(
        "mlia.tools.aiet_wrapper.get_aiet_runner",
        MagicMock(return_value=mock_aiet_runner),
    )

    with expected_error:
        model = test_models_path / "simple_3_layers_model.tflite"
        perf_metrics = estimate_performance(TFLiteModel(str(model)), device)

        assert isinstance(perf_metrics, PerformanceMetrics)
        assert perf_metrics.npu_axi0_rd_data_beat_received == 1
        assert perf_metrics.npu_axi0_wr_data_beat_written == 2
        assert perf_metrics.npu_axi1_rd_data_beat_received == 3
        assert perf_metrics.npu_active_cycles == 4
        assert perf_metrics.npu_idle_cycles == 5
        assert perf_metrics.npu_total_cycles == 6

        assert mock_aiet_runner.is_system_installed.called_once_with(system_name)
        assert mock_aiet_runner.is_application_installed.called_once_with(
            application_name, system_name
        )


def test_estimate_performance_invalid_output(
    test_models_path: Path, monkeypatch: Any
) -> None:
    """Test estimation could not be done if inference produces unexpected output."""
    mock_aiet_runner = MagicMock()
    mock_aiet_runner.is_system_installed.return_value = True
    mock_aiet_runner.is_application_installed.return_value = True

    mock_process = create_mock_process(
        ["Something", "is", "wrong"], ["What a nice error!"]
    )
    mock_aiet_runner.run_application.return_value = RunningCommand(mock_process)

    monkeypatch.setattr(
        "mlia.tools.aiet_wrapper.get_aiet_runner",
        MagicMock(return_value=mock_aiet_runner),
    )

    with pytest.raises(Exception, match="Unable to get performance metrics"):
        model = test_models_path / "simple_3_layers_model.tflite"
        estimate_performance(TFLiteModel(str(model)), EthosU55())


def test_get_aiet_runner() -> None:
    """Test getting aiet runner."""
    aiet_runner = get_aiet_runner()
    assert isinstance(aiet_runner, AIETRunner)


def create_mock_process(stdout: List[str], stderr: List[str]) -> MagicMock:
    """Mock underlying process."""
    mock_process = MagicMock()
    mock_process.poll.return_value = 0
    type(mock_process).stdout = PropertyMock(return_value=iter(stdout))
    type(mock_process).stderr = PropertyMock(return_value=iter(stderr))
    return mock_process
