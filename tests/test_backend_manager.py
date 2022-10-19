# SPDX-FileCopyrightText: Copyright 2022, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for module backend/manager."""
from __future__ import annotations

import base64
import json
from contextlib import ExitStack as does_not_raise
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock
from unittest.mock import PropertyMock

import pytest

from mlia.backend.application import get_application
from mlia.backend.execution import ExecutionContext
from mlia.backend.manager import BackendRunner
from mlia.backend.manager import DeviceInfo
from mlia.backend.manager import estimate_performance
from mlia.backend.manager import ExecutionParams
from mlia.backend.manager import GenericInferenceOutputParser
from mlia.backend.manager import GenericInferenceRunnerEthosU
from mlia.backend.manager import get_generic_runner
from mlia.backend.manager import get_system_name
from mlia.backend.manager import is_supported
from mlia.backend.manager import ModelInfo
from mlia.backend.manager import PerformanceMetrics
from mlia.backend.manager import supported_backends
from mlia.backend.output_consumer import Base64OutputConsumer
from mlia.backend.system import get_system


def _mock_encode_b64(data: dict[str, int]) -> str:
    """
    Encode the given data into a mock base64-encoded string of JSON.

    This reproduces the base64 encoding done in the Corstone applications.

    JSON example:

    ```json
    [{'count': 1,
        'profiling_group': 'Inference',
        'samples': [{'name': 'NPU IDLE', 'value': [612]},
                    {'name': 'NPU AXI0_RD_DATA_BEAT_RECEIVED', 'value': [165872]},
                    {'name': 'NPU AXI0_WR_DATA_BEAT_WRITTEN', 'value': [88712]},
                    {'name': 'NPU AXI1_RD_DATA_BEAT_RECEIVED', 'value': [57540]},
                    {'name': 'NPU ACTIVE', 'value': [520489]},
                    {'name': 'NPU TOTAL', 'value': [521101]}]}]
    ```
    """
    wrapped_data = [
        {
            "count": 1,
            "profiling_group": "Inference",
            "samples": [
                {"name": name, "value": [value]} for name, value in data.items()
            ],
        }
    ]
    json_str = json.dumps(wrapped_data)
    json_bytes = bytearray(json_str, encoding="utf-8")
    json_b64 = base64.b64encode(json_bytes).decode("utf-8")
    tag = Base64OutputConsumer.TAG_NAME
    return f"<{tag}>{json_b64}</{tag}>"


@pytest.mark.parametrize(
    "data, is_ready, result, missed_keys",
    [
        (
            [],
            False,
            {},
            {
                "npu_active_cycles",
                "npu_axi0_rd_data_beat_received",
                "npu_axi0_wr_data_beat_written",
                "npu_axi1_rd_data_beat_received",
                "npu_idle_cycles",
                "npu_total_cycles",
            },
        ),
        (
            ["sample text"],
            False,
            {},
            {
                "npu_active_cycles",
                "npu_axi0_rd_data_beat_received",
                "npu_axi0_wr_data_beat_written",
                "npu_axi1_rd_data_beat_received",
                "npu_idle_cycles",
                "npu_total_cycles",
            },
        ),
        (
            [_mock_encode_b64({"NPU AXI0_RD_DATA_BEAT_RECEIVED": 123})],
            False,
            {"npu_axi0_rd_data_beat_received": 123},
            {
                "npu_active_cycles",
                "npu_axi0_wr_data_beat_written",
                "npu_axi1_rd_data_beat_received",
                "npu_idle_cycles",
                "npu_total_cycles",
            },
        ),
        (
            [
                _mock_encode_b64(
                    {
                        "NPU AXI0_RD_DATA_BEAT_RECEIVED": 1,
                        "NPU AXI0_WR_DATA_BEAT_WRITTEN": 2,
                        "NPU AXI1_RD_DATA_BEAT_RECEIVED": 3,
                        "NPU ACTIVE": 4,
                        "NPU IDLE": 5,
                        "NPU TOTAL": 6,
                    }
                )
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
            set(),
        ),
    ],
)
def test_generic_inference_output_parser(
    data: dict[str, int], is_ready: bool, result: dict, missed_keys: set[str]
) -> None:
    """Test generic runner output parser."""
    parser = GenericInferenceOutputParser()

    for line in data:
        parser.feed(line)

    assert parser.is_ready() == is_ready
    assert parser.result == result
    assert parser.missed_keys() == missed_keys


class TestBackendRunner:
    """Tests for BackendRunner class."""

    @staticmethod
    def _setup_backends(
        monkeypatch: pytest.MonkeyPatch,
        available_systems: list[str] | None = None,
        available_apps: list[str] | None = None,
    ) -> None:
        """Set up backend metadata."""

        def mock_system(system: str) -> MagicMock:
            """Mock the System instance."""
            mock = MagicMock()
            type(mock).name = PropertyMock(return_value=system)
            return mock

        def mock_app(app: str) -> MagicMock:
            """Mock the Application instance."""
            mock = MagicMock()
            type(mock).name = PropertyMock(return_value=app)
            mock.can_run_on.return_value = True
            return mock

        system_mocks = [mock_system(name) for name in (available_systems or [])]
        monkeypatch.setattr(
            "mlia.backend.manager.get_available_systems",
            MagicMock(return_value=system_mocks),
        )

        apps_mock = [mock_app(name) for name in (available_apps or [])]
        monkeypatch.setattr(
            "mlia.backend.manager.get_available_applications",
            MagicMock(return_value=apps_mock),
        )

    @pytest.mark.parametrize(
        "available_systems, system, installed",
        [
            ([], "system1", False),
            (["system1", "system2"], "system1", True),
        ],
    )
    def test_is_system_installed(
        self,
        available_systems: list,
        system: str,
        installed: bool,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test method is_system_installed."""
        backend_runner = BackendRunner()

        self._setup_backends(monkeypatch, available_systems)

        assert backend_runner.is_system_installed(system) == installed

    @pytest.mark.parametrize(
        "available_systems, systems",
        [
            ([], []),
            (["system1"], ["system1"]),
        ],
    )
    def test_installed_systems(
        self,
        available_systems: list[str],
        systems: list[str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test method installed_systems."""
        backend_runner = BackendRunner()

        self._setup_backends(monkeypatch, available_systems)
        assert backend_runner.get_installed_systems() == systems

    @staticmethod
    def test_install_system(monkeypatch: pytest.MonkeyPatch) -> None:
        """Test system installation."""
        install_system_mock = MagicMock()
        monkeypatch.setattr("mlia.backend.manager.install_system", install_system_mock)

        backend_runner = BackendRunner()
        backend_runner.install_system(Path("test_system_path"))

        install_system_mock.assert_called_once_with(Path("test_system_path"))

    @pytest.mark.parametrize(
        "available_systems, systems, expected_result",
        [
            ([], [], False),
            (["system1"], [], False),
            (["system1"], ["system1"], True),
            (["system1", "system2"], ["system1", "system3"], False),
            (["system1", "system2"], ["system1", "system2"], True),
        ],
    )
    def test_systems_installed(
        self,
        available_systems: list[str],
        systems: list[str],
        expected_result: bool,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test method systems_installed."""
        self._setup_backends(monkeypatch, available_systems)

        backend_runner = BackendRunner()

        assert backend_runner.systems_installed(systems) is expected_result

    @pytest.mark.parametrize(
        "available_apps, applications, expected_result",
        [
            ([], [], False),
            (["app1"], [], False),
            (["app1"], ["app1"], True),
            (["app1", "app2"], ["app1", "app3"], False),
            (["app1", "app2"], ["app1", "app2"], True),
        ],
    )
    def test_applications_installed(
        self,
        available_apps: list[str],
        applications: list[str],
        expected_result: bool,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test method applications_installed."""
        self._setup_backends(monkeypatch, [], available_apps)
        backend_runner = BackendRunner()

        assert backend_runner.applications_installed(applications) is expected_result

    @pytest.mark.parametrize(
        "available_apps, applications",
        [
            ([], []),
            (
                ["application1", "application2"],
                ["application1", "application2"],
            ),
        ],
    )
    def test_get_installed_applications(
        self,
        available_apps: list[str],
        applications: list[str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test method get_installed_applications."""
        self._setup_backends(monkeypatch, [], available_apps)

        backend_runner = BackendRunner()
        assert applications == backend_runner.get_installed_applications()

    @staticmethod
    def test_install_application(monkeypatch: pytest.MonkeyPatch) -> None:
        """Test application installation."""
        mock_install_application = MagicMock()
        monkeypatch.setattr(
            "mlia.backend.manager.install_application", mock_install_application
        )

        backend_runner = BackendRunner()
        backend_runner.install_application(Path("test_application_path"))
        mock_install_application.assert_called_once_with(Path("test_application_path"))

    @pytest.mark.parametrize(
        "available_apps, application, installed",
        [
            ([], "system1", False),
            (
                ["application1", "application2"],
                "application1",
                True,
            ),
            (
                [],
                "application1",
                False,
            ),
        ],
    )
    def test_is_application_installed(
        self,
        available_apps: list[str],
        application: str,
        installed: bool,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test method is_application_installed."""
        self._setup_backends(monkeypatch, [], available_apps)

        backend_runner = BackendRunner()
        assert installed == backend_runner.is_application_installed(
            application, "system1"
        )

    @staticmethod
    @pytest.mark.parametrize(
        "execution_params, expected_command",
        [
            (
                ExecutionParams("application_4", "System 4", [], []),
                ["application_4", [], "System 4", []],
            ),
            (
                ExecutionParams(
                    "application_6",
                    "System 6",
                    ["param1=value2"],
                    ["sys-param1=value2"],
                ),
                [
                    "application_6",
                    ["param1=value2"],
                    "System 6",
                    ["sys-param1=value2"],
                ],
            ),
        ],
    )
    def test_run_application_local(
        monkeypatch: pytest.MonkeyPatch,
        execution_params: ExecutionParams,
        expected_command: list[str],
    ) -> None:
        """Test method run_application with local systems."""
        run_app = MagicMock()
        monkeypatch.setattr("mlia.backend.manager.run_application", run_app)

        backend_runner = BackendRunner()
        backend_runner.run_application(execution_params)

        run_app.assert_called_once_with(*expected_command)


@pytest.mark.parametrize(
    "device, system, application, backend, expected_error",
    [
        (
            DeviceInfo(device_type="ethos-u55", mac=32),
            ("Corstone-300: Cortex-M55+Ethos-U55", True),
            ("Generic Inference Runner: Ethos-U55", True),
            "Corstone-300",
            does_not_raise(),
        ),
        (
            DeviceInfo(device_type="ethos-u55", mac=32),
            ("Corstone-300: Cortex-M55+Ethos-U55", False),
            ("Generic Inference Runner: Ethos-U55", False),
            "Corstone-300",
            pytest.raises(
                Exception,
                match=r"System Corstone-300: Cortex-M55\+Ethos-U55 is not installed",
            ),
        ),
        (
            DeviceInfo(device_type="ethos-u55", mac=32),
            ("Corstone-300: Cortex-M55+Ethos-U55", True),
            ("Generic Inference Runner: Ethos-U55", False),
            "Corstone-300",
            pytest.raises(
                Exception,
                match=r"Application Generic Inference Runner: Ethos-U55 "
                r"for the system Corstone-300: Cortex-M55\+Ethos-U55 is not installed",
            ),
        ),
        (
            DeviceInfo(device_type="ethos-u55", mac=32),
            ("Corstone-310: Cortex-M85+Ethos-U55", True),
            ("Generic Inference Runner: Ethos-U55", True),
            "Corstone-310",
            does_not_raise(),
        ),
        (
            DeviceInfo(device_type="ethos-u55", mac=32),
            ("Corstone-310: Cortex-M85+Ethos-U55", False),
            ("Generic Inference Runner: Ethos-U55", False),
            "Corstone-310",
            pytest.raises(
                Exception,
                match=r"System Corstone-310: Cortex-M85\+Ethos-U55 is not installed",
            ),
        ),
        (
            DeviceInfo(device_type="ethos-u55", mac=32),
            ("Corstone-310: Cortex-M85+Ethos-U55", True),
            ("Generic Inference Runner: Ethos-U55", False),
            "Corstone-310",
            pytest.raises(
                Exception,
                match=r"Application Generic Inference Runner: Ethos-U55 "
                r"for the system Corstone-310: Cortex-M85\+Ethos-U55 is not installed",
            ),
        ),
        (
            DeviceInfo(device_type="ethos-u65", mac=512),
            ("Corstone-300: Cortex-M55+Ethos-U65", True),
            ("Generic Inference Runner: Ethos-U65", True),
            "Corstone-300",
            does_not_raise(),
        ),
        (
            DeviceInfo(device_type="ethos-u65", mac=512),
            ("Corstone-300: Cortex-M55+Ethos-U65", False),
            ("Generic Inference Runner: Ethos-U65", False),
            "Corstone-300",
            pytest.raises(
                Exception,
                match=r"System Corstone-300: Cortex-M55\+Ethos-U65 is not installed",
            ),
        ),
        (
            DeviceInfo(device_type="ethos-u65", mac=512),
            ("Corstone-300: Cortex-M55+Ethos-U65", True),
            ("Generic Inference Runner: Ethos-U65", False),
            "Corstone-300",
            pytest.raises(
                Exception,
                match=r"Application Generic Inference Runner: Ethos-U65 "
                r"for the system Corstone-300: Cortex-M55\+Ethos-U65 is not installed",
            ),
        ),
        (
            DeviceInfo(device_type="ethos-u65", mac=512),
            ("Corstone-310: Cortex-M85+Ethos-U65", True),
            ("Generic Inference Runner: Ethos-U65", True),
            "Corstone-310",
            does_not_raise(),
        ),
        (
            DeviceInfo(device_type="ethos-u65", mac=512),
            ("Corstone-310: Cortex-M85+Ethos-U65", False),
            ("Generic Inference Runner: Ethos-U65", False),
            "Corstone-310",
            pytest.raises(
                Exception,
                match=r"System Corstone-310: Cortex-M85\+Ethos-U65 is not installed",
            ),
        ),
        (
            DeviceInfo(device_type="ethos-u65", mac=512),
            ("Corstone-310: Cortex-M85+Ethos-U65", True),
            ("Generic Inference Runner: Ethos-U65", False),
            "Corstone-310",
            pytest.raises(
                Exception,
                match=r"Application Generic Inference Runner: Ethos-U65 "
                r"for the system Corstone-310: Cortex-M85\+Ethos-U65 is not installed",
            ),
        ),
        (
            DeviceInfo(
                device_type="unknown_device",  # type: ignore
                mac=None,  # type: ignore
            ),
            ("some_system", False),
            ("some_application", False),
            "some backend",
            pytest.raises(Exception, match="Unsupported device unknown_device"),
        ),
    ],
)
def test_estimate_performance(
    device: DeviceInfo,
    system: tuple[str, bool],
    application: tuple[str, bool],
    backend: str,
    expected_error: Any,
    test_tflite_model: Path,
    backend_runner: MagicMock,
) -> None:
    """Test getting performance estimations."""
    system_name, system_installed = system
    application_name, application_installed = application

    backend_runner.is_system_installed.return_value = system_installed
    backend_runner.is_application_installed.return_value = application_installed

    mock_context = create_mock_context(
        [
            _mock_encode_b64(
                {
                    "NPU AXI0_RD_DATA_BEAT_RECEIVED": 1,
                    "NPU AXI0_WR_DATA_BEAT_WRITTEN": 2,
                    "NPU AXI1_RD_DATA_BEAT_RECEIVED": 3,
                    "NPU ACTIVE": 4,
                    "NPU IDLE": 5,
                    "NPU TOTAL": 6,
                }
            )
        ]
    )

    backend_runner.run_application.return_value = mock_context

    with expected_error:
        perf_metrics = estimate_performance(
            ModelInfo(test_tflite_model), device, backend
        )

        assert isinstance(perf_metrics, PerformanceMetrics)
        assert perf_metrics == PerformanceMetrics(
            npu_axi0_rd_data_beat_received=1,
            npu_axi0_wr_data_beat_written=2,
            npu_axi1_rd_data_beat_received=3,
            npu_active_cycles=4,
            npu_idle_cycles=5,
            npu_total_cycles=6,
        )

        assert backend_runner.is_system_installed.called_once_with(system_name)
        assert backend_runner.is_application_installed.called_once_with(
            application_name, system_name
        )


@pytest.mark.parametrize("backend", ("Corstone-300", "Corstone-310"))
def test_estimate_performance_insufficient_data(
    backend_runner: MagicMock, test_tflite_model: Path, backend: str
) -> None:
    """Test that performance could not be estimated when not all data presented."""
    backend_runner.is_system_installed.return_value = True
    backend_runner.is_application_installed.return_value = True

    no_total_cycles_output = {
        "NPU AXI0_RD_DATA_BEAT_RECEIVED": 1,
        "NPU AXI0_WR_DATA_BEAT_WRITTEN": 2,
        "NPU AXI1_RD_DATA_BEAT_RECEIVED": 3,
        "NPU ACTIVE": 4,
        "NPU IDLE": 5,
    }
    mock_context = create_mock_context([_mock_encode_b64(no_total_cycles_output)])

    backend_runner.run_application.return_value = mock_context

    with pytest.raises(
        Exception, match="Unable to get performance metrics, insufficient data"
    ):
        device = DeviceInfo(device_type="ethos-u55", mac=32)
        estimate_performance(ModelInfo(test_tflite_model), device, backend)


@pytest.mark.parametrize("backend", ("Corstone-300", "Corstone-310"))
def test_estimate_performance_invalid_output(
    test_tflite_model: Path, backend_runner: MagicMock, backend: str
) -> None:
    """Test estimation could not be done if inference produces unexpected output."""
    backend_runner.is_system_installed.return_value = True
    backend_runner.is_application_installed.return_value = True

    mock_context = create_mock_context(["Something", "is", "wrong"])
    backend_runner.run_application.return_value = mock_context

    with pytest.raises(Exception, match="Unable to get performance metrics"):
        estimate_performance(
            ModelInfo(test_tflite_model),
            DeviceInfo(device_type="ethos-u55", mac=256),
            backend=backend,
        )


def create_mock_process(stdout: list[str], stderr: list[str]) -> MagicMock:
    """Mock underlying process."""
    mock_process = MagicMock()
    mock_process.poll.return_value = 0
    type(mock_process).stdout = PropertyMock(return_value=iter(stdout))
    type(mock_process).stderr = PropertyMock(return_value=iter(stderr))
    return mock_process


def create_mock_context(stdout: list[str]) -> ExecutionContext:
    """Mock ExecutionContext."""
    ctx = ExecutionContext(
        app=get_application("application_1")[0],
        app_params=[],
        system=get_system("System 1"),
        system_params=[],
    )
    ctx.stdout = bytearray("\n".join(stdout).encode("utf-8"))
    return ctx


@pytest.mark.parametrize("backend", ("Corstone-300", "Corstone-310"))
def test_get_generic_runner(backend: str) -> None:
    """Test function get_generic_runner()."""
    device_info = DeviceInfo("ethos-u55", 256)

    runner = get_generic_runner(device_info=device_info, backend=backend)
    assert isinstance(runner, GenericInferenceRunnerEthosU)

    with pytest.raises(RuntimeError):
        get_generic_runner(device_info=device_info, backend="UNKNOWN_BACKEND")


@pytest.mark.parametrize(
    ("backend", "device_type"),
    (
        ("Corstone-300", "ethos-u55"),
        ("Corstone-300", "ethos-u65"),
        ("Corstone-310", "ethos-u55"),
    ),
)
def test_backend_support(backend: str, device_type: str) -> None:
    """Test backend & device support."""
    assert is_supported(backend)
    assert is_supported(backend, device_type)

    assert get_system_name(backend, device_type)

    assert backend in supported_backends()


class TestGenericInferenceRunnerEthosU:
    """Test for the class GenericInferenceRunnerEthosU."""

    @staticmethod
    @pytest.mark.parametrize(
        "device, backend, expected_system, expected_app",
        [
            [
                DeviceInfo("ethos-u55", 256),
                "Corstone-300",
                "Corstone-300: Cortex-M55+Ethos-U55",
                "Generic Inference Runner: Ethos-U55",
            ],
            [
                DeviceInfo("ethos-u65", 256),
                "Corstone-300",
                "Corstone-300: Cortex-M55+Ethos-U65",
                "Generic Inference Runner: Ethos-U65",
            ],
            [
                DeviceInfo("ethos-u55", 256),
                "Corstone-310",
                "Corstone-310: Cortex-M85+Ethos-U55",
                "Generic Inference Runner: Ethos-U55",
            ],
            [
                DeviceInfo("ethos-u65", 256),
                "Corstone-310",
                "Corstone-310: Cortex-M85+Ethos-U65",
                "Generic Inference Runner: Ethos-U65",
            ],
        ],
    )
    def test_artifact_resolver(
        device: DeviceInfo, backend: str, expected_system: str, expected_app: str
    ) -> None:
        """Test artifact resolving based on the provided parameters."""
        generic_runner = get_generic_runner(device, backend)
        assert isinstance(generic_runner, GenericInferenceRunnerEthosU)

        assert generic_runner.system_name == expected_system
        assert generic_runner.app_name == expected_app

    @staticmethod
    def test_artifact_resolver_unsupported_backend() -> None:
        """Test that it should be not possible to use unsupported backends."""
        with pytest.raises(
            RuntimeError, match="Unsupported device ethos-u65 for backend test_backend"
        ):
            get_generic_runner(DeviceInfo("ethos-u65", 256), "test_backend")

    @staticmethod
    @pytest.mark.parametrize("backend", ("Corstone-300", "Corstone-310"))
    def test_inference_should_fail_if_system_not_installed(
        backend_runner: MagicMock, test_tflite_model: Path, backend: str
    ) -> None:
        """Test that inference should fail if system is not installed."""
        backend_runner.is_system_installed.return_value = False

        generic_runner = get_generic_runner(DeviceInfo("ethos-u55", 256), backend)
        with pytest.raises(
            Exception,
            match=r"System Corstone-3[01]0: Cortex-M[58]5\+Ethos-U55 is not installed",
        ):
            generic_runner.run(ModelInfo(test_tflite_model), [])

    @staticmethod
    @pytest.mark.parametrize("backend", ("Corstone-300", "Corstone-310"))
    def test_inference_should_fail_is_apps_not_installed(
        backend_runner: MagicMock, test_tflite_model: Path, backend: str
    ) -> None:
        """Test that inference should fail if apps are not installed."""
        backend_runner.is_system_installed.return_value = True
        backend_runner.is_application_installed.return_value = False

        generic_runner = get_generic_runner(DeviceInfo("ethos-u55", 256), backend)
        with pytest.raises(
            Exception,
            match="Application Generic Inference Runner: Ethos-U55"
            r" for the system Corstone-3[01]0: Cortex-M[58]5\+Ethos-U55 is not "
            r"installed",
        ):
            generic_runner.run(ModelInfo(test_tflite_model), [])


@pytest.fixture(name="backend_runner")
def fixture_backend_runner(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Mock backend runner."""
    backend_runner_mock = MagicMock(spec=BackendRunner)
    monkeypatch.setattr(
        "mlia.backend.manager.get_backend_runner",
        MagicMock(return_value=backend_runner_mock),
    )
    return backend_runner_mock
