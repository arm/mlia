# Copyright 2021, Arm Ltd.
"""Tests for main module."""
import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any
from typing import Callable
from typing import List
from unittest.mock import ANY
from unittest.mock import call
from unittest.mock import MagicMock

import pytest
from mlia.cli.main import CommandInfo
from mlia.cli.main import main
from mlia.config import EthosU55
from mlia.metadata import Operators
from mlia.metrics import MemoryUsage
from mlia.metrics import NPUCycles
from mlia.metrics import PerformanceMetrics
from mlia.optimizations.select import OptimizationSettings
from mlia.utils.proc import working_directory

from tests.utils.logging import clear_loggers


def teardown_function() -> None:
    """Perform action after test completion.

    This function is launched automatically by pytest after each test
    in this module.
    """
    clear_loggers()


def test_option_version(capfd: Any) -> None:
    """Test --version."""
    with pytest.raises(SystemExit) as ex:
        main(["--version"])

    assert ex.type == SystemExit
    assert ex.value.code == 0

    stdout, stderr = capfd.readouterr()
    assert len(stdout.splitlines()) == 1
    assert stderr == ""


def test_operators_command(test_models_path: Path) -> None:
    """Test operators command."""
    model = test_models_path / "simple_3_layers_model.tflite"

    exit_code = main(["operators", str(model)])
    assert exit_code == 0


def test_command_info() -> None:
    """Test properties of CommandInfo object."""

    def super_command() -> None:
        """Activate super power."""

    ci_default = CommandInfo(super_command, ["super"], [], True)
    assert ci_default.command_name == "super_command"
    assert ci_default.command_name_and_aliases == ["super_command", "super"]
    assert ci_default.command_help == "Activate super power [default]"

    ci_non_default = CommandInfo(super_command, ["super"], [], False)
    assert ci_default.command_name == ci_non_default.command_name
    assert (
        ci_default.command_name_and_aliases == ci_non_default.command_name_and_aliases
    )
    assert ci_non_default.command_help == "Activate super power"


def test_default_command(monkeypatch: Any) -> None:
    """Test adding default command."""

    def mock_command(
        func_mock: MagicMock, name: str, with_working_dir: bool
    ) -> Callable[..., None]:
        """Mock cli command."""

        def f(*args: Any, **kwargs: Any) -> None:
            """Sample command."""
            func_mock(*args, **kwargs)

        def g(working_dir: str, **kwargs: Any) -> None:
            """Another sample command."""
            func_mock(working_dir=working_dir, **kwargs)

        ret_func = g if with_working_dir else f
        ret_func.__name__ = name

        return ret_func  # type: ignore

    default_command = MagicMock()
    non_default_command = MagicMock()

    def default_command_params(parser: argparse.ArgumentParser) -> None:
        """Add parameters for default command."""
        parser.add_argument("--sample")
        parser.add_argument("--default_arg", default="123")

    def non_default_command_params(parser: argparse.ArgumentParser) -> None:
        """Add parameters for non default command."""
        parser.add_argument("--param")

    monkeypatch.setattr(
        "mlia.cli.main.get_commands",
        MagicMock(
            return_value=[
                CommandInfo(
                    func=mock_command(default_command, "default_command", True),
                    aliases=["command1"],
                    opt_groups=[default_command_params],
                    is_default=True,
                ),
                CommandInfo(
                    func=mock_command(
                        non_default_command, "non_default_command", False
                    ),
                    aliases=["command2"],
                    opt_groups=[non_default_command_params],
                    is_default=False,
                ),
            ]
        ),
    )

    main(["--working-dir", "test_work_dir", "--sample", "1"])
    main(["command2", "--param", "test"])

    default_command.assert_called_once_with(
        working_dir="test_work_dir", sample="1", default_arg="123"
    )
    non_default_command.assert_called_once_with(param="test")


@pytest.mark.parametrize(
    "args",
    [
        ["operators", "--supported-ops-report"],
        ["operators", "--supported-ops-report", "--mac", "32"],
        ["operators", "--supported-ops-report", "--device", "ethos-u65"],
    ],
)
def test_operators_command_gen_supported_report(
    args: List[str], tmp_path: Path
) -> None:
    """Test supported operators report generation."""
    with working_directory(tmp_path):
        exit_code = main(args)

        assert exit_code == 0

        md_file = tmp_path / "SUPPORTED_OPS.md"
        assert md_file.is_file()
        assert md_file.stat().st_size > 0


@pytest.mark.parametrize(
    "args",
    [
        ["performance"],
        ["performance", "--device", "ethos-u65"],
        ["performance", "--device", "ethos-u55", "--mac", "32"],
    ],
)
def test_performance_command(
    args: List[str], test_models_path: Path, monkeypatch: Any
) -> None:
    """Test performance command."""
    model = test_models_path / "simple_3_layers_model.tflite"
    mock_performance_estimation(monkeypatch)

    exit_code = main(args + [str(model)])
    assert exit_code == 0


def test_performance_custom_vela_init(
    test_resources_path: Path, test_models_path: Path, monkeypatch: Any
) -> None:
    """Test performance command with custom vela.ini."""
    vela_ini = test_resources_path / "vela/sample_vela.ini"
    model = test_models_path / "simple_3_layers_model.tflite"

    mock_performance_estimation(monkeypatch)

    args = [
        "performance",
        str(model),
        "--config",
        str(vela_ini),
        "--system-config",
        "Ethos_U55_High_End_Embedded",
        "--memory-mode",
        "Shared_Sram",
    ]

    exit_code = main(args)
    assert exit_code == 0


@pytest.mark.parametrize("extra_params", [[], ["--quantize"]])
def test_keras_to_tflite_command(
    test_models_path: Path, extra_params: List[str], tmp_path: Path
) -> None:
    """Test keras_to_flite command."""
    model = test_models_path / "simple_model.h5"

    exit_code = main(
        ["keras_to_tflite", str(model), *extra_params, "--out-path", str(tmp_path)]
    )
    assert exit_code == 0


def mock_optimize_and_compare(monkeypatch: Any) -> None:
    """Mock optimize_and_compare function."""
    perf_metrics = PerformanceMetrics(
        EthosU55(), NPUCycles(0, 0, 0, 0, 0, 0), MemoryUsage(0, 0, 0, 0, 0)
    )

    monkeypatch.setattr(
        "mlia.cli.commands.optimize_and_compare",
        MagicMock(return_value=(perf_metrics, perf_metrics)),
    )


def mock_operators_compatibility(monkeypatch: Any) -> None:
    """Mock supported operators check."""
    monkeypatch.setattr(
        "mlia.cli.commands.supported_operators", MagicMock(return_value=Operators([]))
    )


def mock_get_optimizer(monkeypatch: Any) -> MagicMock:
    """Mock get_optimizer function."""
    mock = MagicMock()
    monkeypatch.setattr("mlia.cli.commands.get_optimizer", mock)

    return mock


def mock_performance_estimation(monkeypatch: Any, verbose: bool = False) -> None:
    """Mock performance estimation."""
    if verbose:
        logger = logging.getLogger("mlia.mock.perf")
        logger.setLevel(logging.INFO)
        logger.addHandler(logging.StreamHandler(sys.stdout))
        logger.info("Mocking performance estimation")
    perf_metrics = PerformanceMetrics(
        EthosU55(), NPUCycles(0, 0, 0, 0, 0, 0), MemoryUsage(0, 0, 0, 0, 0)
    )

    monkeypatch.setattr(
        "mlia.performance.ethosu_performance_metrics",
        MagicMock(return_value=perf_metrics),
    )


@pytest.mark.parametrize(
    "device, optimization_type, optimization_target",
    [
        ["ethos-u55", "pruning", "0.5"],
        ["ethos-u65", "clustering", "32"],
    ],
)
def test_optimization_command(
    test_models_path: Path,
    device: str,
    optimization_type: str,
    optimization_target: str,
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    """Test keras_to_flite command."""
    model = test_models_path / "simple_model.h5"
    mock_performance_estimation(monkeypatch)

    exit_code = main(
        [
            "--working-dir",
            str(tmp_path),
            "optimization",
            str(model),
            "--device",
            device,
            "--optimization-type",
            optimization_type,
            "--optimization-target",
            optimization_target,
        ]
    )

    assert exit_code == 0

    assert (tmp_path / "original_model.tflite").is_file()
    assert (tmp_path / "optimized_model.tflite").is_file()


@pytest.mark.parametrize(
    "extra_params, expected_exit_code, expected_opt_settings",
    [
        (
            ["--optimization-type", "pruning", "--optimization-target", "0.5"],
            0,
            [
                OptimizationSettings(
                    optimization_type="pruning",
                    optimization_target=0.5,
                    layers_to_optimize=None,
                )
            ],
        ),
        (
            [
                "--optimization-type",
                "pruning,clustering",
                "--optimization-target",
                "0.5,32",
            ],
            0,
            [
                OptimizationSettings(
                    optimization_type="pruning",
                    optimization_target=0.5,
                    layers_to_optimize=None,
                ),
                OptimizationSettings(
                    optimization_type="clustering",
                    optimization_target=32,
                    layers_to_optimize=None,
                ),
            ],
        ),
        (
            [
                "--optimization-type",
                " pruning, clustering",
                "--optimization-target",
                "0.5, 32 ",
            ],
            0,
            [
                OptimizationSettings(
                    optimization_type="pruning",
                    optimization_target=0.5,
                    layers_to_optimize=None,
                ),
                OptimizationSettings(
                    optimization_type="clustering",
                    optimization_target=32,
                    layers_to_optimize=None,
                ),
            ],
        ),
        (
            [
                "--optimization-type",
                "pruning,clustering",
                "--optimization-target",
                "0.5",
            ],
            1,
            None,
        ),
    ],
)
def test_all_tests_command(
    tmp_path: Path,
    test_models_path: Path,
    monkeypatch: Any,
    extra_params: List[str],
    expected_exit_code: int,
    expected_opt_settings: List[OptimizationSettings],
) -> None:
    """Test all_tests command."""
    model = test_models_path / "simple_model.h5"

    mock_optimize_and_compare(monkeypatch)
    mock_operators_compatibility(monkeypatch)
    get_optimizer_mock = mock_get_optimizer(monkeypatch)

    args = ["--working-dir", str(tmp_path), "all_tests", *extra_params, str(model)]

    exit_code = main(args)
    assert exit_code == expected_exit_code

    if expected_exit_code == 0:
        get_optimizer_mock.assert_called_once()
        get_optimizer_mock.assert_has_calls([call(ANY, expected_opt_settings)])


@pytest.mark.parametrize("output_format", ["plain_text", "csv", "json"])
def test_all_tests_command_output(
    tmp_path: Path,
    monkeypatch: Any,
    output_format: str,
    test_models_path: Path,
) -> None:
    """Test all_tests command can produce correct output file."""
    model = test_models_path / "simple_model.h5"
    output = tmp_path / "report.all_command"

    mock_performance_estimation(monkeypatch)

    exit_code = main(
        [
            "--working-dir",
            str(tmp_path),
            "all_tests",
            "--device",
            "ethos-u55",
            str(model),
            "--output-format",
            output_format,
            "--output",
            str(output),
        ]
    )

    assert exit_code == 0
    assert output.is_file()
    assert output.stat().st_size > 0

    if output_format == "json":
        with open(output) as file:
            parsed_json = json.load(file)
            assert all(
                metric in parsed_json
                for metric in [
                    "device",
                    "operators_stats",
                    "operators",
                    "performance_metrics",
                    "advice",
                ]
            )


args_ops = [
    [
        [
            "performance",
            "--device",
            "ethos-u55",
            "--mac",
            "256",
            "--verbose",
        ],
        True,
        "Mocking performance estimation",
        0,
        "simple_3_layers_model.tflite",
    ],
    [
        [
            "performance",
            "--device",
            "ethos-u55",
            "--mac",
            "256",
            "--verbose",
        ],
        True,
        "Traceback",
        1,
        "xyz",
    ],
    [
        [
            "operators",
            "--device",
            "ethos-u55",
            "--mac",
            "256",
            "--verbose",
        ],
        False,
        "mlia.tools.vela",
        0,
        "simple_3_layers_model.tflite",
    ],
    [
        [
            "optimization",
            "--device",
            "ethos-u55",
            "--optimization-type",
            "pruning",
            "--optimization-target",
            "0.5",
            "--mac",
            "256",
            "--verbose",
        ],
        False,
        "tensorflow - Compiled the loaded model",
        0,
        "simple_model.h5",
    ],
    [
        [
            "all",
            "--device",
            "ethos-u55",
            "--mac",
            "256",
            "--verbose",
        ],
        False,
        "tensorflow - Compiled the loaded model",
        0,
        "simple_model.h5",
    ],
]


@pytest.mark.parametrize(
    "args_main, mock_perf_verbose, expected_output, expected_exit_code, model_name",
    args_ops,
)
def test_perf_ops_opt_all_command_verbose(
    args_main: List[str],
    mock_perf_verbose: bool,
    expected_output: str,
    expected_exit_code: int,
    test_models_path: Path,
    model_name: str,
    monkeypatch: Any,
    capfd: Any,
) -> None:
    """Test all four commands in verbose mode."""
    model = test_models_path / model_name
    mock_performance_estimation(monkeypatch, mock_perf_verbose)

    exit_code = main(args_main + [str(model)])
    assert exit_code == expected_exit_code

    out, _ = capfd.readouterr()
    assert expected_output in out
