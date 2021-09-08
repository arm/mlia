# Copyright 2021, Arm Ltd.
"""Tests for main module."""
import logging
import pathlib
import sys
from pathlib import Path
from typing import Any
from typing import List
from unittest.mock import MagicMock

import pytest
from mlia.cli.main import main
from mlia.config import EthosU55
from mlia.metrics import MemoryUsage
from mlia.metrics import NPUCycles
from mlia.metrics import PerformanceMetrics
from mlia.utils.general import save_keras_model
from mlia.utils.proc import working_directory

from tests.utils.generate_keras_model import generate_keras_model


def clear_loggers() -> None:
    """Close the log handlers."""
    for _, logger in logging.Logger.manager.loggerDict.items():  # type: ignore
        if not isinstance(logger, logging.PlaceHolder):
            for handler in logger.handlers:
                handler.close()
                logger.removeHandler(handler)


def teardown_function() -> None:
    """Call the function to close log handlers for pytest."""
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
        main(args)

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


@pytest.mark.parametrize(
    "quantize",
    [True, False],
)
def test_keras_to_tflite_command(quantize: bool, tmp_path: pathlib.Path) -> None:
    """Test keras_to_flite command."""
    model = generate_keras_model()
    temp_file = tmp_path / "test_keras_to_tflite_command.h5"
    save_keras_model(model, temp_file)

    if quantize:
        exit_code = main(
            [
                "keras_to_tflite",
                str(temp_file),
                "--quantize",
                "--out-path",
                str(tmp_path),
            ]
        )
    else:
        exit_code = main(
            ["keras_to_tflite", str(temp_file), "--out-path", str(tmp_path)]
        )

    assert exit_code == 0


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
    device: str,
    optimization_type: str,
    optimization_target: str,
    monkeypatch: Any,
    tmp_path: pathlib.Path,
) -> None:
    """Test keras_to_flite command."""
    model = generate_keras_model()
    temp_file = tmp_path / "test_optimization_command.h5"
    save_keras_model(model, temp_file)

    mock_performance_estimation(monkeypatch)

    exit_code = main(
        [
            "--working-dir",
            str(tmp_path),
            "optimization",
            str(temp_file),
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


def test_all_tests_command(tmp_path: pathlib.Path, monkeypatch: Any) -> None:
    """Test all_tests command."""
    model = generate_keras_model()
    temp_file = tmp_path / "test_model_optimization_command.h5"
    save_keras_model(model, temp_file)

    mock_performance_estimation(monkeypatch)

    exit_code = main(
        [
            "--working-dir",
            str(tmp_path),
            "all_tests",
            "--device",
            "ethos-u55",
            str(temp_file),
        ]
    )

    assert exit_code == 0


@pytest.mark.parametrize(
    "args_main, mock_perf_verbose, expected_output",
    [
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
        ],
    ],
)
def test_ops_perf_command_verbose(
    args_main: List[str],
    mock_perf_verbose: bool,
    expected_output: str,
    test_models_path: Path,
    monkeypatch: Any,
    capfd: Any,
) -> None:
    """Test ops commands in verbose mode."""
    model = test_models_path / "simple_3_layers_model.tflite"
    mock_performance_estimation(monkeypatch, mock_perf_verbose)

    exit_code = main(args_main + [str(model)])
    assert exit_code == 0
    out, _ = capfd.readouterr()

    assert expected_output in out
    teardown_function()


args_opt_all = [
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
    ["all", "--device", "ethos-u55", "--mac", "256", "--verbose"],
]


@pytest.mark.parametrize("args", args_opt_all)
def test_opt_all_command_verbose(
    args: List[str],
    test_models_path: Path,
    capfd: Any,
    monkeypatch: Any,
) -> None:
    """Test opt and all commands in verbose mode."""
    model = generate_keras_model()
    temp_file = test_models_path / "test_model_optimization_command.h5"
    save_keras_model(model, temp_file)

    mock_performance_estimation(monkeypatch)

    exit_code = main(args + [str(temp_file)])
    assert exit_code == 0
    out, _ = capfd.readouterr()
    assert "tensorflow - Compiled the loaded model" in out
    teardown_function()
