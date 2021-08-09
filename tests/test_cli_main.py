# Copyright 2021, Arm Ltd.
"""Tests for main module."""
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
    "args",
    [
        ["--optimization-type", "pruning", "--optimization-target", "0.5"],
        ["--optimization-type", "clustering", "--optimization-target", "32"],
        [
            "--optimization-type",
            "pruning",
            "--optimization-target",
            "0.5",
            "--layers-to-optimize",
            "conv1",
            "conv2",
        ],
        [
            "--optimization-type",
            "clustering",
            "--optimization-target",
            "32",
            "--layers-to-optimize",
            "conv1",
            "conv2",
        ],
    ],
)
def test_model_optimization_command(args: List[str]) -> None:
    """Test operators command."""
    model = generate_keras_model()
    model_path = save_keras_model(model)

    exit_code = main(["model_optimization", model_path] + args)
    assert exit_code == 0


@pytest.mark.parametrize(
    "quantize",
    [True, False],
)
def test_keras_to_tflite_command(quantize: bool) -> None:
    """Test keras_to_flite command."""
    model = generate_keras_model()
    model_path = save_keras_model(model)

    if quantize:
        exit_code = main(["keras_to_tflite", model_path, "--quantize"])
    else:
        exit_code = main(["keras_to_tflite", model_path])

    assert exit_code == 0


def mock_performance_estimation(monkeypatch: Any) -> None:
    """Mock performance estimation."""
    perf_metrics = PerformanceMetrics(
        EthosU55(), NPUCycles(0, 0, 0, 0, 0, 0), MemoryUsage(0, 0, 0, 0, 0)
    )

    monkeypatch.setattr(
        "mlia.performance.ethosu_performance_metrics",
        MagicMock(return_value=perf_metrics),
    )
