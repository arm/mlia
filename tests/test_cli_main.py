# Copyright 2021, Arm Ltd.
"""Module for running tests.

This file contains tests which will be execute by pytest.
Please refer to official pytest documentation.

https://docs.pytest.org/en/latest/contents.html
"""
from pathlib import Path
from typing import Any
from typing import List

import pytest
from mlia.cli.main import main
from mlia.utils.general import save_keras_model

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
        ["performance"],
        ["performance", "--device", "ethos-u65"],
        ["performance", "--device", "ethos-u55", "--mac", "32"],
    ],
)
def test_performance_command(args: List[str], test_models_path: Path) -> None:
    """Test performance command."""
    model = test_models_path / "simple_3_layers_model.tflite"

    exit_code = main(args + [str(model)])
    assert exit_code == 0


@pytest.mark.parametrize(
    "args",
    [
        ["--optimization_type", "pruning", "--optimization_target", "0.5"],
        ["--optimization_type", "clustering", "--optimization_target", "32"],
        [
            "--optimization_type",
            "pruning",
            "--optimization_target",
            "0.5",
            "--layers_to_optimize",
            "conv1",
            "conv2",
        ],
        [
            "--optimization_type",
            "clustering",
            "--optimization_target",
            "32",
            "--layers_to_optimize",
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
