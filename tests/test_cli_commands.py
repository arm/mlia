"""Tests for cli.commands module."""
from pathlib import Path

from mlia.cli.main import main


def test_operators_command(test_models_path: Path) -> None:
    """Test operators command."""
    model = test_models_path / "simple_3_layers_model.tflite"

    exit_code = main(["operators", str(model)])
    assert exit_code == 0
