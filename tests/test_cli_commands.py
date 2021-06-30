"""Tests for cli.commands module."""
import pytest
from mlia.cli.commands import performance


def test_command_no_device() -> None:
    """Test that command should fail if no device provided."""
    with pytest.raises(Exception, match="Device is not provided"):
        performance("some_model.tflite")


def test_command_unknown_device() -> None:
    """Test that command should fail if unknown device passed."""
    with pytest.raises(Exception, match="Unsupported device unknown"):
        performance("some_model.tflite", device="unknown")
