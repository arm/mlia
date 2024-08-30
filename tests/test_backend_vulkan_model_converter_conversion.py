# SPDX-FileCopyrightText: Copyright 2023-2024, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: LicenseRef-LICENSE
"""Tests for NGP Graph Compiler config."""
from __future__ import annotations

from pathlib import Path
from typing import Generator
from unittest.mock import MagicMock

import pytest

from mlia.backend.vulkan_model_converter.conversion import VulkanModelConverter
from mlia.utils.proc import Command


@pytest.fixture(name="vulkan_model_converter")
def fixture_vulkan_model_converter(
    tmp_path: Path,
) -> Generator[VulkanModelConverter, None, None]:
    """Create a mock instance of the Vulkan Model Converter for testing."""
    vmc = VulkanModelConverter(tmp_path / "backend-vulkan-model-converter")
    yield vmc


def test_vulkan_model_converter_no_output_dir(
    vulkan_model_converter: VulkanModelConverter,
    tmp_path: Path,
) -> None:
    """Test for class VulkanModelConverter with an invalid output directory."""
    with pytest.raises(NotADirectoryError):
        vulkan_model_converter(tmp_path / "model.tflite", tmp_path / "output")


def test_vulkan_model_converter_success(
    vulkan_model_converter: VulkanModelConverter,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test for class VulkanModelConverter with successful execution."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    monkeypatch.setattr(
        "mlia.backend.vulkan_model_converter.conversion.VulkanModelConverter."
        "_create_front_end_command",
        lambda _, __, tosa_file: Command(["touch", str(tosa_file)]),
    )
    monkeypatch.setattr(
        "mlia.backend.vulkan_model_converter.conversion.VulkanModelConverter."
        "_create_back_end_command",
        lambda _, __, vgf_file: Command(["touch", str(vgf_file)]),
    )

    vgf_file = vulkan_model_converter(tmp_path / "model.tflite", output_dir)

    assert vgf_file.is_file()


def test_vulkan_model_converter_front_end_fail(
    vulkan_model_converter: VulkanModelConverter,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test for class VulkanModelConverter when the front end execution fails."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    monkeypatch.setattr(
        "mlia.backend.vulkan_model_converter.conversion.VulkanModelConverter."
        "_create_front_end_command",
        MagicMock(
            return_value=Command(
                [
                    "echo",
                    '"Faking a run of Vulkan Model Converter front end..."',
                ]
            )
        ),
    )

    with pytest.raises(FileNotFoundError):
        vulkan_model_converter(tmp_path / "model.tflite", output_dir)


def test_vulkan_model_converter_back_end_fail(
    vulkan_model_converter: VulkanModelConverter,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test for class VulkanModelConverter when the back end execution fails."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    monkeypatch.setattr(
        "mlia.backend.vulkan_model_converter.conversion.VulkanModelConverter."
        "_create_front_end_command",
        lambda _, __, tosa_file: Command(["touch", str(tosa_file)]),
    )
    monkeypatch.setattr(
        "mlia.backend.vulkan_model_converter.conversion.VulkanModelConverter."
        "_create_back_end_command",
        MagicMock(
            return_value=Command(
                [
                    "echo",
                    '"Faking a run of Vulkan Model Converter back end..."',
                ]
            )
        ),
    )

    with pytest.raises(FileNotFoundError):
        vulkan_model_converter(tmp_path / "model.tflite", output_dir)


def test_vulkan_model_converter_create_front_end_command(
    vulkan_model_converter: VulkanModelConverter,
    tmp_path: Path,
) -> None:
    """Test for function _create_front_end_command of VulkanModelConverter."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    in_file = tmp_path / "in"
    out_file = tmp_path / "out"

    cmd = vulkan_model_converter._create_front_end_command(  # pylint: disable=protected-access
        in_file, out_file
    )

    assert cmd.cmd
    assert all(isinstance(arg, str) for arg in cmd.cmd)
    assert cmd.cmd[0] == str(
        vulkan_model_converter.converter_path / vulkan_model_converter.FRONT_END_EXE
    )
    assert str(in_file) in cmd.cmd
    assert str(out_file) in cmd.cmd


def test_vulkan_model_converter_create_back_end_command(
    vulkan_model_converter: VulkanModelConverter,
    tmp_path: Path,
) -> None:
    """Test for function _create_back_end_command of VulkanModelConverter."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    in_file = tmp_path / "in"
    out_file = tmp_path / "out"
    cmd = vulkan_model_converter._create_back_end_command(  # pylint: disable=protected-access
        in_file, out_file
    )

    assert cmd.cmd
    assert all(isinstance(arg, str) for arg in cmd.cmd)
    assert cmd.cmd[0] == str(
        vulkan_model_converter.converter_path / vulkan_model_converter.BACK_END_EXE
    )
    assert str(in_file) in cmd.cmd
    assert str(out_file) in cmd.cmd
