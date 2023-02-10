# SPDX-FileCopyrightText: Copyright 2022-2023, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for module backend/manager."""
from __future__ import annotations

import base64
from pathlib import Path
from typing import Generator
from unittest.mock import MagicMock

import pytest

from mlia.backend.corstone.performance import build_corstone_command
from mlia.backend.corstone.performance import estimate_performance
from mlia.backend.corstone.performance import GenericInferenceOutputParser
from mlia.backend.corstone.performance import get_metrics
from mlia.backend.corstone.performance import PerformanceMetrics
from mlia.backend.errors import BackendExecutionFailed
from mlia.utils.proc import Command


def encode_b64(data: str) -> str:
    """Encode data in base64 format."""
    return base64.b64encode(data.encode()).decode()


def valid_fvp_output() -> list[str]:
    """Return valid FVP output that could be succesfully parsed."""
    json_data = """[
    {
        "profiling_group": "Inference",
        "count": 1,
        "samples": [
            {"name": "NPU IDLE", "value": [2]},
            {"name": "NPU AXI0_RD_DATA_BEAT_RECEIVED", "value": [4]},
            {"name": "NPU AXI0_WR_DATA_BEAT_WRITTEN", "value": [5]},
            {"name": "NPU AXI1_RD_DATA_BEAT_RECEIVED", "value": [6]},
            {"name": "NPU ACTIVE", "value": [1]},
            {"name": "NPU TOTAL", "value": [3]}
        ]
    }
]"""

    return [
        "some output",
        f"<metrics>{encode_b64(json_data)}</metrics>",
        "some_output",
    ]


def test_generic_inference_output_parser_success() -> None:
    """Test successful generic inference output parsing."""
    output_parser = GenericInferenceOutputParser()
    for line in valid_fvp_output():
        output_parser(line)

    assert output_parser.get_metrics() == PerformanceMetrics(1, 2, 3, 4, 5, 6)


@pytest.mark.parametrize(
    "wrong_fvp_output",
    [
        [],
        ["NPU IDLE: 123"],
        ["<metrics>123</metrics>"],
    ],
)
def test_generic_inference_output_parser_failure(wrong_fvp_output: list[str]) -> None:
    """Test unsuccessful generic inference output parsing."""
    output_parser = GenericInferenceOutputParser()

    for line in wrong_fvp_output:
        output_parser(line)

    with pytest.raises(ValueError, match="Unable to parse output and get metrics"):
        output_parser.get_metrics()


@pytest.mark.parametrize(
    "backend_path, fvp, target, mac, model, profile, expected_command",
    [
        [
            Path("backend_path"),
            "corstone-300",
            "ethos-u55",
            256,
            Path("model.tflite"),
            "default",
            Command(
                [
                    "backend_path/FVP_Corstone_SSE-300_Ethos-U55",
                    "-a",
                    "apps/backends/applications/"
                    "inference_runner-sse-300-22.08.02-ethos-U55-Default-noTA/"
                    "ethos-u-inference_runner.axf",
                    "--data",
                    "model.tflite@0x90000000",
                    "-C",
                    "ethosu.num_macs=256",
                    "-C",
                    "mps3_board.telnetterminal0.start_telnet=0",
                    "-C",
                    "mps3_board.uart0.out_file='-'",
                    "-C",
                    "mps3_board.uart0.shutdown_on_eot=1",
                    "-C",
                    "mps3_board.visualisation.disable-visualisation=1",
                    "--stat",
                ]
            ),
        ],
    ],
)
def test_build_corsone_command(
    monkeypatch: pytest.MonkeyPatch,
    backend_path: Path,
    fvp: str,
    target: str,
    mac: int,
    model: Path,
    profile: str,
    expected_command: Command,
) -> None:
    """Test function build_corstone_command."""
    monkeypatch.setattr(
        "mlia.backend.corstone.performance.get_mlia_resources", lambda: Path("apps")
    )

    command = build_corstone_command(backend_path, fvp, target, mac, model, profile)
    assert command == expected_command


def test_get_metrics_wrong_fvp() -> None:
    """Test that command construction should fail for wrong FVP."""
    with pytest.raises(
        BackendExecutionFailed, match=r"Unable to construct a command line for some_fvp"
    ):
        get_metrics(
            Path("backend_path"),
            "some_fvp",
            "ethos-u55",
            256,
            Path("model.tflite"),
        )


def test_estimate_performance(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test function estimate_performance."""
    mock_repository = MagicMock()
    mock_repository.get_backend_settings.return_value = Path("backend_path"), {
        "profile": "default"
    }

    monkeypatch.setattr(
        "mlia.backend.corstone.performance.get_backend_repository",
        lambda: mock_repository,
    )

    def command_output_mock(_command: Command) -> Generator[str, None, None]:
        """Mock FVP output."""
        yield from valid_fvp_output()

    monkeypatch.setattr("mlia.utils.proc.command_output", command_output_mock)

    result = estimate_performance(
        "ethos-u55", 256, Path("model.tflite"), "corstone-300"
    )
    assert result == PerformanceMetrics(1, 2, 3, 4, 5, 6)

    mock_repository.get_backend_settings.assert_called_once()
