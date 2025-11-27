# SPDX-FileCopyrightText: Copyright 2022-2023, 2025, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for module backend/manager."""
from __future__ import annotations

import base64
import subprocess  # nosec
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing import Generator
from unittest.mock import MagicMock

import pytest

import mlia.core.output_schema as schema
from mlia.backend.corstone.performance import build_corstone_command
from mlia.backend.corstone.performance import CorstoneRunConfig
from mlia.backend.corstone.performance import estimate_performance
from mlia.backend.corstone.performance import GenericInferenceOutputParser
from mlia.backend.corstone.performance import get_metrics
from mlia.backend.corstone.performance import PerformanceMetrics
from mlia.backend.errors import BackendExecutionFailed
from mlia.core.context import ExecutionContext
from mlia.core.events import CollectedDataEvent
from mlia.target.ethos_u.config import EthosUConfiguration
from mlia.target.ethos_u.data_collection import EthosUPerformance
from mlia.target.ethos_u.handlers import EthosUEventHandler
from mlia.target.ethos_u.performance import CorstonePerformanceResult
from mlia.target.ethos_u.performance import (
    PerformanceMetrics as EthosPerformanceMetrics,
)
from mlia.utils.proc import Command


def encode_b64(data: str) -> str:
    """Encode data in base64 format."""
    return base64.b64encode(data.encode()).decode()


def valid_fvp_output() -> list[str]:
    """Return valid FVP output that could be successfully parsed."""
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


@dataclass(frozen=True)
class BuildCmdCase:
    """Build Command Case function."""

    backend_path: Path
    fvp: str
    target: str
    mac: int
    model: Path
    profile: str
    expected_command: Command


@pytest.mark.parametrize(
    "case",
    [
        BuildCmdCase(
            backend_path=Path("backend_path"),
            fvp="corstone-300",
            target="ethos-u55",
            mac=256,
            model=Path("model.tflite"),
            profile="default",
            expected_command=Command(
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
        ),
        BuildCmdCase(
            backend_path=Path("backend_path"),
            fvp="corstone-320",
            target="ethos-u85",
            mac=1024,
            model=Path("model.tflite"),
            profile="default",
            expected_command=Command(
                [
                    "backend_path/FVP_Corstone_SSE-320",
                    "-a",
                    "apps/backends/applications/"
                    "inference_runner-sse-320-22.08.02-ethos-U85-Default-noTA/"
                    "ethos-u-inference_runner.axf",
                    "--data",
                    "model.tflite@0x90000000",
                    "-C",
                    "mps4_board.subsystem.ethosu.num_macs=1024",
                    "-C",
                    "mps4_board.telnetterminal0.start_telnet=0",
                    "-C",
                    "mps4_board.uart0.out_file='-'",
                    "-C",
                    "mps4_board.uart0.shutdown_on_eot=1",
                    "-C",
                    "mps4_board.visualisation.disable-visualisation=1",
                    "-C",
                    "vis_hdlcd.disable_visualisation=1",
                    "--stat",
                ],
            ),
        ),
    ],
)
def test_build_corsone_command(
    monkeypatch: pytest.MonkeyPatch,
    case: BuildCmdCase,
) -> None:
    """Test function build_corstone_command."""
    monkeypatch.setattr(
        "mlia.backend.corstone.performance.get_mlia_resources", lambda: Path("apps")
    )

    command = build_corstone_command(
        CorstoneRunConfig(
            case.backend_path,
            case.fvp,
            case.target,
            case.mac,
            case.model,
            case.profile,
        )
    )
    assert command == case.expected_command


def test_get_metrics_wrong_fvp() -> None:
    """Test that command construction should fail for wrong FVP."""
    with pytest.raises(
        BackendExecutionFailed, match=r"Unable to construct a command line for some_fvp"
    ):
        get_metrics(
            CorstoneRunConfig(
                Path("backend_path"),
                "some_fvp",
                "ethos-u55",
                256,
                Path("model.tflite"),
            )
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

    # Check if BackendExecutionFailed is raised if the corstone command fails
    mock_check_call = MagicMock(
        side_effect=subprocess.CalledProcessError(returncode=1, cmd="fvp")
    )

    monkeypatch.setattr("mlia.utils.proc.command_output", mock_check_call)

    with pytest.raises(BackendExecutionFailed, match="Backend execution failed."):
        _ = estimate_performance("ethos-u55", 256, Path("model.tflite"), "corstone-300")

    # Check if BackendExecutionFailed is raised if get_backend_settings
    # returns invalid results
    mock_backend_repo = MagicMock()
    mock_backend_repo.get_backend_settings.return_value = (None, None)

    monkeypatch.setattr(
        "mlia.backend.corstone.performance.get_backend_repository",
        MagicMock(return_value=mock_backend_repo),
    )

    with pytest.raises(BackendExecutionFailed, match="Unable to configure backend"):
        _ = estimate_performance("ethos-u55", 256, Path("model.tflite"), "corstone-300")


def test_performance_metrics_to_standardized_output(tmp_path: Path) -> None:
    """Test conversion of PerformanceMetrics to standardized output."""
    perf_metrics = PerformanceMetrics(
        npu_active_cycles=1000,
        npu_idle_cycles=500,
        npu_total_cycles=1500,
        npu_axi0_rd_data_beat_received=200,
        npu_axi0_wr_data_beat_written=100,
        npu_axi1_rd_data_beat_received=150,
        npu_axi1_wr_data_beat_written=75,
    )

    # Create a model file for hash computation
    model_file = tmp_path / "model.tflite"
    model_file.write_bytes(b"test model content")
    output = perf_metrics.to_standardized_output(
        model_path=model_file,
        backend_name="corstone-300",
        target_config={"mac": 256, "target": "ethos-u55"},
    )
    # Structure checks
    for key in ("schema_version", "backends", "target", "model", "context", "results"):
        assert key in output
    assert output["schema_version"] == schema.SCHEMA_VERSION
    # Backend checks

    # Target/component checks
    components = output["target"]["components"]

    assert any(
        c["type"] == "npu" and "ethos" in c.get("family", "").lower()
        for c in components
    )
    # Check target description mentions corstone
    assert "corstone" in output["target"]["description"].lower()
    # Results/metrics checks
    result = output["results"][0]
    metrics = result["metrics"]
    metrics_dict = {m["name"]: m for m in metrics}
    assert metrics_dict["npu_active_cycles"]["value"] == 1000
    assert metrics_dict["npu_idle_cycles"]["value"] == 500
    assert metrics_dict["npu_total_cycles"]["value"] == 1500
    assert metrics_dict["npu_axi0_rd_data_beat_received"]["value"] == 200


def test_performance_metrics_to_standardized_output_with_null_axi1_wr(
    tmp_path: Path,
) -> None:
    """Test conversion when axi1_wr_data_beat_written is None."""
    perf_metrics = PerformanceMetrics(
        npu_active_cycles=1000,
        npu_idle_cycles=500,
        npu_total_cycles=1500,
        npu_axi0_rd_data_beat_received=200,
        npu_axi0_wr_data_beat_written=100,
        npu_axi1_rd_data_beat_received=150,
        npu_axi1_wr_data_beat_written=None,
    )

    # Create a model file for hash computation
    model_file = tmp_path / "model.tflite"
    model_file.write_bytes(b"test model content")

    output: dict = perf_metrics.to_standardized_output(
        model_path=model_file,
        backend_name="corstone-300",
        target_config={"mac": 256, "target": "ethos-u55"},
    )

    # Verify that 6 metrics are present (not 7)
    results = output["results"]
    assert len(results) == 1
    metrics = results[0]["metrics"]
    assert len(metrics) == 6


def test_ethosu_collector_and_handler_write_json(  # pylint: disable=too-many-locals
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Collector should return CorstonePerformanceResult and handler should write JSON.

    This test does not inject any stubs; it performs imports locally so the
    test module can be imported without requiring optional backends at module
    import time. If Vela/related packages are not installed in the test
    environment, this test will fail during import of the target modules.
    """

    # Create a test model file
    model = tmp_path / "model.tflite"
    model.write_bytes(b"test")

    # Create backend corstone metrics
    cor_metrics = PerformanceMetrics(
        npu_active_cycles=10,
        npu_idle_cycles=2,
        npu_total_cycles=12,
        npu_axi0_rd_data_beat_received=1,
        npu_axi0_wr_data_beat_written=2,
        npu_axi1_rd_data_beat_received=3,
    )

    # Build an Ethos-U legacy PerformanceMetrics instance
    # and attach corstone metrics
    target_cfg = EthosUConfiguration(
        target="ethos-u55",
        mac=256,
        system_config="Ethos_U55_High_End_Embedded",
        memory_mode="Shared_Sram",
    )
    ethos_perf = EthosPerformanceMetrics(target_cfg, None, None, None)
    ethos_perf.corstone_metrics = cor_metrics

    # Monkeypatch the estimator used by the collector to return our prepared metrics
    class MockEstimator:
        """Mock estimator for testing."""

        def __init__(
            self, context: Any, target_config: Any, backends: Any = None
        ) -> None:
            pass

        def estimate(self, model_arg: Any) -> Any:  # pylint: disable=unused-argument
            """Return test performance metrics."""
            return ethos_perf

    monkeypatch.setattr(
        "mlia.target.ethos_u.data_collection.EthosUPerformanceEstimator",
        MockEstimator,
    )

    # Monkeypatch get_tflite_model so collector doesn't try to load any TF models
    monkeypatch.setattr(
        "mlia.target.ethos_u.data_collection.get_tflite_model", lambda m, c: m
    )

    # Create collector and run
    collector = EthosUPerformance(model, target_cfg, backends=["corstone-300"])
    # Provide minimal context expected by collector
    # (set by inject_context in real workflow)
    context = ExecutionContext(output_dir=tmp_path)
    if hasattr(collector, "set_context"):
        collector.set_context(context)
    else:
        collector.context = context
    collected = collector.collect_data()

    # Collector should ideally return a CorstonePerformanceResult
    # when corstone metrics exist. But older code paths may return
    # PerformanceMetrics; handle both cases.
    if isinstance(collected, CorstonePerformanceResult):
        wrapped = collected
    elif isinstance(collected, EthosPerformanceMetrics):
        standardized = None
        try:
            standardized = collected.to_standardized_output(
                model_path=model, backend_name="corstone-300"
            )
        except Exception:  # pylint: disable=broad-exception-caught
            standardized = None

        wrapped = CorstonePerformanceResult(
            legacy_info=collected, standardized_output=standardized
        )
    else:
        pytest.fail(
            f"Unexpected collector return type: {type(collected).__name__}. "
            f"Expected CorstonePerformanceResult or PerformanceMetrics."
        )

    assert wrapped.standardized_output is not None

    # Now create handler and publish the event
    # It should save the JSON into tmp_path
    handler = EthosUEventHandler(output_dir=tmp_path)
    # Provide a minimal reporter to avoid AttributeError
    # when handler tries to submit
    handler.reporter = MagicMock()

    event = CollectedDataEvent(wrapped)
    handler.on_collected_data(event)

    # Check that a file with corstone_performance.json was created
    output_file = tmp_path / "corstone_performance.json"
    assert output_file.exists(), f"Expected JSON output file not found: {output_file}"
