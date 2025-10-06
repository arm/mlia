# SPDX-FileCopyrightText: Copyright 2022-2025, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for module vela/performance."""
from pathlib import Path
from unittest.mock import MagicMock

import pytest

try:
    import ethosu.vela  # noqa: F401
except ImportError:
    pytest.skip(
        "All tests require ethosu.vela package to be installed", allow_module_level=True
    )
else:
    # Only reference ethosu.vela if it was successfully imported
    _ = ethosu.vela

from mlia.backend.vela.compiler import compile_model  # noqa: E402
from mlia.backend.vela.performance import estimate_performance  # noqa: E402
from mlia.backend.vela.performance import layer_metrics  # noqa: E402
from mlia.backend.vela.performance import LayerwisePerfInfo  # noqa: E402
from mlia.backend.vela.performance import parse_layerwise_perf_csv  # noqa: E402
from mlia.backend.vela.performance import PerformanceMetrics  # noqa: E402
from mlia.target.ethos_u.config import EthosUConfiguration  # noqa: E402
from mlia.utils.filesystem import recreate_directory  # noqa: E402


def test_estimate_performance(test_tflite_model: Path) -> None:
    """Test getting performance estimations."""
    target_config = EthosUConfiguration.load_profile("ethos-u55-256")
    assert (
        target_config.compiler_options is not None
    ), "Vela should be available in tests"
    perf_metrics = estimate_performance(
        test_tflite_model, target_config.compiler_options
    )

    assert isinstance(perf_metrics, PerformanceMetrics)


def test_estimate_performance_csv_parser_called(
    monkeypatch: pytest.MonkeyPatch, test_tflite_model: Path
) -> None:
    """Test that estimate_performance from backend.vela.performance is called."""
    target_config = EthosUConfiguration.load_profile("ethos-u55-256")
    assert (
        target_config.compiler_options is not None
    ), "Vela should be available in tests"
    csv_file_name = target_config.compiler_options.output_dir / (
        test_tflite_model.stem + "_per-layer.csv"
    )
    mock = MagicMock()
    monkeypatch.setattr("mlia.backend.vela.performance.parse_layerwise_perf_csv", mock)
    estimate_performance(test_tflite_model, target_config.compiler_options)
    mock.assert_called_with(vela_csv_file=csv_file_name, metrics=layer_metrics)


LAYERWISE_TMP_DATA_STR = """
TFLite_operator,NNG Operator,SRAM Usage,Peak%,Op Cycles,Network%,NPU,SRAM AC,DRAM AC,OnFlash AC,OffFlash AC,MAC Count,Network%,Util%,Name
CONV_2D,Conv2DBias,11936,54.65201465201465,7312.0,17.648194632168373,7312.0,2000.0,0.0,0.0,0.0,73008,8.653353814644136,3.9002666849015313,sequential/conv1/Relu;sequential/conv1/Conv2D
MAX_POOL_2D,MaxPool,10944,50.10989010989011,2992.0,7.22147132651091,1330.0,2992.0,0.0,0.0,0.0,6912,0.819252432155658,0.9024064171122994,sequential/max_pooling2d/MaxPool
""".strip()

LAYERWISE_TMP_DATA_MISSING_HEADER_STR = """
TFLite_operator,NNG Operator,Peak%,Op Cycles,Network%,NPU,SRAM AC,DRAM AC,OnFlash AC,OffFlash AC,MAC Count,Network%,Util%,Name
CONV_2D,Conv2DBias,54.65201465201465,7312.0,17.648194632168373,7312.0,2000.0,0.0,0.0,0.0,73008,8.653353814644136,3.9002666849015313,sequential/conv1/Relu;sequential/conv1/Conv2D
MAX_POOL_2D,MaxPool,50.10989010989011,2992.0,7.22147132651091,1330.0,2992.0,0.0,0.0,0.0,6912,0.819252432155658,0.9024064171122994,sequential/max_pooling2d/MaxPool
""".strip()

LAYERWISE_MULTI_HEADER_TMP_DATA_STR = """
TFLite_operator,NNG Operator,SRAM Usage,Peak%,Op Cycles,Network%,NPU,SRAM AC,DRAM AC,OnFlash AC,OffFlash AC,MAC Count,Network%,Util%,Name
CONV_2D,Conv2DBias,11936,54.65201465201465,7312.0,17.648194632168373,7312.0,2000.0,0.0,0.0,0.0,73008,8.653353814644136,3.9002666849015313,sequential/conv1/Relu;sequential/conv1/Conv2D
TFLite_operator,NNG Operator,SRAM Usage,Peak%,Op Cycles,Network%,NPU,SRAM AC,DRAM AC,OnFlash AC,OffFlash AC,MAC Count,Network%,Util%,Name
MAX_POOL_2D,MaxPool,10944,50.10989010989011,2992.0,7.22147132651091,1330.0,2992.0,0.0,0.0,0.0,6912,0.819252432155658,0.9024064171122994,sequential/max_pooling2d/MaxPool
""".strip()


TMP_DATA_EXPECTED_STRING = "\
Name: sequential/conv1/Relu;sequential/conv1/Conv2D, \
TFLite_operator: CONV_2D, \
SRAM Usage: 11936, \
Op Cycles: 7312, \
NPU: 7312, \
SRAM AC: 2000, \
DRAM AC: 0, \
OnFlash AC: 0, \
OffFlash AC: 0, \
MAC Count: 73008, \
Util%: 3.9002666849015313, \
\
Name: sequential/max_pooling2d/MaxPool, \
TFLite_operator: MAX_POOL_2D, \
SRAM Usage: 10944, \
Op Cycles: 2992, \
NPU: 1330, \
SRAM AC: 2992, \
DRAM AC: 0, \
OnFlash AC: 0, \
OffFlash AC: 0, \
MAC Count: 6912, \
Util%: 0.9024064171122994, \
"


@pytest.mark.parametrize(
    "input_csv_content, expected_output",
    [
        (LAYERWISE_TMP_DATA_STR, TMP_DATA_EXPECTED_STRING),
        (
            LAYERWISE_MULTI_HEADER_TMP_DATA_STR,
            TMP_DATA_EXPECTED_STRING,
        ),
    ],
)
def test_estimate_performance_parse_layerwise_csv_file(
    test_csv_file: Path, input_csv_content: str, expected_output: str
) -> None:
    """Test that parsing a csv file produces a LayerwisePerfInfo object."""
    with open(test_csv_file, "w", encoding="utf8") as csv_file:
        csv_file.write(input_csv_content)
    layerwise_object = parse_layerwise_perf_csv(test_csv_file, layer_metrics)
    strings_to_check_layerwise_object = repr(layerwise_object)
    assert isinstance(layerwise_object, LayerwisePerfInfo)
    assert expected_output == strings_to_check_layerwise_object


def test_estimate_performance_parse_layerwise_csv_file_with_missing_headers(
    test_csv_file: Path,
) -> None:
    """Test that ensures a KeyError
    is raised when a csv file is parsed with missing headers.
    """
    with open(test_csv_file, "w", encoding="utf8") as csv_file:
        csv_file.write(LAYERWISE_TMP_DATA_MISSING_HEADER_STR)
    with pytest.raises(KeyError, match="Generated CSV missing expected headers"):
        parse_layerwise_perf_csv(test_csv_file, layer_metrics)


def test_estimate_performance_parse_layerwise_csv_file_missing_file() -> None:
    """Test that ensures a FileNotFoundError
    is raised when a non-existent csv file is parsed.
    """
    with pytest.raises(
        FileNotFoundError, match="CSV File not found at missing_file.csv"
    ):
        parse_layerwise_perf_csv(Path("missing_file.csv"), layer_metrics)


def test_estimate_performance_parse_layerwise_empty_csv_file(
    empty_test_csv_file: Path,
) -> None:
    """Test that ensures that if an empty csv file
    is parsed, we return an empty layerwise object.
    """
    empty_test_csv_file.touch()
    layerwise_object = parse_layerwise_perf_csv(empty_test_csv_file, layer_metrics)
    assert isinstance(layerwise_object, LayerwisePerfInfo)
    assert len(layerwise_object.layerwise_info) == 0


def test_read_invalid_model(test_tflite_invalid_model: Path) -> None:
    """Test that reading invalid model should fail with exception."""
    with pytest.raises(
        Exception, match=f"Unable to read model {test_tflite_invalid_model}"
    ):
        target_config = EthosUConfiguration.load_profile("ethos-u55-256")
        assert (
            target_config.compiler_options is not None
        ), "Vela should be available in tests"
        estimate_performance(test_tflite_invalid_model, target_config.compiler_options)


def test_compile_invalid_model(
    test_tflite_model: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test that if model could not be compiled then correct exception raised."""

    mock_compiler = MagicMock()
    mock_compiler.side_effect = Exception("Bad model!")

    monkeypatch.setattr("mlia.backend.vela.compiler.main", mock_compiler)

    model_path = tmp_path / "optimized_model.tflite"
    with pytest.raises(
        Exception, match="Model could not be optimized with Vela compiler"
    ):
        target_config = EthosUConfiguration.load_profile("ethos-u55-256")
        assert (
            target_config.compiler_options is not None
        ), "Vela should be available in tests"
        recreate_directory(Path(target_config.compiler_options.output_dir))
        compile_model(test_tflite_model, target_config.compiler_options)

    assert not model_path.exists()
