# SPDX-FileCopyrightText: Copyright 2023-2024, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: LicenseRef-LICENSE
"""Tests for NGP Graph Compiler performance estimation."""
from __future__ import annotations

import copy
from pathlib import Path

import pytest

from mlia.backend.ngp_graph_compiler.output_parsing import NGPDebugDatabaseParser
from mlia.backend.ngp_graph_compiler.output_parsing import NGPPerformanceDatabaseParser
from mlia.backend.ngp_graph_compiler.statistics import NGPOperatorPerformanceStats
from mlia.backend.ngp_graph_compiler.statistics import NGPPerformanceStats


def test_sanitize_memory_fields_expected_input() -> None:
    """Sanitize memory attribute of NGPOperator."""
    op_stats = NGPOperatorPerformanceStats(
        op_id=[33],
        op_cycles=15,
        total_cycles=18,
        memory={
            "Undefined": {
                "readBytes": 0,
                "writeBytes": 0,
                "trafficCycles": 0,
            },
            "Internal": {
                "readBytes": 0,
                "writeBytes": 0,
                "trafficCycles": 0,
            },
            "L1": {
                "readBytes": 4,
                "writeBytes": 6,
                "trafficCycles": 43530,
            },
            "L2": {
                "readBytes": 5,
                "writeBytes": 7,
                "trafficCycles": 456570,
            },
            "SystemCache": {
                "readBytes": 1,
                "writeBytes": 2,
                "trafficCycles": 3,
            },
            "DRAM": {
                "readBytes": 312369600,
                "writeBytes": 31104000,
                "trafficCycles": 7056279,
            },
        },
        utilization=[
            {"sectionName": "OutputWriter", "hwUtil": 1},
            {"sectionName": "VectorEngine", "hwUtil": 1},
        ],
        operators=["foo"],
    )
    op_stats.sanitize_memory_fields()

    assert op_stats.memory == {
        "L1": {"readBytes": 4, "writeBytes": 6, "trafficCycles": 43530},
        "L2": {"readBytes": 5, "writeBytes": 7, "trafficCycles": 456570},
        "SystemCache": {
            "readBytes": 1,
            "writeBytes": 2,
            "trafficCycles": 3,
        },
        "DRAM": {
            "readBytes": 312369600,
            "writeBytes": 31104000,
            "trafficCycles": 7056279,
        },
    }


def test_sanitize_memory_fields_missing_memory_name_input() -> None:
    """Throw error when the memoryName key is absent."""
    op_stats = NGPOperatorPerformanceStats(
        op_id=[33],
        op_cycles=15,
        total_cycles=18,
        memory={
            "wrong_key": {
                "readBytes": 0,
                "writeBytes": 0,
                "trafficCycles": 0,
            },
        },
        utilization=[
            {"sectionName": "OutputWriter", "hwUtil": 1},
            {"sectionName": "VectorEngine", "hwUtil": 1},
        ],
        operators=["foo"],
    )
    with pytest.raises(KeyError):
        op_stats.sanitize_memory_fields()


def test_sanitize_utilization_fields_expected_input() -> None:
    """Sanitize utilization attribute of the NGP Operator."""
    op_stats = NGPOperatorPerformanceStats(
        op_id=[33],
        op_cycles=15,
        total_cycles=18,
        memory={
            "Undefined": {
                "readBytes": 0,
                "writeBytes": 0,
                "trafficCycles": 0,
            },
            "Internal": {
                "readBytes": 0,
                "writeBytes": 0,
                "trafficCycles": 0,
            },
            "L1": {
                "readBytes": 4,
                "writeBytes": 6,
                "trafficCycles": 43530,
            },
            "L2": {
                "readBytes": 5,
                "writeBytes": 7,
                "trafficCycles": 456570,
            },
            "SystemCache": {
                "readBytes": 1,
                "writeBytes": 2,
                "trafficCycles": 3,
            },
            "DRAM": {
                "readBytes": 312369600,
                "writeBytes": 31104000,
                "trafficCycles": 7056279,
            },
        },
        utilization=[
            {"sectionName": "OutputWriter", "hwUtil": 1},
            {"sectionName": "VectorEngine", "hwUtil": 1},
            {"sectionName": "OutputWriter", "hwUtil": 0.65625},
            {"sectionName": "VectorEngine", "hwUtil": 0.875},
            {"sectionName": "ConvolutionEngine", "hwUtil": 0.875},
            {"sectionName": "WeightDecoder", "hwUtil": 1},
            {"sectionName": "InputReader", "hwUtil": 1},
            {"sectionName": "InputReader", "hwUtil": 1},
        ],
        operators=["foo"],
    )
    op_stats.sanitize_utilization_fields()

    assert op_stats.utilization == [
        {"sectionName": "OutputWriter", "hwUtil": "0.828"},
        {"sectionName": "VectorEngine", "hwUtil": "0.938"},
        {"sectionName": "ConvolutionEngine", "hwUtil": "0.875"},
        {"sectionName": "WeightDecoder", "hwUtil": "1.0"},
        {"sectionName": "InputReader", "hwUtil": "1.0"},
    ]


def test_sanitize_utilization_fields_additional_utilization_field_input() -> None:
    """Sanitize utilization attribute when a new sectionName is added."""
    op_stats = NGPOperatorPerformanceStats(
        op_id=[33],
        op_cycles=15,
        total_cycles=18,
        memory={
            "Undefined": {
                "readBytes": 0,
                "writeBytes": 0,
                "trafficCycles": 0,
            },
            "Internal": {
                "readBytes": 0,
                "writeBytes": 0,
                "trafficCycles": 0,
            },
            "SystemCache": {
                "readBytes": 1,
                "writeBytes": 2,
                "trafficCycles": 3,
            },
            "DRAM": {
                "readBytes": 312369600,
                "writeBytes": 31104000,
                "trafficCycles": 7056279,
            },
        },
        utilization=[
            {"sectionName": "OutputWriter", "hwUtil": 1},
            {"sectionName": "VectorEngine", "hwUtil": 1},
            {"sectionName": "OutputWriter", "hwUtil": 0.65625},
            {"sectionName": "VectorEngine", "hwUtil": 0.875},
            {"sectionName": "ConvolutionEngine", "hwUtil": 0.875},
            {"sectionName": "Bar", "hwUtil": 1},
            {"sectionName": "Bar", "hwUtil": 1},
            {"sectionName": "Foo", "hwUtil": 1},
        ],
        operators=["foo"],
    )
    op_stats.sanitize_utilization_fields()

    assert op_stats.utilization == [
        {"sectionName": "OutputWriter", "hwUtil": "0.828"},
        {"sectionName": "VectorEngine", "hwUtil": "0.938"},
        {"sectionName": "ConvolutionEngine", "hwUtil": "0.875"},
        {"sectionName": "Bar", "hwUtil": "1.0"},
        {"sectionName": "Foo", "hwUtil": "1.0"},
    ]


def test_sanitize_utilization_fields_missing_keys_input() -> None:
    """Throw error when the sectionName key is missing from the utilization attibute."""
    op_stats = NGPOperatorPerformanceStats(
        op_id=[33],
        op_cycles=15,
        total_cycles=18,
        memory={
            "DRAM": {
                "readBytes": 312369600,
                "writeBytes": 31104000,
                "trafficCycles": 7056279,
            },
        },
        utilization=[
            {"bar": "OutputWriter", "foo": 1},
        ],
        operators=["foo"],
    )

    with pytest.raises(KeyError):
        op_stats.sanitize_utilization_fields()


def test_merge_ngp_operator_performance_stats() -> None:
    """Merge the performance stats of two performance operators."""
    op_stats_1 = NGPOperatorPerformanceStats(
        op_id=[33],
        op_cycles=15,
        total_cycles=18,
        memory={
            "L1": {
                "readBytes": 4,
                "writeBytes": 6,
                "trafficCycles": 43530,
            },
            "L2": {
                "readBytes": 5,
                "writeBytes": 7,
                "trafficCycles": 456570,
            },
            "SystemCache": {
                "readBytes": 1,
                "writeBytes": 2,
                "trafficCycles": 3,
            },
            "DRAM": {
                "readBytes": 312369600,
                "writeBytes": 31104000,
                "trafficCycles": 7056279,
            },
        },
        utilization=[
            {"sectionName": "OutputWriter", "hwUtil": 1},
            {"sectionName": "VectorEngine", "hwUtil": 1},
            {"sectionName": "ConvolutionEngine", "hwUtil": 0.875},
            {"sectionName": "WeightDecoder", "hwUtil": 1},
            {"sectionName": "InputReader", "hwUtil": 1},
        ],
        operators=["foo", "bar"],
    )

    op_stats_2 = NGPOperatorPerformanceStats(
        op_id=[11],
        op_cycles=5345,
        total_cycles=465,
        memory={
            "L1": {
                "readBytes": 564,
                "writeBytes": 3,
                "trafficCycles": 876,
            },
            "L2": {
                "readBytes": 65,
                "writeBytes": 87,
                "trafficCycles": 985,
            },
            "SystemCache": {
                "readBytes": 12,
                "writeBytes": 34,
                "trafficCycles": 54,
            },
            "DRAM": {
                "readBytes": 5675656,
                "writeBytes": 567567,
                "trafficCycles": 6876,
            },
        },
        utilization=[
            {"sectionName": "OutputWriter", "hwUtil": 1},
            {"sectionName": "VectorEngine", "hwUtil": 1},
        ],
        operators=["foo", "bar"],
    )
    op_stats_1_copy = copy.deepcopy(op_stats_1)
    op_stats_2_copy = copy.deepcopy(op_stats_2)

    op_stats_1.merge(op_stats_2_copy)
    op_stats_2.merge(op_stats_1_copy)

    assert set(op_stats_1.op_id) == set(op_stats_2.op_id) == {11, 33}
    assert op_stats_1.op_cycles == op_stats_2.op_cycles == 5360
    assert op_stats_1.total_cycles == op_stats_2.total_cycles == 483
    assert op_stats_1.memory == op_stats_2.memory
    assert op_stats_1.utilization == op_stats_2.utilization
    assert op_stats_1.operators == op_stats_2.operators


def test_merge_different_location_strings_error() -> None:
    """Throw an error when the same chain is mapped to different location strings."""
    op_stats_1 = NGPOperatorPerformanceStats(
        op_id=[33],
        op_cycles=15,
        total_cycles=18,
        memory={
            "L1": {
                "readBytes": 4,
                "writeBytes": 6,
                "trafficCycles": 43530,
            },
            "L2": {
                "readBytes": 5,
                "writeBytes": 7,
                "trafficCycles": 456570,
            },
            "SystemCache": {
                "readBytes": 1,
                "writeBytes": 2,
                "trafficCycles": 3,
            },
            "DRAM": {
                "readBytes": 312369600,
                "writeBytes": 31104000,
                "trafficCycles": 7056279,
            },
        },
        utilization=[
            {"sectionName": "OutputWriter", "hwUtil": 1},
            {"sectionName": "VectorEngine", "hwUtil": 1},
            {"sectionName": "OutputWriter", "hwUtil": 0.65625},
            {"sectionName": "VectorEngine", "hwUtil": 0.875},
            {"sectionName": "ConvolutionEngine", "hwUtil": 0.875},
            {"sectionName": "WeightDecoder", "hwUtil": 1},
            {"sectionName": "InputReader", "hwUtil": 1},
            {"sectionName": "InputReader", "hwUtil": 1},
        ],
        operators=[{"foo": "bar"}],
    )

    op_stats_2 = NGPOperatorPerformanceStats(
        op_id=[11],
        op_cycles=5345,
        total_cycles=465,
        memory={
            "L1": {
                "readBytes": 564,
                "writeBytes": 3,
                "trafficCycles": 876,
            },
            "L2": {
                "readBytes": 65,
                "writeBytes": 87,
                "trafficCycles": 985,
            },
            "SystemCache": {
                "readBytes": 12,
                "writeBytes": 34,
                "trafficCycles": 54,
            },
            "DRAM": {
                "readBytes": 5675656,
                "writeBytes": 567567,
                "trafficCycles": 6876,
            },
        },
        utilization=[
            {"sectionName": "OutputWriter", "hwUtil": 1},
            {"sectionName": "VectorEngine", "hwUtil": 1},
        ],
        operators=[{"foo": "test"}],
    )

    with pytest.raises(ValueError):
        op_stats_1.merge(op_stats_2)


def test_merge_missing_memory_name_input() -> None:
    """Throw an error when the memoryName key is missing."""
    op_stats_1 = NGPOperatorPerformanceStats(
        op_id=[33],
        op_cycles=15,
        total_cycles=18,
        memory={
            "L1": {
                "readBytes": 4,
                "writeBytes": 6,
                "trafficCycles": 43530,
            },
            "L2": {
                "readBytes": 5,
                "writeBytes": 7,
                "trafficCycles": 456570,
            },
            "SystemCache": {
                "readBytes": 1,
                "writeBytes": 2,
                "trafficCycles": 3,
            },
            "DRAM": {
                "readBytes": 312369600,
                "writeBytes": 31104000,
                "trafficCycles": 7056279,
            },
        },
        utilization=[
            {"sectionName": "OutputWriter", "hwUtil": 1},
            {"sectionName": "VectorEngine", "hwUtil": 1},
            {"sectionName": "OutputWriter", "hwUtil": 0.65625},
            {"sectionName": "VectorEngine", "hwUtil": 0.875},
            {"sectionName": "ConvolutionEngine", "hwUtil": 0.875},
            {"sectionName": "WeightDecoder", "hwUtil": 1},
            {"sectionName": "InputReader", "hwUtil": 1},
            {"sectionName": "InputReader", "hwUtil": 1},
        ],
        operators=["foo", "bar"],
    )

    op_stats_2 = NGPOperatorPerformanceStats(
        op_id=[33],
        op_cycles=15,
        total_cycles=18,
        memory={
            "wrong_key": {
                "readBytes": 0,
                "writeBytes": 0,
                "trafficCycles": 0,
            },
        },
        utilization=[
            {"sectionName": "OutputWriter", "hwUtil": 1},
            {"sectionName": "VectorEngine", "hwUtil": 1},
        ],
        operators=["foo", "bar"],
    )

    with pytest.raises(KeyError):
        op_stats_1.merge(op_stats_2)


def test_process_stats_per_chain(test_resources_path: Path) -> None:
    """Test that we can find all location strings."""
    debug_db_file = str(
        test_resources_path / "ngp/ds_cnn_large_fully_quantized_int8_debug_database.dat"
    )
    ddb_parser = NGPDebugDatabaseParser(Path(debug_db_file))
    debug_db = ddb_parser.parse_debug_database()

    perf_db_file = str(
        test_resources_path
        / "ngp/ds_cnn_large_fully_quantized_int8_performance_database.dat"
    )

    pdb_parser = NGPPerformanceDatabaseParser(Path(perf_db_file))
    performance_db = pdb_parser.parse_performance_database()

    operator_types_mapping = {
        "Identity": "identity_op_type",
        "model/re_lu_7/Relu": "RELU",
    }

    performance_stats = NGPPerformanceStats(
        debug_db, performance_db, operator_types_mapping
    )
    performance_stats_per_chain = performance_stats.process_stats_per_chain()

    # One chain per stripe, no accumulation of statistics
    performance_stats_chain_901 = NGPOperatorPerformanceStats(
        op_id=["26"],
        op_cycles=18,
        total_cycles=218,
        memory={
            "L1": {
                "readBytes": 0,
                "writeBytes": 0,
                "trafficCycles": 0,
            },
            "L2": {
                "readBytes": 0,
                "writeBytes": 0,
                "trafficCycles": 0,
            },
            "SystemCache": {
                "readBytes": 0,
                "writeBytes": 0,
                "trafficCycles": 0,
            },
            "DRAM": {
                "readBytes": 320,
                "writeBytes": 12,
                "trafficCycles": 10,
            },
        },
        utilization=[
            {"sectionName": "OutputWriter", "hwUtil": "1.0"},
            {"sectionName": "VectorEngine", "hwUtil": "0.25"},
            {"sectionName": "TransformUnit", "hwUtil": "0.25"},
            {"sectionName": "InputReader", "hwUtil": "0.125"},
        ],
        operators=[
            {"opLocation": ["Identity"], "opType": "identity_op_type"},
            {"opLocation": ["Identity"], "opType": "identity_op_type"},
            {"opLocation": ["Identity"], "opType": "identity_op_type"},
            {"opLocation": ["Identity"], "opType": "identity_op_type"},
            {"opLocation": ["Identity"], "opType": "identity_op_type"},
        ],
    )

    assert performance_stats_per_chain["901"] == performance_stats_chain_901

    # One chain shared by two stripes, accumulation of statistics
    # Note: the debug db was edited manually to create this scenario
    performance_stats_per_chain_907 = NGPOperatorPerformanceStats(
        op_id=["13", "12"],
        op_cycles=46,
        total_cycles=474,
        memory={
            "L1": {
                "readBytes": 0,
                "writeBytes": 60,
                "trafficCycles": 1,
            },
            "L2": {
                "readBytes": 0,
                "writeBytes": 0,
                "trafficCycles": 0,
            },
            "SystemCache": {
                "readBytes": 0,
                "writeBytes": 0,
                "trafficCycles": 0,
            },
            "DRAM": {
                "readBytes": 4940,
                "writeBytes": 60,
                "trafficCycles": 155,
            },
        },
        utilization=[
            {"sectionName": "OutputWriter", "hwUtil": "0.219"},
            {"sectionName": "VectorEngine", "hwUtil": "0.375"},
            {"sectionName": "InputReader", "hwUtil": "0.368"},
            {"sectionName": "ConvolutionEngine", "hwUtil": "0.094"},
            {"sectionName": "WeightDecoder", "hwUtil": "1.0"},
        ],
        operators=[
            {"opLocation": ["Identity"], "opType": "identity_op_type"},
        ],
    )

    assert performance_stats_per_chain["907"] == performance_stats_per_chain_907

    # One chain shared by three stripes, accumulation of statistics
    # One TOSA op maps to multiple tflite location strings
    # Note: the debug db was edited manually to create this scenario
    performance_stats_per_chain_619 = NGPOperatorPerformanceStats(
        op_id=["7", "8", "9"],
        op_cycles=3042,
        total_cycles=11822,
        memory={
            "L1": {
                "readBytes": 0,
                "writeBytes": 53820,
                "trafficCycles": 336,
            },
            "L2": {
                "readBytes": 0,
                "writeBytes": 0,
                "trafficCycles": 0,
            },
            "SystemCache": {
                "readBytes": 0,
                "writeBytes": 0,
                "trafficCycles": 0,
            },
            "DRAM": {
                "readBytes": 409530,
                "writeBytes": 53820,
                "trafficCycles": 9518,
            },
        },
        utilization=[
            {"sectionName": "OutputWriter", "hwUtil": "0.625"},
            {"sectionName": "VectorEngine", "hwUtil": "0.833"},
            {"sectionName": "ConvolutionEngine", "hwUtil": "0.575"},
            {"sectionName": "WeightDecoder", "hwUtil": "1.0"},
            {"sectionName": "InputReader", "hwUtil": "0.864"},
        ],
        operators=[
            {
                "opLocation": ["model/re_lu_7/Relu;test_location_string"],
                "opType": "<unknown>",
            },
            {"opLocation": ["model/re_lu_7/Relu"], "opType": "RELU"},
        ],
    )

    assert performance_stats_per_chain["619"] == performance_stats_per_chain_619


def test_track_op(test_resources_path: Path) -> None:
    """Test that we can track location strings from a chain id."""
    debug_db_file = str(
        test_resources_path / "ngp/ds_cnn_large_fully_quantized_int8_debug_database.dat"
    )
    ddb_parser = NGPDebugDatabaseParser(Path(debug_db_file))
    debug_db = ddb_parser.parse_debug_database()

    perf_db_file = str(
        test_resources_path
        / "ngp/ds_cnn_large_fully_quantized_int8_performance_database.dat"
    )

    pdb_parser = NGPPerformanceDatabaseParser(Path(perf_db_file))
    performance_db = pdb_parser.parse_performance_database()
    performance_stats = NGPPerformanceStats(
        debug_db=debug_db, performance_db=performance_db, operator_types_mapping={}
    )

    chain_op_id, location_strings = performance_stats.track_op("26")

    assert chain_op_id == "901"
    assert location_strings == [
        ["Identity"],
        ["Identity"],
        ["Identity"],
        ["Identity"],
        ["Identity"],
    ]

    chain_op_id, location_strings = performance_stats.track_op("0")
    assert chain_op_id == "605"
    # pylint: disable=line-too-long
    assert location_strings == [
        [
            "deeplabv3plus_mbnV2__1080p/Conv_Relu6/Relu6;deeplabv3plus_mbnV2__1080p/Conv_BN/FusedBatchNormV3;deeplabv3plus_mbnV2__1080p/expanded_conv_5_project/Conv2D;deeplabv3plus_mbnV2__1080p/Conv/Conv2D"
        ],
        [
            "deeplabv3plus_mbnV2__1080p/Conv_Relu6/Relu6;deeplabv3plus_mbnV2__1080p/Conv_BN/FusedBatchNormV3;deeplabv3plus_mbnV2__1080p/expanded_conv_5_project/Conv2D;deeplabv3plus_mbnV2__1080p/Conv/Conv2D"
        ],
    ]
