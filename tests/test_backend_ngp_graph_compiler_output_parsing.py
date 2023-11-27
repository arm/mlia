# SPDX-FileCopyrightText: Copyright 2023, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: LicenseRef-LICENSE
"""Tests for NGP Graph Compiler performance estimation."""
from __future__ import annotations

from pathlib import Path

import pytest

from mlia.backend.ngp_graph_compiler.output_parsing import extract_cdata
from mlia.backend.ngp_graph_compiler.output_parsing import extract_field
from mlia.backend.ngp_graph_compiler.output_parsing import NGPPerformanceDatabase
from mlia.backend.ngp_graph_compiler.output_parsing import SubtableColumnParser


def test_extract_cdata() -> None:
    """Test util method to extract cdata."""

    with pytest.raises(Exception) as exc_info:
        extract_cdata("""<![CDATA[a]]>....<![CDATA[b]]>""")
        assert str(exc_info.value) == "No single CDATA section"

    assert "abc" == extract_cdata(
        """line1
    line2
    <![CDATA[
    abc
    ]]>
    line3
    """
    )


def test_parse_contents() -> None:
    """Testing with a complete csv body."""
    contents = """
    "id", "opCycles", "totalCycles", "memoryName;readBytes;writeBytes;trafficCycles", "sectionName;hwUtil"
    26, 18, 212, Undefined;0;0;0;Internal;0;0;0;L1;0;0;0;L2;0;0;0;SystemCache;0;0;0;DRAM;320;12;10;, OutputWriter;1;VectorEngine;0.25;VectorEngine;0.25;VectorEngine;0.25;TransformUnit;0.25;TransformUnit;0.25;InputReader;0.0625;InputReader;0.0625;InputReader;0.25;
    25, 4, 13, Undefined;0;0;0;Internal;0;0;0;L1;0;4;0;L2;0;0;0;SystemCache;0;0;0;DRAM;128;4;4;, OutputWriter;0.0625;VectorEngine;0.125;VectorEngine;0.125;VectorEngine;0.125;VectorEngine;0.125;InputReader;0.0625;InputReader;0.0625;
    """.strip()

    pdb = NGPPerformanceDatabase()
    assert pdb.parse_contents(contents) == [
        {
            "Memory": [
                {
                    "memoryName": "Undefined",
                    "readBytes": "0",
                    "trafficCycles": "0",
                    "writeBytes": "0",
                },
                {
                    "memoryName": "Internal",
                    "readBytes": "0",
                    "trafficCycles": "0",
                    "writeBytes": "0",
                },
                {
                    "memoryName": "L1",
                    "readBytes": "0",
                    "trafficCycles": "0",
                    "writeBytes": "0",
                },
                {
                    "memoryName": "L2",
                    "readBytes": "0",
                    "trafficCycles": "0",
                    "writeBytes": "0",
                },
                {
                    "memoryName": "SystemCache",
                    "readBytes": "0",
                    "trafficCycles": "0",
                    "writeBytes": "0",
                },
                {
                    "memoryName": "DRAM",
                    "readBytes": "320",
                    "trafficCycles": "10",
                    "writeBytes": "12",
                },
            ],
            "Utilization": [
                {"hwUtil": "1", "sectionName": "OutputWriter"},
                {"hwUtil": "0.25", "sectionName": "VectorEngine"},
                {"hwUtil": "0.25", "sectionName": "VectorEngine"},
                {"hwUtil": "0.25", "sectionName": "VectorEngine"},
                {"hwUtil": "0.25", "sectionName": "TransformUnit"},
                {"hwUtil": "0.25", "sectionName": "TransformUnit"},
                {"hwUtil": "0.0625", "sectionName": "InputReader"},
                {"hwUtil": "0.0625", "sectionName": "InputReader"},
                {"hwUtil": "0.25", "sectionName": "InputReader"},
            ],
            "id": 26,
            "opCycles": 18,
            "totalCycles": 212,
        },
        {
            "Memory": [
                {
                    "memoryName": "Undefined",
                    "readBytes": "0",
                    "trafficCycles": "0",
                    "writeBytes": "0",
                },
                {
                    "memoryName": "Internal",
                    "readBytes": "0",
                    "trafficCycles": "0",
                    "writeBytes": "0",
                },
                {
                    "memoryName": "L1",
                    "readBytes": "0",
                    "trafficCycles": "0",
                    "writeBytes": "4",
                },
                {
                    "memoryName": "L2",
                    "readBytes": "0",
                    "trafficCycles": "0",
                    "writeBytes": "0",
                },
                {
                    "memoryName": "SystemCache",
                    "readBytes": "0",
                    "trafficCycles": "0",
                    "writeBytes": "0",
                },
                {
                    "memoryName": "DRAM",
                    "readBytes": "128",
                    "trafficCycles": "4",
                    "writeBytes": "4",
                },
            ],
            "Utilization": [
                {"hwUtil": "0.0625", "sectionName": "OutputWriter"},
                {"hwUtil": "0.125", "sectionName": "VectorEngine"},
                {"hwUtil": "0.125", "sectionName": "VectorEngine"},
                {"hwUtil": "0.125", "sectionName": "VectorEngine"},
                {"hwUtil": "0.125", "sectionName": "VectorEngine"},
                {"hwUtil": "0.0625", "sectionName": "InputReader"},
                {"hwUtil": "0.0625", "sectionName": "InputReader"},
            ],
            "id": 25,
            "opCycles": 4,
            "totalCycles": 13,
        },
    ]


def test_subtable_column() -> None:
    """Test"""
    parser = SubtableColumnParser(
        "Meminfo", "memoryName;readBytes;writeBytes;trafficCycles"
    )

    assert parser.sub_columns == [
        "memoryName",
        "readBytes",
        "writeBytes",
        "trafficCycles",
    ]

    result = parser(
        "Undefined;0;0;0;Internal;0;0;0;L1;0;0;0;L2;0;0;0;SystemCache;0;0;0;DRAM;320;12;10;"  # pylint: disable=line-too-long
    )

    assert result == [
        {
            "memoryName": "Undefined",
            "readBytes": "0",
            "writeBytes": "0",
            "trafficCycles": "0",
        },
        {
            "memoryName": "Internal",
            "readBytes": "0",
            "writeBytes": "0",
            "trafficCycles": "0",
        },
        {"memoryName": "L1", "readBytes": "0", "writeBytes": "0", "trafficCycles": "0"},
        {"memoryName": "L2", "readBytes": "0", "writeBytes": "0", "trafficCycles": "0"},
        {
            "memoryName": "SystemCache",
            "readBytes": "0",
            "writeBytes": "0",
            "trafficCycles": "0",
        },
        {
            "memoryName": "DRAM",
            "readBytes": "320",
            "writeBytes": "12",
            "trafficCycles": "10",
        },
    ]


def test_parse_file(test_resources_path: Path) -> None:
    """Parsing the whole file."""
    perf_db_file = str(
        test_resources_path
        / "ngp/ds_cnn_large_fully_quantized_int8_performance_database.dat"
    )
    records = NGPPerformanceDatabase().load(Path(perf_db_file))
    assert len(records) == 27
    assert records[0] == {
        "Memory": [
            {
                "memoryName": "Undefined",
                "readBytes": "0",
                "trafficCycles": "0",
                "writeBytes": "0",
            },
            {
                "memoryName": "Internal",
                "readBytes": "0",
                "trafficCycles": "0",
                "writeBytes": "0",
            },
            {
                "memoryName": "L1",
                "readBytes": "0",
                "trafficCycles": "0",
                "writeBytes": "0",
            },
            {
                "memoryName": "L2",
                "readBytes": "0",
                "trafficCycles": "0",
                "writeBytes": "0",
            },
            {
                "memoryName": "SystemCache",
                "readBytes": "0",
                "trafficCycles": "0",
                "writeBytes": "0",
            },
            {
                "memoryName": "DRAM",
                "readBytes": "320",
                "trafficCycles": "10",
                "writeBytes": "12",
            },
        ],
        "Utilization": [
            {"hwUtil": "1", "sectionName": "OutputWriter"},
            {"hwUtil": "0.25", "sectionName": "VectorEngine"},
            {"hwUtil": "0.25", "sectionName": "VectorEngine"},
            {"hwUtil": "0.25", "sectionName": "VectorEngine"},
            {"hwUtil": "0.25", "sectionName": "TransformUnit"},
            {"hwUtil": "0.25", "sectionName": "TransformUnit"},
            {"hwUtil": "0.0625", "sectionName": "InputReader"},
            {"hwUtil": "0.0625", "sectionName": "InputReader"},
            {"hwUtil": "0.25", "sectionName": "InputReader"},
        ],
        "id": 26,
        "opCycles": 18,
        "totalCycles": 212,
    }


def test_column_parsers() -> None:
    """Test if column parsers are set up properly."""
    pdb = NGPPerformanceDatabase()
    parsers = pdb._column_parsers  # pylint: disable=protected-access
    col1 = "memoryName;readBytes;writeBytes;trafficCycles"
    col2 = "sectionName;hwUtil"
    assert parsers == {
        col1: SubtableColumnParser("Memory", col1),
        col2: SubtableColumnParser("Utilization", col2),
    }


@pytest.mark.parametrize(
    "variant",
    [
        "'field' ",
        '  "field" ',
        ' "field"',
        "field",
    ],
)
def test_extract_field(variant: str) -> None:
    """Test extract field."""
    assert "field" == extract_field(variant)
