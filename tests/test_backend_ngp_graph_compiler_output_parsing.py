# SPDX-FileCopyrightText: Copyright 2023-2024, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: LicenseRef-LICENSE
"""Tests for NGP Graph Compiler performance estimation."""
from __future__ import annotations

import csv
from pathlib import Path

import pytest

from mlia.backend.ngp_graph_compiler.output_parsing import NGPDebugDatabaseParser
from mlia.backend.ngp_graph_compiler.output_parsing import NGPOutputParser
from mlia.backend.ngp_graph_compiler.output_parsing import NGPPerformanceDatabaseParser
from mlia.backend.ngp_graph_compiler.output_parsing import SubtableColumnParser


def test_load(test_resources_path: Path) -> None:
    """Load a file into an NGP output parser."""
    perf_db_file = str(
        test_resources_path
        / "ngp/ds_cnn_large_fully_quantized_int8_performance_database.dat"
    )

    debug_db_file = str(
        test_resources_path / "ngp/ds_cnn_large_fully_quantized_int8_debug_database.dat"
    )

    parser = NGPOutputParser()

    loaded_perf_db = parser.load(Path(perf_db_file))
    assert loaded_perf_db == parser.raw_xmlish

    loaded_debug_db = parser.load(Path(debug_db_file))
    assert loaded_debug_db == parser.raw_xmlish


def test_get_csv_reader() -> None:
    """Read string into csv."""
    contents = """
    "id", "opCycles", "totalCycles", "memoryName;readBytes;writeBytes;trafficCycles", "sectionName;hwUtil"
    26, 18, 212, Undefined;0;0;0;Internal;0;0;0;L1;0;0;0;L2;0;0;0;SystemCache;0;0;0;DRAM;320;12;10;, OutputWriter;1;VectorEngine;0.25;VectorEngine;0.25;VectorEngine;0.25;TransformUnit;0.25;TransformUnit;0.25;InputReader;0.0625;InputReader;0.0625;InputReader;0.25;
    25, 4, 13, Undefined;0;0;0;Internal;0;0;0;L1;0;4;0;L2;0;0;0;SystemCache;0;0;0;DRAM;128;4;4;, OutputWriter;0.0625;VectorEngine;0.125;VectorEngine;0.125;VectorEngine;0.125;VectorEngine;0.125;InputReader;0.0625;InputReader;0.0625;
    """.strip()
    parser = NGPOutputParser()
    csv_reader = parser.get_csv_reader(table_data=contents)
    expected_csv_reader = csv.reader(contents.splitlines())

    for actual, expected in zip(csv_reader, expected_csv_reader):
        assert actual == expected


def test_get_csv_headers() -> None:
    """Extract the headers from a csv reader."""
    contents = """
    "id", "opCycles", "totalCycles", "memoryName;readBytes;writeBytes;trafficCycles", "sectionName;hwUtil"
    26, 18, 212, Undefined;0;0;0;Internal;0;0;0;L1;0;0;0;L2;0;0;0;SystemCache;0;0;0;DRAM;320;12;10;, OutputWriter;1;VectorEngine;0.25;VectorEngine;0.25;VectorEngine;0.25;TransformUnit;0.25;TransformUnit;0.25;InputReader;0.0625;InputReader;0.0625;InputReader;0.25;
    25, 4, 13, Undefined;0;0;0;Internal;0;0;0;L1;0;4;0;L2;0;0;0;SystemCache;0;0;0;DRAM;128;4;4;, OutputWriter;0.0625;VectorEngine;0.125;VectorEngine;0.125;VectorEngine;0.125;VectorEngine;0.125;InputReader;0.0625;InputReader;0.0625;
    """.strip()
    parser = NGPOutputParser()
    csv_reader = parser.get_csv_reader(table_data=contents)
    csv_headers = parser.get_csv_headers(csv_reader=csv_reader)
    assert csv_headers == [
        "id",
        "opCycles",
        "totalCycles",
        "memoryName;readBytes;writeBytes;trafficCycles",
        "sectionName;hwUtil",
    ]


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
    parser = NGPOutputParser()
    assert "field" == parser.extract_field(variant)


def test_extract_cdata() -> None:
    """Test util method to extract cdata."""
    parser = NGPOutputParser()
    with pytest.raises(Exception) as exc_info:
        parser.extract_cdata("<![CDATA[a]]>....<![CDATA[b]]>")
        assert str(exc_info.value) == "No single CDATA section"

    with pytest.raises(Exception) as exc_info:
        parser.extract_cdata("foo")
        assert str(exc_info.value) == "No single CDATA section"

    assert "abc" == parser.extract_cdata(
        """line1
    line2
    <![CDATA[
    abc
    ]]>
    line3
    """
    )


def test_performance_database_parser_from_file(test_resources_path: Path) -> None:
    """Parse the whole file."""
    perf_db_file = str(
        test_resources_path
        / "ngp/ds_cnn_large_fully_quantized_int8_performance_database.dat"
    )
    parser = NGPPerformanceDatabaseParser()
    parser.load(Path(perf_db_file))
    records = parser.parse_performance_database()
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


def test_register_sub_table() -> None:
    """Add a subtable to the performance db parser."""
    parser = NGPPerformanceDatabaseParser()
    column_parsers = parser.register_sub_table(title="foo", header="bar")
    assert column_parsers == parser.column_parsers


def test_parse_performance_database() -> None:
    """Testing with a CDATA xml body."""
    contents = """
    <![CDATA[
    "id", "opCycles", "totalCycles", "memoryName;readBytes;writeBytes;trafficCycles", "sectionName;hwUtil"
    26, 18, 212, Undefined;0;0;0;Internal;0;0;0;L1;0;0;0;L2;0;0;0;SystemCache;0;0;0;DRAM;320;12;10;, OutputWriter;1;VectorEngine;0.25;VectorEngine;0.25;VectorEngine;0.25;TransformUnit;0.25;TransformUnit;0.25;InputReader;0.0625;InputReader;0.0625;InputReader;0.25;
    25, 4, 13, Undefined;0;0;0;Internal;0;0;0;L1;0;4;0;L2;0;0;0;SystemCache;0;0;0;DRAM;128;4;4;, OutputWriter;0.0625;VectorEngine;0.125;VectorEngine;0.125;VectorEngine;0.125;VectorEngine;0.125;InputReader;0.0625;InputReader;0.0625;
    ]]>
    """.strip()
    pdb_parser = NGPPerformanceDatabaseParser()
    pdb_parser.raw_xmlish = contents
    assert pdb_parser.parse_performance_database() == [
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


def test_make_parsed_db_performance_db() -> None:
    """
    Test the performance_db has the required fields.
    """
    contents = """
    "id", "opCycles", "totalCycles", "memoryName;readBytes;writeBytes;trafficCycles", "sectionName;hwUtil"
    26, 18, 212, Undefined;0;0;0;Internal;0;0;0;L1;0;0;0;L2;0;0;0;SystemCache;0;0;0;DRAM;320;12;10;, OutputWriter;1;VectorEngine;0.25;VectorEngine;0.25;VectorEngine;0.25;TransformUnit;0.25;TransformUnit;0.25;InputReader;0.0625;InputReader;0.0625;InputReader;0.25;
    25, 4, 13, Undefined;0;0;0;Internal;0;0;0;L1;0;4;0;L2;0;0;0;SystemCache;0;0;0;DRAM;128;4;4;, OutputWriter;0.0625;VectorEngine;0.125;VectorEngine;0.125;VectorEngine;0.125;VectorEngine;0.125;InputReader;0.0625;InputReader;0.0625;
    """.strip()
    parser = NGPPerformanceDatabaseParser()
    reader = parser.get_csv_reader(table_data=contents)
    headers = parser.get_csv_headers(csv_reader=reader)
    int_column_parsers = parser.set_column_parsers(headers=headers, content_type=int)
    parser.make_parsed_db(
        csv_reader=reader, headers=headers, column_parsers=int_column_parsers
    )

    assert parser.performance_db == [
        {
            "id": 26,
            "opCycles": 18,
            "totalCycles": 212,
            "Memory": [
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
                {
                    "memoryName": "L1",
                    "readBytes": "0",
                    "writeBytes": "0",
                    "trafficCycles": "0",
                },
                {
                    "memoryName": "L2",
                    "readBytes": "0",
                    "writeBytes": "0",
                    "trafficCycles": "0",
                },
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
            ],
            "Utilization": [
                {"sectionName": "OutputWriter", "hwUtil": "1"},
                {"sectionName": "VectorEngine", "hwUtil": "0.25"},
                {"sectionName": "VectorEngine", "hwUtil": "0.25"},
                {"sectionName": "VectorEngine", "hwUtil": "0.25"},
                {"sectionName": "TransformUnit", "hwUtil": "0.25"},
                {"sectionName": "TransformUnit", "hwUtil": "0.25"},
                {"sectionName": "InputReader", "hwUtil": "0.0625"},
                {"sectionName": "InputReader", "hwUtil": "0.0625"},
                {"sectionName": "InputReader", "hwUtil": "0.25"},
            ],
        },
        {
            "id": 25,
            "opCycles": 4,
            "totalCycles": 13,
            "Memory": [
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
                {
                    "memoryName": "L1",
                    "readBytes": "0",
                    "writeBytes": "4",
                    "trafficCycles": "0",
                },
                {
                    "memoryName": "L2",
                    "readBytes": "0",
                    "writeBytes": "0",
                    "trafficCycles": "0",
                },
                {
                    "memoryName": "SystemCache",
                    "readBytes": "0",
                    "writeBytes": "0",
                    "trafficCycles": "0",
                },
                {
                    "memoryName": "DRAM",
                    "readBytes": "128",
                    "writeBytes": "4",
                    "trafficCycles": "4",
                },
            ],
            "Utilization": [
                {"sectionName": "OutputWriter", "hwUtil": "0.0625"},
                {"sectionName": "VectorEngine", "hwUtil": "0.125"},
                {"sectionName": "VectorEngine", "hwUtil": "0.125"},
                {"sectionName": "VectorEngine", "hwUtil": "0.125"},
                {"sectionName": "VectorEngine", "hwUtil": "0.125"},
                {"sectionName": "InputReader", "hwUtil": "0.0625"},
                {"sectionName": "InputReader", "hwUtil": "0.0625"},
            ],
        },
    ]


def test_debug_database_parser_from_file(test_resources_path: Path) -> None:
    """Parse the whole file."""
    debug_db_file = str(
        test_resources_path / "ngp/ds_cnn_large_fully_quantized_int8_debug_database.dat"
    )
    parser = NGPDebugDatabaseParser()
    parser.load(Path(debug_db_file))
    records = parser.parse_debug_database()
    assert len(records) == 7
    assert records["fused_op_id_to_tosa_op_ids"]["531"] == ["398"]
    assert records["fused_op_id_to_tosa_op_ids"]["497"] == ["394", "461"]
    assert records["chain_op_id_to_fused_op_ids"]["639"] == [
        "591",
        "593",
        "511",
        "513",
        "517",
    ]


def test_parse_debug_database() -> None:
    """Test the debug database has the required key-value pairs."""
    contents = """<?xml version='1.0' encoding='utf-8' ?>
    <![CDATA[\n"id", "api_id"\n]]>\n</table>\n<table name="fused_op_id">
    <![CDATA[\n"id", "tosa_op_ids"\n531, 334;\n557, 335;\n499, 394;462;;\n]]>\n</table>\n<table name="chain_op_id">
    <![CDATA[\n"id", "fused_op_ids"\n603, 531;557;\n605, 533;559;\n607, 535;561;\n637, 589;591;509;511;515;\n]]>\n<table name="stripe_op_id">
    <![CDATA[\n"id", "chain_op_id", "cascade_op_id"\n0, 603, 1693;\n1, 605, 1691;\n]]>
    </table>\n</debug>"""
    parser = NGPDebugDatabaseParser()
    parser.raw_xmlish = contents
    records = parser.parse_debug_database()
    assert len(records) == 4
    assert records["fused_op_id_to_tosa_op_ids"]["531"] == ["334"]
    assert records["fused_op_id_to_tosa_op_ids"]["499"] == ["394", "462"]
    assert records["chain_op_id_to_fused_op_ids"]["637"] == [
        "589",
        "591",
        "509",
        "511",
        "515",
    ]
    assert records["stripe_op_id_to_chain_op_id"]["0"] == ["603"]
    assert records["stripe_op_id_to_cascade_op_id"]["0"] == ["1693"]


def test_parse_debug_database_invalid_num_db_headers() -> None:
    """Test error is raised if the debug database has too many headers."""
    contents = """<?xml version='1.0' encoding='utf-8' ?>
    <![CDATA[\n"id", "api_id"\n]]>\n</table>\n<table name="fused_op_id">
    <![CDATA[\n"id", "chain_op_id", "cascade_op_id", "foo"\n0, 603, 1693, 100\n1, 605, 1691, 200\n<table name="chain_op_id">
    <![CDATA[\n"id", "fused_op_ids"\n603, 531;557;\n605, 533;559;\n607, 535;561;\n637, 589;591;509;511;515;\n]]>
    </table>\n</debug>"""
    parser = NGPDebugDatabaseParser()
    parser.raw_xmlish = contents
    with pytest.raises(Exception) as exc_info:
        parser.parse_debug_database()
        assert str(exc_info.value) == "Unsupported number of headers."


def test_make_parsed_db_debug_db() -> None:
    """Test the debug database has the required key-value pairs."""
    contents = """<?xml version='1.0' encoding='utf-8' ?>
    <![CDATA[\n"id", "api_id"\n]]>\n</table>\n<table name="fused_op_id">
    <![CDATA[\n"id", "tosa_op_ids"\n531, 334;\n557, 335;\n499, 394;462;;\n]]>\n</table>\n<table name="chain_op_id">
    <![CDATA[\n"id", "fused_op_ids"\n603, 531;557;\n605, 533;559;\n607, 535;561;\n637, 589;591;509;511;515;\n]]>\n<table name="stripe_op_id">
    <![CDATA[\n"id", "chain_op_id", "cascade_op_id"\n0, 603, 1693;\n1, 605, 1691;\n]]>
    </table>\n</debug>"""
    parser = NGPDebugDatabaseParser()
    table_elements = contents.split('<table name="')[1:]
    for table_element in table_elements:
        table_name = table_element.split('">')[0]
        table_data = parser.extract_cdata(table_element)
        reader = parser.get_csv_reader(table_data=table_data)
        headers = parser.get_csv_headers(csv_reader=reader)
        headers[0] = table_name + "_to_" + headers[1]
        parser.make_parsed_db(csv_reader=reader, headers=headers)

    assert len(parser.debug_db) == 4
    assert parser.debug_db["fused_op_id_to_tosa_op_ids"]["531"] == ["334"]
    assert parser.debug_db["fused_op_id_to_tosa_op_ids"]["499"] == ["394", "462"]
    assert parser.debug_db["chain_op_id_to_fused_op_ids"]["637"] == [
        "589",
        "591",
        "509",
        "511",
        "515",
    ]
    assert parser.debug_db["stripe_op_id_to_chain_op_id"]["0"] == ["603"]


def test_get_performance_stats(test_resources_path: Path) -> None:
    """Test that given a performance db we can find all the corresponding TOSA ids."""
    debug_db_file = str(
        test_resources_path / "ngp/ds_cnn_large_fully_quantized_int8_debug_database.dat"
    )
    ddb_parser = NGPDebugDatabaseParser()
    ddb_parser.load(Path(debug_db_file))
    ddb_parser.parse_debug_database()

    perf_db_file = str(
        test_resources_path
        / "ngp/ds_cnn_large_fully_quantized_int8_performance_database.dat"
    )

    pdb_parser = NGPPerformanceDatabaseParser()
    pdb_parser.load(Path(perf_db_file))
    performance_db = pdb_parser.parse_performance_database()

    with pytest.raises(Exception) as exc_info:
        performance_stats = ddb_parser.get_performance_stats(
            performance_db=performance_db, target="foo"
        )
        assert str(exc_info.value) == "Tracking operation foo is not supported."

    performance_stats = ddb_parser.get_performance_stats(performance_db=performance_db)

    assert performance_stats[0] == {
        "id": 26,
        "opCycles": 18,
        "totalCycles": 212,
        "Memory": [
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
            {
                "memoryName": "L1",
                "readBytes": "0",
                "writeBytes": "0",
                "trafficCycles": "0",
            },
            {
                "memoryName": "L2",
                "readBytes": "0",
                "writeBytes": "0",
                "trafficCycles": "0",
            },
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
        ],
        "Utilization": [
            {"sectionName": "OutputWriter", "hwUtil": "1"},
            {"sectionName": "VectorEngine", "hwUtil": "0.25"},
            {"sectionName": "VectorEngine", "hwUtil": "0.25"},
            {"sectionName": "VectorEngine", "hwUtil": "0.25"},
            {"sectionName": "TransformUnit", "hwUtil": "0.25"},
            {"sectionName": "TransformUnit", "hwUtil": "0.25"},
            {"sectionName": "InputReader", "hwUtil": "0.0625"},
            {"sectionName": "InputReader", "hwUtil": "0.0625"},
            {"sectionName": "InputReader", "hwUtil": "0.25"},
        ],
        "tosa_op_ids": [["399", "467"], ["401", "470"], ["402"]],
    }


def test_track_op() -> None:
    """Test that given a stripe op id we can track the corresponding TOSA ops."""
    contents = """<?xml version='1.0' encoding='utf-8' ?>
    <![CDATA[\n"id", "api_id"\n]]>\n</table>\n<table name="fused_op_id">
    <![CDATA[\n"id", "tosa_op_ids"\n531, 334;\n557, 335;\n499, 394;462;;\n]]>\n</table>\n<table name="chain_op_id">
    <![CDATA[\n"id", "fused_op_ids"\n603, 531;557;\n605, 533;559;\n607, 535;561;\n637, 589;591;509;511;515;\n]]>\n<table name="stripe_op_id">
    <![CDATA[\n"id", "chain_op_id", "cascade_op_id"\n0, 603, 1693;\n1, 605, 1691;\n]]>
    </table>\n</debug>"""
    parser = NGPDebugDatabaseParser()
    parser.raw_xmlish = contents
    parser.parse_debug_database()

    with pytest.raises(Exception) as exc_info:
        parser.track_op(stripe_op_id="0", target="foo")
        assert str(exc_info.value) == "Tracking operation foo is not supported."

    assert parser.track_op(stripe_op_id="0") == [["334"], ["335"]]


def test_subtable_column() -> None:
    """Test the subtable columns have the expected labels."""
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


def test_column_parsers() -> None:
    """Test if column parsers are set up properly."""
    pdb = NGPPerformanceDatabaseParser()
    parsers = pdb.column_parsers  # pylint: disable=protected-access
    col1 = "memoryName;readBytes;writeBytes;trafficCycles"
    col2 = "sectionName;hwUtil"
    assert parsers == {
        col1: SubtableColumnParser("Memory", col1),
        col2: SubtableColumnParser("Utilization", col2),
    }
