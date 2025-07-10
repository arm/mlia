# SPDX-FileCopyrightText: Copyright 2023-2025, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: LicenseRef-LICENSE
"""Tests for Neural Technology reporters."""
from functools import partial
from pathlib import Path
from typing import List

import pytest
from rich.console import Console

from mlia.backend.nx_graph_compiler.config import NXGraphCompilerConfig
from mlia.backend.nx_graph_compiler.output_parsing import NXDebugDatabaseParser
from mlia.backend.nx_graph_compiler.output_parsing import NXPerformanceDatabaseParser
from mlia.backend.nx_graph_compiler.performance import NXGraphCompilerOutputFiles
from mlia.backend.nx_graph_compiler.performance import (
    NXGraphCompilerPerformanceMetrics,
)
from mlia.backend.nx_graph_compiler.statistics import NXPerformanceStats
from mlia.backend.vulkan_model_converter.compat import NXModelCompatibilityInfo
from mlia.core.reporting import Table
from mlia.target.neural_technology.config import NeuralTechnologyConfiguration
from mlia.target.neural_technology.reporters import neural_technology_formatters
from mlia.target.neural_technology.reporters import report_target
from mlia.utils.console import remove_ascii_codes


def test_report_target() -> None:
    """Test function report_target()."""
    report = report_target(
        NeuralTechnologyConfiguration.load_profile("neural-technology")
    )
    assert report.to_plain_text()


def assert_table_contents(report: Table, json: dict) -> None:
    """Assert that a given Table renders the expected JSON output."""
    assert isinstance(report, Table)
    assert report.to_json() == json


def assert_table_lines(report: Table, expected_lines: list) -> None:
    """Assert that a given Table renders the expected JSON output.

    In case of failure, it renders actual and expected textual tables in a form
    that's easy to overview and can directly be used as "golden" data in the test.
    """
    assert isinstance(report, Table)
    actual_lines = remove_ascii_codes(report.to_plain_text()).split("\n")

    def to_diff_string(lines: List[str]) -> str:
        test_line = [f'          "{line}",' for line in lines]
        return ("\n").join(test_line)

    actual = to_diff_string(actual_lines)
    expected = to_diff_string(expected_lines)
    assert actual_lines == expected_lines, f"Expected:\n{expected}\n\nActual:\n{actual}"


def test_nx_graph_compiler_reporting(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test function neural_technology_formatters() with Neural Accelerator performance data."""

    performance_contents = """
    <![CDATA[
    "id", "opCycles", "totalCycles", "memoryName;readBytes;writeBytes;trafficCycles", "sectionName;hwUtil"
    26, 18, 212, Undefined;0;0;0;Internal;0;0;0;L1;0;0;0;L2;0;0;0;SystemCache;0;0;0;DRAM;320;12;10;, OutputWriter;1;VectorEngine;0.25;VectorEngine;0.25;VectorEngine;0.25;TransformUnit;0.25;TransformUnit;0.25;InputReader;0.0625;InputReader;0.0625;InputReader;0.25;
    25, 4, 13, Undefined;0;0;0;Internal;0;0;0;L1;0;4;0;L2;0;0;0;SystemCache;0;0;0;DRAM;128;4;4;, OutputWriter;0.0625;VectorEngine;0.125;VectorEngine;0.125;VectorEngine;0.125;VectorEngine;0.125;InputReader;0.0625;InputReader;0.0625;
    ]]>
    """.strip()

    debug_contents = """
<?xml version='1.0' encoding='utf-8' ?>
<debug_database>
<regor_version>1.0.0</regor_version>
<table name="tosa_op_id">
<![CDATA[
"id", "tosa_op", "api_labels"
1077, DepthwiseConv2D, deeplabv3plus_mbnV2__1080p/expanded_conv_8_depthwise_relu/Relu6;deeplabv3plus_mbnV2__1080p/expanded_conv_8_depthwise_BN/FusedBatchNormV3;deeplabv3plus_mbnV2__1080p/expanded_conv_10_depthwise/depthwise;deeplabv3plus_mbnV2__1080p/expanded_conv_8_depthwise/depthwise;
1078, Rescale, deeplabv3plus_mbnV2__1080p/expanded_conv_8_depthwise_relu/Relu6;deeplabv3plus_mbnV2__1080p/expanded_conv_8_depthwise_BN/FusedBatchNormV3;deeplabv3plus_mbnV2__1080p/expanded_conv_10_depthwise/depthwise;deeplabv3plus_mbnV2__1080p/expanded_conv_8_depthwise/depthwise;
1079, Conv2D, deeplabv3plus_mbnV2__1080p/expanded_conv_8_project_BN/FusedBatchNormV3;deeplabv3plus_mbnV2__1080p/expanded_conv_9_project/Conv2D;deeplabv3plus_mbnV2__1080p/expanded_conv_8_project/Conv2D;
1080, Rescale, deeplabv3plus_mbnV2__1080p/expanded_conv_8_project_BN/FusedBatchNormV3;deeplabv3plus_mbnV2__1080p/expanded_conv_9_project/Conv2D;deeplabv3plus_mbnV2__1080p/expanded_conv_8_project/Conv2D;
1083, Rescale, deeplabv3plus_mbnV2__1080p/expanded_conv_8_add/add;
1081, Rescale, deeplabv3plus_mbnV2__1080p/expanded_conv_8_add/add;
1084, Add, deeplabv3plus_mbnV2__1080p/expanded_conv_8_add/add;
1085, Rescale, deeplabv3plus_mbnV2__1080p/expanded_conv_8_add/add;
]]>
</table>
<table name="fused_op_id">
<![CDATA[
"id", "tosa_op_ids"
1361, 1077;
1473, 1078;
1363, 1079;
1475, 1080;
1299, 1083;1081;1084;1085;
]]>
</table>
<table name="chain_op_id">
<![CDATA[
"id", "fused_op_ids"
1613, 1361;1473;
1617, 1363;1475;1299;
]]>
</table>
<table name="stripe_op_id">
<![CDATA[
"id", "op_id", "cascade_op_id"
25, 1613, 2196
26, 1617, 2194
]]>
</table>
""".strip()
    # pylint: disable=line-too-long
    operator_types_mapping = {
        "deeplabv3plus_mbnV2__1080p/expanded_conv_8_depthwise_relu/Relu6;deeplabv3plus_mbnV2__1080p/expanded_conv_8_depthwise_BN/FusedBatchNormV3;deeplabv3plus_mbnV2__1080p/expanded_conv_10_depthwise/depthwise;deeplabv3plus_mbnV2__1080p/expanded_conv_8_depthwise/depthwise": "type1",
        "deeplabv3plus_mbnV2__1080p/expanded_conv_8_project_BN/FusedBatchNormV3;deeplabv3plus_mbnV2__1080p/expanded_conv_9_project/Conv2D;deeplabv3plus_mbnV2__1080p/expanded_conv_8_project/Conv2D": "type_2",
        "deeplabv3plus_mbnV2__1080p/expanded_conv_8_add/add": "type_3",
    }

    performance_db_parser = NXPerformanceDatabaseParser()
    performance_db_parser.raw_xmlish = performance_contents
    performance_db = performance_db_parser.parse_performance_database()
    debug_db_parser = NXDebugDatabaseParser()
    debug_db_parser.raw_xmlish = debug_contents
    debug_db = debug_db_parser.parse_debug_database()

    sys_cfg, compiler_cfg = Path("system-config"), Path("compiler-config")
    cfg = NXGraphCompilerConfig(sys_cfg, compiler_cfg)

    ignored_path = Path("ignored")

    metrics = NXGraphCompilerPerformanceMetrics(
        backend_config=cfg,
        output_files=NXGraphCompilerOutputFiles(
            ignored_path,
            ignored_path,
            ignored_path,
            ignored_path,
            ignored_path,
            ignored_path,
            ignored_path,
        ),
        performance_db_parser=performance_db_parser,
        performance_metrics=NXPerformanceStats(
            debug_db=debug_db,
            performance_db=performance_db,
            operator_types_mapping=operator_types_mapping,
        ).process_stats_per_chain(),
    )

    monkeypatch.setattr("mlia.utils.console.Console", partial(Console, width=80))

    formatter = neural_technology_formatters(metrics)
    report = formatter(metrics)
    assert isinstance(report, Table)

    assert_table_lines(
        report,
        [
            # pylint: disable=C0301
            "Neural Accelerator raw performance report:",
            "┌────┬──────┬──────┬──────┬───────┬──────┬───────┬──────┬───────┬──────┬───────┐",
            "│    │ TFL… │ TFL… │      │       │      │       │      │       │      │       │",
            "│    │ Ope… │ Ope… │ Ope… │ Total │ HW   │ HW    │ Mem… │ Read  │ Wri… │ Traf… │",
            "│ ID │ Loc… │ Type │ Cyc… │ Cycl… │ Sec… │ Util… │ Name │ bytes │ byt… │ cycl… │",
            "╞════╪══════╪══════╪══════╪═══════╪══════╪═══════╪══════╪═══════╪══════╪═══════╡",
            "│ 25 │ dee… │ typ… │ 4    │ 13    │ Out… │ 0.062 │ L1   │ 0     │ 4    │ 0     │",
            "│    │ p/e… │ typ… │      │       │ Vec… │ 0.125 │ L2   │ 0     │ 0    │ 0     │",
            "│    │ se_… │      │      │       │ Inp… │ 0.062 │ Sys… │ 0     │ 0    │ 0     │",
            "│    │ us_… │      │      │       │      │       │ DRAM │ 128   │ 4    │ 4     │",
            "│    │ con… │      │      │       │      │       │      │       │      │       │",
            "│    │ Bat… │      │      │       │      │       │      │       │      │       │",
            "│    │ _mb… │      │      │       │      │       │      │       │      │       │",
            "│    │ nv_… │      │      │       │      │       │      │       │      │       │",
            "│    │ ;de… │      │      │       │      │       │      │       │      │       │",
            "│    │ 0p/… │      │      │       │      │       │      │       │      │       │",
            "│    │ ise… │      │      │       │      │       │      │       │      │       │",
            "│    │ dee… │      │      │       │      │       │      │       │      │       │",
            "│    │ p/e… │      │      │       │      │       │      │       │      │       │",
            "│    │ se_… │      │      │       │      │       │      │       │      │       │",
            "│    │ us_… │      │      │       │      │       │      │       │      │       │",
            "│    │ con… │      │      │       │      │       │      │       │      │       │",
            "│    │ Bat… │      │      │       │      │       │      │       │      │       │",
            "│    │ _mb… │      │      │       │      │       │      │       │      │       │",
            "│    │ nv_… │      │      │       │      │       │      │       │      │       │",
            "│    │ ;de… │      │      │       │      │       │      │       │      │       │",
            "│    │ 0p/… │      │      │       │      │       │      │       │      │       │",
            "│    │ ise… │      │      │       │      │       │      │       │      │       │",
            "├────┼──────┼──────┼──────┼───────┼──────┼───────┼──────┼───────┼──────┼───────┤",
            "│ 26 │ dee… │ typ… │ 18   │ 212   │ Out… │ 1.0   │ L1   │ 0     │ 0    │ 0     │",
            "│    │ p/e… │ typ… │      │       │ Vec… │ 0.25  │ L2   │ 0     │ 0    │ 0     │",
            "│    │ _BN… │ typ… │      │       │ Tra… │ 0.25  │ Sys… │ 0     │ 0    │ 0     │",
            "│    │ lab… │ typ… │      │       │ Inp… │ 0.125 │ DRAM │ 320   │ 12   │ 10    │",
            "│    │ pan… │ typ… │      │       │      │       │      │       │      │       │",
            "│    │ v2D… │ typ… │      │       │      │       │      │       │      │       │",
            "│    │ 108… │      │      │       │      │       │      │       │      │       │",
            "│    │ jec… │      │      │       │      │       │      │       │      │       │",
            "│    │ dee… │      │      │       │      │       │      │       │      │       │",
            "│    │ p/e… │      │      │       │      │       │      │       │      │       │",
            "│    │ _BN… │      │      │       │      │       │      │       │      │       │",
            "│    │ lab… │      │      │       │      │       │      │       │      │       │",
            "│    │ pan… │      │      │       │      │       │      │       │      │       │",
            "│    │ v2D… │      │      │       │      │       │      │       │      │       │",
            "│    │ 108… │      │      │       │      │       │      │       │      │       │",
            "│    │ jec… │      │      │       │      │       │      │       │      │       │",
            "│    │ dee… │      │      │       │      │       │      │       │      │       │",
            "│    │ p/e… │      │      │       │      │       │      │       │      │       │",
            "│    │ dee… │      │      │       │      │       │      │       │      │       │",
            "│    │ p/e… │      │      │       │      │       │      │       │      │       │",
            "│    │ dee… │      │      │       │      │       │      │       │      │       │",
            "│    │ p/e… │      │      │       │      │       │      │       │      │       │",
            "│    │ dee… │      │      │       │      │       │      │       │      │       │",
            "│    │ p/e… │      │      │       │      │       │      │       │      │       │",
            "└────┴──────┴──────┴──────┴───────┴──────┴───────┴──────┴───────┴──────┴───────┘",
            # pylint: enable=C0301
        ],
    )


def test_nx_compatibility_reporting(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test function neural_technology_formatters() with Neural Accelerator compatibility data."""

    comp_info = NXModelCompatibilityInfo({"/myop1": "COMP2D", "/myop4": "NMS"})
    comp_info.add_lowered_to_tosa("/myop1", "tosaop1")
    comp_info.add_lowered_to_tosa("/myop2", "tosa.custom")
    comp_info.add_lowered_to_tosa("/myop3", "tosaop3")
    comp_info.add_lowering_error("/myop4", "Error occured when lowering")

    formatter = neural_technology_formatters(comp_info)
    report = formatter(comp_info)
    assert isinstance(report, Table)

    monkeypatch.setattr("mlia.utils.console.Console", partial(Console, width=80))
    assert_table_lines(
        report,
        [
            # pylint: disable=C0301
            "Operators:",
            "┌───┬───────────────────┬───────────────┬──────────────┬──────────────────┐",
            "│ # │ Operator location │ Operator type │ NX placement │ NX compatibility │",
            "╞═══╪═══════════════════╪═══════════════╪══════════════╪══════════════════╡",
            "│ 1 │ /myop1            │ COMP2D        │ NE           │ TOSA             │",
            "├───┼───────────────────┼───────────────┼──────────────┼──────────────────┤",
            "│ 2 │ /myop2            │ Unknown       │ EE           │ Shader           │",
            "├───┼───────────────────┼───────────────┼──────────────┼──────────────────┤",
            "│ 3 │ /myop3            │ Unknown       │ NE           │ TOSA             │",
            "├───┼───────────────────┼───────────────┼──────────────┼──────────────────┤",
            "│ 4 │ /myop4            │ NMS           │ FAIL         │ Non-NX           │",
            "└───┴───────────────────┴───────────────┴──────────────┴──────────────────┘",
            # pylint: enable=C0301
        ],
    )


def test_neural_technology_formatters_invalid_data() -> None:
    """Test neural_technology_formatters() with invalid input."""
    with pytest.raises(
        Exception,
        match=r"^Unable to find appropriate formatter for .*",
    ):
        neural_technology_formatters(12)


# %%
