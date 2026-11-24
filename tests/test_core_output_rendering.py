# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the compact check console report."""

from io import StringIO
from pathlib import Path

from rich.console import Console

from mlia.core.application_settings import DARK
from mlia.core.output_rendering import check_report


def test_check_report_summarizes_and_groups_findings(tmp_path: Path) -> None:
    """The console stays compact while retaining distinct actionable findings."""
    output = {
        "model": {"name": "network.tflite", "hash": "secret"},
        "target": {"profile_name": "ethos-u", "configuration": {"secret": 1}},
        "backends": [{"id": "vela", "name": "Vela", "version": "4.0"}],
        "results": [
            {
                "kind": "compatibility",
                "status": "ok",
                "producer": "vela",
                "entities": [
                    {"id": "a", "kind": "source_operator", "placement": "NPU"},
                    {"id": "b", "kind": "source_operator", "placement": "NPU"},
                ],
            },
            {
                "kind": "performance",
                "status": "partial",
                "producer": "vela",
                "mode": "simulated",
                "warnings": ["Some measurements unavailable"],
                "metrics": [{"name": "cycles", "value": 123456}],
                "advice": [
                    {
                        "severity": "warning",
                        "message": "Review high-cycle layers",
                        "affected_entity_ids": ["a"],
                    },
                    {
                        "severity": "warning",
                        "message": "Review high-cycle layers",
                        "affected_entity_ids": ["b"],
                    },
                ],
            },
        ],
    }

    report = check_report(output, tmp_path).plain

    assert "Model: network.tflite" in report
    assert "Target: ethos-u" in report
    assert "Results: Compatibility, Performance" in report
    assert "All 2 source operators are placed on the NPU." in report
    assert "Status: Partial with warnings" in report
    assert "Producer: Vela 4.0" in report
    assert "Scope: Simulated results" in report
    assert report.count("Review high-cycle layers") == 1
    assert "Affects: a, b." in report
    assert "Some measurements unavailable" in report
    assert "Details in full results: checks, advice and entities." in report
    assert "Details in full results: advice, breakdowns and entities." in report
    assert "path below" not in report
    assert str(tmp_path / "mlia-output.json") in report
    assert "123456" not in report
    assert "secret" not in report


def test_check_report_keeps_failed_reason_and_only_requested_result(
    tmp_path: Path,
) -> None:
    """Failed checks show their reason without inventing a second analysis."""
    report = check_report(
        {
            "results": [
                {
                    "kind": "performance",
                    "status": "failed",
                    "producer": "backend",
                    "errors": ["Backend could not read the model"],
                }
            ]
        },
        tmp_path,
    ).plain

    assert "Status: Failed" in report
    assert "Backend could not read the model" in report
    assert "COMPATIBILITY FINDINGS" not in report
    assert "No warnings or advice" not in report


def test_check_report_names_unknown_producer_version(tmp_path: Path) -> None:
    """An unknown backend version is identified as a version."""
    report = check_report(
        {
            "backends": [
                {
                    "id": "nx",
                    "name": "Neural Accelerator (NX) Performance Estimator",
                    "version": "unknown",
                }
            ],
            "results": [{"kind": "performance", "status": "ok", "producer": "nx"}],
        },
        tmp_path,
    ).plain

    assert (
        "Producer: Neural Accelerator (NX) Performance Estimator (unknown version)"
        in report
    )


def test_check_report_includes_failing_check_reasons(tmp_path: Path) -> None:
    """Failure diagnostics in checks must not be hidden by the empty fallback."""
    report = check_report(
        {
            "results": [
                {
                    "kind": "compatibility",
                    "status": "incompatible",
                    "producer": "vela",
                    "entities": [{"id": "op-1", "name": "Conv2D"}],
                    "checks": [
                        {
                            "id": "npu_support_op-1",
                            "status": "fail",
                            "entity_id": "op-1",
                            "details": {
                                "reasons": [
                                    {
                                        "description": "Unsupported kernel",
                                        "detail": "Size exceeds the limit",
                                    }
                                ]
                            },
                        }
                    ],
                }
            ]
        },
        tmp_path,
    ).plain

    assert "ERROR · Unsupported kernel: Size exceeds the limit" in report
    assert "Affects: Conv2D (op-1)." in report
    assert "No further reason" not in report


def test_check_report_sized_to_console_width(tmp_path: Path) -> None:
    """Rules and box lines fit even when the terminal is narrow."""
    width = 42
    report = check_report(
        {
            "results": [
                {
                    "kind": "compatibility",
                    "status": "incompatible",
                    "producer": "a very long backend producer name",
                }
            ]
        },
        tmp_path,
        width=width,
    ).plain

    assert all(
        len(line) <= width
        for line in report.splitlines()
        if line.startswith(("── ", "╭", "│", "╰"))
    )
    assert "╭" + "─" * (width - 2) + "╮" in report

    tiny = check_report(
        {"results": [{"kind": "performance", "status": "ok"}]},
        tmp_path,
        width=16,
    ).plain
    assert "╭" not in tiny
    assert all(len(line) <= 16 for line in tiny.splitlines() if line.startswith("── "))


def test_check_report_names_one_affected_operator(tmp_path: Path) -> None:
    """A single affected entity is identified without requiring a JSON lookup."""
    report = check_report(
        {
            "results": [
                {
                    "kind": "performance",
                    "status": "ok",
                    "producer": "backend",
                    "entities": [
                        {
                            "id": "source_operator/segment_0/spirv-42",
                            "kind": "source_operator",
                            "name": "Conv2D",
                        }
                    ],
                    "advice": [
                        {
                            "severity": "warning",
                            "message": "Review low MAC utilisation",
                            "affected_entity_ids": [
                                "source_operator/segment_0/spirv-42"
                            ],
                        }
                    ],
                }
            ]
        },
        tmp_path,
    ).plain

    assert "Affects: Conv2D (source_operator/segment_0/spirv-42)." in report


def test_check_report_limits_operator_names(tmp_path: Path) -> None:
    """Long findings stay short while pointing to complete JSON identities."""
    report = check_report(
        {
            "results": [
                {
                    "kind": "compatibility",
                    "status": "partial",
                    "producer": "backend",
                    "advice": [
                        {
                            "severity": "warning",
                            "message": "Review placement",
                            "affected_entity_ids": ["A", "B", "C", "D", "E"],
                        }
                    ],
                }
            ]
        },
        tmp_path,
    ).plain

    assert "Affects: A, B, C, and 2 more (see mlia-output.json)." in report
    assert "Affects: A, B, C, D" not in report


def test_check_report_styles_headings_and_severity(tmp_path: Path) -> None:
    """Semantic styles let configured light and dark themes supply the colors."""
    report = check_report(
        {
            "results": [
                {
                    "kind": "performance",
                    "status": "ok",
                    "producer": "backend",
                    "warnings": ["Check this layer"],
                }
            ]
        },
        tmp_path,
    )
    styles = {span.style for span in report.spans}
    assert "tbl.border" in styles
    assert "tbl.name" in styles
    assert "tbl.heading" in styles
    assert "tbl.highlight" in styles

    stream = StringIO()
    Console(
        file=stream,
        force_terminal=True,
        no_color=False,
        color_system="truecolor",
        theme=DARK,
    ).print(report)
    assert "\x1b[1;38;2;107;155;248m" in stream.getvalue()  # blue section title
    assert "\x1b[1;38;2;246;155;79m" in stream.getvalue()  # orange result title


def test_check_report_highlights_suggested_command(tmp_path: Path) -> None:
    """Follow-up commands sit on their own line in the help screen's blue."""
    command = "mlia check model.tflite --target-profile ethos-u85-2048 --performance"
    report = check_report(
        {
            "results": [
                {
                    "kind": "compatibility",
                    "status": "ok",
                    "producer": "backend",
                    "advice": [
                        {
                            "severity": "info",
                            "message": (
                                "Check performance with this command:  " + command
                            ),
                        }
                    ],
                }
            ]
        },
        tmp_path,
    )

    assert "command:\n  " + command in report.plain
    command_start = report.plain.index(command)
    assert any(
        span.style == "tbl.name"
        and span.start <= command_start
        and span.end >= command_start + len(command)
        for span in report.spans
    )
    stream = StringIO()
    Console(
        file=stream,
        force_terminal=True,
        no_color=False,
        color_system="truecolor",
        theme=DARK,
        width=200,
    ).print(report)
    assert "\x1b[1;38;2;107;155;248m" + command in stream.getvalue()
