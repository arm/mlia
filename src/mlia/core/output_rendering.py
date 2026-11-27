# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""JSON serialization and compact console rendering of MLIA output."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rich.cells import cell_len, set_cell_size
from rich.text import Text

from mlia.core.reporting import CustomJSONEncoder

STANDARDIZED_OUTPUT_FILENAME = "mlia-output.json"


def standardized_output_to_json(output: dict[str, Any]) -> str:
    """Serialize standardized output for JSON presentation."""
    return json.dumps(output, indent=4, cls=CustomJSONEncoder)


MAX_REPORT_WIDTH = 78


def check_report(output: dict[str, Any], output_dir: Path, *, width: int = 78) -> Text:
    """Summarize completed analyses and point to their canonical JSON data."""
    width = min(MAX_REPORT_WIDTH, max(4, width))
    boxed_results = width >= 20
    report = Text()
    _heading(report, "ML INFERENCE ADVISOR", width=width)
    model = output.get("model")
    target = output.get("target")
    results = output.get("results")
    results = results if isinstance(results, list) else []
    _line(report, "Model", _field(model, "name"), width=width)
    _line(report, "Target", _field(target, "profile_name"), width=width)
    _line(
        report,
        "Results",
        ", ".join(
            str(result["kind"]).capitalize()
            for result in results
            if isinstance(result, dict) and result.get("kind")
        )
        or "None",
        width=width,
    )

    backends = output.get("backends")
    backends = backends if isinstance(backends, list) else []
    for index, result in enumerate(results, 1):
        if not isinstance(result, dict):
            continue
        kind = str(result.get("kind") or "result")
        _heading(
            report,
            f"RESULT {index}: {kind.upper()}",
            width=width,
            boxed=boxed_results,
        )
        status = str(result.get("status") or "unknown")
        if status == "ok":
            status_text = "Compatible" if kind == "compatibility" else "Completed"
        else:
            status_text = status.capitalize()
        status_text = result.get("status_summary") or status_text
        _line(report, "Status", status_text, width=width, boxed=boxed_results)
        producer = str(result.get("producer") or "Unknown")
        for backend in backends:
            if isinstance(backend, dict) and backend.get("id") == producer:
                producer = str(backend.get("name") or producer)
                if backend.get("version"):
                    version = str(backend["version"])
                    producer += (
                        " (unknown version)"
                        if version.lower() == "unknown"
                        else f" {version}"
                    )
                break
        _line(report, "Producer", producer, width=width, boxed=boxed_results)
        if kind == "performance" and result.get("mode"):
            scope = f"{str(result['mode']).capitalize()} results"
            _line(report, "Scope", scope, width=width, boxed=boxed_results)
        if boxed_results:
            report.append("\n╰" + "─" * (width - 2) + "╯", style="tbl.border")
        _heading(report, f"{kind.upper()} FINDINGS AND ADVICE", width=width)
        _findings(report, result, kind)

    _heading(report, "FULL RESULTS", width=width)
    report.append("\n\n")
    report.append("Full metrics, operator details and breakdowns are available in:\n")
    report.append(str(output_dir / STANDARDIZED_OUTPUT_FILENAME), style="tbl.name")
    return report


def _field(value: object, key: str) -> str:
    return str(value.get(key) or "Unknown") if isinstance(value, dict) else "Unknown"


def _fit(value: str, cells: int) -> str:
    """Fit a label inside a rule or box without splitting a wide character."""
    if cells <= 0:
        return ""
    if cell_len(value) > cells:
        return set_cell_size(value, cells - 1) + "…"
    return value


def _heading(report: Text, title: str, *, width: int, boxed: bool = False) -> None:
    """Append a short section heading with distinct title and rule styles."""
    if report:
        report.append("\n\n")
    title = _fit(title, width - 4)
    if boxed:
        report.append("╭" + "─" * (width - 2) + "╮\n", style="tbl.border")
        report.append("│ ", style="tbl.border")
        report.append(title, style="tbl.heading")
        report.append(" " * (width - cell_len(title) - 3) + "│", style="tbl.border")
        return
    report.append("── ", style="tbl.border")
    report.append(title, style="tbl.name")
    report.append(" " + "─" * (width - cell_len(title) - 4), style="tbl.border")


def _line(
    report: Text, label: str, value: str, *, width: int, boxed: bool = False
) -> None:
    report.append("\n")
    if boxed:
        report.append("│ ", style="tbl.border")
    report.append(f"{label}: ", style="tbl.border")
    if boxed:
        value = _fit(value, width - cell_len(label) - 6)
    report.append(value, style="tbl.version" if label == "Status" else "tbl.name")
    if boxed:
        padding = " " * (width - cell_len(label) - cell_len(value) - 5)
        report.append(padding + "│", style="tbl.border")


def _findings(report: Text, result: dict[str, Any], kind: str) -> None:
    """Group repeated advice while preserving distinct diagnostics."""
    advice = result.get("advice")
    warnings = result.get("warnings")
    errors = result.get("errors")
    entities = result.get("entities")
    entities_by_id = {
        str(entity["id"]): entity
        for entity in (entities if isinstance(entities, list) else [])
        if isinstance(entity, dict) and entity.get("id")
    }
    findings: dict[tuple[str, str], set[str]] = {}
    if isinstance(advice, list):
        for item in advice:
            if not isinstance(item, dict) or not item.get("message"):
                continue
            key = (str(item.get("severity") or "info"), str(item["message"]))
            ids = item.get("affected_entity_ids")
            findings.setdefault(key, set()).update(ids if isinstance(ids, list) else [])
    if isinstance(warnings, list):
        for warning in warnings:
            findings.setdefault(("warning", str(warning)), set())
    if isinstance(errors, list):
        for error in errors:
            findings.setdefault(("error", str(error)), set())
    checks = result.get("checks")
    if isinstance(checks, list):
        for check in checks:
            if not isinstance(check, dict) or check.get("status") not in (
                "fail",
                "partial",
            ):
                continue
            details = check.get("details")
            reasons = details.get("reasons") if isinstance(details, dict) else None
            if not isinstance(reasons, list):
                continue
            for reason in reasons:
                if isinstance(reason, dict):
                    description = str(
                        reason.get("description") or reason.get("category") or ""
                    )
                    detail = str(reason.get("detail") or "")
                    message = (
                        f"{description}: {detail}"
                        if description and detail and detail != description
                        else description or detail
                    )
                else:
                    message = str(reason) if isinstance(reason, str) else ""
                if message:
                    key = (
                        "error" if check["status"] == "fail" else "warning",
                        message,
                    )
                    ids = findings.setdefault(key, set())
                    if check.get("entity_id"):
                        ids.add(str(check["entity_id"]))

    if not findings:
        if result.get("status") in ("failed", "partial", "incompatible"):
            report.append("\n\nNo further reason or recovery advice was reported.")
        else:
            report.append("\n\nNo warnings or advice were reported.")
    else:
        for (severity, message), ids in findings.items():
            report.append("\n\n")
            severity_style = {
                "warning": "tbl.highlight",
                "error": "tbl.error",
            }.get(severity, "tbl.name")
            report.append(severity.upper(), style=severity_style)
            report.append(" · ", style="tbl.border")
            _finding_message(report, message)
            if ids:
                labels = []
                for entity_id in sorted(ids)[:3]:
                    entity = entities_by_id.get(entity_id, {})
                    name = str(entity.get("name") or entity_id)
                    labels.append(
                        f"{name} ({entity_id})" if name != entity_id else entity_id
                    )
                remaining = len(ids) - len(labels)
                report.append("\nAffects: " + ", ".join(labels))
                if remaining:
                    report.append(
                        f", and {remaining} more (see {STANDARDIZED_OUTPUT_FILENAME})"
                    )
                report.append(".")
    report.append("\n\nDetails in full results: ", style="tbl.border")
    report.append(
        "checks, advice and entities."
        if kind == "compatibility"
        else "advice, breakdowns and entities."
    )


def _finding_message(report: Text, message: str) -> None:
    """Place a suggested MLIA command on a distinct, colored line."""
    description, marker, command = message.partition("mlia ")
    if not marker or not command:
        report.append(message)
        return
    report.append(description.rstrip())
    if description:
        report.append("\n  ")
    report.append(marker + command.strip(), style="tbl.name")
