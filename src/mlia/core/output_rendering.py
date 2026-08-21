# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Target-agnostic rendering for MLIA standardized output."""

from __future__ import annotations

import json
from typing import Any

from mlia.core.reporting import CustomJSONEncoder
from mlia.utils.console import create_section_header, produce_table


def standardized_output_to_json(output: dict[str, Any]) -> str:
    """Serialize standardized output for JSON presentation."""
    return json.dumps(output, indent=4, cls=CustomJSONEncoder)


def standardized_output_to_text(output: dict[str, Any]) -> str:
    """Render standardized output as target-agnostic human-readable text."""
    sections = [create_section_header("ML Inference Advisor started").rstrip()]
    sections.extend(
        section
        for section in (
            _target_section(output.get("target")),
            _model_section(output.get("model")),
            _backends_section(output.get("backends")),
            _results_section(output.get("results")),
        )
        if section
    )
    return "\n\n".join(sections)


def _target_section(target: object) -> str | None:
    if not isinstance(target, dict) or not target:
        return None
    rows: list[tuple[str, str]] = []
    _append(rows, "Profile", target.get("profile_name"))
    _append(rows, "Target type", target.get("target_type"))
    _append(rows, "Description", target.get("description"))
    configuration = target.get("configuration")
    if isinstance(configuration, dict):
        for key, value in configuration.items():
            _append(rows, _title(str(key)), value)
    components = target.get("components")
    if isinstance(components, list) and components:
        _append(
            rows, "Components", ", ".join(_component_name(item) for item in components)
        )
    if not rows:
        return None
    return "Target information:\n" + produce_table(rows, table_style="nested")


def _model_section(model: object) -> str | None:
    if not isinstance(model, dict) or not model:
        return None
    rows: list[tuple[str, str]] = []
    _append(rows, "Name", model.get("name"))
    _append(rows, "Format", model.get("format"))
    _append(rows, "Hash", model.get("hash"))
    _append(rows, "Size", _format_with_unit(model.get("size_bytes"), "bytes"))
    if not rows:
        return None
    return "Model information:\n" + produce_table(rows, table_style="nested")


def _backends_section(backends: object) -> str | None:
    if not isinstance(backends, list) or not backends:
        return None
    rows = []
    for backend in backends:
        if isinstance(backend, dict):
            rows.append(
                (
                    str(backend.get("id") or "-"),
                    str(backend.get("name") or "-"),
                    str(backend.get("version") or "-"),
                )
            )
    if not rows:
        return None
    return "Backends:\n" + produce_table(
        rows, headers=["ID", "Name", "Version"], table_style="default"
    )


def _results_section(results: object) -> str | None:
    if not isinstance(results, list) or not results:
        return None
    rendered = [create_section_header("Model Analysis Results").rstrip()]
    for index, result in enumerate(results, start=1):
        if isinstance(result, dict):
            rendered.append(_result_section(index, result))
    return "\n\n".join(section for section in rendered if section)


def _result_section(index: int, result: dict[str, Any]) -> str:
    title = f"Result {index}: {_title(str(result.get('kind') or 'result'))}"
    rows: list[tuple[str, str]] = []
    _append(rows, "Status", result.get("status"))
    _append(rows, "Producer", result.get("producer"))
    _append(rows, "Mode", result.get("mode"))
    lines = [title, produce_table(rows, table_style="nested") if rows else ""]
    for key in ("warnings", "errors"):
        values = result.get(key)
        if isinstance(values, list) and values:
            lines.append(_list_block(_title(key), values))
    metrics = _metrics_table(result.get("metrics"))
    if metrics:
        lines.append("Metrics:\n" + metrics)
    entities = _entities_table(result.get("entities"))
    if entities:
        lines.append("Entities:\n" + entities)
    breakdowns = _breakdowns_table(result.get("breakdowns"))
    if breakdowns:
        lines.append("Breakdowns:\n" + breakdowns)
    checks = _checks_table(result.get("checks"))
    if checks:
        lines.append("Checks:\n" + checks)
    advice = _advice_block(result.get("advice"))
    if advice:
        lines.append("Advice:\n" + advice)
    return "\n".join(line for line in lines if line)


def _metrics_table(metrics: object) -> str | None:
    if not isinstance(metrics, list) or not metrics:
        return None
    rows = []
    for metric in metrics:
        if isinstance(metric, dict):
            rows.append(
                (
                    str(metric.get("name") or "-"),
                    _metric_value(metric),
                    str(metric.get("unit") or ""),
                )
            )
    return produce_table(rows, headers=["Metric", "Value", "Unit"]) if rows else None


def _entities_table(entities: object) -> str | None:
    if not isinstance(entities, list) or not entities:
        return None
    rows = []
    truncated = len(entities) > 20
    for entity in entities[:20]:
        if isinstance(entity, dict):
            rows.append(
                (
                    str(entity.get("name") or "-"),
                    str(entity.get("scope") or "-"),
                    str(entity.get("location") or "-"),
                    str(entity.get("placement") or "-"),
                )
            )
    if not rows:
        return None
    table = produce_table(rows, headers=["Name", "Scope", "Location", "Placement"])
    if truncated:
        table += f"\nShowing 20 of {len(entities)} entities."
    return table


def _breakdowns_table(breakdowns: object) -> str | None:
    if not isinstance(breakdowns, list) or not breakdowns:
        return None
    rows = []
    truncated = len(breakdowns) > 20
    for breakdown in breakdowns[:20]:
        if not isinstance(breakdown, dict):
            continue
        metrics = breakdown.get("metrics")
        metric_text = ""
        if isinstance(metrics, list):
            metric_text = ", ".join(
                (
                    f"{metric.get('name')}={_metric_value(metric)} "
                    f"{metric.get('unit', '')}"
                ).strip()
                for metric in metrics
                if isinstance(metric, dict)
            )
        rows.append(
            (
                str(breakdown.get("name") or "-"),
                str(breakdown.get("scope") or "-"),
                str(breakdown.get("location") or "-"),
                metric_text or "-",
            )
        )
    if not rows:
        return None
    table = produce_table(rows, headers=["Name", "Scope", "Location", "Metrics"])
    if truncated:
        table += f"\nShowing 20 of {len(breakdowns)} breakdowns."
    return table


def _checks_table(checks: object) -> str | None:
    if not isinstance(checks, list) or not checks:
        return None
    rows = []
    truncated = len(checks) > 20
    for check in checks[:20]:
        if isinstance(check, dict):
            rows.append((str(check.get("id") or "-"), str(check.get("status") or "-")))
    if not rows:
        return None
    table = produce_table(rows, headers=["Check", "Status"])
    if truncated:
        table += f"\nShowing 20 of {len(checks)} checks."
    return table


def _advice_block(advice: object) -> str | None:
    if not isinstance(advice, list) or not advice:
        return None
    rows = []
    for item in advice:
        if isinstance(item, dict):
            rows.append(
                (
                    str(item.get("severity") or "info"),
                    str(item.get("category") or "-"),
                    str(item.get("message") or ""),
                )
            )
    return (
        produce_table(
            rows, headers=["Severity", "Category", "Message"], table_style="no_borders"
        )
        if rows
        else None
    )


def _list_block(title: str, values: list[object]) -> str:
    return title + ":\n" + "\n".join(f"  - {value}" for value in values)


def _metric_value(metric: dict[str, Any]) -> str:
    if "value" in metric:
        return str(metric.get("value"))
    if metric.get("availability"):
        reason = metric.get("reason")
        return str(reason or metric.get("availability"))
    return "-"


def _component_name(component: object) -> str:
    if not isinstance(component, dict):
        return str(component)
    values = [
        component.get("type"),
        component.get("family"),
        component.get("model"),
        component.get("variant"),
    ]
    return " ".join(str(value) for value in values if value is not None)


def _append(rows: list[tuple[str, str]], key: str, value: object) -> None:
    if value is not None and value != "":
        rows.append((key, str(value)))


def _title(value: str) -> str:
    return value.replace("_", " ").title()


def _format_with_unit(value: object, unit: str) -> str | None:
    if value is None:
        return None
    return f"{value} {unit}"
