# SPDX-FileCopyrightText: Copyright 2022-2023, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Reports module."""
from __future__ import annotations

from pathlib import Path
from typing import Any
from typing import Callable

from mlia.backend.tosa_checker.compat import Operator
from mlia.backend.tosa_checker.compat import TOSACompatibilityInfo
from mlia.core.advice_generation import Advice
from mlia.core.metadata import Metadata
from mlia.core.metadata import MLIAMetadata
from mlia.core.metadata import ModelMetadata
from mlia.core.reporters import report_advice
from mlia.core.reporting import Cell
from mlia.core.reporting import Column
from mlia.core.reporting import CompoundReport
from mlia.core.reporting import Format
from mlia.core.reporting import NestedReport
from mlia.core.reporting import Report
from mlia.core.reporting import ReportItem
from mlia.core.reporting import Table
from mlia.nn.tensorflow.tflite_compat import TFLiteCompatibilityInfo
from mlia.target.common.reporters import report_tflite_compatiblity
from mlia.target.tosa.config import TOSAConfiguration
from mlia.target.tosa.metadata import TOSAMetadata
from mlia.utils.console import style_improvement
from mlia.utils.types import is_list_of


class MetadataDisplay(Metadata):  # pylint: disable=too-few-public-methods
    """TOSA metadata display items."""

    def __init__(self, model_path: Path) -> None:
        """Init MetadataDisplay."""
        self.model_path = model_path
        super().__init__("Metadata")

    def get_metadata(self) -> dict:
        """Combine all necessary elements for display."""
        all_data = {
            data_dict.name: data_dict.data_dict
            for data_dict in (
                TOSAMetadata(),
                MLIAMetadata(),
                ModelMetadata(self.model_path),
            )
        }
        return all_data


def report_target(target_config: TOSAConfiguration) -> Report:
    """Generate report for the target."""
    return NestedReport(
        "Target information",
        "target",
        [
            ReportItem("Target", alias="target", value=target_config.target),
        ],
    )


def report_metadata(data: MetadataDisplay) -> Report:
    """Generate report for the metadata."""
    items: list[ReportItem] = []

    for key, sub_dict in data.data_dict.items():
        nested_items = [
            ReportItem(key, alias=key, value=val) for key, val in sub_dict.items()
        ]
        items.append(ReportItem(key, alias=key, nested_items=nested_items))

    return NestedReport("Metadata", "metadata", items)


def report_tosa_operators(ops: list[Operator]) -> Report:
    """Generate report for the operators."""
    return Table(
        [
            Column("#", only_for=["plain_text"]),
            Column(
                "Operator location",
                alias="operator_location",
                fmt=Format(wrap_width=30),
            ),
            Column("Operator name", alias="operator_name", fmt=Format(wrap_width=20)),
            Column(
                "TOSA compatibility",
                alias="is_tosa_compatible",
                fmt=Format(wrap_width=25),
            ),
        ],
        [
            (
                index + 1,
                op.location,
                op.name,
                Cell(
                    op.is_tosa_compatible,
                    Format(
                        style=style_improvement(op.is_tosa_compatible),
                        str_fmt=lambda v: "Compatible" if v else "Not compatible",
                    ),
                ),
            )
            for index, op in enumerate(ops)
        ],
        name="Operators",
        alias="operators",
    )


def report_tosa_exception(exc: Exception | None) -> Report:
    """Generate report for exception thrown by tosa."""
    return NestedReport(
        "TOSA exception",
        "exception",
        [
            ReportItem("TOSA exception", alias="exception", value=repr(exc)),
        ],
    )


def report_tosa_errors(err: list[str] | None) -> Report:
    """Generate report for errors thrown by tosa."""
    message = "".join(err) if err else None
    return NestedReport(
        "TOSA stderr",
        "stderr",
        [
            ReportItem(
                "TOSA stderr",
                alias="stderr",
                value=message,
            ),
        ],
    )


def report_tosa_compatibility(compat_info: TOSACompatibilityInfo) -> Report:
    """Generate combined report for all compatibility info."""
    report_ops = report_tosa_operators(compat_info.operators)
    report_exception = report_tosa_exception(compat_info.exception)

    report_errors = report_tosa_errors(compat_info.errors)
    return CompoundReport([report_ops, report_exception, report_errors])


def tosa_formatters(data: Any) -> Callable[[Any], Report]:
    """Find appropriate formatter for the provided data."""
    if is_list_of(data, Advice):
        return report_advice

    if isinstance(data, TOSAConfiguration):
        return report_target

    if isinstance(data, MetadataDisplay):
        return report_metadata

    if is_list_of(data, Operator):
        return report_tosa_operators

    if isinstance(data, TOSACompatibilityInfo):
        return report_tosa_compatibility

    if isinstance(data, TFLiteCompatibilityInfo):
        return report_tflite_compatiblity

    raise RuntimeError(f"Unable to find appropriate formatter for {data}.")
