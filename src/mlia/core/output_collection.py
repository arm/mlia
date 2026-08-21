# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Canonical standardized output collection for MLIA workflows."""

from __future__ import annotations

import logging
from typing import Any, cast

logger = logging.getLogger(__name__)


class StandardizedOutputCollector:
    """Collect complete canonical structured outputs from workflow data."""

    def __init__(self) -> None:
        """Create an empty collector."""
        self.standardized_outputs: list[Any] = []

    def submit_data_item(self, data_item: Any) -> None:
        """Submit a workflow data item to the canonical output stream."""
        standardized_output = getattr(data_item, "standardized_output", None)
        if standardized_output:
            self.standardized_outputs.append(standardized_output)

    def build_output(self) -> dict[str, Any] | None:
        """Return the canonical standardized output for the workflow run."""
        if not self.standardized_outputs:
            return None

        if len(self.standardized_outputs) == 1:
            output = self.standardized_outputs[0]
        else:
            output = self._merge_standardized_outputs(self.standardized_outputs)

        return cast(dict[str, Any], output)

    @staticmethod
    def _merge_standardized_outputs(outputs: list[Any]) -> dict[str, Any]:
        """Merge complete standardized outputs without modifying their results."""
        merged: dict[str, Any] = {
            "results": [],
            "model": {},
            "target": {},
            "context": {},
            "backends": [],
        }
        for output in outputs:
            if not isinstance(output, dict):
                continue
            if (
                merged.get("schema_version")
                and output.get("schema_version")
                and output["schema_version"] != merged["schema_version"]
            ):
                logger.warning(
                    "Merging standardized outputs with mismatched schema_version: "
                    "%s (kept) vs %s (ignored).",
                    merged["schema_version"],
                    output["schema_version"],
                )
            if "results" in output:
                merged["results"].extend(output.get("results", []))
            if "backends" in output:
                merged["backends"].extend(output.get("backends", []))
            for key in (
                "model",
                "target",
                "context",
                "schema_version",
                "run_id",
                "timestamp",
                "tool",
                "extensions",
            ):
                if output.get(key) and not merged.get(key):
                    merged[key] = output[key]
        return merged
