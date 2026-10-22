# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Module for various helpers."""

from __future__ import annotations

from typing import Any

from mlia.core.helpers import ActionResolver


def _get_target_option_args(target_args: dict[str, Any] | None) -> list[str]:
    """Return CLI target profile arguments for follow-up command suggestions."""
    if not target_args:
        return []

    target_profile = target_args.get("target_profile")
    if target_profile is None:
        return []

    return ["--target-profile", str(target_profile)]


class CLIActionResolver(ActionResolver):
    """Helper class for generating cli commands."""

    def __init__(self, args: dict[str, Any]) -> None:
        """Init action resolver."""
        self.args = args

    def apply_optimizations(self, **kwargs: Any) -> list[str]:
        """Return no optimization command details for the CLI resolver."""
        return []

    def check_performance(self) -> list[str]:
        """Return command details for checking performance."""
        model_path, target_opts = self._get_model_and_target_opts()
        if not model_path:
            return []

        return [
            "Check the estimated performance by running the following command: ",
            f"mlia check {model_path}{target_opts} --performance",
        ]

    def check_operator_compatibility(self) -> list[str]:
        """Return command details for op compatibility."""
        model_path, target_opts = self._get_model_and_target_opts()
        if not model_path:
            return []

        return [
            "Try running the following command to verify that:",
            f"mlia check {model_path}{target_opts}",
        ]

    def operator_compatibility_details(self) -> list[str]:
        """Return command details for op compatibility."""
        return ["For more details, run: mlia check --help"]

    def _get_model_and_target_opts(
        self, separate_target_opts: bool = True
    ) -> tuple[str | None, str]:
        """Get model and target options."""
        target_opts = " ".join(_get_target_option_args(self.args))
        if separate_target_opts and target_opts:
            target_opts = f" {target_opts}"

        model_path = self.args.get("model")
        return model_path, target_opts
