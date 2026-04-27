# SPDX-FileCopyrightText: Copyright 2022-2024, 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Module for various helpers."""

from __future__ import annotations

import importlib
from pathlib import Path
from shutil import copy
from typing import Any, cast

from mlia.cli.options import get_target_profile_opts
from mlia.core.helpers import ActionResolver


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
        target_opts = " ".join(get_target_profile_opts(self.args))
        if separate_target_opts and target_opts:
            target_opts = f" {target_opts}"

        model_path = self.args.get("model")
        return model_path, target_opts


def copy_profile_file_to_output_dir(
    target_profile: str | Path, output_dir: str | Path, profile_to_copy: str
) -> bool:
    """Copy the target profile file to the output directory."""
    get_func_name = "get_builtin_" + profile_to_copy + "_path"
    get_func = getattr(importlib.import_module("mlia.target.config"), get_func_name)
    is_func_name = "is_builtin_" + profile_to_copy
    is_func = getattr(importlib.import_module("mlia.target.config"), is_func_name)
    profile_file_path = (
        get_func(cast(str, target_profile))
        if is_func(target_profile)
        else Path(target_profile)
    )
    output_file_path = f"{output_dir}/{profile_file_path.stem}.toml"
    try:
        copy(profile_file_path, output_file_path)
        return True
    except OSError as err:
        raise RuntimeError(f"Failed to copy {profile_to_copy} file: {err}") from err
