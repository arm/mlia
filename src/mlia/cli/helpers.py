# SPDX-FileCopyrightText: Copyright 2022-2024, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Module for various helpers."""
from __future__ import annotations

import importlib
from pathlib import Path
from shutil import copy
from typing import Any
from typing import cast

from mlia.cli.options import get_target_profile_opts
from mlia.core.helpers import ActionResolver
from mlia.nn.select import OptimizationSettings
from mlia.nn.tensorflow.utils import is_keras_model
from mlia.utils.types import is_list_of


class CLIActionResolver(ActionResolver):
    """Helper class for generating cli commands."""

    def __init__(self, args: dict[str, Any]) -> None:
        """Init action resolver."""
        self.args = args

    @staticmethod
    def _general_optimization_command(model_path: str | None) -> list[str]:
        """Return general optimization command description."""
        keras_note = []
        if model_path is None or not is_keras_model(model_path):
            model_path = "/path/to/keras_model"
            keras_note = ["Note: you will need a Keras model for that."]

        return [
            *keras_note,
            f"For example: mlia optimize {model_path} --pruning --clustering "
            "--pruning-target 0.5 --clustering-target 32",
            "For more info: mlia optimize --help",
        ]

    @staticmethod
    def _specific_optimization_command(
        model_path: str,
        target_opts: str,
        opt_settings: list[OptimizationSettings],
    ) -> list[str]:
        """Return specific optimization command description."""
        opt_types = " ".join("--" + opt.optimization_type for opt in opt_settings)
        opt_targs_strings = [
            "--pruning-target",
            "--clustering-target",
            "--rewrite-target",
        ]
        opt_targs = ",".join(
            f"{opt_targs_strings[i]} {opt.optimization_target}"
            for i, opt in enumerate(opt_settings)
        )

        return [
            "For more info: mlia optimize --help",
            "Optimization command: "
            f"mlia optimize {model_path}{target_opts} {opt_types} {opt_targs}",
        ]

    def apply_optimizations(self, **kwargs: Any) -> list[str]:
        """Return command details for applying optimizations."""
        model_path, target_opts = self._get_model_and_target_opts()

        if (opt_settings := kwargs.pop("opt_settings", None)) is None:
            return self._general_optimization_command(model_path)

        if is_list_of(opt_settings, OptimizationSettings) and model_path:
            return self._specific_optimization_command(
                model_path, target_opts, opt_settings
            )

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

    def optimization_details(self) -> list[str]:
        """Return command details for optimization."""
        return ["For more info, see: mlia optimize --help"]

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
