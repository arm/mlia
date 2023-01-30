# SPDX-FileCopyrightText: Copyright 2022-2023, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Module for the CLI options."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any
from typing import Callable

from mlia.cli.config import DEFAULT_CLUSTERING_TARGET
from mlia.cli.config import DEFAULT_PRUNING_TARGET
from mlia.cli.config import get_available_backends
from mlia.cli.config import is_corstone_backend
from mlia.core.common import FormattedFilePath
from mlia.utils.filesystem import get_supported_profile_names


def add_check_category_options(parser: argparse.ArgumentParser) -> None:
    """Add check category type options."""
    parser.add_argument(
        "--performance", action="store_true", help="Perform performance checks."
    )

    parser.add_argument(
        "--compatibility",
        action="store_true",
        help="Perform compatibility checks. (default)",
    )


def add_target_options(
    parser: argparse.ArgumentParser,
    profiles_to_skip: list[str] | None = None,
    required: bool = True,
) -> None:
    """Add target specific options."""
    target_profiles = get_supported_profile_names()
    if profiles_to_skip:
        target_profiles = [tp for tp in target_profiles if tp not in profiles_to_skip]

    target_group = parser.add_argument_group("target options")
    target_group.add_argument(
        "-t",
        "--target-profile",
        choices=target_profiles,
        required=required,
        default="",
        help="Target profile that will set the target options "
        "such as target, mac value, memory mode, etc. "
        "For the values associated with each target profile "
        "please refer to the documentation.",
    )


def add_multi_optimization_options(parser: argparse.ArgumentParser) -> None:
    """Add optimization specific options."""
    multi_optimization_group = parser.add_argument_group("optimization options")

    multi_optimization_group.add_argument(
        "--pruning", action="store_true", help="Apply pruning optimization."
    )

    multi_optimization_group.add_argument(
        "--clustering", action="store_true", help="Apply clustering optimization."
    )

    multi_optimization_group.add_argument(
        "--pruning-target",
        type=float,
        help="Sparsity to be reached during optimization "
        f"(default: {DEFAULT_PRUNING_TARGET})",
    )

    multi_optimization_group.add_argument(
        "--clustering-target",
        type=int,
        help="Number of clusters to reach during optimization "
        f"(default: {DEFAULT_CLUSTERING_TARGET})",
    )


def add_model_options(parser: argparse.ArgumentParser) -> None:
    """Add model specific options."""
    parser.add_argument("model", help="TensorFlow Lite model or Keras model")


def add_output_options(parser: argparse.ArgumentParser) -> None:
    """Add output specific options."""
    output_group = parser.add_argument_group("output options")
    output_group.add_argument(
        "-o",
        "--output",
        type=Path,
        help=("Name of the file where the report will be saved."),
    )

    output_group.add_argument(
        "--json",
        action="store_true",
        help=("Format to use for the output (requires --output argument to be set)."),
    )


def add_debug_options(parser: argparse.ArgumentParser) -> None:
    """Add debug options."""
    debug_group = parser.add_argument_group("debug options")
    debug_group.add_argument(
        "-d",
        "--debug",
        default=False,
        action="store_true",
        help="Produce verbose output",
    )


def add_keras_model_options(parser: argparse.ArgumentParser) -> None:
    """Add model specific options."""
    model_group = parser.add_argument_group("Keras model options")
    model_group.add_argument("model", help="Keras model")


def add_backend_install_options(parser: argparse.ArgumentParser) -> None:
    """Add options for the backends configuration."""

    def valid_directory(param: str) -> Path:
        """Check if passed string is a valid directory path."""
        if not (dir_path := Path(param)).is_dir():
            parser.error(f"Invalid directory path {param}")

        return dir_path

    parser.add_argument(
        "--path", type=valid_directory, help="Path to the installed backend"
    )
    parser.add_argument(
        "--i-agree-to-the-contained-eula",
        default=False,
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--force",
        default=False,
        action="store_true",
        help="Force reinstalling backend in the specified path",
    )
    parser.add_argument(
        "--noninteractive",
        default=False,
        action="store_true",
        help="Non interactive mode with automatic confirmation of every action",
    )
    parser.add_argument(
        "name",
        help="Name of the backend to install",
    )


def add_backend_uninstall_options(parser: argparse.ArgumentParser) -> None:
    """Add options for the backends configuration."""
    parser.add_argument(
        "name",
        help="Name of the installed backend",
    )


def add_backend_options(
    parser: argparse.ArgumentParser, backends_to_skip: list[str] | None = None
) -> None:
    """Add evaluation options."""
    available_backends = get_available_backends()

    def only_one_corstone_checker() -> Callable:
        """
        Return a callable to check that only one Corstone backend is passed.

        Raises an exception when more than one Corstone backend is passed.
        """
        num_corstones = 0

        def check(backend: str) -> str:
            """Count Corstone backends and raise an exception if more than one."""
            nonlocal num_corstones
            if is_corstone_backend(backend):
                num_corstones = num_corstones + 1
                if num_corstones > 1:
                    raise argparse.ArgumentTypeError(
                        "There must be only one Corstone backend in the argument list."
                    )
            return backend

        return check

    # Remove backends to skip
    if backends_to_skip:
        available_backends = [
            x for x in available_backends if x not in backends_to_skip
        ]

    evaluation_group = parser.add_argument_group("backend options")
    evaluation_group.add_argument(
        "-b",
        "--backend",
        help="Backends to use for evaluation.",
        action="append",
        choices=available_backends,
        type=only_one_corstone_checker(),
    )


def parse_output_parameters(path: Path | None, json: bool) -> FormattedFilePath | None:
    """Parse and return path and file format as FormattedFilePath."""
    if not path and json:
        raise argparse.ArgumentError(
            None,
            "To enable JSON output you need to specify the output path. "
            "(e.g. --output out.json --json)",
        )
    if not path:
        return None
    if json:
        return FormattedFilePath(path, "json")

    return FormattedFilePath(path, "plain_text")


def parse_optimization_parameters(
    pruning: bool = False,
    clustering: bool = False,
    pruning_target: float | None = None,
    clustering_target: int | None = None,
    layers_to_optimize: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Parse provided optimization parameters."""
    opt_types = []
    opt_targets = []

    if clustering_target and not clustering:
        raise argparse.ArgumentError(
            None,
            "To enable clustering optimization you need to include the "
            "`--clustering` flag in your command.",
        )

    if not pruning_target:
        pruning_target = DEFAULT_PRUNING_TARGET

    if not clustering_target:
        clustering_target = DEFAULT_CLUSTERING_TARGET

    if (pruning is False and clustering is False) or pruning:
        opt_types.append("pruning")
        opt_targets.append(pruning_target)

    if clustering:
        opt_types.append("clustering")
        opt_targets.append(clustering_target)

    optimizer_params = [
        {
            "optimization_type": opt_type.strip(),
            "optimization_target": float(opt_target),
            "layers_to_optimize": layers_to_optimize,
        }
        for opt_type, opt_target in zip(opt_types, opt_targets)
    ]

    return optimizer_params


def get_target_profile_opts(device_args: dict | None) -> list[str]:
    """Get non default values passed as parameters for the target profile."""
    if not device_args:
        return []

    parser = argparse.ArgumentParser()
    add_target_options(parser, required=False)
    args = parser.parse_args([])

    params_name = {
        action.dest: param_name
        for param_name, action in parser._option_string_actions.items()  # pylint: disable=protected-access
    }

    non_default = [
        arg_name
        for arg_name, arg_value in device_args.items()
        if arg_name in args and vars(args)[arg_name] != arg_value
    ]

    def construct_param(name: str, value: Any) -> list[str]:
        """Construct parameter."""
        if isinstance(value, list):
            return [str(item) for v in value for item in [name, v]]

        return [name, str(value)]

    return [
        item
        for name in non_default
        for item in construct_param(params_name[name], device_args[name])
    ]
