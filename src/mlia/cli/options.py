# SPDX-FileCopyrightText: Copyright 2022, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Module for the CLI options."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any
from typing import Callable

from mlia.cli.config import get_available_backends
from mlia.cli.config import get_default_backends
from mlia.cli.config import is_corstone_backend
from mlia.utils.filesystem import get_supported_profile_names
from mlia.utils.types import is_number


def add_target_options(
    parser: argparse.ArgumentParser, profiles_to_skip: list[str] | None = None
) -> None:
    """Add target specific options."""
    target_profiles = get_supported_profile_names()
    if profiles_to_skip:
        target_profiles = [tp for tp in target_profiles if tp not in profiles_to_skip]

    default_target_profile = None
    default_help = ""
    if target_profiles:
        default_target_profile = target_profiles[0]
        default_help = " (default: %(default)s)"

    target_group = parser.add_argument_group("target options")
    target_group.add_argument(
        "--target-profile",
        choices=target_profiles,
        default=default_target_profile,
        help="Target profile that will set the target options "
        "such as target, mac value, memory mode, etc. "
        f"For the values associated with each target profile "
        f" please refer to the documenation {default_help}.",
    )


def add_multi_optimization_options(parser: argparse.ArgumentParser) -> None:
    """Add optimization specific options."""
    multi_optimization_group = parser.add_argument_group("optimization options")

    multi_optimization_group.add_argument(
        "--optimization-type",
        default="pruning,clustering",
        help="List of the optimization types separated by comma (default: %(default)s)",
    )
    multi_optimization_group.add_argument(
        "--optimization-target",
        default="0.5,32",
        help="""List of the optimization targets separated by comma,
             (for pruning this is sparsity between (0,1),
             for clustering this is the number of clusters (positive integer))
             (default: %(default)s)""",
    )


def add_optional_tflite_model_options(parser: argparse.ArgumentParser) -> None:
    """Add optional model specific options."""
    model_group = parser.add_argument_group("TensorFlow Lite model options")
    # make model parameter optional
    model_group.add_argument(
        "model", nargs="?", help="TensorFlow Lite model (optional)"
    )


def add_tflite_model_options(parser: argparse.ArgumentParser) -> None:
    """Add model specific options."""
    model_group = parser.add_argument_group("TensorFlow Lite model options")
    model_group.add_argument("model", help="TensorFlow Lite model")


def add_output_options(parser: argparse.ArgumentParser) -> None:
    """Add output specific options."""
    valid_extensions = ["csv", "json"]

    def check_extension(filename: str) -> str:
        """Check extension of the provided file."""
        suffix = Path(filename).suffix
        if suffix.startswith("."):
            suffix = suffix[1:]

        if suffix.lower() not in valid_extensions:
            parser.error(f"Unsupported format '{suffix}'")

        return filename

    output_group = parser.add_argument_group("output options")
    output_group.add_argument(
        "--output",
        type=check_extension,
        help=(
            "Name of the file where report will be saved. "
            "Report format is automatically detected based on the file extension. "
            f"Supported formats are: {', '.join(valid_extensions)}"
        ),
    )


def add_debug_options(parser: argparse.ArgumentParser) -> None:
    """Add debug options."""
    debug_group = parser.add_argument_group("debug options")
    debug_group.add_argument(
        "--verbose", default=False, action="store_true", help="Produce verbose output"
    )


def add_keras_model_options(parser: argparse.ArgumentParser) -> None:
    """Add model specific options."""
    model_group = parser.add_argument_group("Keras model options")
    model_group.add_argument("model", help="Keras model")


def add_custom_supported_operators_options(parser: argparse.ArgumentParser) -> None:
    """Add custom options for the command 'operators'."""
    parser.add_argument(
        "--supported-ops-report",
        action="store_true",
        default=False,
        help=(
            "Generate the SUPPORTED_OPS.md file in the "
            "current working directory and exit"
        ),
    )


def add_backend_options(parser: argparse.ArgumentParser) -> None:
    """Add options for the backends configuration."""

    def valid_directory(param: str) -> Path:
        """Check if passed string is a valid directory path."""
        if not (dir_path := Path(param)).is_dir():
            parser.error(f"Invalid directory path {param}")

        return dir_path

    subparsers = parser.add_subparsers(title="Backend actions", dest="backend_action")
    subparsers.required = True

    install_subparser = subparsers.add_parser(
        "install", help="Install backend", allow_abbrev=False
    )
    install_type_group = install_subparser.add_mutually_exclusive_group()
    install_type_group.required = True
    install_type_group.add_argument(
        "--path", type=valid_directory, help="Path to the installed backend"
    )
    install_type_group.add_argument(
        "--download",
        default=False,
        action="store_true",
        help="Download and install backend",
    )
    install_subparser.add_argument(
        "--i-agree-to-the-contained-eula",
        default=False,
        action="store_true",
        help=argparse.SUPPRESS,
    )
    install_subparser.add_argument(
        "--noninteractive",
        default=False,
        action="store_true",
        help="Non interactive mode with automatic confirmation of every action",
    )
    install_subparser.add_argument(
        "name",
        nargs="?",
        help="Name of the backend to install",
    )

    subparsers.add_parser("status", help="Show backends status")


def add_evaluation_options(parser: argparse.ArgumentParser) -> None:
    """Add evaluation options."""
    available_backends = get_available_backends()
    default_backends = get_default_backends()

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

    evaluation_group = parser.add_argument_group("evaluation options")
    evaluation_group.add_argument(
        "--evaluate-on",
        help="Backends to use for evaluation (default: %(default)s)",
        nargs="*",
        choices=available_backends,
        default=default_backends,
        type=only_one_corstone_checker(),
    )


def parse_optimization_parameters(
    optimization_type: str,
    optimization_target: str,
    sep: str = ",",
    layers_to_optimize: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Parse provided optimization parameters."""
    if not optimization_type:
        raise Exception("Optimization type is not provided")

    if not optimization_target:
        raise Exception("Optimization target is not provided")

    opt_types = optimization_type.split(sep)
    opt_targets = optimization_target.split(sep)

    if len(opt_types) != len(opt_targets):
        raise Exception("Wrong number of optimization targets and types")

    non_numeric_targets = [
        opt_target for opt_target in opt_targets if not is_number(opt_target)
    ]
    if len(non_numeric_targets) > 0:
        raise Exception("Non numeric value for the optimization target")

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
    add_target_options(parser)
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
