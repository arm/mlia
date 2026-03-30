# SPDX-FileCopyrightText: Copyright 2022-2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Module for the CLI options."""

from __future__ import annotations

import argparse
import importlib
import pkgutil
from pathlib import Path
from typing import Any, Callable, Sequence, TypedDict

import mlia.backend
from mlia.backend.manager import get_available_backends
from mlia.core.common import AdviceCategory
from mlia.core.typing import OutputFormat
from mlia.target.registry import builtin_profile_names
from mlia.target.registry import registry as target_registry


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
    supported_advice: Sequence[AdviceCategory] | None = None,
    required: bool = True,
) -> None:
    """Add target specific options."""
    target_profiles = builtin_profile_names()

    if supported_advice:

        def is_advice_supported(profile: str, advice: Sequence[AdviceCategory]) -> bool:
            """
            Collect all target profiles that support the advice.

            This means target profiles that...
            - have the right target prefix, e.g. "ethos-u55..." to avoid loading
              all target profiles
            - support any of the required advice
            """
            for target, info in target_registry.items.items():
                if profile.startswith(target):
                    return any(info.is_supported(adv) for adv in advice)
            return False

        target_profiles = [
            profile
            for profile in target_profiles
            if is_advice_supported(profile, supported_advice)
        ]

    target_group = parser.add_argument_group("target options")
    target_group.add_argument(
        "-t",
        "--target-profile",
        required=required,
        help="Built-in target profile or path to the custom target profile. "
        "Target profile that will set the target options "
        "such as target, mac value, memory mode, etc. "
        "To see all available target profiles, use 'mlia-target list'.",
    )


def add_model_options(parser: argparse.ArgumentParser) -> None:
    """Add model specific options."""
    parser.add_argument("model", help="TensorFlow Lite model or Keras model")


def add_output_options(parser: argparse.ArgumentParser) -> None:
    """Add output specific options."""
    output_group = parser.add_argument_group("output options")
    output_group.add_argument(
        "--json",
        action="store_true",
        help=("Print the output in JSON format."),
    )


class BackendOptionSpec(TypedDict):
    """Describe backend option metadata discovered from CLI mappings."""

    module: str
    backend: str
    config_key: str
    cli_option: str
    full_cli_option: str
    dest: str
    type: type
    help: str


def discover_backend_option_specs() -> list[BackendOptionSpec]:
    """Return backend option metadata derived from CONFIG_TO_CLI_OPTION."""
    specs: list[BackendOptionSpec] = []
    backend_package_path = mlia.backend.__path__
    for _, module_name, is_pkg in pkgutil.iter_modules(backend_package_path):
        if not is_pkg:
            continue

        for submodule in ["config", "compiler", "__init__"]:
            try:
                full_name = f"mlia.backend.{module_name}.{submodule}"
                module = importlib.import_module(full_name)
            except (ImportError, AttributeError):
                continue

            if not hasattr(module, "CONFIG_TO_CLI_OPTION"):
                continue

            config_mapping = module.CONFIG_TO_CLI_OPTION
            for config_key, cli_option in config_mapping.items():
                specs.append(
                    {
                        "module": module_name,
                        "backend": module_name.replace("_", "-"),
                        "config_key": config_key,
                        "cli_option": cli_option,
                        "full_cli_option": (
                            f"--{module_name.replace('_', '-')}"
                            f".{cli_option.lstrip('-')}"
                        ),
                        "dest": f"{module_name}_{config_key}",
                        "type": Path,
                        "help": f"Overrides the {cli_option} backend option.",
                    }
                )

    return specs


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


def add_backend_install_options(parser: argparse.ArgumentParser) -> None:
    """Add options for the backends configuration."""

    def valid_directory(param: str) -> Path:
        """Check if passed string is a valid directory path."""
        if not (dir_path := Path(param)).is_dir():
            parser.error(f"Invalid directory path {param}")

        return dir_path

    parser.add_argument(
        "--path",
        type=valid_directory,
        help="Path to the installed backend",
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
        "names",
        nargs="+",
        help="Names of the backends to install",
    )


def add_backend_uninstall_options(parser: argparse.ArgumentParser) -> None:
    """Add options for the backends configuration."""
    parser.add_argument(
        "names",
        nargs="+",
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

        def is_corstone_backend_name(backend_name: str) -> bool:
            """Detect Corstone backends by name without importing plugin modules."""
            name = backend_name.casefold()
            return name.startswith("corstone-") or name.startswith("corstone")

        def check(backend: str) -> str:
            """Count Corstone backends and raise an exception if more than one."""
            nonlocal num_corstones
            if is_corstone_backend_name(backend):
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


def add_output_directory(parser: argparse.ArgumentParser) -> None:
    """Add parameter for the output directory."""
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Path to the directory where MLIA will create "
        "output directory 'mlia-output' "
        "for storing artifacts, e.g. logs, target profiles and model files. "
        "If not specified then 'mlia-output' directory will be created "
        "in the current working directory.",
    )


def get_target_profile_opts(target_args: dict | None) -> list[str]:
    """Get non default values passed as parameters for the target profile."""
    if not target_args:
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
        for arg_name, arg_value in target_args.items()
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
        for item in construct_param(params_name[name], target_args[name])
    ]


def get_output_format(args: argparse.Namespace) -> OutputFormat:
    """Return the OutputFormat depending on the CLI flags."""
    output_format: OutputFormat = "plain_text"
    if "json" in args and args.json:
        output_format = "json"
    return output_format
