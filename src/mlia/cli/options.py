# Copyright 2021, Arm Ltd.
"""Module for the CLI options."""
import argparse
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

from mlia.utils.filesystem import get_supported_profile_names
from mlia.utils.types import is_number


def add_target_options(parser: argparse.ArgumentParser) -> None:
    """Add target specific options."""
    target_profiles = get_supported_profile_names()

    default_target = None
    default_help = ""
    if target_profiles:
        default_target = target_profiles[0]
        default_help = " (default: %(default)s)"

    target_group = parser.add_argument_group("target options")
    target_group.add_argument(
        "--target",
        choices=target_profiles,
        default=default_target,
        help="Target profile that will set the default device options"
        "such as device type, mac value, memory mode, etc."
        "For the values associated with each target profile,"
        f"see: resources/profiles.json{default_help}.",
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
    model_group = parser.add_argument_group("TFLite model options")
    # make model parameter optional
    model_group.add_argument("model", nargs="?", help="TFLite model (optional)")


def add_tflite_model_options(parser: argparse.ArgumentParser) -> None:
    """Add model specific options."""
    model_group = parser.add_argument_group("TFLite model options")
    model_group.add_argument("model", help="TFLite model")


def add_output_options(parser: argparse.ArgumentParser) -> None:
    """Add output specific options."""
    output_group = parser.add_argument_group("output options")
    output_group.add_argument(
        "--output",
        help=(
            "Name of the file where report will be saved. "
            "The report is always displayed the standard output "
            "formatted as plain text. "
            "Valid file extensions(formats) are {.txt, .json, .csv}, "
            "anything else will be formatted as plain text."
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


def parse_optimization_parameters(
    optimization_type: str,
    optimization_target: str,
    sep: str = ",",
    layers_to_optimize: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
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


def get_target_opts(device_args: Optional[Dict]) -> List[str]:
    """Get non default values passed as parameters for the target."""
    if not device_args:
        return []

    dummy_parser = argparse.ArgumentParser()
    add_target_options(dummy_parser)
    args = dummy_parser.parse_args([])

    params_name = {
        action.dest: param_name
        for param_name, action in dummy_parser._option_string_actions.items()  # pylint: disable=protected-access
    }

    non_default = [
        arg_name
        for arg_name, arg_value in device_args.items()
        if arg_name in args and vars(args)[arg_name] != arg_value
    ]

    def construct_param(name: str, value: Any) -> List[str]:
        """Construct parameter."""
        if isinstance(value, list):
            return [str(item) for v in value for item in [name, v]]

        return [name, str(value)]

    return [
        item
        for name in non_default
        for item in construct_param(params_name[name], device_args[name])
    ]
