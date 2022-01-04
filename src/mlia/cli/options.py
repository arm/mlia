# Copyright 2021, Arm Ltd.
"""Module for the CLI options."""
import argparse
import os
from typing import Any
from typing import List
from typing import Tuple

from mlia.utils.filesystem import get_supported_profile_names


def add_target_options(parser: argparse.ArgumentParser) -> None:
    """Add target specific options."""
    target_group = parser.add_argument_group("target options")
    target_group.add_argument(
        "--target",
        choices=get_supported_profile_names(),
        help="""Target profile that will set the default device options
                such as device type, mac value, memory mode, etc..
                For the values associated with each target profile,
                see: resources/profiles.json.""",
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
        action=OutputFormatAction,
        help=(
            "Name of the file where report will be saved. "
            "The report is always displayed the standard output "
            "formatted as plain text. "
            "Valid file extensions(formats) are {.txt,.json,.csv}, "
            "anything else will be formatted as plain text."
        ),
    )


class OutputFormatAction(argparse.Action):  # pylint: disable=too-few-public-methods
    """Argparse action for --output option."""

    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: Any,
        option_string: str = None,
    ) -> None:
        """Add the output file, and derive the format from file extension."""
        output_formats = {".txt": "plain_text", ".json": "json", ".csv": "csv"}
        setattr(namespace, self.dest, values)
        file_ext = os.path.splitext(values)[1]
        if file_ext in output_formats:
            setattr(namespace, "output_format", output_formats[file_ext])


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


def add_quantize_option(parser: argparse.ArgumentParser) -> None:
    """Add quantization if needed."""
    quantization_group = parser.add_argument_group(
        "quantization_opts", "Quantization options"
    )
    quantization_group.add_argument(
        "--quantized",
        default=False,
        action="store_true",
        help="""Quantizes model to int8 if provided.
        Leaves it as fp32 if otherwise.""",
    )


def add_out_path(parser: argparse.ArgumentParser) -> None:
    """Add option for output path instead of temporary directory."""
    out_path_group = parser.add_argument_group("out path")
    out_path_group.add_argument(
        "--out-path",
        help="""Add the folder where you want to save the files created
        (if none specified, they will be saved in the current directory)""",
        default=None,
    )


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
    optimization_type: str, optimization_target: str, sep: str = ","
) -> List[Tuple[str, float]]:
    """Parse provided optimization parameters."""
    if not optimization_type:
        raise Exception("Optimization type is not provided")

    if not optimization_target:
        raise Exception("Optimization target is not provided")

    opt_types = optimization_type.split(sep)
    opt_targets = optimization_target.split(sep)

    if len(opt_types) != len(opt_targets):
        raise Exception("Wrong number of optimization targets and types")

    optimizer_params = [
        (opt_type.strip(), float(opt_target))
        for opt_type, opt_target in zip(opt_types, opt_targets)
    ]

    return optimizer_params
