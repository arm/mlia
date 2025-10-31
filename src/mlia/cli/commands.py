# SPDX-FileCopyrightText: Copyright 2022-2025, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""CLI commands module.

This module contains functions which implement main app
functionality.

Before running them from scripts 'logging' module should
be configured. Function 'setup_logging' from module
'mli.core.logging' could be used for that, e.g.

>>> from mlia.api import ExecutionContext
>>> from mlia.core.logging import setup_logging
>>> setup_logging(verbose=True)
>>> import mlia.cli.commands as mlia
>>> mlia.check(ExecutionContext(), "ethos-u55-256",
                   "path/to/model")
"""
from __future__ import annotations

import logging
from pathlib import Path

from mlia.api import ExecutionContext
from mlia.api import get_advice
from mlia.backend.manager import get_installation_manager
from mlia.cli.command_validators import validate_backend
from mlia.cli.command_validators import validate_check_target_profile
from mlia.cli.command_validators import validate_optimize_target_profile
from mlia.cli.options import parse_optimization_parameters
from mlia.utils.console import create_section_header

logger = logging.getLogger(__name__)

CONFIG = create_section_header("ML Inference Advisor configuration")


def check(
    ctx: ExecutionContext,
    target_profile: str,
    model: str | None = None,
    compatibility: bool = False,
    performance: bool = False,
    backend: list[str] | None = None,
) -> None:
    """Generate a full report on the input model.

    This command runs a series of tests in order to generate a
    comprehensive report/advice:

        - converts the input Keras model into TensorFlow Lite format
        - checks the model for operator compatibility on the specified target
        - generates a final report on the steps above
        - provides advice on how to (possibly) improve the inference performance

    :param ctx: execution context
    :param target_profile: target profile identifier. Will load appropriate parameters
            from the profile.json file based on this argument.
    :param model: path to the Keras model
    :param compatibility: flag that identifies whether to run compatibility checks
    :param performance: flag that identifies whether to run performance checks
    :param backend: list of the backends to use for evaluation

    Example:
        Run command for the target profile ethos-u55-256 to verify both performance
        and operator compatibility.

        >>> from mlia.api import ExecutionContext
        >>> from mlia.core.logging import setup_logging
        >>> setup_logging()
        >>> from mlia.cli.commands import check
        >>> check(ExecutionContext(), "ethos-u55-256",
                      "model.h5", compatibility=True, performance=True)
    """
    if not model:
        raise ValueError("Model is not provided.")

    # Set category based on checks to perform (i.e. "compatibility" and/or
    # "performance").
    # If no check type is specified, "compatibility" is the default category.
    if compatibility and performance:
        category = {"compatibility", "performance"}
    elif performance:
        category = {"performance"}
    else:
        category = {"compatibility"}

    validate_check_target_profile(target_profile, category)
    validated_backend = validate_backend(target_profile, backend)

    get_advice(
        target_profile,
        model,
        category,
        context=ctx,
        backends=validated_backend,
    )


def optimize(  # pylint: disable=too-many-locals,too-many-arguments
    ctx: ExecutionContext,
    target_profile: str,
    model: str,
    pruning: bool,
    clustering: bool,
    pruning_target: float | None,
    clustering_target: int | None,
    optimization_profile: str | None = None,
    rewrite: bool | None = None,
    rewrite_target: str | None = None,
    rewrite_start: str | None = None,
    rewrite_end: str | None = None,
    layers_to_optimize: list[str] | None = None,
    backend: list[str] | None = None,
    dataset: Path | None = None,
) -> None:
    """Show the performance improvements (if any) after applying the optimizations.

    This command applies the selected optimization techniques (up to the
    indicated targets) and generates a report with advice on how to improve
    the inference performance (if possible).

    :param ctx: execution context
    :param target_profile: target profile identifier. Will load appropriate parameters
            from the profile.json file based on this argument.
    :param model: path to the TensorFlow Lite model
    :param pruning: perform pruning optimization (default if no option specified)
    :param clustering: perform clustering optimization
    :param clustering_target: clustering optimization target
    :param pruning_target: pruning optimization target
    :param layers_to_optimize: list of the layers of the model which should be
           optimized, if None then all layers are used
    :param backend: list of the backends to use for evaluation

    Example:
        Run command for the target profile ethos-u55-256 and
        the provided TensorFlow Lite model and print report on the standard output

        >>> from mlia.core.logging import setup_logging
        >>> from mlia.api import ExecutionContext
        >>> setup_logging()
        >>> from mlia.cli.commands import optimize
        >>> optimize(ExecutionContext(),
                         target_profile="ethos-u55-256",
                         model="model.tflite", pruning=True,
                         clustering=False, pruning_target=0.5,
                         clustering_target=None)
    """
    opt_params = (
        parse_optimization_parameters(  # pylint: disable=too-many-function-args
            pruning,
            clustering,
            pruning_target,
            clustering_target,
            rewrite,
            rewrite_target,
            rewrite_start,
            rewrite_end,
            layers_to_optimize,
            dataset,
        )
    )

    validate_optimize_target_profile(target_profile)
    validated_backend = validate_backend(target_profile, backend)

    get_advice(
        target_profile,
        model,
        {"optimization"},
        optimization_targets=opt_params,
        optimization_profile=optimization_profile,
        context=ctx,
        backends=validated_backend,
    )


def backend_install(
    names: list[str],
    path: Path | None = None,
    i_agree_to_the_contained_eula: bool = False,
    noninteractive: bool = False,
    force: bool = False,
) -> None:
    """Install backend."""
    logger.info(CONFIG)

    manager = get_installation_manager(noninteractive)

    if path is not None:
        if len(names) != 1:
            raise ValueError("Exactly one backend name is required.")
        manager.install_from(path, names[0], force)
    else:
        eula_agreement = not i_agree_to_the_contained_eula
        manager.download_and_install(names, eula_agreement, force)


def backend_uninstall(names: list[str]) -> None:
    """Uninstall backend."""
    logger.info(CONFIG)

    manager = get_installation_manager(noninteractive=True)
    manager.uninstall(names)


def backend_list() -> None:
    """List backends status."""
    logger.info(CONFIG)

    manager = get_installation_manager(noninteractive=True)
    manager.show_env_details()
