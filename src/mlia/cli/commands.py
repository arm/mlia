# SPDX-FileCopyrightText: Copyright 2022-2026, Arm Limited and/or its affiliates.
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

from mlia.api import ExecutionContext, get_advice
from mlia.backend.manager import get_installation_manager
from mlia.cli.command_validators import validate_check_target_profile
from mlia.core.reporting import Column, Format, Table
from mlia.plugins.plugins import BACKEND_PLUGIN_GROUP, TARGET_PLUGIN_GROUP
from mlia.plugins.registry import list_entry_points
from mlia.target.config import get_builtin_target_profile_path, load_profile
from mlia.target.registry import profiles_by_target
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
    i_agree_to_the_contained_eula: bool = False,
    noninteractive: bool = False,
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
    :param i_agree_to_the_contained_eula: flag indicating EULA acceptance
    :param noninteractive: flag indicating noninteractive mode

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

    get_advice(
        target_profile,
        model,
        category,
        context=ctx,
        backends=backend,
        accept_eula=True
        if i_agree_to_the_contained_eula
        else False
        if noninteractive
        else None,
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
        eula_agreement = i_agree_to_the_contained_eula
        manager.download_and_install(names, eula_agreement, force)


def backend_uninstall(names: list[str]) -> None:
    """Uninstall backend."""
    logger.info(CONFIG)

    manager = get_installation_manager(noninteractive=True)
    manager.uninstall(names)


def backend_list() -> None:
    """List backends status."""
    logger.info(CONFIG)

    _log_plugin_list("Backend Plugins", BACKEND_PLUGIN_GROUP)

    manager = get_installation_manager(noninteractive=True)
    manager.show_env_details()


def target_list() -> None:
    """List available target profiles."""
    logger.info(CONFIG)

    _log_plugin_list("Target Plugins", TARGET_PLUGIN_GROUP)

    grouped_profiles = profiles_by_target()

    logger.info("Available Target Profiles\n")

    for target_type, profile_names in grouped_profiles.items():
        rows = []

        for profile_name in profile_names:
            try:
                profile_path = get_builtin_target_profile_path(profile_name)
                profile_data = load_profile(profile_path)

                description = profile_data.get("description", "")

                rows.append((profile_name, description if description else "-"))

            except Exception:  # pylint: disable=broad-except
                rows.append((profile_name, "-"))

        table = Table(
            columns=[
                Column("Profile"),
                Column("Description", fmt=Format(wrap_width=60)),
            ],
            rows=rows,
            name=f"{target_type.upper()}:",
        )

        logger.info("%s\n", table.to_plain_text())


def _log_plugin_list(title: str, group: str) -> None:
    """Log available plugins for a given entry point group."""
    plugins = list_entry_points(group)
    if not plugins:
        logger.info("%s\nNo plugins found.\n", title)
        return

    rows = [
        (
            plugin.name,
            plugin.value,
            plugin.dist_name or "-",
            plugin.dist_version or "-",
        )
        for plugin in plugins
    ]
    table = Table(
        columns=[
            Column("Name"),
            Column("Entry Point", fmt=Format(wrap_width=60)),
            Column("Package"),
            Column("Version"),
        ],
        rows=rows,
        name=f"{title}:",
    )
    logger.info("%s\n", table.to_plain_text())
