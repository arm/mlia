# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Backend CLI option metadata."""

import copy
from pathlib import Path
from typing import TypedDict, cast

import click

from mlia.backend.config import BackendCliOption
from mlia.backend.registry import registry


class BackendOptionSpec(TypedDict):
    """Describe backend option metadata derived from backend configurations."""

    module: str
    backend: str
    config_key: str
    click_option: click.Option


def _backend_option_dest(module_name: str, config_key: str) -> str:
    """Return the Click destination for a backend option."""
    return f"{module_name}_{config_key}"


def _namespace_backend_option_decl(backend_name: str, option_decl: str) -> str:
    """Prefix long backend option declarations with the backend namespace."""
    if not option_decl.startswith("--"):
        raise TypeError("Typed backend CLI options must only declare long options.")

    namespace = f"--{backend_name}."
    if option_decl.startswith(namespace):
        return option_decl

    return f"{namespace}{option_decl.lstrip('-')}"


def _legacy_backend_click_option(
    backend_name: str,
    module_name: str,
    config_key: str,
    cli_option: str,
) -> BackendOptionSpec:
    """Return the legacy Click option declaration."""
    if not isinstance(cli_option, str):
        raise TypeError(
            "Backend plugin 0.0.1 CLI option "
            f"'{backend_name}.{config_key}' must be a string."
        )

    full_cli_option = f"--{backend_name}.{cli_option.lstrip('-')}"
    dest = _backend_option_dest(module_name, config_key)
    help_text = f"Overrides the {cli_option} backend option."
    click_option = click.Option(
        [full_cli_option, dest],
        default=None,
        type=click.Path(path_type=Path),
        help=help_text,
    )
    return {
        "module": module_name,
        "backend": backend_name,
        "config_key": config_key,
        "click_option": click_option,
    }


def _typed_backend_click_option(
    backend_name: str,
    module_name: str,
    config_key: str,
    cli_option: click.Option,
) -> BackendOptionSpec:
    """Return the typed Click option declaration."""
    if not isinstance(cli_option, click.Option):
        raise TypeError(
            "Backend plugin 0.0.2 CLI option "
            f"'{backend_name}.{config_key}' must be a Click option."
        )

    click_option = copy.copy(cli_option)
    click_option.opts = [
        _namespace_backend_option_decl(backend_name, option_decl)
        for option_decl in cli_option.opts
    ]
    click_option.secondary_opts = [
        _namespace_backend_option_decl(backend_name, option_decl)
        for option_decl in cli_option.secondary_opts
    ]
    click_option.name = _backend_option_dest(module_name, config_key)

    return {
        "module": module_name,
        "backend": backend_name,
        "config_key": config_key,
        "click_option": click_option,
    }


def _backend_click_option(
    backend_name: str,
    module_name: str,
    config_key: str,
    cli_option: BackendCliOption,
    plugin_interface_version: str,
) -> BackendOptionSpec:
    """Return the Click option for a backend option declaration."""
    if plugin_interface_version == "0.0.1":
        return _legacy_backend_click_option(
            backend_name, module_name, config_key, cast(str, cli_option)
        )
    if plugin_interface_version == "0.0.2":
        return _typed_backend_click_option(
            backend_name, module_name, config_key, cast(click.Option, cli_option)
        )

    raise ValueError(
        f"Unsupported backend plugin interface version '{plugin_interface_version}' "
        f"for backend '{backend_name}'."
    )


def discover_backend_option_specs() -> list[BackendOptionSpec]:
    """Return option metadata from registered backend configurations."""
    specs: list[BackendOptionSpec] = []
    for backend_name, backend_configuration in registry.items.items():
        module_name = backend_name.replace("-", "_")
        plugin_interface_version = (
            registry.plugin_interface_versions.get(backend_name) or "0.0.1"
        )
        for config_key, cli_option in backend_configuration.cli_options.items():
            specs.append(
                _backend_click_option(
                    backend_name,
                    module_name,
                    config_key,
                    cli_option,
                    plugin_interface_version,
                )
            )

    return specs
