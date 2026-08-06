# SPDX-FileCopyrightText: Copyright 2022-2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""CLI commands module."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated, Any

import click
import typer
from rich.panel import Panel
from rich.table import Table

from mlia import __version__
from mlia.backend.options import (
    BackendOptionSpec,
    discover_backend_option_specs,
)
from mlia.backend.registry import get_selectable_backends
from mlia.cli.completion import target_profile_names
from mlia.cli.helpers import CLIActionResolver
from mlia.cli.settings import get_environment, new_settings
from mlia.core.context import ExecutionContext
from mlia.core.logging import setup_logging
from mlia.core.settings import ApplicationSettings
from mlia.plugins.plugins import BACKEND_PLUGIN_GROUP, TARGET_PLUGIN_GROUP
from mlia.plugins.registry import list_entry_points
from mlia.utils.console import create_section_header

logger = logging.getLogger(__name__)

CONFIG = create_section_header("ML Inference Advisor configuration")
LIST_TABLE_WIDTH = 120
MLIA_HELP_EPILOG = (
    "Plugin discovery:\n"
    "  Run 'mlia target list' and 'mlia backend list' to see installed plugin "
    "capabilities.\n"
    "  Browse MLIA repositories: "
    "https://github.com/orgs/arm/repositories?q=mlia"
)


def _reshape_backend_options(
    backend_option_specs: list[BackendOptionSpec],
    backend_cli_options: dict[str, object],
) -> dict[str, dict[str, object]]:
    """Reshape flat CLI backend option values into backend config overrides."""
    backend_options: dict[str, dict[str, object]] = {}
    for spec in backend_option_specs:
        option_name = _backend_option_name(spec["click_option"])
        value = backend_cli_options.get(option_name)
        if value is None:
            continue

        backend_options.setdefault(spec["backend"], {})[spec["config_key"]] = value

    return backend_options


def _backend_option_name(click_option: click.Option) -> str:
    """Return the internal Click option name for backend option handling."""
    if click_option.name:
        return click_option.name
    raise ValueError("Backend CLI option must have an internal name.")


class BackendOptionCommand(typer.core.TyperCommand):
    """Typer command that adds backend-specific check options."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the command with discovered backend-specific options."""
        self._backend_option_specs = discover_backend_option_specs()
        params = list(kwargs["params"])
        params.extend(spec["click_option"] for spec in self._backend_option_specs)
        command_kwargs = {**kwargs, "params": params}
        super().__init__(*args, **command_kwargs)

    def invoke(self, ctx: click.Context) -> Any:
        """Collect dynamic backend options before invoking the Typer callback."""
        backend_options = _reshape_backend_options(
            self._backend_option_specs, ctx.params
        )
        for spec in self._backend_option_specs:
            option_name = _backend_option_name(spec["click_option"])
            ctx.params.pop(option_name, None)
        ctx.obj = new_settings(
            source=ctx.ensure_object(ApplicationSettings),
            backend_options=backend_options,
        )
        return super().invoke(ctx)


def complete_backend_names(incomplete: str) -> list[str]:
    """Complete registered selectable backend names."""
    return [
        backend
        for backend in sorted(get_selectable_backends())
        if backend.startswith(incomplete)
    ]


def complete_target_profile_names(incomplete: str) -> list[str]:
    """Complete packaged target profile names."""
    return [
        profile for profile in target_profile_names() if profile.startswith(incomplete)
    ]


def _setup_command_logging(debug: bool) -> None:
    """Set up logging for commands without an execution context."""
    setup_logging(verbose=debug)


def _create_check_context(
    *,
    model: str,
    target_profile: str,
    backend: list[str] | None,
    performance: bool,
    compatibility: bool,
    output_dir: Path | None,
    json_output: bool,
    i_agree_to_the_contained_eula: bool,
    noninteractive: bool,
    debug: bool,
    backend_options: dict[str, dict[str, object]] | None = None,
) -> ExecutionContext:
    """Create an execution context for the live Typer check command."""
    cli_args: dict[str, Any] = {
        "model": model,
        "target_profile": target_profile,
        "backend": backend,
        "performance": performance,
        "compatibility": compatibility,
        "output_dir": output_dir,
        "json": json_output,
        "i_agree_to_the_contained_eula": i_agree_to_the_contained_eula,
        "noninteractive": noninteractive,
        "backend_options": backend_options or {},
    }
    try:
        execution_context = ExecutionContext(
            verbose=debug,
            action_resolver=CLIActionResolver(cli_args),
            output_format="json" if json_output else "plain_text",
            output_dir=output_dir,
        )
    except Exception as err:  # pylint: disable=broad-except
        typer.echo(f"Error: {err}", err=True)
        raise typer.Exit(code=1) from err

    setup_logging(
        execution_context.logs_path,
        execution_context.verbose,
        execution_context.output_format,
    )
    return execution_context


def _log_plugin_table(settings: ApplicationSettings, title: str, group: str) -> None:
    """Log available plugins for a given entry point group."""
    plugins = list_entry_points(group)
    if not plugins:
        settings.console.print("No plugins found.", style="warning")
        return

    table = Table(
        box=None,
        border_style="tbl.border",
        header_style="tbl.header",
        title_style="tbl.title",
        show_lines=False,
        expand=True,
    )

    table.add_column(
        "Name",
        style="bold cyan",
        no_wrap=True,
    )
    table.add_column(
        "Entry Point",
        style="yellow",
        overflow="fold",
        max_width=(LIST_TABLE_WIDTH // 2),
    )
    table.add_column(
        "Package",
    )
    table.add_column(
        "Version",
        style="tbl.border",
        no_wrap=True,
    )

    for plugin in plugins:
        table.add_row(
            plugin.name,
            plugin.value,
            plugin.dist_name or "-",
            plugin.dist_version or "-",
        )

    width = min(LIST_TABLE_WIDTH, settings.console.size.width)

    settings.console.print(
        Panel(
            table,
            title=title,
            width=width,
            border_style="tbl.border",
            title_align="left",
        )
    )


def _create_list_table(settings: ApplicationSettings) -> Table:
    """Create a CLI list table with the shared list command style."""
    return Table(
        box=None,
        border_style="tbl.border",
        header_style="tbl.header",
        title_style="tbl.title",
        show_lines=False,
        expand=True,
    )


def _print_list_table(settings: ApplicationSettings, title: str, table: Table) -> None:
    """Print a CLI list table in the shared panel style."""
    width = min(LIST_TABLE_WIDTH, settings.console.size.width)

    settings.console.print(
        Panel.fit(
            table,
            width=width,
            title=title,
            title_align="left",
            border_style="tbl.border",
        )
    )


def format_target_info(settings: ApplicationSettings) -> None:
    """List available target profiles."""
    from mlia.target.config import get_builtin_target_profile_path, load_profile
    from mlia.target.registry import profiles_by_target

    logger.info(CONFIG)

    _log_plugin_table(settings, "Target Plugins", TARGET_PLUGIN_GROUP)

    grouped_profiles = profiles_by_target()

    logger.info("Available Target Profiles\n")

    for target_type, profile_names in grouped_profiles.items():
        table = _create_list_table(settings)

        table.add_column("Profile", style="tbl.name", no_wrap=True)
        table.add_column("Description", max_width=(LIST_TABLE_WIDTH // 2))

        for profile_name in profile_names:
            try:
                profile_path = get_builtin_target_profile_path(profile_name)
                profile_data = load_profile(profile_path)

                description = profile_data.get("description", "")
                table.add_row(
                    profile_name,
                    description if description else "-",
                )

            except Exception:
                table.add_row(profile_name, "-")

        _print_list_table(settings, f"{target_type.upper()}:", table)


def format_backend_info(settings: ApplicationSettings) -> None:
    """List available backend plugins and installation status."""
    from mlia.api import list_backends

    logger.info(CONFIG)
    _log_plugin_table(settings, "Backend Plugins", BACKEND_PLUGIN_GROUP)

    rows = [
        (
            backend["name"],
            "yes" if backend["installed"] else "no",
            "yes" if backend["could_be_installed"] else "no",
        )
        for backend in list_backends()
    ]

    if not rows:
        settings.console.print(
            Panel.fit(
                "No backends found.",
                title="Backends",
                border_style="tbl.border",
            )
        )
        return

    table = _create_list_table(settings)

    table.add_column("Name", style="tbl.name", no_wrap=True)
    table.add_column("Installed", no_wrap=True)
    table.add_column("Installable", no_wrap=True)

    for row in rows:
        table.add_row(*row)

    _print_list_table(settings, "Backends", table)


mlia_app = typer.Typer(
    no_args_is_help=True,
    epilog=MLIA_HELP_EPILOG,
    context_settings={
        "help_option_names": ["-h", "--help"],
    },
)

backend_app = typer.Typer(
    no_args_is_help=True,
    context_settings={
        "help_option_names": ["-h", "--help"],
    },
)

target_app = typer.Typer(
    no_args_is_help=True,
    context_settings={
        "help_option_names": ["-h", "--help"],
    },
)

mlia_app.add_typer(
    backend_app,
    name="backend",
    help="Manage MLIA backends",
)

mlia_app.add_typer(
    target_app,
    name="target",
    help="Manage MLIA targets",
)


def version_callback(value: bool) -> None:
    """Print the package version and exit when the version flag is set."""
    if value:
        typer.echo(f"MLIA {__version__}")
        raise typer.Exit()


def debug_option() -> Any:
    """Return the shared debug option declaration."""
    return typer.Option(
        get_environment().get("DEBUG", "") != "",
        "--debug",
        "-d",
        help="Produce verbose output",
    )


@mlia_app.callback()
def mlia_app_main(
    ctx: typer.Context,
    version: bool = typer.Option(
        False,
        "--version",
        "-v",
        callback=version_callback,
        is_eager=True,
        help="Show the MLIA version",
    ),
) -> None:
    """Configure top-level MLIA CLI callback options."""
    src = ctx.obj if isinstance(ctx.obj, ApplicationSettings) else None
    ctx.obj = new_settings(source=src)


@backend_app.callback()
def backend_app_main(ctx: typer.Context) -> None:
    """Configure the backend CLI namespace."""
    src = ctx.obj if isinstance(ctx.obj, ApplicationSettings) else None
    ctx.obj = new_settings(source=src)


@target_app.callback()
def target_app_main(ctx: typer.Context) -> None:
    """Configure the target CLI namespace."""
    src = ctx.obj if isinstance(ctx.obj, ApplicationSettings) else None
    ctx.obj = new_settings(source=src)


def check(
    model: Annotated[str, typer.Argument(help="Model to check")],
    target_profile: Annotated[
        str,
        typer.Option(
            "--target-profile",
            "-t",
            help="Set the target profile",
            autocompletion=complete_target_profile_names,
        ),
    ],
    output_dir: Annotated[
        Path | None,
        typer.Option("--output-dir", help="Set an output directory"),
    ] = None,
    backend: Annotated[
        list[str] | None,
        typer.Option(
            "--backend",
            "-b",
            help="Set backend profiles to use for evaluation",
            autocompletion=complete_backend_names,
        ),
    ] = None,
    performance: Annotated[
        bool,
        typer.Option(
            "--performance",
            help="Estimate the performance of the model",
        ),
    ] = False,
    compatibility: Annotated[
        bool,
        typer.Option(
            "--compatibility",
            help="Perform compatibility checks (default)",
        ),
    ] = False,
    json_output: Annotated[
        bool,
        typer.Option(
            "--json",
            help="Output results in a JSON format",
        ),
    ] = False,
    i_agree_to_the_contained_eula: Annotated[
        bool,
        typer.Option(
            "--i-agree-to-the-contained-eula",
            help="Agree to the contained EULA",
        ),
    ] = False,
    noninteractive: Annotated[
        bool,
        typer.Option("--noninteractive", help="Run check without interaction"),
    ] = False,
    debug: bool = debug_option(),
    backend_options: dict[str, dict[str, object]] | None = None,
    ctx: typer.Context | None = None,
) -> None:
    """Generate advice for the input model."""
    from mlia.api import get_advice
    from mlia.cli.command_validators import validate_check_target_profile

    category = set()
    if compatibility:
        category.add("compatibility")

    if performance:
        category.add("performance")

    if not category:
        category.add("compatibility")

    execution_context = _create_check_context(
        model=model,
        target_profile=target_profile,
        backend=backend,
        performance=performance,
        compatibility=compatibility,
        output_dir=output_dir,
        json_output=json_output,
        i_agree_to_the_contained_eula=i_agree_to_the_contained_eula,
        noninteractive=noninteractive,
        debug=debug,
        backend_options=backend_options,
    )

    if not validate_check_target_profile(target_profile, category):
        raise typer.Exit(code=0)

    get_advice(
        target_profile,
        model,
        category,
        context=execution_context,
        backends=backend,
        accept_eula=True
        if i_agree_to_the_contained_eula
        else False
        if noninteractive
        else None,
        backend_options=backend_options,
    )


@mlia_app.command(
    "check",
    cls=BackendOptionCommand,
    help="Generate compatibility/performance advice for a model",
    no_args_is_help=True,
)
def check_command(
    ctx: typer.Context,
    model: Annotated[str, typer.Argument(help="Model to check")],
    target_profile: Annotated[
        str,
        typer.Option(
            "--target-profile",
            "-t",
            help="Set the target profile",
            autocompletion=complete_target_profile_names,
        ),
    ],
    output_dir: Annotated[
        Path | None,
        typer.Option("--output-dir", help="Set an output directory"),
    ] = None,
    backend: Annotated[
        list[str] | None,
        typer.Option(
            "--backend",
            "-b",
            help="Set backend profiles to use for evaluation",
            autocompletion=complete_backend_names,
        ),
    ] = None,
    performance: Annotated[
        bool,
        typer.Option(
            "--performance",
            help="Estimate the performance of the model",
        ),
    ] = False,
    compatibility: Annotated[
        bool,
        typer.Option(
            "--compatibility",
            help="Perform compatibility checks (default)",
        ),
    ] = False,
    json_output: Annotated[
        bool,
        typer.Option(
            "--json",
            help="Output results in a JSON format",
        ),
    ] = False,
    i_agree_to_the_contained_eula: Annotated[
        bool,
        typer.Option(
            "--i-agree-to-the-contained-eula",
            help="Agree to the contained EULA",
        ),
    ] = False,
    noninteractive: Annotated[
        bool,
        typer.Option("--noninteractive", help="Run check without interaction"),
    ] = False,
    debug: bool = debug_option(),
) -> None:
    """Typer entry point for the check command."""
    backend_options = {}
    if ctx is not None and isinstance(ctx.obj, ApplicationSettings):
        backend_options = ctx.obj.backend_options

    check(
        ctx=ctx,
        model=model,
        output_dir=output_dir,
        target_profile=target_profile,
        backend=backend,
        performance=performance,
        compatibility=compatibility,
        json_output=json_output,
        i_agree_to_the_contained_eula=i_agree_to_the_contained_eula,
        noninteractive=noninteractive,
        debug=debug,
        backend_options=backend_options,
    )


@backend_app.command("install")
def backend_install(
    names: list[str] = typer.Argument(
        ...,
        help="Backend names to install",
        autocompletion=complete_backend_names,
    ),
    path: Path | None = typer.Option(
        None,
        "--path",
        help="Install from a local backend package",
    ),
    accept_eula: bool = typer.Option(
        False,
        "--accept-eula",
        help="Accept any required EULAs",
    ),
    noninteractive: bool = typer.Option(
        False,
        "--noninteractive",
        help="Run without interactive prompts",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help="Reinstall if already installed",
    ),
    debug: bool = debug_option(),
) -> None:
    """Install backend packages."""
    from mlia.api import install_backends

    _setup_command_logging(debug)
    install_backends(
        names=names,
        path=path,
        accept_eula=accept_eula,
        noninteractive=noninteractive,
        force=force,
    )


@backend_app.command("uninstall")
def backend_uninstall(
    names: list[str] = typer.Argument(
        ...,
        help="Backend names to uninstall",
        autocompletion=complete_backend_names,
    ),
    debug: bool = debug_option(),
) -> None:
    """Uninstall backend packages."""
    from mlia.api import uninstall_backends

    _setup_command_logging(debug)
    uninstall_backends(names)


@backend_app.command("list")
def backend_list(
    ctx: typer.Context,
    debug: bool = debug_option(),
) -> None:
    """List available backend plugins and backend installation status."""
    _setup_command_logging(debug)
    format_backend_info(ctx.obj)


@target_app.command("list")
def target_list(
    ctx: typer.Context,
    debug: bool = debug_option(),
) -> None:
    """List available target plugins and built-in target profiles."""
    _setup_command_logging(debug)
    format_target_info(ctx.obj)
