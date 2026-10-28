# SPDX-FileCopyrightText: Copyright 2022-2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""CLI commands module."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any, cast

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
from mlia.check import CheckError, CheckRequest, execute_check, prepare_check
from mlia.cli.completion import target_profile_names
from mlia.cli.settings import get_environment, new_settings
from mlia.core.errors import ConfigurationError
from mlia.core.logging import setup_logging
from mlia.core.output_rendering import (
    standardized_output_to_json,
)
from mlia.core.settings import ApplicationSettings
from mlia.plugins.analysis import AnalysisPluginRegistry
from mlia.plugins.plugins import (
    BACKEND_PLUGIN_GROUP,
    TARGET_PLUGIN_GROUP,
    load_analysis_plugins,
)
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
ANALYSIS_PLUGIN_ARGS_META_KEY = "mlia.analysis_plugin_args"


def _reshape_backend_options(
    backend_option_specs: list[BackendOptionSpec],
    backend_cli_options: dict[str, object],
) -> dict[str, dict[str, object]]:
    """Reshape flat CLI backend option values into backend config overrides."""
    backend_options: dict[str, dict[str, object]] = {}
    for spec in backend_option_specs:
        option_name = _click_option_name(spec["click_option"])
        value = backend_cli_options.get(option_name)
        if value is None:
            continue

        backend_options.setdefault(spec["backend"], {})[spec["config_key"]] = value

    return backend_options


def _click_option_name(click_option: click.Option) -> str:
    """Return the internal name of a dynamic Click option."""
    if click_option.name:
        return click_option.name
    raise ValueError("Dynamic CLI option must have an internal name.")


def _load_analysis_plugin_registry() -> AnalysisPluginRegistry:
    """Load analysis plugins into a new command-scoped registry."""
    registry = AnalysisPluginRegistry()
    load_analysis_plugins(registry)
    return registry


def _exit_with_help_on_empty_args(ctx: click.Context, args: list[str]) -> None:
    """Print help and exit when arguments are empty outside shell completion."""
    if not args and not ctx.resilient_parsing:
        typer.echo(ctx.get_help())
        ctx.exit(2)


class HelpOnEmptyGroup(typer.core.TyperGroup):
    """Typer group that prints help to stdout for empty argument lists."""

    def parse_args(self, ctx: click.Context, args: list[str]) -> list[str]:
        """Print help for empty arguments without disrupting shell completion."""
        _exit_with_help_on_empty_args(ctx, args)
        return super().parse_args(ctx, args)


class CheckOptionCommand(typer.core.TyperCommand):
    """Typer command that adds dynamic options to the check command."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the command with backend and analysis-plugin options."""
        self._backend_option_specs = discover_backend_option_specs()
        self._analysis_plugin_registry = _load_analysis_plugin_registry()
        self._analysis_plugin_options = self._analysis_plugin_registry.cli_options()

        params = list(kwargs["params"])
        params.extend(spec["click_option"] for spec in self._backend_option_specs)
        params.extend(self._analysis_plugin_options)
        command_kwargs = {**kwargs, "params": params}
        super().__init__(*args, **command_kwargs)

    def prepare_request(self, ctx: click.Context) -> CheckRequest:
        """Collect dynamic options and run the shared Typer preparation callback."""
        backend_options = _reshape_backend_options(
            self._backend_option_specs, ctx.params
        )
        for spec in self._backend_option_specs:
            ctx.params.pop(_click_option_name(spec["click_option"]), None)

        analysis_plugin_args = {
            _click_option_name(option): ctx.params.pop(_click_option_name(option), None)
            for option in self._analysis_plugin_options
        }
        ctx.meta[ANALYSIS_PLUGIN_ARGS_META_KEY] = analysis_plugin_args
        ctx.obj = new_settings(
            source=ctx.ensure_object(ApplicationSettings),
            backend_options=backend_options,
        )
        return cast(CheckRequest, super().invoke(ctx))

    def invoke(self, ctx: click.Context) -> None:
        """Prepare a request and dispatch to the selected frontend."""
        request = self.prepare_request(ctx)
        settings = ctx.ensure_object(ApplicationSettings)
        analysis_plugin_registry = self._analysis_plugin_registry
        try:
            if ctx.params["tui"]:
                try:
                    from mlia_tui import launch
                except ModuleNotFoundError as err:
                    if err.name != "mlia_tui":
                        raise
                    raise CheckError(
                        "The optional TUI package is not installed. "
                        "Install it in this environment with 'pip install mlia-tui'."
                    ) from err
                result = launch(request)
                if result is None:
                    return
                if result.outcome == "skipped":
                    typer.echo(result.skip_reason or "Check skipped.", err=True)
                elif request.json_output and result.output is not None:
                    settings.console.out(
                        standardized_output_to_json(result.output), highlight=False
                    )
            else:
                execute_check(
                    request,
                    console=settings.console,
                    analysis_plugin_registry=analysis_plugin_registry,
                )
        except ConfigurationError as err:
            if not ctx.params["tui"] and not isinstance(err, CheckError):
                raise
            exit_code = err.exit_code if isinstance(err, CheckError) else 1
            prefix = "" if not ctx.params["tui"] and exit_code == 2 else "Error: "
            typer.echo(f"{prefix}{err}", err=True)
            raise typer.Exit(code=exit_code) from err

    def parse_args(self, ctx: click.Context, args: list[str]) -> list[str]:
        """Print help for empty arguments without disrupting shell completion."""
        _exit_with_help_on_empty_args(ctx, args)
        return super().parse_args(ctx, args)


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
    setup_logging(verbose=debug, cli_mode=True)


def _replace_unencodable_characters(text: str, encoding: str) -> str:
    """Replace characters unsupported by the configured console encoding."""
    return text.encode(encoding, errors="replace").decode(encoding)


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
        style="tbl.name",
        no_wrap=True,
    )
    table.add_column(
        "Entry Point",
        style="tbl.highlight",
        overflow="fold",
        max_width=(LIST_TABLE_WIDTH // 2),
    )
    table.add_column(
        "Package",
    )
    table.add_column(
        "Version",
        style="tbl.version",
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
    cls=HelpOnEmptyGroup,
    no_args_is_help=True,
    epilog=MLIA_HELP_EPILOG,
    context_settings={
        "help_option_names": ["-h", "--help"],
    },
)

backend_app = typer.Typer(
    cls=HelpOnEmptyGroup,
    no_args_is_help=True,
    context_settings={
        "help_option_names": ["-h", "--help"],
    },
)

target_app = typer.Typer(
    cls=HelpOnEmptyGroup,
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


def _require_cli_target(
    ctx: click.Context, param: click.Parameter, value: str | None
) -> str | None:
    """Reject a missing CLI target before unspecified plugin options are resolved."""
    if value is None and not ctx.resilient_parsing and not ctx.params.get("tui"):
        raise click.MissingParameter(ctx=ctx, param=param)
    return value


@mlia_app.command(
    "check",
    cls=CheckOptionCommand,
    help="Generate compatibility/performance advice for a model",
    epilog=(
        "Examples:\n\n"
        "  mlia target list\n\n"
        "  mlia check model.tflite --target-profile <profile> --compatibility\n\n"
        "Replace <profile> with a listed profile name or a custom profile path."
    ),
    no_args_is_help=True,
)
def check_command(
    ctx: typer.Context,
    target_profile: Annotated[
        str | None,
        typer.Option(
            "--target-profile",
            "-t",
            help=(
                "Hardware-analysis configuration: a named profile or path to a "
                "custom profile. Run 'mlia target list' to see profiles supplied "
                "by installed plugins. Required unless --tui is supplied."
            ),
            autocompletion=complete_target_profile_names,
            callback=_require_cli_target,
        ),
    ] = None,
    model: Annotated[str | None, typer.Argument(help="Model to check")] = None,
    output_dir: Annotated[
        Path | None,
        typer.Option("--output-dir", "--out-dir", help="Set an output directory"),
    ] = None,
    backend: Annotated[
        list[str] | None,
        typer.Option(
            "--backend",
            "-b",
            help=(
                "Implementation that performs the analysis. Normally uses the "
                "target's default backends; repeat this option to select "
                "compatible backends explicitly. Run 'mlia backend list' to see "
                "available backends and their installation status."
            ),
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
    profiling_data: Annotated[
        list[Path] | None,
        typer.Option(
            "--profiling-data",
            help=(
                "Use measured profiling data as an input source; repeat for "
                "ordered inputs"
            ),
        ),
    ] = None,
    companion_model: Annotated[
        Path | None,
        typer.Option(
            "--companion-model",
            help=(
                "Path to a supplementary model used by the selected target/backend "
                "during analysis. See the plugin documentation for supported "
                "formats and usage."
            ),
        ),
    ] = None,
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
        typer.Option(
            "--noninteractive",
            help="Do not prompt for backend installation (also applies with --tui)",
        ),
    ] = False,
    tui: Annotated[
        bool,
        typer.Option(
            "--tui",
            help="Configure and run the check in the optional TUI",
            is_eager=True,  # Resolve the launch mode before validating the target.
        ),
    ] = False,
    debug: bool = debug_option(),
) -> CheckRequest:
    """Prepare the check request from resolved Typer values."""
    settings = ctx.obj if isinstance(ctx.obj, ApplicationSettings) else new_settings()
    plugin_args = cast(
        Mapping[str, object],
        ctx.meta[ANALYSIS_PLUGIN_ARGS_META_KEY],
    )

    return prepare_check(
        model=model,
        target_profile=target_profile,
        output_dir=output_dir,
        backend=backend,
        performance=performance,
        compatibility=compatibility,
        profiling_data=profiling_data,
        companion_model=companion_model,
        json_output=json_output,
        i_agree_to_the_contained_eula=i_agree_to_the_contained_eula,
        noninteractive=noninteractive,
        debug=debug,
        settings=settings,
        plugin_args=plugin_args,
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
