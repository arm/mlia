# SPDX-FileCopyrightText: Copyright 2022-2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""CLI commands module."""

from __future__ import annotations

import importlib
import json
import logging
import re
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, cast

import click
import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from typer.completion import completion_init, show_callback

from mlia import __version__
from mlia.backend.options import (
    BackendOptionSpec,
    discover_backend_option_specs,
)
from mlia.backend.registry import get_selectable_backends
from mlia.cli.completion import target_profile_names
from mlia.cli.settings import get_environment, new_settings
from mlia.core.errors import ConfigurationError
from mlia.core.logging import setup_logging
from mlia.core.settings import ApplicationSettings
from mlia.plugins.analysis import AnalysisPluginRegistry
from mlia.plugins.plugins import (
    ANALYSIS_PLUGIN_GROUP,
    BACKEND_PLUGIN_GROUP,
    TARGET_PLUGIN_GROUP,
    TRANSFORMER_PLUGIN_GROUP,
    load_analysis_plugins,
)
from mlia.plugins.registry import list_entry_points
from mlia.utils.boundaries.filesystem import install_shell_completion_with_notice
from mlia.utils.console import create_section_header

if TYPE_CHECKING:
    from mlia.check import CheckRequest
else:
    # Typer resolves callback annotations while building completion commands.
    CheckRequest = Any

logger = logging.getLogger(__name__)

completion_init()

CONFIG = create_section_header("ML Inference Advisor configuration")
LIST_TABLE_WIDTH = 120
PLUGIN_URL = "https://github.com/orgs/arm/repositories?q=mlia"
NO_PLUGINS_URL = "https://github.com/search?q=org%3Aarm+mlia-*&type=repositories"
NO_PLUGINS_MESSAGE = (
    "MLIA requires plugins to analyse models. Plugins provide access to targets, "
    "backends and model converters.\nFind available plugins and installation "
    f"instructions: {NO_PLUGINS_URL}"
)
MLIA_HELP_EPILOG = (
    "Tip: Run 'mlia <command> --help' for options and examples, "
    "such as 'mlia check --help'.\n\n"
    f"Browse MLIA repositories: {PLUGIN_URL}"
)
PLUGIN_DISCOVERY_FAILURE = (
    "MLIA couldn't detect installed plugins. Run 'mlia target list' and "
    "'mlia backend list' to check plugin discovery. Find plugins and "
    f"installation instructions at: {PLUGIN_URL}"
)
PLUGIN_HELP_EXAMPLES = {
    "ethos_u": (
        "ethos-u85-2048",
        "vela",
        "--compatibility",
        "perform a compatibility check",
    ),
    "neural_technology": (
        "neural-technology",
        "nx-performance-estimator",
        "--performance",
        "estimate performance",
    ),
}
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


def _no_plugins_installed() -> bool:
    """Return whether no supported MLIA plugin entry points are installed."""
    return not any(
        list_entry_points(group)
        for group in (
            TARGET_PLUGIN_GROUP,
            BACKEND_PLUGIN_GROUP,
            TRANSFORMER_PLUGIN_GROUP,
            ANALYSIS_PLUGIN_GROUP,
        )
    )


def _print_no_plugins_message(
    settings: ApplicationSettings, *, stderr: bool = False
) -> None:
    """Show installation guidance with an orange first line."""
    console = (
        Console(stderr=True, no_color=not settings.color, width=settings.console.width)
        if stderr
        else settings.console
    )
    console.print(
        "No MLIA plugins are installed.",
        style=settings.console.get_style("warning"),
    )
    console.print(NO_PLUGINS_MESSAGE)


def execute_check(*args: Any, **kwargs: Any) -> Any:
    """Load check execution only when an analysis actually runs."""
    return importlib.import_module("mlia.check").execute_check(*args, **kwargs)


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


class RootHelpGroup(HelpOnEmptyGroup):
    """Root command group with an installed-plugin example in its help."""

    def get_help_option(self, ctx: click.Context) -> click.Option | None:
        """Explain how to find help for an individual command."""
        option = super().get_help_option(ctx)
        if option is not None:
            option.help = (
                "Show help for this command. Tip: Run mlia <command> --help "
                "for command-specific help, for example: 'mlia check --help'."
            )
            if self.rich_markup_mode == "rich":
                from typer import rich_utils

                option.help = _markup_quoted_help_commands(
                    option.help, rich_utils.STYLE_OPTION
                )
        return option

    def get_help(self, ctx: click.Context) -> str:
        """Append a plugin-specific example to the root command help."""
        from typer import rich_utils

        original_help = {name: command.help for name, command in self.commands.items()}
        original_epilog = self.epilog
        if self.rich_markup_mode == "rich":
            for command in self.commands.values():
                if command.help:
                    command.help = _markup_quoted_help_commands(
                        command.help, rich_utils.STYLE_OPTION
                    )
            if self.epilog:
                self.epilog = _markup_quoted_help_commands(
                    self.epilog, rich_utils.STYLE_OPTION
                )
                self.epilog = self.epilog.replace(
                    "Tip:", f"[{rich_utils.STYLE_USAGE}]Tip:[/]", 1
                )
        try:
            help_text = super().get_help(ctx)
        finally:
            for name, command in self.commands.items():
                command.help = original_help[name]
            self.epilog = original_epilog
        example = _root_help_example()
        if self.rich_markup_mode == "rich":
            highlight = _help_rgb(rich_utils.STYLE_USAGE)
            option_color = _help_rgb(rich_utils.STYLE_OPTION)
            example = example.replace(
                "Example Usage:",
                click.style("Example Usage:", fg=highlight, bold=True),
                1,
            )
            example = _style_quoted_help_commands(example, option_color)
        return f"{help_text.rstrip()}\n\n{example}" if help_text else example


def _help_rgb(style: str) -> str | tuple[int, int, int]:
    """Convert a Rich color to a format accepted by Click."""
    color = style.rsplit(" ", 1)[-1]
    if not color.startswith("#"):
        return color
    hex_color = color.lstrip("#")
    return (
        int(hex_color[0:2], 16),
        int(hex_color[2:4], 16),
        int(hex_color[4:6], 16),
    )


def _style_quoted_help_commands(
    help_text: str, color: str | tuple[int, int, int]
) -> str:
    """Color command examples and option names within quoted help text."""
    return re.sub(
        r"(?<=')(?:mlia\s+[^']+|check\s+[^']+|--[a-z-]+)(?=')",
        lambda match: click.style(
            re.sub(r"\x1b\[[0-9;]*m", "", match.group()), fg=color, bold=True
        ),
        help_text,
    )


def _markup_quoted_help_commands(help_text: str, style: str) -> str:
    """Add Rich styling to quoted commands before Typer renders help panels."""
    return re.sub(
        r"'((?:mlia\s+[^']+|check\s+[^']+|--[a-z-]+))'",
        lambda match: f"'[{style}]{match.group(1)}[/]'",
        help_text,
    )


def _root_help_example() -> str:
    """Build an example from installed target entry-point metadata only."""
    try:
        targets = list_entry_points(TARGET_PLUGIN_GROUP)
        if not targets:
            return (
                "No target plugins are installed. Install a target plugin to "
                f"enable model analysis. Find plugins at: {PLUGIN_URL}"
            )

        for target in sorted(
            targets, key=lambda plugin: plugin.name != "neural_technology"
        ):
            if example := PLUGIN_HELP_EXAMPLES.get(target.name):
                profile, backend, analysis_flag, action = example
                return (
                    "Example Usage: Run 'mlia check model.ext "
                    f"--target-profile {profile} --backend {backend} "
                    f"{analysis_flag}' to {action}."
                )
    except Exception:  # pylint: disable=broad-except
        logger.debug("Could not discover plugins for root help", exc_info=True)
        return PLUGIN_DISCOVERY_FAILURE
    return (
        "Explore your installed target plugins: run 'mlia target list' to find "
        "a target profile, and 'mlia backend list' to choose a backend for your model."
    )


class CheckOptionCommand(typer.core.TyperCommand):
    """Typer command that adds dynamic options to the check command."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the command with backend and analysis-plugin options."""
        self._dynamic_options_loaded = False
        super().__init__(*args, **kwargs)

    def get_params(self, ctx: click.Context) -> list[click.Parameter]:
        """Discover dynamic options only when parsing or displaying this command."""
        if not self._dynamic_options_loaded:
            self._backend_option_specs = discover_backend_option_specs()
            self._analysis_plugin_registry = _load_analysis_plugin_registry()
            self._analysis_plugin_options = self._analysis_plugin_registry.cli_options()
            self.params.extend(
                spec["click_option"] for spec in self._backend_option_specs
            )
            self.params.extend(self._analysis_plugin_options)
            self._dynamic_options_loaded = True
        return super().get_params(ctx)

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
        if _no_plugins_installed():
            _print_no_plugins_message(
                ctx.ensure_object(ApplicationSettings), stderr=True
            )
            ctx.exit(1)
        check_error = importlib.import_module("mlia.check").CheckError

        request = self.prepare_request(ctx)
        settings = ctx.ensure_object(ApplicationSettings)
        analysis_plugin_registry = self._analysis_plugin_registry
        try:
            if ctx.params["tui"]:
                try:
                    launch = importlib.import_module("mlia_tui").launch
                except ModuleNotFoundError as err:
                    if err.name != "mlia_tui":
                        raise
                    raise check_error(
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
                        importlib.import_module(
                            "mlia.core.output_rendering"
                        ).standardized_output_to_json(result.output),
                        highlight=False,
                    )
            else:
                execute_check(
                    request,
                    console=settings.console,
                    analysis_plugin_registry=analysis_plugin_registry,
                )
        except ConfigurationError as err:
            exit_code = err.exit_code if isinstance(err, check_error) else 1
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
    if _no_plugins_installed():
        _print_no_plugins_message(settings)
        return
    target_config = importlib.import_module("mlia.target.config")
    target_registry = importlib.import_module("mlia.target.registry")

    logger.info(CONFIG)

    _log_plugin_table(settings, "Target Plugins", TARGET_PLUGIN_GROUP)

    grouped_profiles = target_registry.profiles_by_target()

    logger.info("Available Target Profiles\n")

    for target_type, profile_names in grouped_profiles.items():
        table = _create_list_table(settings)

        table.add_column("Profile", style="tbl.name", no_wrap=True)
        table.add_column("Description", max_width=(LIST_TABLE_WIDTH // 2))

        for profile_name in profile_names:
            try:
                profile_path = target_config.get_builtin_target_profile_path(
                    profile_name
                )
                profile_data = target_config.load_profile(profile_path)

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
    if _no_plugins_installed():
        _print_no_plugins_message(settings)
        return
    api = importlib.import_module("mlia.api")

    logger.info(CONFIG)
    _log_plugin_table(settings, "Backend Plugins", BACKEND_PLUGIN_GROUP)

    rows = [
        (
            backend["name"],
            "yes" if backend["installed"] else "no",
            "yes" if backend["could_be_installed"] else "no",
        )
        for backend in api.list_backends()
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
    cls=RootHelpGroup,
    add_completion=False,
    no_args_is_help=True,
    epilog=MLIA_HELP_EPILOG,
    context_settings={
        "help_option_names": ["-h", "--help"],
    },
)

backend_app = typer.Typer(
    cls=HelpOnEmptyGroup,
    add_completion=False,
    no_args_is_help=True,
    context_settings={
        "help_option_names": ["-h", "--help"],
    },
)

target_app = typer.Typer(
    cls=HelpOnEmptyGroup,
    add_completion=False,
    no_args_is_help=True,
    context_settings={
        "help_option_names": ["-h", "--help"],
    },
)

mlia_app.add_typer(
    backend_app,
    name="backend",
    help=(
        "List analysis backends and their installation status, or install and "
        "uninstall them. Use 'mlia backend list' to see what's available."
    ),
)

mlia_app.add_typer(
    target_app,
    name="target",
    help=(
        "List installed target plugins and their available profiles. Use "
        "'mlia target list' to find a profile for 'check --target-profile'."
    ),
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


def enable_completion_callback(
    ctx: click.Context, param: click.Parameter, value: Any
) -> Any:
    """Delegate completion installation to the logged filesystem gateway."""
    if value and not ctx.resilient_parsing:
        _setup_command_logging(debug=False)
    return install_shell_completion_with_notice(logger, ctx, param, value)


def enable_completion_option() -> Any:
    """Return the option for setting up shell completion."""
    return typer.Option(
        None,
        "--enable-completion",
        callback=enable_completion_callback,
        expose_value=False,
        help="Set up Tab completion for your shell",
    )


def print_completion_script_option() -> Any:
    """Return the option for printing the shell completion script."""
    return typer.Option(
        None,
        "--print-completion-script",
        callback=show_callback,
        expose_value=False,
        help="Print the shell completion script for manual setup",
    )


@mlia_app.callback()
def mlia_app_main(
    ctx: typer.Context,
    enable_completion: bool = enable_completion_option(),
    print_completion_script: bool = print_completion_script_option(),
    version: bool = typer.Option(
        False,
        "--version",
        "-v",
        callback=version_callback,
        is_eager=True,
        help="Show the MLIA version.",
    ),
) -> None:
    """Configure top-level MLIA CLI callback options."""
    src = ctx.obj if isinstance(ctx.obj, ApplicationSettings) else None
    ctx.obj = new_settings(source=src)


@backend_app.callback()
def backend_app_main(
    ctx: typer.Context,
    enable_completion: bool = enable_completion_option(),
    print_completion_script: bool = print_completion_script_option(),
) -> None:
    """Configure the backend CLI namespace."""
    src = ctx.obj if isinstance(ctx.obj, ApplicationSettings) else None
    ctx.obj = new_settings(source=src)


@target_app.callback()
def target_app_main(
    ctx: typer.Context,
    enable_completion: bool = enable_completion_option(),
    print_completion_script: bool = print_completion_script_option(),
) -> None:
    """Configure the target CLI namespace."""
    src = ctx.obj if isinstance(ctx.obj, ApplicationSettings) else None
    ctx.obj = new_settings(source=src)


def _require_cli_target(
    ctx: click.Context, param: click.Parameter, value: str | None
) -> str | None:
    """Reject a missing CLI target before unspecified plugin options are resolved."""
    if value is None and not ctx.resilient_parsing and not ctx.params.get("tui"):
        if _no_plugins_installed():
            _print_no_plugins_message(
                ctx.ensure_object(ApplicationSettings), stderr=True
            )
            ctx.exit(1)
        raise click.MissingParameter(ctx=ctx, param=param)
    return value


@mlia_app.command(
    "check",
    cls=CheckOptionCommand,
    help=(
        "Check model compatibility with a target, or estimate performance with "
        "'--performance'. Select a target with '--target-profile' and optionally "
        "choose a backend with '--backend'."
    ),
    epilog=(
        "Examples:\n\n"
        "  mlia target list\n\n"
        "  mlia check <model.ext> --target-profile <profile> --compatibility\n\n"
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
    execution_profile: Annotated[
        str | None,
        typer.Option(
            "--execution-profile", help="Named runtime execution profile or TOML file"
        ),
    ] = None,
    show_plan: Annotated[
        bool,
        typer.Option(
            "--show-plan",
            help="Preview runtime selections without installing or running them",
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

    return cast(
        CheckRequest,
        importlib.import_module("mlia.check").prepare_check(
            model=model,
            execution_profile=execution_profile,
            show_plan=show_plan,
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
        ),
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
    _setup_command_logging(debug)
    try:
        importlib.import_module("mlia.api").install_backends(
            names=names,
            path=path,
            accept_eula=accept_eula,
            noninteractive=noninteractive,
            force=force,
        )
    except ConfigurationError as err:
        typer.echo(f"Error: {err}", err=True)
        raise typer.Exit(code=1) from err


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
    _setup_command_logging(debug)
    try:
        importlib.import_module("mlia.api").uninstall_backends(names)
    except ConfigurationError as err:
        typer.echo(f"Error: {err}", err=True)
        raise typer.Exit(code=1) from err


@backend_app.command("list")
def backend_list(
    ctx: typer.Context,
    runtimes: Annotated[
        bool,
        typer.Option("--runtimes", help="List supplied isolated runtime definitions"),
    ] = False,
    debug: bool = debug_option(),
) -> None:
    """List available backend plugins and backend installation status."""
    _setup_command_logging(debug)
    if runtimes:
        try:
            typer.echo(
                json.dumps(
                    importlib.import_module("mlia.api").list_runtimes(), indent=2
                )
            )
        except ConfigurationError as err:
            typer.echo(f"Error: {err}", err=True)
            raise typer.Exit(code=1) from err
    else:
        format_backend_info(ctx.obj)


@target_app.command("list")
def target_list(
    ctx: typer.Context,
    debug: bool = debug_option(),
) -> None:
    """List available target plugins and built-in target profiles."""
    _setup_command_logging(debug)
    format_target_info(ctx.obj)
