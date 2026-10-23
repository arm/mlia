# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Prepared checks shared by command-line and interactive frontends.

Preparation captures resolved inputs without creating output directories or
running an analysis. Execution is the sole owner of the check lifecycle.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from io import StringIO
from pathlib import Path
from typing import Any, Literal, TextIO

from rich.console import Console

from mlia.core.application_settings import (
    NO_COLOR,
    THEME_COLORS,
    _rich_theme,
    new_settings,
)
from mlia.core.check_actions import CLIActionResolver
from mlia.core.context import ExecutionContext
from mlia.core.errors import ConfigurationError
from mlia.core.logging import close_configured_handlers, setup_logging
from mlia.core.output_persistence import persist_standardized_output
from mlia.core.output_rendering import (
    standardized_output_to_json,
    standardized_output_to_text,
)
from mlia.core.settings import ApplicationSettings, FilteringSettings, ThemeName
from mlia.plugins.analysis import AnalysisPluginRegistry, AnalysisRunResult
from mlia.plugins.plugins import load_analysis_plugins

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CheckSettings:
    """Resolved configuration without console state.

    Option values belong to their backend or plugin and may contain live
    resources. Only the MLIA-owned settings mappings are copied.
    """

    color: bool = True
    theme: ThemeName = "dark"
    backend_options: dict[str, Any] = field(default_factory=dict)
    core_settings: dict[str, Any] = field(default_factory=dict)
    filtering: FilteringSettings = field(default_factory=FilteringSettings)
    plugin_settings: dict[str, dict[str, Any]] = field(default_factory=dict)


@dataclass(frozen=True)
class CheckRequest:
    """Inputs to a check, before backend selection and execution.

    Freezing the record does not freeze option values. Callers own their
    lifetime and must keep resources open until execution finishes. Frontends
    that transfer requests between processes must validate transport support.
    """

    target_profile: str | None = None
    model: str | None = None
    output_dir: Path | None = None
    backend: tuple[str, ...] | None = None
    performance: bool = False
    compatibility: bool = False
    profiling_data: tuple[Path, ...] | None = None
    json_output: bool = False
    i_agree_to_the_contained_eula: bool = False
    noninteractive: bool = False
    debug: bool = False
    settings: CheckSettings = field(default_factory=CheckSettings)
    plugin_args: dict[str, object] = field(default_factory=dict)
    companion_model: Path | None = None

    @property
    def categories(self) -> tuple[str, ...]:
        """Return selected categories, including the shared default."""
        return tuple(
            name
            for name, selected in (
                ("compatibility", self.compatibility),
                ("performance", self.performance),
            )
            if selected
        ) or ("compatibility",)


@dataclass(frozen=True)
class CheckResult:
    """Check outcome, output, and its persisted location."""

    output: dict[str, object] | None
    output_dir: Path | None
    report: str = ""
    outcome: Literal["completed", "skipped"] = "completed"
    skip_reason: str | None = None


class CheckError(ConfigurationError):
    """Check failure with the corresponding command exit status."""

    def __init__(self, message: str, exit_code: int = 1) -> None:
        """Initialize a frontend-independent failure."""
        super().__init__(message)
        self.exit_code = exit_code


def _copy_settings(settings: ApplicationSettings | CheckSettings) -> CheckSettings:
    """Copy MLIA-owned mappings while retaining plugin-owned option values."""
    return CheckSettings(
        color=settings.color,
        theme=settings.theme,
        backend_options={
            name: dict(options) for name, options in settings.backend_options.items()
        },
        core_settings=dict(settings.core_settings),
        filtering=settings.filtering,
        plugin_settings={
            name: dict(options) for name, options in settings.plugin_settings.items()
        },
    )


def prepare_check(
    *,
    target_profile: str | None = None,
    model: str | None = None,
    output_dir: Path | None = None,
    backend: Sequence[str] | None = None,
    performance: bool = False,
    compatibility: bool = False,
    profiling_data: Sequence[Path] | None = None,
    json_output: bool = False,
    i_agree_to_the_contained_eula: bool = False,
    noninteractive: bool = False,
    debug: bool = False,
    settings: ApplicationSettings | None = None,
    plugin_args: Mapping[str, object] | None = None,
    companion_model: Path | None = None,
) -> CheckRequest:
    """Capture inputs without starting a check or touching its output.

    Copy owned option mappings, preserving plugin values and resource identity.
    """
    resolved = settings if settings is not None else new_settings()
    return CheckRequest(
        target_profile=target_profile,
        model=model,
        output_dir=output_dir,
        backend=tuple(backend) if backend is not None else None,
        performance=performance,
        compatibility=compatibility,
        profiling_data=tuple(profiling_data) if profiling_data is not None else None,
        json_output=json_output,
        i_agree_to_the_contained_eula=i_agree_to_the_contained_eula,
        noninteractive=noninteractive,
        debug=debug,
        settings=_copy_settings(resolved),
        plugin_args=dict(plugin_args or {}),
        companion_model=companion_model,
    )


def _run_analysis_plugins(
    registry: AnalysisPluginRegistry,
    args: Mapping[str, object],
    context: ExecutionContext,
    parameters: dict[str, Any],
    output: dict[str, object] | None,
    settings: ApplicationSettings,
) -> None:
    """Run enabled post-analysis plugins after persisting and rendering output."""
    result = AnalysisRunResult(
        output=output,
        context=context,
        args=args,
        command_name="check",
        parameters=parameters,
    )
    for plugin in registry.enabled_plugins(args):
        logger.debug("Running analysis plugin '%s'", plugin.name)
        plugin.run(replace(result, settings=settings.for_plugin(plugin.name)))


def _emit_standardized_output(
    settings: ApplicationSettings,
    context: ExecutionContext,
    output: dict[str, object] | None,
) -> None:
    """Render canonical output to the selected frontend console."""
    if output is not None:
        text = (
            standardized_output_to_json(output)
            if context.output_format == "json"
            else standardized_output_to_text(output)
        )
        settings.console.out(
            text.encode(settings.console.encoding, errors="replace").decode(
                settings.console.encoding
            ),
            highlight=False,
        )


def execute_check(
    request: CheckRequest,
    *,
    console: Console | None = None,
    output_stream: TextIO | None = None,
    analysis_plugin_registry: AnalysisPluginRegistry | None = None,
    prompt: Callable[[str], str] | None = None,
) -> CheckResult:
    """Execute, persist, render, and postprocess a prepared check.

    An explicit console takes precedence over ``output_stream``. Otherwise,
    the configured theme and colors apply to the supplied stream, which remains
    open. Without either, rendered output is returned in ``report``.
    The caller owns containment of third-party stdout/stderr and cancellation.
    Every exit closes MLIA-configured logging handlers, including skipped or
    incomplete checks, as in ordinary CLI execution.
    """
    from mlia.api import get_advice
    from mlia.core.check_validation import validate_check_target_profile

    try:
        if not request.target_profile:
            raise CheckError(
                "Missing target profile. Select a target profile before running.",
                exit_code=2,
            )
        if not request.model and not request.profiling_data:
            raise CheckError("Missing argument 'MODEL'", exit_code=2)
        if not validate_check_target_profile(
            request.target_profile, set(request.categories)
        ):
            return CheckResult(
                None,
                None,
                outcome="skipped",
                skip_reason=(
                    f"Checks skipped: target profile '{request.target_profile}' "
                    "does not support the requested checks: "
                    f"{', '.join(request.categories)}."
                ),
            )
        report = StringIO()
        snapshot = _copy_settings(request.settings)
        selected_console = console or Console(
            file=output_stream if output_stream is not None else report,
            no_color=not snapshot.color,
            theme=_rich_theme(THEME_COLORS[snapshot.theme])
            if snapshot.color
            else NO_COLOR,
        )
        settings = ApplicationSettings(console=selected_console, **vars(snapshot))
        parameters = {
            name: getattr(request, name)
            for name in (
                "model",
                "target_profile",
                "output_dir",
                "backend",
                "performance",
                "compatibility",
                "profiling_data",
                "companion_model",
                "json_output",
                "i_agree_to_the_contained_eula",
                "noninteractive",
                "debug",
            )
        }
        parameters["backend"] = (
            list(request.backend) if request.backend is not None else None
        )
        parameters["profiling_data"] = (
            list(request.profiling_data) if request.profiling_data is not None else None
        )
        cli_args = {
            **parameters,
            "json": parameters["json_output"],
            "backend_options": settings.backend_options,
        }
        cli_args.pop("json_output")
        cli_args.pop("debug")
        output_format: Literal["json", "plain_text"] = (
            "json" if request.json_output else "plain_text"
        )
        setup_logging(verbose=request.debug, output_format=output_format)
        try:
            context = ExecutionContext(
                verbose=request.debug,
                action_resolver=CLIActionResolver(cli_args),
                output_format=output_format,
                output_dir=request.output_dir,
            )
        except Exception as err:
            raise CheckError(str(err)) from err
        setup_logging(context.logs_path, context.verbose, context.output_format)
        companion_args: dict[str, Any] = {}
        if request.companion_model is not None:
            companion_args["companion_model"] = request.companion_model
        output = get_advice(
            request.target_profile,
            request.model,
            set(request.categories),
            profiling_data=parameters["profiling_data"],
            **companion_args,
            context=context,
            backends=parameters["backend"],
            accept_eula=True
            if request.i_agree_to_the_contained_eula
            else False
            if request.noninteractive
            else None,
            backend_options=settings.backend_options,
            settings=settings,
            prompt=prompt,
        )
        if output is not None:
            persist_standardized_output(output, context.output_dir)
        _emit_standardized_output(settings, context, output)
        registry = analysis_plugin_registry
        if registry is None:
            registry = AnalysisPluginRegistry()
            load_analysis_plugins(registry)
        try:
            _run_analysis_plugins(
                registry,
                dict(request.plugin_args),
                context,
                parameters,
                output,
                settings,
            )
        except CheckError:
            raise
        except ConfigurationError as err:
            raise CheckError(str(err)) from err
        return CheckResult(output, context.output_dir, report.getvalue())
    finally:
        close_configured_handlers()
