# SPDX-FileCopyrightText: Copyright 2022-2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Module for the API functions."""

from __future__ import annotations

import argparse
import copy
import io
import logging
import os
import sys
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator, cast

from mlia.backend.config import BackendType
from mlia.backend.manager import get_installation_manager
from mlia.backend.registry import registry as backend_registry
from mlia.cli.command_validators import validate_backend
from mlia.cli.options import discover_backend_option_specs
from mlia.core.advisor import InferenceAdvisor
from mlia.core.common import AdviceCategory
from mlia.core.context import ExecutionContext
from mlia.core.errors import (
    ConfigurationError,
    FunctionalityNotSupportedError,
    InternalError,
    UnsupportedConfigurationError,
)
from mlia.core.handlers import WorkflowEventsHandler
from mlia.core.logging import setup_logging
from mlia.core.output_validation import collect_validation_errors
from mlia.target.config import get_builtin_target_profile_path, load_profile
from mlia.target.registry import (
    get_optimization_profile,
    get_target,
    profile,
    profiles_by_target,
)
from mlia.target.registry import registry as target_registry
from mlia.target.registry import supported_advice as target_supported_advice
from mlia.target.registry import supported_backends as target_supported_backends
from mlia.utils.filesystem import temp_directory
from mlia.utils.logging import process_raw_output

logger = logging.getLogger(__name__)


class ValidationMode(Enum):
    """Schema validation mode for run_advisor."""

    STRICT = "strict"
    WARN = "warn"
    OFF = "off"


if TYPE_CHECKING:
    from torch import nn


def _require_torch_module(model: object) -> nn.Module:
    try:
        import torch
    except ImportError as err:
        raise ConfigurationError(
            "The 'torch' package is required when passing nn.Module inputs."
        ) from err

    if not isinstance(model, torch.nn.Module):
        raise ConfigurationError("Model must be a path or torch.nn.Module.")
    return model


def _target_supports_torch_module(target: str) -> bool:
    target_info = target_registry.items.get(target)
    if target_info is None:
        return False
    return target_info.supports_torch_module


def _torch_module_backend_config(target_profile: str) -> tuple[str | None, str]:
    target = get_target(target_profile)
    target_info = target_registry.items.get(target)
    if target_info is None:
        return None, "enable_quantization"
    return (
        target_info.torch_module_backend,
        target_info.torch_module_quantization_option,
    )


def _export_module_to_pt2(
    module: nn.Module, example_inputs: tuple[Any, ...], output_dir: Path
) -> Path:
    try:
        import torch
    except ImportError as err:
        raise ConfigurationError(
            "The 'torch' package is required when exporting nn.Module inputs."
        ) from err

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "model.pt2"
    try:
        exported = torch.export.export(module, args=example_inputs)
        torch.export.save(exported, output_path)
    except Exception as err:
        raise ConfigurationError(
            f"Failed to export torch.nn.Module with torch.export: {err}"
        ) from err

    return output_path


def _validate_module_input(
    model: object,
    example_inputs: tuple[Any, ...] | None,
    enable_quantization: bool,
    target_profile: str,
) -> tuple[nn.Module | None, tuple[Any, ...] | None]:
    if isinstance(model, (str, Path)):
        if enable_quantization:
            raise ConfigurationError(
                "enable_quantization is only supported for torch.nn.Module inputs."
            )
        return None, None

    module = _require_torch_module(model)
    if example_inputs is None:
        raise ConfigurationError(
            "example_inputs is required for torch.nn.Module inputs."
        )
    if not isinstance(example_inputs, tuple):
        raise ConfigurationError(
            "example_inputs must be a tuple of positional arguments."
        )

    try:
        target = get_target(target_profile)
    except ValueError as err:
        raise ConfigurationError(str(err)) from err
    if not _target_supports_torch_module(target):
        raise UnsupportedConfigurationError(
            "Unsupported target for torch.nn.Module input",
            f"torch.nn.Module inputs are not supported for target '{target}'.",
        )

    return module, example_inputs


def _resolve_model_for_run(
    model: str | Path | nn.Module,
    module_input: nn.Module | None,
    module_example_inputs: tuple[Any, ...] | None,
    output_dir: Path | None,
) -> str:
    if module_input is None:
        if not isinstance(model, (str, Path)):
            raise InternalError("Expected model path when module input is absent.")
        return os.fspath(model)
    if output_dir is None:
        raise InternalError(
            "Output directory is required to export torch.nn.Module inputs."
        )
    return os.fspath(
        _export_module_to_pt2(module_input, module_example_inputs or (), output_dir)
    )


def _normalize_validation_mode(validation: ValidationMode | str) -> ValidationMode:
    if isinstance(validation, ValidationMode):
        return validation
    if isinstance(validation, str):
        try:
            return ValidationMode(validation)
        except ValueError as err:
            raise ConfigurationError(
                "validation must be one of 'strict', 'warn', or 'off'."
            ) from err
    raise ConfigurationError("validation must be a ValidationMode or string.")


def _resolve_output_base_dir(
    write_output_files: bool, output_dir: str | Path | None
) -> Path | None:
    """Resolve the base output directory when output files should be written."""
    if not write_output_files or output_dir is None:
        return None

    output_base = Path(output_dir).expanduser()
    if not output_base.is_absolute():
        output_base = (Path.cwd() / output_base).resolve()
    return output_base


def run_advisor(
    advice_category: str,
    target_profile: str,
    model: str | Path | nn.Module,
    *,
    backends: list[str] | None = None,
    example_inputs: tuple[Any, ...] | None = None,
    enable_quantization: bool = False,
    write_output_files: bool = False,
    output_dir: str | Path | None = None,
    logs_dir: str | Path | None = None,
    verbose: bool = False,
    context: ExecutionContext | None = None,
    backend_options: dict[str, dict[str, Any]] | None = None,
    validation: ValidationMode | str = ValidationMode.WARN,
) -> dict[str, object]:
    """Run MLIA analysis and return standardized JSON-compatible output.

    This API supports compatibility and performance categories only. It mirrors
    CLI backend selection behavior, returns standardized output in-memory, and
    raises ConfigurationError for invalid inputs. The returned dict follows
    MLIA's standardized output schema.

    Args:
        advice_category: Compatibility or performance advice category string.
        target_profile: Target profile name or path.
        model: Model path or torch.nn.Module input. For torch.nn.Module inputs,
            MLIA exports a temporary .pt2 artifact for analysis; the standardized
            output reports `model.format` as "pt2" and `model.hash` for that
            exported artifact.
        backends: Optional list of backend names. Defaults to CLI behavior.
        example_inputs: Required for torch.nn.Module inputs.
        enable_quantization: Only valid for torch.nn.Module inputs.
        write_output_files: Whether to keep output files on disk.
        output_dir: Output directory when write_output_files is true.
        logs_dir: Optional logs directory for file logging. When provided, MLIA
            configures file logging to logs_dir/mlia.log. If omitted, MLIA does
            not configure logging and defers to caller configuration. MLIA uses
            Python logging under the "mlia" namespace (e.g., "mlia", "mlia.api").
        verbose: Enable verbose file logging when logs_dir is provided.
        context: Optional ExecutionContext template to derive advanced
            execution components from. API-controlled fields such as output
            directory, output format, and event handlers are still owned by
            run_advisor().
        backend_options: Optional per-backend configuration overrides.
        validation: Schema validation mode. Use ValidationMode.STRICT (raise),
            ValidationMode.WARN (log and return output), or ValidationMode.OFF
            (skip validation). Default is ValidationMode.WARN.

    Returns:
        Standardized output as a JSON-compatible dict.

    Raises:
        ConfigurationError: Invalid inputs or unsupported advice category.
        UnsupportedConfigurationError: Target/backend combinations unsupported.
        FunctionalityNotSupportedError: Deprecated backend cannot satisfy request.
        InternalError: Unexpected failures or missing standardized output.

    Logging and output behavior:
        - API mode does not print MLIA reports to stdout/stderr.
        - MLIA uses Python logging under the ``mlia`` namespace.
        - API execution captures direct third-party stdio writes and reroutes
          them to the ``mlia.api`` logger where possible.
        - CLI behavior is unchanged by this API path.
    """
    validation_mode = _normalize_validation_mode(validation)
    advice_set = {advice_category}
    advice_category_enum = AdviceCategory.from_string(advice_set)
    if AdviceCategory.OPTIMIZATION in advice_category_enum:
        raise ConfigurationError(
            "Optimization advice is not supported by run_advisor()."
        )
    module_input, module_example_inputs = _validate_module_input(
        model,
        example_inputs,
        enable_quantization,
        target_profile,
    )

    output_base = _resolve_output_base_dir(write_output_files, output_dir)

    try:
        selected_backends = validate_backend(target_profile, backends)
    except argparse.ArgumentError as err:
        raise ConfigurationError(str(err)) from err
    except ValueError as err:
        raise ConfigurationError(str(err)) from err
    except Exception as err:
        raise UnsupportedConfigurationError(str(err)) from err

    backend_options_local = (
        copy.deepcopy(backend_options) if backend_options is not None else {}
    )
    _validate_backend_options(backend_options_local)
    if module_input is not None and not enable_quantization:
        backend_name, option_key = _torch_module_backend_config(target_profile)
        if backend_name:
            backend_options_local.setdefault(backend_name, {})[option_key] = False
    _validate_deprecated_backends(selected_backends, advice_category_enum)

    inputs = _RunAdvisorInputs(
        advice_set=advice_set,
        advice_category_enum=advice_category_enum,
        model=model,
        module_input=module_input,
        module_example_inputs=module_example_inputs,
        logs_dir=logs_dir,
        verbose=verbose,
        target_profile=target_profile,
        selected_backends=selected_backends,
        backend_options=backend_options_local,
        validation=validation_mode,
    )

    if write_output_files:
        local_context = _create_api_execution_context(
            output_base, context, advice_category_enum
        )
        return _run_advisor_with_context(local_context, inputs)

    try:
        with temp_directory(suffix="mlia-api") as temp_base:
            local_context = _create_api_execution_context(
                temp_base, context, advice_category_enum
            )
            return _run_advisor_with_context(local_context, inputs)
    except OSError as err:
        raise InternalError(
            f"Unable to create temporary output directory: {err}."
        ) from err


def _get_api_event_handler(
    target: str, output_dir: Path | None
) -> WorkflowEventsHandler:
    """Resolve API handler for target.

    Handlers returned from the target registry for API execution must be
    configured in collect-only mode so standardized JSON output is built in
    memory without invoking text report generation paths.
    """
    target_info = target_registry.items.get(target)
    if target_info is None or target_info.event_handler_factory is None:
        raise ConfigurationError(
            f"No API event handler is registered for target '{target}'."
        )
    handler = target_info.event_handler_factory(output_dir)
    if not handler.collect_only:
        raise ConfigurationError(
            f"API event handler for target '{target}' must be created with "
            "collect_only=True."
        )
    return handler


@dataclass(frozen=True)
class _RunAdvisorInputs:
    advice_set: set[str]
    advice_category_enum: set[AdviceCategory]
    model: str | Path | nn.Module
    module_input: nn.Module | None
    module_example_inputs: tuple[Any, ...] | None
    logs_dir: str | Path | None
    verbose: bool
    target_profile: str
    selected_backends: list[str]
    backend_options: dict[str, dict[str, Any]]
    validation: ValidationMode


def _create_api_execution_context(
    output_base_dir: Path | None,
    existing_context: ExecutionContext | None,
    advice_category: set[AdviceCategory],
) -> ExecutionContext:
    """Create a fresh API execution context for this run."""
    if existing_context is not None:
        return ExecutionContext(
            advice_category=advice_category,
            output_format="json",
            output_dir=output_base_dir,
            event_publisher=existing_context.event_publisher,
            action_resolver=existing_context.action_resolver,
        )
    return ExecutionContext(
        advice_category=advice_category,
        output_format="json",
        output_dir=output_base_dir,
    )


def _run_advisor_with_context(
    local_context: ExecutionContext,
    inputs: _RunAdvisorInputs,
) -> dict[str, object]:
    logs_path = _resolve_logs_dir(inputs.logs_dir)
    if logs_path is not None:
        setup_logging(logs_path, verbose=inputs.verbose, output_format="json")

    target = get_target(inputs.target_profile)
    local_context.event_handlers = [
        _get_api_event_handler(target, local_context.output_dir),
    ]
    if not local_context.event_handlers:
        raise InternalError("No event handlers configured for API execution.")

    model_for_run = _resolve_model_for_run(
        inputs.model,
        inputs.module_input,
        inputs.module_example_inputs,
        local_context.output_dir,
    )

    with _capture_external_output():
        get_advice(
            target_profile=inputs.target_profile,
            model=model_for_run,
            category=inputs.advice_set,
            context=local_context,
            backends=inputs.selected_backends,
            backend_options=inputs.backend_options,
        )
    handler = cast(WorkflowEventsHandler, local_context.event_handlers[0])
    if handler.output is None:
        _raise_if_deprecated_output_missing(
            inputs.selected_backends, inputs.target_profile
        )
        raise InternalError(
            "Standardized output is missing for target-profile "
            f"'{inputs.target_profile}' with backends {inputs.selected_backends}."
        )
    if inputs.module_input is not None:
        _override_model_name_for_module(handler.output, inputs.module_input)
    _strip_cli_arguments(handler.output)
    if inputs.validation != ValidationMode.OFF:
        errors = collect_validation_errors(handler.output)
        if errors:
            logger.debug(
                "Schema validation failed for target-profile '%s' with backends %s: %s",
                inputs.target_profile,
                inputs.selected_backends,
                "; ".join(errors),
                exc_info=True,
            )
            logger.warning(
                "Schema validation failed for target-profile '%s' with backends %s: "
                "%d issue(s).",
                inputs.target_profile,
                inputs.selected_backends,
                len(errors),
            )
            error_counts: dict[str, int] = {}
            for error in errors:
                error_counts[error] = error_counts.get(error, 0) + 1
            for message, count in error_counts.items():
                logger.warning("%dx %s", count, message)
            if inputs.validation == ValidationMode.STRICT:
                summary = (
                    "Schema validation failed for target-profile "
                    f"'{inputs.target_profile}' with backends "
                    f"{inputs.selected_backends}: {len(errors)} issue(s)."
                )
                raise InternalError(summary) from None
    return handler.output


@contextmanager
def _capture_external_output() -> Iterator[None]:
    """Capture non-logger stdout/stderr output during API execution.

    This is API-only containment to prevent direct console noise from
    third-party runtimes. Captured output is rerouted to the ``mlia.api``
    logger. This helper is only used from ``run_advisor`` and does not alter
    CLI output behavior.
    """

    def consume(line: str) -> None:
        logger.debug(line.rstrip())

    with ExitStack() as stack:
        for stream_name in ("__stdout__", "__stderr__"):
            output = getattr(sys, stream_name, None)
            if output is None:
                continue
            try:
                stack.enter_context(process_raw_output(consume, output))
            except (AttributeError, OSError, ValueError, io.UnsupportedOperation):
                continue
        yield


def _strip_cli_arguments(output: dict[str, object]) -> None:
    context = output.get("context")
    if isinstance(context, dict):
        context.pop("cli_arguments", None)


def _override_model_name_for_module(
    output: dict[str, object], module_input: nn.Module
) -> None:
    model = output.get("model")
    if not isinstance(model, dict):
        return
    module_name = module_input.__class__.__name__ or "torch.nn.Module"
    model["name"] = module_name


def _resolve_logs_dir(logs_dir: str | Path | None) -> Path | None:
    if logs_dir is None:
        return None

    logs_path = Path(logs_dir).expanduser()
    if not logs_path.is_absolute():
        logs_path = (Path.cwd() / logs_path).resolve()

    if logs_path.exists() and not logs_path.is_dir():
        raise ConfigurationError(f"Logs path '{logs_path}' is not a directory.")

    try:
        logs_path.mkdir(parents=True, exist_ok=True)
    except OSError as err:
        raise ConfigurationError(f"Unable to create logs directory: {err}.") from err

    if not os.access(logs_path, os.W_OK):
        raise ConfigurationError(f"Logs directory '{logs_path}' is not writable.")

    return logs_path


def _validate_deprecated_backends(
    backends: list[str], advice_categories: set[AdviceCategory]
) -> None:
    for backend in backends:
        backend_config = backend_registry.items.get(backend)
        if backend_config is None or not backend_config.is_deprecated:
            continue
        supports_any = any(
            backend_config.is_supported(advice, check_system=False)
            for advice in advice_categories
        )
        if supports_any:
            continue
        message = backend_config.deprecated_message or "Backend is deprecated."
        raise FunctionalityNotSupportedError(
            "Deprecated backend",
            f"{message} Backend '{backend}' cannot support the requested advice.",
        )


def _raise_if_deprecated_output_missing(
    backends: list[str], target_profile: str
) -> None:
    deprecated = [
        backend
        for backend in backends
        if backend_registry.items.get(backend)
        and backend_registry.items[backend].is_deprecated
    ]
    if not deprecated:
        return
    if len(deprecated) == 1:
        backend = deprecated[0]
        message = (
            backend_registry.items[backend].deprecated_message
            or "Backend is deprecated."
        )
        raise FunctionalityNotSupportedError(
            "Deprecated backend",
            f"{message} Backend '{backend}' does not emit standardized output for "
            f"target-profile '{target_profile}'.",
        )
    raise FunctionalityNotSupportedError(
        "Deprecated backend",
        "Deprecated backends do not emit standardized output for API execution: "
        f"{', '.join(deprecated)}.",
    )


def list_targets() -> list[dict[str, object]]:
    """List available targets with supported advice/backends and profiles.

    Returns:
        A list of target entries with names, profiles, supported backends, and
        supported advice categories.
    """
    targets: list[dict[str, object]] = []
    profiles = profiles_by_target()

    advice_order = (
        AdviceCategory.COMPATIBILITY,
        AdviceCategory.PERFORMANCE,
        AdviceCategory.OPTIMIZATION,
    )

    for target in target_registry.names():
        supported = set(target_supported_advice(target))
        supported_advice = [
            str(advice.name).lower() for advice in advice_order if advice in supported
        ]
        targets.append(
            {
                "target": target,
                "pretty_name": target_registry.pretty_name(target),
                "profiles": profiles.get(target, []),
                "supported_backends": sorted(target_supported_backends(target)),
                "supported_advice": supported_advice,
            }
        )

    return targets


def list_target_profiles() -> dict[str, list[dict[str, object]]]:
    """List available target profiles grouped by target.

    Returns:
        Mapping from target identifier to a list of profile entries with names
        and descriptions.
    """
    target_profiles: dict[str, list[dict[str, object]]] = {}
    profiles = profiles_by_target()

    for target, profile_names in profiles.items():
        profile_entries: list[dict[str, object]] = []
        for profile_name in profile_names:
            description = ""
            try:
                profile_path = get_builtin_target_profile_path(profile_name)
                profile_data = load_profile(profile_path)
                description = profile_data.get("description", "")
            except Exception:  # nosec B112
                description = ""

            profile_entries.append(
                {
                    "name": profile_name,
                    "description": description,
                }
            )
        target_profiles[target] = profile_entries

    return target_profiles


def list_backends() -> list[dict[str, object]]:
    """List available backends with install status and description.

    Returns:
        A list of backend entries with install status and descriptions.
    """
    manager = get_installation_manager(noninteractive=True)
    backends: list[dict[str, object]] = []

    for backend, config in backend_registry.items.items():
        if not config.selectable or not config.supported_advice:
            continue

        if config.type == BackendType.BUILTIN:
            installed = True
            could_be_installed = True
        elif config.installation:
            installed = config.installation.already_installed
            could_be_installed = config.installation.could_be_installed
        else:
            installed = manager.backend_installed(backend)
            could_be_installed = False

        backends.append(
            {
                "name": backend,
                "description": backend_registry.pretty_name(backend),
                "installed": installed,
                "could_be_installed": could_be_installed,
            }
        )

    return sorted(backends, key=lambda item: str(item["name"]))


def list_backend_options() -> list[dict[str, object]]:
    """List backend options derived from backend CLI metadata.

    Returns:
        A list of backend option entries, grouped by backend.
    """
    backend_options: dict[str, dict[str, dict[str, str]]] = {}

    def type_name(option_type: type | None) -> str:
        if option_type is Path:
            return "path"
        if option_type is bool:
            return "bool"
        if option_type is int:
            return "int"
        if option_type is float:
            return "float"
        if option_type is str or option_type is None:
            return "str"
        return getattr(option_type, "__name__", "str")

    for spec in discover_backend_option_specs():
        backend_name: str = spec["backend"]
        if backend_name not in backend_registry.items:
            continue
        config_key: str = spec["config_key"]
        backend_options.setdefault(backend_name, {})
        backend_options[backend_name].setdefault(
            config_key,
            {
                "config_key": config_key,
                "type": type_name(spec["type"]),
                "description": spec["help"],
            },
        )

    return [
        {"backend": backend, "options": list(options.values())}
        for backend, options in sorted(backend_options.items())
    ]


def _validate_backend_options(backend_options: dict[str, dict[str, Any]]) -> None:
    if not backend_options:
        return

    allowed: dict[str, set[str]] = {}
    for entry in list_backend_options():
        backend_name = cast(str, entry["backend"])
        option_entries = cast(list[dict[str, object]], entry["options"])
        option_keys = {cast(str, opt["config_key"]) for opt in option_entries}
        allowed[backend_name] = option_keys

    for backend_name, options_for_backend in backend_options.items():
        if backend_name not in allowed:
            raise ConfigurationError(
                f"Unknown backend in backend_options: '{backend_name}'."
            )
        for key in options_for_backend.keys():
            if key not in allowed[backend_name]:
                raise ConfigurationError(
                    f"Unknown backend option '{key}' for backend '{backend_name}'."
                )


def supported_backends(target_profile: str | Path) -> list[str]:
    """List backends supported by a target profile.

    Args:
        target_profile: Target profile name or path.

    Returns:
        List of backend identifier strings.

    Raises:
        ConfigurationError: If the target profile is invalid.
    """
    try:
        target = get_target(target_profile)
    except ValueError as err:
        raise ConfigurationError(str(err)) from err
    return target_supported_backends(target)


def get_advice(
    target_profile: str,
    model: str | Path | nn.Module,
    category: set[str],
    optimization_profile: str | None = None,
    optimization_targets: list[dict[str, Any]] | None = None,
    context: ExecutionContext | None = None,
    backends: list[str] | None = None,
    backend_options: dict[str, dict[str, Any]] | None = None,
) -> None:
    """Get the advice.

    This function represents an entry point to the library API.

    Based on provided parameters it will collect and analyze the data
    and produce the advice.

    :param target_profile: target profile identifier
    :param model: path to the NN model
    :param category: set of categories of the advice. MLIA supports three categories:
           "compatibility", "performance", "optimization". If not provided
           category "compatibility" is used by default.
    :param optimization_targets: optional model optimization targets that
           could be used for generating advice in "optimization" category.
    :param context: optional parameter which represents execution context,
           could be used for advanced use cases
    :param backends: A list of backends that should be used for the given
           target. Default settings will be used if None.
    :param backend_options: Optional dictionary of backend-specific options
           discovered from CLI arguments. Backend parameters are defined in each
           backend's CONFIG_TO_CLI_OPTION and automatically exposed as CLI options.

    Examples:
        NB: Before launching MLIA, the logging functionality should be configured!

        Getting the advice for the provided target profile and the model

        >>> get_advice("ethos-u55-256", "path/to/the/model",
                       {"optimization", "compatibility"})

        Getting the advice for the category "performance".

        >>> get_advice("ethos-u55-256", "path/to/the/model", {"performance"})

    """
    advice_category = AdviceCategory.from_string(category)

    if context is not None:
        context.advice_category = advice_category

    if context is None:
        context = ExecutionContext(advice_category=advice_category)

    advisor = get_advisor(
        context,
        target_profile,
        model,
        optimization_targets=optimization_targets,
        optimization_profile=optimization_profile,
        backends=backends,
        backend_options=backend_options,
    )
    advisor.run(context)


def get_advisor(
    context: ExecutionContext,
    target_profile: str | Path,
    model: str | Path | nn.Module,
    **extra_args: Any,
) -> InferenceAdvisor:
    """Find appropriate advisor for the target."""
    if extra_args.get("optimization_profile"):
        extra_args["optimization_profile"] = get_optimization_profile(
            extra_args["optimization_profile"]
        )
    target = profile(target_profile).target
    factory_function = target_registry.items[target].advisor_factory_func
    return factory_function(
        context,
        target_profile,
        model,
        **extra_args,
    )
