# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for shared check preparation and execution."""

import logging
from collections.abc import Callable
from dataclasses import replace
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, call

import pytest

import mlia.api
from mlia import check as check_api
from mlia.check import CheckError, CheckRequest, execute_check, prepare_check
from mlia.core import check_validation
from mlia.core.context import ExecutionContext
from mlia.core.logging import setup_logging
from mlia.core.settings import ApplicationSettings, ThemeName
from mlia.plugins.analysis import AnalysisPluginRegistry


@pytest.mark.parametrize(
    "compatibility, performance, expected_category",
    [
        pytest.param(True, True, {"compatibility", "performance"}, id="both"),
        pytest.param(True, False, {"compatibility"}, id="compatibility-only"),
        pytest.param(False, True, {"performance"}, id="performance-only"),
        pytest.param(False, False, {"compatibility"}, id="default"),
    ],
)
def test_check_category_combinations(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    compatibility: bool,
    performance: bool,
    expected_category: set[str],
) -> None:
    """The executor forwards the shared default or explicit analysis categories."""
    get_advice = MagicMock(return_value=None)
    monkeypatch.setattr(mlia.api, "get_advice", get_advice)
    monkeypatch.setattr(
        check_validation, "validate_check_target_profile", MagicMock(return_value=True)
    )
    request = prepare_check(
        model="model.tflite",
        output_dir=tmp_path,
        target_profile="ethos-u55-256",
        compatibility=compatibility,
        performance=performance,
    )
    execute_check(request, analysis_plugin_registry=AnalysisPluginRegistry())

    assert get_advice.call_args.args[:3] == (
        "ethos-u55-256",
        "model.tflite",
        expected_category,
    )


@pytest.mark.parametrize(
    "request_factory", [prepare_check, CheckRequest], ids=["prepared", "direct"]
)
@pytest.mark.parametrize(
    "model, target, profiling_data, message",
    [
        ("model.tflite", None, None, "Missing target profile"),
        (None, "tosa", None, "Missing argument 'MODEL'"),
        (None, "tosa", (), "Missing argument 'MODEL'"),
        ("", "tosa", None, "Missing argument 'MODEL'"),
    ],
    ids=["missing-target", "missing-model", "empty-profiling", "empty-model"],
)
def test_incomplete_check_preserves_output(
    tmp_path: Path,
    request_factory: Callable[..., CheckRequest],
    model: str | None,
    target: str | None,
    profiling_data: tuple[Path, ...] | None,
    message: str,
) -> None:
    """Incomplete requests preserve existing output and release logging resources."""
    destination = tmp_path / "existing"
    output = destination / "mlia-output"
    output.mkdir(parents=True)
    sentinel = output / "prior-result.txt"
    sentinel.write_bytes(b"previous analysis\n")
    request = request_factory(
        model=model,
        target_profile=target,
        profiling_data=profiling_data,
        output_dir=destination,
    )
    setup_logging(tmp_path)
    file_handler = next(
        handler
        for handler in logging.getLogger("mlia").handlers
        if isinstance(handler, logging.FileHandler)
    )
    stream = file_handler.stream

    with pytest.raises(CheckError, match=message) as error:
        execute_check(request)

    assert error.value.exit_code == 2
    assert list(output.iterdir()) == [sentinel]
    assert sentinel.read_bytes() == b"previous analysis\n"
    assert stream.closed
    assert file_handler not in logging.getLogger("mlia").handlers

    fresh_destination = tmp_path / "not-created"
    with pytest.raises(CheckError, match=message):
        execute_check(replace(request, output_dir=fresh_destination))

    assert not fresh_destination.exists()


def test_check_accepts_profiling_without_model(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Profiling-only input reaches analysis without requiring a model."""
    get_advice = MagicMock(return_value=None)
    monkeypatch.setattr(mlia.api, "get_advice", get_advice)
    profiling_data = tmp_path / "profiling.json"
    request = prepare_check(
        target_profile="tosa", profiling_data=[profiling_data], output_dir=tmp_path
    )

    result = execute_check(request, analysis_plugin_registry=AnalysisPluginRegistry())

    assert result.outcome == "completed"
    assert get_advice.call_args.args[1] is None
    assert get_advice.call_args.kwargs["profiling_data"] == [profiling_data]


def test_check_preserves_option_identity_and_owns_settings_mappings(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Requests copy their mappings while preserving values owned by plugins."""
    token = object()
    plugin_args = {"token": token}
    settings = ApplicationSettings(
        backend_options={"example": {"token": token}},
        core_settings={"token": token},
        plugin_settings={"example": {"token": token}},
    )
    get_advice = MagicMock(return_value=None)
    plugin = MagicMock()
    plugin.name = "example"
    plugin.enabled.return_value = True
    registry = AnalysisPluginRegistry()
    registry.register(plugin)
    monkeypatch.setattr(mlia.api, "get_advice", get_advice)

    request = prepare_check(
        target_profile="tosa",
        model="model.tflite",
        output_dir=tmp_path,
        settings=settings,
        plugin_args=plugin_args,
    )
    plugin_args.clear()
    settings.backend_options["example"].clear()
    settings.core_settings.clear()
    settings.plugin_settings["example"].clear()
    execute_check(request, analysis_plugin_registry=registry)

    assert get_advice.call_args.kwargs["backend_options"] == {
        "example": {"token": token}
    }
    run_settings = get_advice.call_args.kwargs["settings"]
    assert run_settings.backend_options["example"]["token"] is token
    assert run_settings.core_settings["token"] is token
    assert run_settings.plugin_settings["example"]["token"] is token
    plugin_result = plugin.run.call_args.args[0]
    assert plugin_result.args["token"] is token
    assert plugin_result.settings["token"] is token
    plugin_result.args.clear()
    run_settings.backend_options["example"].clear()
    run_settings.plugin_settings["example"].clear()
    run_settings.core_settings.clear()
    assert request.settings.backend_options == {"example": {"token": token}}
    assert request.settings.core_settings == {"token": token}
    assert request.settings.plugin_settings == {"example": {"token": token}}
    assert request.plugin_args == {"token": token}


def test_unsupported_check_is_skipped(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Unsupported categories skip analysis without creating an output directory."""
    get_advice = MagicMock()
    monkeypatch.setattr(mlia.api, "get_advice", get_advice)
    destination = tmp_path / "not-created"
    request = prepare_check(
        target_profile="tosa",
        model="model.tflite",
        performance=True,
        output_dir=destination,
    )

    result = execute_check(request)

    assert result.outcome == "skipped"
    assert result.skip_reason == (
        "Checks skipped: target profile 'tosa' does not support the requested "
        "checks: performance."
    )
    assert result.output is None
    assert result.output_dir is None
    assert not destination.exists()
    get_advice.assert_not_called()


def test_check_without_canonical_output_is_completed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An advisor may complete without returning standardized output."""
    monkeypatch.setattr(
        mlia.api,
        "get_advice",
        MagicMock(
            side_effect=lambda *args, settings, **kwargs: settings.console.print(
                "Analysis complete", style="tbl.name"
            )
        ),
    )
    request = prepare_check(
        target_profile="tosa", model="model.tflite", output_dir=tmp_path
    )

    result = execute_check(request, analysis_plugin_registry=AnalysisPluginRegistry())

    assert result.outcome == "completed"
    assert result.skip_reason is None
    assert result.output is None
    assert result.output_dir == tmp_path / "mlia-output"
    assert result.report == "Analysis complete\n"


@pytest.mark.parametrize(
    "theme, color, expected_style",
    [
        pytest.param("light", True, "bold #0157ff", id="light"),
        pytest.param("dark", True, "bold #6b9bf8", id="dark"),
        pytest.param("light", False, "none", id="no-color"),
    ],
)
def test_check_output_stream_preserves_configured_styles(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    theme: ThemeName,
    color: bool,
    expected_style: str,
) -> None:
    """Caller streams receive named-style output using the prepared settings."""
    stream = StringIO()
    get_advice = MagicMock(
        side_effect=lambda *args, settings, **kwargs: settings.console.print(
            "Analysis complete", style="tbl.name"
        )
    )
    monkeypatch.setattr(mlia.api, "get_advice", get_advice)
    request = prepare_check(
        target_profile="tosa",
        model="model.tflite",
        output_dir=tmp_path,
        settings=ApplicationSettings(theme=theme, color=color),
    )

    result = execute_check(
        request, output_stream=stream, analysis_plugin_registry=AnalysisPluginRegistry()
    )

    console = get_advice.call_args.kwargs["settings"].console
    assert str(console.get_style("tbl.name")) == expected_style
    assert console.no_color is not color
    assert not stream.closed
    assert stream.getvalue() == "Analysis complete\n"
    assert result.report == ""


def test_check_uses_application_console(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Standardized output uses the caller's console without markup highlighting."""
    console = MagicMock()
    console.encoding = "utf-8"
    output: dict[str, object] = {"result": "[literal]"}
    render = MagicMock(return_value="[literal]")
    monkeypatch.setattr(check_api, "standardized_output_to_text", render)
    monkeypatch.setattr(mlia.api, "get_advice", MagicMock(return_value=output))
    request = prepare_check(
        target_profile="tosa", model="model.tflite", output_dir=tmp_path
    )

    execute_check(
        request, console=console, analysis_plugin_registry=AnalysisPluginRegistry()
    )

    render.assert_called_once_with(output)
    console.out.assert_called_once_with("[literal]", highlight=False)


def test_check_replaces_unencodable_characters(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Standardized output respects the caller's console encoding."""
    console = MagicMock()
    console.encoding = "cp1252"
    output: dict[str, object] = {"result": "table"}
    render = MagicMock(return_value="┌─┐ café")
    monkeypatch.setattr(check_api, "standardized_output_to_text", render)
    monkeypatch.setattr(mlia.api, "get_advice", MagicMock(return_value=output))
    request = prepare_check(
        target_profile="tosa", model="model.tflite", output_dir=tmp_path
    )

    execute_check(
        request, console=console, analysis_plugin_registry=AnalysisPluginRegistry()
    )

    console.out.assert_called_once_with("??? café", highlight=False)


@pytest.mark.parametrize(
    "noninteractive, accept_eula, expected_policy",
    [(False, False, None), (True, False, False), (True, True, True)],
    ids=["interactive", "noninteractive-decline", "noninteractive-accept"],
)
def test_check_forwards_installation_policy(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    noninteractive: bool,
    accept_eula: bool,
    expected_policy: bool | None,
) -> None:
    """The executor forwards the CLI's installation policy to the advice API."""
    get_advice = MagicMock(return_value=None)
    prompt = MagicMock(return_value="yes")
    monkeypatch.setattr(
        check_validation,
        "validate_check_target_profile",
        MagicMock(return_value=True),
    )
    monkeypatch.setattr(mlia.api, "get_advice", get_advice)

    execute_check(
        prepare_check(
            model="model.tflite",
            target_profile="ethos-u55-256",
            output_dir=tmp_path,
            backend=["corstone-300"],
            noninteractive=noninteractive,
            i_agree_to_the_contained_eula=accept_eula,
        ),
        analysis_plugin_registry=AnalysisPluginRegistry(),
        prompt=prompt,
    )

    assert get_advice.call_args.kwargs["backends"] == ["corstone-300"]
    assert get_advice.call_args.kwargs["accept_eula"] is expected_policy
    assert get_advice.call_args.kwargs["prompt"] is prompt
    assert isinstance(get_advice.call_args.kwargs["context"], ExecutionContext)


def test_check_passes_cli_context_settings(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    get_advice = MagicMock(return_value=None)
    events: list[str] = []
    setup_logging = MagicMock(
        side_effect=lambda *args, **kwargs: events.append("logging")
    )
    execution_context = check_api.ExecutionContext

    def create_context(**kwargs: object) -> ExecutionContext:
        events.append("context")
        return execution_context(**kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(check_validation, "validate_check_target_profile", MagicMock())
    monkeypatch.setattr(mlia.api, "get_advice", get_advice)
    monkeypatch.setattr(check_api, "setup_logging", setup_logging)
    monkeypatch.setattr(check_api, "ExecutionContext", create_context)

    execute_check(
        prepare_check(
            model="model.tflite",
            target_profile="ethos-u55-256",
            output_dir=tmp_path,
            json_output=True,
            debug=True,
        ),
        analysis_plugin_registry=AnalysisPluginRegistry(),
    )

    context = get_advice.call_args.kwargs["context"]
    assert isinstance(context, ExecutionContext)
    assert context.output_dir == tmp_path / "mlia-output"
    assert context.output_format == "json"
    assert context.verbose is True
    assert events == ["logging", "context", "logging"]
    assert setup_logging.call_args_list == [
        call(verbose=True, output_format="json"),
        call(context.logs_path, True, "json"),
    ]
