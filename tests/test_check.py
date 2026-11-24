# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for shared check preparation and execution."""

import logging
import sys
from collections.abc import Callable
from dataclasses import replace
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, call

import pytest
from rich.console import Console
from rich.text import Text

import mlia.api
from mlia import check as check_api
from mlia.check import CheckError, CheckRequest, execute_check, prepare_check
from mlia.core import check_validation
from mlia.core.context import ExecutionContext
from mlia.core.logging import setup_logging
from mlia.core.settings import ApplicationSettings, ThemeName
from mlia.plugins.analysis import AnalysisPluginRegistry
from mlia.utils.logging import log_boundary_action


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


@pytest.mark.parametrize(
    "request_factory", [prepare_check, CheckRequest], ids=["prepared", "direct"]
)
@pytest.mark.parametrize("with_companion", [False, True])
def test_check_forwards_companion_to_analysis_and_plugins(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    request_factory: Callable[..., CheckRequest],
    with_companion: bool,
) -> None:
    """Shared execution preserves companions and ordinary analysis call signatures."""
    companion = tmp_path / "model.pte" if with_companion else None
    get_advice = MagicMock(return_value=None)
    monkeypatch.setattr(mlia.api, "get_advice", get_advice)
    monkeypatch.setattr(
        check_validation, "validate_check_target_profile", MagicMock(return_value=True)
    )
    plugin = MagicMock()
    plugin.name = "example"
    plugin.enabled.return_value = True
    registry = AnalysisPluginRegistry()
    registry.register(plugin)
    request = request_factory(
        target_profile="neural-technology",
        model="model.etrecord",
        companion_model=companion,
        profiling_data=[tmp_path / "model.etdump"],
        performance=True,
        output_dir=tmp_path,
    )

    execute_check(request, analysis_plugin_registry=registry)

    if with_companion:
        assert get_advice.call_args.kwargs["companion_model"] == companion
    else:
        assert "companion_model" not in get_advice.call_args.kwargs
    plugin.run.assert_called_once()
    assert plugin.run.call_args.args[0].parameters["companion_model"] == companion


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


@pytest.mark.parametrize(
    ("encoding", "rendered", "expected"),
    [
        ("utf-8", "[literal]", "[literal]"),
        ("cp1252", "┌─┐ café", "??? café"),
    ],
)
def test_check_uses_application_console(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    encoding: str,
    rendered: str,
    expected: str,
) -> None:
    """Render literal text with the caller's console and encoding."""
    console = MagicMock()
    console.encoding = encoding
    console.width = 52
    output: dict[str, object] = {"result": "[literal]"}
    render = MagicMock(return_value=Text(rendered))
    monkeypatch.setattr(check_api, "check_report", render)
    monkeypatch.setattr(mlia.api, "get_advice", MagicMock(return_value=output))
    request = prepare_check(
        target_profile="tosa", model="model.tflite", output_dir=tmp_path
    )

    execute_check(
        request, console=console, analysis_plugin_registry=AnalysisPluginRegistry()
    )

    render.assert_called_once_with(output, tmp_path / "mlia-output", width=52)
    console.print.assert_called_once()
    assert console.print.call_args.args[0].plain == expected
    assert console.print.call_args.kwargs == {"highlight": False}


@pytest.mark.parametrize("verbose", [False, True])
def test_check_report_spacing_after_progress(tmp_path: Path, verbose: bool) -> None:
    """The report starts cleanly, with a gap when debug progress precedes it."""
    stream = StringIO()
    console = Console(file=stream, no_color=True, width=100)
    settings = ApplicationSettings(console=console)
    context = ExecutionContext(output_dir=tmp_path, verbose=verbose)
    if verbose:
        stream.write("Completed: Checking performance\n")

    check_api._emit_standardized_output(
        settings,
        context,
        {"results": [{"kind": "performance", "status": "ok"}]},
        preceded_by_progress=True,
    )

    expected_start = (
        "Completed: Checking performance\n\n── ML INFERENCE ADVISOR"
        if verbose
        else "\n── ML INFERENCE ADVISOR"
    )
    assert stream.getvalue().startswith(expected_start)


def test_check_keeps_routine_progress_in_log_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Default checks show the report while retaining detailed execution logs."""

    def analyze(*_args: object, **_kwargs: object) -> dict[str, object]:
        logger = logging.getLogger("mlia.test")
        logger.info("Checking performance...")
        log_boundary_action(logger, "Running NX Performance Estimator")
        return {"results": [{"kind": "performance", "status": "ok"}]}

    monkeypatch.setattr(mlia.api, "get_advice", analyze)
    request = prepare_check(
        target_profile="tosa", model="model.tflite", output_dir=tmp_path
    )

    result = execute_check(request, analysis_plugin_registry=AnalysisPluginRegistry())

    assert result.report.startswith("── ML INFERENCE ADVISOR")
    assert "Checking performance..." not in result.report
    assert "Running NX Performance Estimator" not in result.report
    log = (tmp_path / "mlia-output" / "logs" / "mlia.log").read_text()
    assert "Checking performance..." in log
    assert "Running NX Performance Estimator" in log


def test_cli_check_shows_named_progress_without_debug(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Release mode reports that work has started before analysis returns."""
    monkeypatch.setattr(
        mlia.api,
        "get_advice",
        MagicMock(return_value={"results": [{"kind": "performance", "status": "ok"}]}),
    )
    request = prepare_check(
        target_profile="tosa", model="model.tflite", output_dir=tmp_path
    )

    execute_check(
        request,
        console=Console(file=sys.stdout, no_color=True, width=100),
        show_progress=True,
        analysis_plugin_registry=AnalysisPluginRegistry(),
    )

    progress = capsys.readouterr().out
    assert "Preparing model check..." in progress
    assert "Model check complete." in progress
    assert progress.index("Preparing model check...") < progress.index(
        "Model check complete."
    )
    assert progress.index("Model check complete.") < progress.index(
        "ML INFERENCE ADVISOR"
    )
    assert "Model check complete.\n\n── ML INFERENCE ADVISOR" in progress


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

    registry = AnalysisPluginRegistry()
    plugin = MagicMock()
    plugin.name = "runtime-consumer"
    plugin.enabled.return_value = True
    registry.register(plugin)

    execute_check(
        prepare_check(
            model="model.tflite",
            target_profile="ethos-u55-256",
            output_dir=tmp_path,
            backend=["corstone-300"],
            noninteractive=noninteractive,
            i_agree_to_the_contained_eula=accept_eula,
        ),
        analysis_plugin_registry=registry,
        prompt=prompt,
    )

    assert get_advice.call_args.kwargs["backends"] == ["corstone-300"]
    assert get_advice.call_args.kwargs["accept_eula"] is expected_policy
    assert get_advice.call_args.kwargs["prompt"] is prompt
    assert isinstance(get_advice.call_args.kwargs["context"], ExecutionContext)

    post_result = plugin.run.call_args.args[0]
    assert post_result.prompt is prompt
    assert post_result.parameters["noninteractive"] is noninteractive
    assert post_result.parameters["i_agree_to_the_contained_eula"] is accept_eula


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
    console = get_advice.call_args.kwargs["settings"].console
    assert setup_logging.call_args_list == [
        call(
            verbose=True,
            output_format="json",
            cli_mode=True,
            progress_only=False,
            console=console,
        ),
        call(
            context.logs_path,
            True,
            "json",
            cli_mode=True,
            progress_only=False,
            console=console,
        ),
    ]
