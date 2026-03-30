# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for core API functions in the split repo."""

from __future__ import annotations

import argparse
import io
import sys
import types
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, cast
from unittest.mock import MagicMock

import pytest

from mlia.api import (
    ValidationMode,
    _capture_external_output,
    _export_module_to_pt2,
    _get_api_event_handler,
    _normalize_validation_mode,
    _raise_if_deprecated_output_missing,
    _require_torch_module,
    _resolve_logs_dir,
    _resolve_model_for_run,
    _target_supports_torch_module,
    _torch_module_backend_config,
    _validate_backend_options,
    _validate_deprecated_backends,
    _validate_module_input,
    get_advice,
    get_advisor,
    list_backend_options,
    list_backends,
    list_target_profiles,
    list_targets,
    run_advisor,
    supported_backends,
)
from mlia.backend.config import BackendConfiguration, BackendType
from mlia.backend.registry import registry as backend_registry
from mlia.core.common import AdviceCategory
from mlia.core.context import ExecutionContext
from mlia.core.errors import (
    ConfigurationError,
    FunctionalityNotSupportedError,
    InternalError,
    UnsupportedConfigurationError,
)
from mlia.core.handlers import WorkflowEventsHandler
from mlia.target.config import TargetInfo
from mlia.target.registry import registry as target_registry


class _FakeHandler:  # pylint: disable=too-few-public-methods
    """Minimal workflow handler stub."""

    collect_only = True
    output: dict[str, object] = {"schema_version": "1.0.0", "results": []}


def _patch_common_run_advisor_dependencies(
    monkeypatch: pytest.MonkeyPatch,
    handler: object | None = None,
) -> None:
    monkeypatch.setattr("mlia.api.validate_backend", lambda _tp, _b: [])
    monkeypatch.setattr("mlia.api.get_target", lambda target_profile: target_profile)
    monkeypatch.setattr(
        "mlia.api._get_api_event_handler",
        lambda _target, _output_dir: handler or _FakeHandler(),
    )


def test_list_targets_shape() -> None:
    """Ensure list_targets returns the expected shape."""
    result = list_targets()
    assert isinstance(result, list)

    registry_targets = set(target_registry.names())
    for entry in result:
        assert set(entry.keys()) == {
            "target",
            "pretty_name",
            "profiles",
            "supported_backends",
            "supported_advice",
        }
        assert entry["target"] in registry_targets
        assert isinstance(entry["pretty_name"], str)
        assert isinstance(entry["profiles"], list)
        assert isinstance(entry["supported_backends"], list)
        assert isinstance(entry["supported_advice"], list)


def test_list_target_profiles_shape() -> None:
    """Ensure list_target_profiles returns the expected shape."""
    result = list_target_profiles()

    for target, entries in result.items():
        assert isinstance(target, str)
        assert isinstance(entries, list)
        for entry in entries:
            assert set(entry.keys()) == {"name", "description"}
            assert isinstance(entry["name"], str)
            assert isinstance(entry["description"], str)


def test_list_backends_shape() -> None:
    """Ensure list_backends returns the expected shape."""
    result = list_backends()
    assert isinstance(result, list)

    expected_names = {
        name
        for name, cfg in backend_registry.items.items()
        if cfg.selectable and cfg.supported_advice
    }
    assert {entry["name"] for entry in result} == expected_names
    for entry in result:
        assert set(entry.keys()) == {
            "name",
            "description",
            "installed",
            "could_be_installed",
        }
        assert isinstance(entry["name"], str)
        assert isinstance(entry["description"], str)
        assert isinstance(entry["installed"], bool)
        assert isinstance(entry["could_be_installed"], bool)


def test_list_backend_options_shape() -> None:
    """Ensure list_backend_options returns the expected shape."""
    result = list_backend_options()
    assert isinstance(result, list)
    for entry in result:
        assert set(entry.keys()) == {"backend", "options"}
        assert entry["backend"] in backend_registry.items
        assert isinstance(entry["options"], list)
        for option in entry["options"]:
            assert set(option.keys()) == {"config_key", "type", "description"}
            assert isinstance(option["config_key"], str)
            assert isinstance(option["type"], str)
            assert isinstance(option["description"], str)


def test_list_backend_options_uses_discovered_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure backend option metadata comes from CLI discovery."""
    monkeypatch.setattr(
        backend_registry,
        "items",
        {
            "vela": BackendConfiguration(
                supported_advice=[AdviceCategory.COMPATIBILITY],
                supported_systems=None,
                backend_type=BackendType.BUILTIN,
                installation=None,
            )
        },
    )
    monkeypatch.setattr(
        "mlia.api.discover_backend_option_specs",
        lambda: [
            {
                "module": "vela",
                "backend": "vela",
                "config_key": "config_file",
                "cli_option": "--config",
                "full_cli_option": "--vela.config",
                "dest": "vela_config_file",
                "type": int,
                "help": "Uses the CLI-discovered help text.",
            }
        ],
    )

    assert list_backend_options() == [
        {
            "backend": "vela",
            "options": [
                {
                    "config_key": "config_file",
                    "type": "int",
                    "description": "Uses the CLI-discovered help text.",
                }
            ],
        }
    ]


def test_list_backend_options_skips_unregistered_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure specs for unavailable backends are ignored."""
    monkeypatch.setattr(
        "mlia.api.discover_backend_option_specs",
        lambda: [
            {
                "module": "missing_backend",
                "backend": "missing-backend",
                "config_key": "path",
                "cli_option": "--path",
                "full_cli_option": "--missing-backend.path",
                "dest": "missing_backend_path",
                "type": Path,
                "help": "Missing backend option.",
            }
        ],
    )

    assert list_backend_options() == []


def test_validate_backend_options_unknown_backend() -> None:
    """Ensure unknown backend names are rejected."""
    with pytest.raises(ConfigurationError, match="Unknown backend in backend_options"):
        _validate_backend_options({"nonexistent-backend": {"some_key": "value"}})


def test_validate_backend_options_unknown_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure unknown backend option keys are rejected."""
    backend_name = "backend-a"
    monkeypatch.setattr(
        "mlia.api.list_backend_options",
        lambda: [{"backend": backend_name, "options": [{"config_key": "known"}]}],
    )
    with pytest.raises(
        ConfigurationError, match="Unknown backend option 'unknown_key'"
    ):
        _validate_backend_options({backend_name: {"unknown_key": "value"}})


def test_validate_deprecated_backends_ignores_unknown_backend() -> None:
    """Ensure unknown backends do not trigger deprecated checks."""
    _validate_deprecated_backends(["unknown-backend"], {AdviceCategory.COMPATIBILITY})


def test_run_advisor_validation_strict_raises(
    monkeypatch: pytest.MonkeyPatch, test_tflite_model: Path
) -> None:
    """Ensure strict validation rejects invalid standardized output."""

    class FakeHandler:  # pylint: disable=too-few-public-methods
        output = {"invalid": "schema"}

    _patch_common_run_advisor_dependencies(monkeypatch, handler=FakeHandler())
    monkeypatch.setattr("mlia.api.get_advice", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        "mlia.api.collect_validation_errors", lambda _data: ["invalid schema"]
    )

    with pytest.raises(InternalError, match="Schema validation failed"):
        run_advisor(
            "compatibility",
            "tosa",
            test_tflite_model,
            validation=ValidationMode.STRICT,
        )


def test_run_advisor_validation_warn_logs(
    monkeypatch: pytest.MonkeyPatch,
    test_tflite_model: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Ensure warn validation logs but returns output."""

    class FakeHandler:  # pylint: disable=too-few-public-methods
        output = {"invalid": "schema"}

    _patch_common_run_advisor_dependencies(monkeypatch, handler=FakeHandler())
    monkeypatch.setattr("mlia.api.get_advice", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        "mlia.api.collect_validation_errors", lambda _data: ["invalid schema"]
    )
    caplog.set_level("WARNING")

    output = run_advisor(
        "compatibility",
        "tosa",
        test_tflite_model,
        validation=ValidationMode.WARN,
    )
    assert output == FakeHandler.output
    assert any(
        "Schema validation failed" in record.message for record in caplog.records
    )


def test_run_advisor_validation_off_skips(
    monkeypatch: pytest.MonkeyPatch, test_tflite_model: Path
) -> None:
    """Ensure validation can be disabled."""

    class FakeHandler:  # pylint: disable=too-few-public-methods
        output = {"invalid": "schema"}

    _patch_common_run_advisor_dependencies(monkeypatch, handler=FakeHandler())
    monkeypatch.setattr("mlia.api.get_advice", lambda *args, **kwargs: None)
    validate_mock = MagicMock(return_value=[])
    monkeypatch.setattr("mlia.api.collect_validation_errors", validate_mock)

    output = run_advisor(
        "compatibility",
        "tosa",
        test_tflite_model,
        validation=ValidationMode.OFF,
    )
    assert output == FakeHandler.output
    validate_mock.assert_not_called()


def test_run_advisor_happy_path_uses_temp_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, test_tflite_model: Path
) -> None:
    """Ensure run_advisor uses temp output dir when write_output_files is false."""
    captured: dict[str, Path] = {}

    @contextmanager
    def fake_temp_dir(suffix: str) -> Iterator[Path]:
        path = tmp_path / f"temp-{suffix}"
        path.mkdir()
        captured["temp_dir"] = path
        yield path

    def fake_get_handler(_target: str, output_dir: Path | None) -> _FakeHandler:
        assert output_dir is not None
        captured["output_dir"] = output_dir
        return _FakeHandler()

    monkeypatch.setattr("mlia.api.temp_directory", fake_temp_dir)
    _patch_common_run_advisor_dependencies(monkeypatch)
    monkeypatch.setattr("mlia.api._get_api_event_handler", fake_get_handler)
    monkeypatch.setattr("mlia.api.get_advice", lambda *args, **kwargs: None)
    monkeypatch.setattr("mlia.api.collect_validation_errors", lambda _data: [])

    output = run_advisor("compatibility", "tosa", test_tflite_model)

    assert output == {"schema_version": "1.0.0", "results": []}
    assert captured["output_dir"] == captured["temp_dir"] / "mlia-output"


def test_run_advisor_happy_path_resolves_output_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, test_tflite_model: Path
) -> None:
    """Ensure run_advisor resolves relative output_dir when write_output_files."""
    captured: dict[str, Path] = {}

    def fake_get_handler(_target: str, output_dir: Path | None) -> _FakeHandler:
        assert output_dir is not None
        captured["output_dir"] = output_dir
        return _FakeHandler()

    monkeypatch.chdir(tmp_path)
    _patch_common_run_advisor_dependencies(monkeypatch)
    monkeypatch.setattr("mlia.api._get_api_event_handler", fake_get_handler)
    monkeypatch.setattr("mlia.api.get_advice", lambda *args, **kwargs: None)
    monkeypatch.setattr("mlia.api.collect_validation_errors", lambda _data: [])

    output = run_advisor(
        "compatibility",
        "tosa",
        test_tflite_model,
        write_output_files=True,
        output_dir="out",
    )

    assert output == {"schema_version": "1.0.0", "results": []}
    assert captured["output_dir"] == (tmp_path / "out").resolve() / "mlia-output"


def test_run_advisor_coerces_model_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, test_tflite_model: Path
) -> None:
    """Ensure run_advisor coerces Path model inputs to str."""
    captured: dict[str, object] = {}

    def fake_get_advice(*_args: object, **kwargs: object) -> None:
        captured["model"] = kwargs["model"]

    _patch_common_run_advisor_dependencies(monkeypatch)
    monkeypatch.setattr("mlia.api.get_advice", fake_get_advice)
    monkeypatch.setattr("mlia.api.collect_validation_errors", lambda _data: [])

    run_advisor(
        "compatibility",
        "tosa",
        Path(test_tflite_model),
        write_output_files=True,
        output_dir=tmp_path,
    )

    assert isinstance(captured["model"], str)


def test_run_advisor_uses_existing_context(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, test_tflite_model: Path
) -> None:
    """Ensure run_advisor derives a fresh API context from an existing one."""
    captured: dict[str, ExecutionContext] = {}
    context = ExecutionContext(output_dir=tmp_path / "existing")

    def capture_context(*_args: object, **kwargs: object) -> None:
        captured["context"] = cast(ExecutionContext, kwargs["context"])

    _patch_common_run_advisor_dependencies(monkeypatch)
    monkeypatch.setattr("mlia.api.get_advice", capture_context)
    monkeypatch.setattr("mlia.api.collect_validation_errors", lambda _data: [])

    run_advisor(
        "compatibility",
        "tosa",
        test_tflite_model,
        context=context,
        write_output_files=True,
        output_dir=tmp_path / "api-out",
    )

    assert captured["context"] is not context
    assert captured["context"].advice_category == {AdviceCategory.COMPATIBILITY}
    assert captured["context"].output_dir == (tmp_path / "api-out" / "mlia-output")
    assert captured["context"].event_publisher is context.event_publisher
    assert captured["context"].action_resolver is context.action_resolver
    assert context.output_dir == tmp_path / "existing" / "mlia-output"


def test_run_advisor_existing_context_does_not_override_temp_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, test_tflite_model: Path
) -> None:
    """Temp-dir semantics should still apply when a context is provided."""
    captured: dict[str, object] = {}
    context = ExecutionContext(output_dir=tmp_path / "existing")

    @contextmanager
    def fake_temp_dir(suffix: str) -> Iterator[Path]:
        path = tmp_path / f"temp-{suffix}"
        path.mkdir()
        captured["temp_dir"] = path
        yield path

    def capture_context(*_args: object, **kwargs: object) -> None:
        captured["context"] = kwargs["context"]

    monkeypatch.setattr("mlia.api.temp_directory", fake_temp_dir)
    _patch_common_run_advisor_dependencies(monkeypatch)
    monkeypatch.setattr("mlia.api.get_advice", capture_context)
    monkeypatch.setattr("mlia.api.collect_validation_errors", lambda _data: [])

    run_advisor("compatibility", "tosa", test_tflite_model, context=context)

    local_context = cast(ExecutionContext, captured["context"])
    assert local_context is not context
    assert local_context.output_dir == cast(Path, captured["temp_dir"]) / "mlia-output"
    assert context.output_dir == tmp_path / "existing" / "mlia-output"


def test_run_advisor_ignores_existing_context_event_handlers(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, test_tflite_model: Path
) -> None:
    """Existing context event handlers should not affect API execution."""

    class EmptyHandlerContext(ExecutionContext):
        @property
        def event_handlers(self) -> list | None:
            return []

        @event_handlers.setter
        def event_handlers(self, _handlers: list) -> None:
            return

    context = EmptyHandlerContext(output_dir=tmp_path)
    _patch_common_run_advisor_dependencies(monkeypatch)
    monkeypatch.setattr("mlia.api.get_advice", lambda *args, **kwargs: None)
    monkeypatch.setattr("mlia.api.collect_validation_errors", lambda _data: [])

    output = run_advisor("compatibility", "tosa", test_tflite_model, context=context)

    assert output == {"schema_version": "1.0.0", "results": []}


def test_run_advisor_temp_dir_failure(
    monkeypatch: pytest.MonkeyPatch, test_tflite_model: Path
) -> None:
    """Ensure temp directory failures raise InternalError."""

    class FailTempDir:  # pylint: disable=too-few-public-methods
        """Context manager that fails on entry."""

        def __enter__(self) -> Path:
            raise OSError("no temp dir")

        def __exit__(self, *_args: object) -> None:
            return None

    def fail_temp_dir(*_args: object, **_kwargs: object) -> FailTempDir:
        return FailTempDir()

    monkeypatch.setattr("mlia.api.temp_directory", fail_temp_dir)
    _patch_common_run_advisor_dependencies(monkeypatch)
    monkeypatch.setattr("mlia.api.get_advice", lambda *args, **kwargs: None)
    monkeypatch.setattr("mlia.api.collect_validation_errors", lambda _data: [])

    with pytest.raises(InternalError, match="temporary output directory"):
        run_advisor("compatibility", "tosa", test_tflite_model)


def test_run_advisor_rejects_missing_output(
    monkeypatch: pytest.MonkeyPatch, test_tflite_model: Path
) -> None:
    """Ensure run_advisor rejects missing standardized output."""

    class FakeHandler:  # pylint: disable=too-few-public-methods
        output = None

    monkeypatch.setattr("mlia.api.validate_backend", lambda _tp, _b: ["legacy-backend"])
    monkeypatch.setitem(
        backend_registry.items,
        "legacy-backend",
        BackendConfiguration(
            supported_advice=[AdviceCategory.COMPATIBILITY],
            supported_systems=None,
            backend_type=BackendType.BUILTIN,
            installation=None,
            is_deprecated=True,
            deprecated_message="Legacy backend.",
        ),
    )
    monkeypatch.setattr("mlia.api.get_target", lambda target_profile: target_profile)
    monkeypatch.setattr(
        "mlia.api._get_api_event_handler",
        lambda _target, _output_dir: FakeHandler(),
    )
    monkeypatch.setattr("mlia.api.get_advice", lambda *args, **kwargs: None)

    with pytest.raises(FunctionalityNotSupportedError, match="Deprecated backend"):
        run_advisor("compatibility", "tosa", test_tflite_model)


def test_run_advisor_missing_output_non_deprecated(
    monkeypatch: pytest.MonkeyPatch, test_tflite_model: Path
) -> None:
    """Ensure non-deprecated backends still raise InternalError for missing output."""

    class FakeHandler:  # pylint: disable=too-few-public-methods
        output = None

    _patch_common_run_advisor_dependencies(monkeypatch, handler=FakeHandler())
    monkeypatch.setattr("mlia.api.get_advice", lambda *args, **kwargs: None)

    with pytest.raises(InternalError, match="Standardized output is missing"):
        run_advisor("compatibility", "tosa", test_tflite_model)


def test_run_advisor_strips_cli_arguments(
    monkeypatch: pytest.MonkeyPatch, test_tflite_model: Path
) -> None:
    """Ensure run_advisor removes cli_arguments from context."""

    class FakeHandler:  # pylint: disable=too-few-public-methods
        output = {
            "schema_version": "1.0.0",
            "results": [],
            "context": {"cli_arguments": ["run.py", "model.tflite"]},
        }

    _patch_common_run_advisor_dependencies(monkeypatch, handler=FakeHandler())
    monkeypatch.setattr("mlia.api.get_advice", lambda *args, **kwargs: None)
    monkeypatch.setattr("mlia.api.collect_validation_errors", lambda _data: [])

    output = run_advisor("compatibility", "tosa", test_tflite_model)

    context = output.get("context")
    assert not (isinstance(context, dict) and "cli_arguments" in context)


def test_run_advisor_rejects_enable_quantization_for_non_module(
    test_tflite_model: Path,
) -> None:
    """Ensure enable_quantization is rejected for non-module inputs."""
    with pytest.raises(ConfigurationError, match="enable_quantization"):
        run_advisor(
            "compatibility",
            "tosa",
            test_tflite_model,
            enable_quantization=True,
        )


def test_run_advisor_rejects_optimization_category(test_tflite_model: Path) -> None:
    """Ensure run_advisor rejects optimization advice."""
    with pytest.raises(ConfigurationError, match="Optimization advice"):
        run_advisor("optimization", "tosa", test_tflite_model)


def test_run_advisor_maps_argparse_error(
    monkeypatch: pytest.MonkeyPatch, test_tflite_model: Path
) -> None:
    """Ensure argparse errors map to ConfigurationError."""
    parser = argparse.ArgumentParser()
    argument = parser.add_argument("--foo")

    def raise_error(*_args: object, **_kwargs: object) -> list[str]:
        raise argparse.ArgumentError(argument, "bad argument")

    monkeypatch.setattr("mlia.api.validate_backend", raise_error)

    with pytest.raises(ConfigurationError, match="bad argument"):
        run_advisor("compatibility", "tosa", test_tflite_model)


def test_run_advisor_maps_value_error(
    monkeypatch: pytest.MonkeyPatch, test_tflite_model: Path
) -> None:
    """Ensure ValueError maps to ConfigurationError."""

    def raise_error(*_args: object, **_kwargs: object) -> list[str]:
        raise ValueError("bad value")

    monkeypatch.setattr("mlia.api.validate_backend", raise_error)

    with pytest.raises(ConfigurationError, match="bad value"):
        run_advisor("compatibility", "tosa", test_tflite_model)


def test_run_advisor_maps_generic_error(
    monkeypatch: pytest.MonkeyPatch, test_tflite_model: Path
) -> None:
    """Ensure generic exceptions map to UnsupportedConfigurationError."""

    def raise_error(*_args: object, **_kwargs: object) -> list[str]:
        raise RuntimeError("boom")

    monkeypatch.setattr("mlia.api.validate_backend", raise_error)

    with pytest.raises(UnsupportedConfigurationError, match="boom"):
        run_advisor("compatibility", "tosa", test_tflite_model)


def test_get_api_event_handler_missing_factory(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure missing event handler factory raises ConfigurationError."""
    missing = TargetInfo(
        supported_backends=[],
        default_backends=[],
        advisor_factory_func=MagicMock(),
        target_profile_cls=cast(Any, MagicMock()),
        event_handler_factory=None,
    )
    monkeypatch.setitem(target_registry.items, "missing-target", missing)

    with pytest.raises(ConfigurationError, match="No API event handler is registered"):
        _get_api_event_handler("missing-target", None)


def test_list_target_profiles_fallback_on_load_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Ensure list_target_profiles handles load failures gracefully."""
    monkeypatch.setattr("mlia.api.profiles_by_target", lambda: {"foo": ["bar"]})
    monkeypatch.setattr(
        "mlia.api.get_builtin_target_profile_path",
        lambda _name: tmp_path / "missing.toml",
    )

    def raise_error(_path: Path) -> dict[str, str]:
        raise ValueError("load failure")

    monkeypatch.setattr("mlia.api.load_profile", raise_error)

    profiles = list_target_profiles()
    assert profiles["foo"][0]["name"] == "bar"
    assert profiles["foo"][0]["description"] == ""


def test_list_target_profiles_reads_descriptions(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Profile descriptions should come from the loaded profile data."""
    monkeypatch.setattr("mlia.api.profiles_by_target", lambda: {"foo": ["bar"]})
    monkeypatch.setattr(
        "mlia.api.get_builtin_target_profile_path",
        lambda _name: tmp_path / "bar.toml",
    )
    monkeypatch.setattr(
        "mlia.api.load_profile",
        lambda _path: {"description": "Profile description"},
    )

    assert list_target_profiles() == {
        "foo": [{"name": "bar", "description": "Profile description"}]
    }


def test_list_backends_uses_installation_manager(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure list_backends queries the installation manager when needed."""

    class FakeManager:  # pylint: disable=too-few-public-methods
        def backend_installed(self, _backend: str) -> bool:
            return False

    backend_name = "custom-backend"
    config = BackendConfiguration(
        supported_advice=[AdviceCategory.COMPATIBILITY],
        supported_systems=None,
        backend_type=BackendType.CUSTOM,
        installation=None,
        selectable=True,
    )

    monkeypatch.setitem(backend_registry.items, backend_name, config)
    monkeypatch.setitem(backend_registry.pretty_names, backend_name, "Custom Backend")
    monkeypatch.setattr(
        "mlia.api.get_installation_manager", lambda noninteractive=True: FakeManager()
    )

    entries = list_backends()
    entry = next(item for item in entries if item["name"] == backend_name)
    assert entry["installed"] is False
    assert entry["could_be_installed"] is False


def test_supported_backends_matches_registry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure supported_backends delegates to target registry helpers."""
    monkeypatch.setattr("mlia.api.get_target", lambda _profile: "target-a")
    monkeypatch.setattr(
        "mlia.api.target_supported_backends",
        lambda _target: ["backend-a", "backend-b"],
    )

    assert supported_backends("profile-a") == ["backend-a", "backend-b"]


def test_supported_backends_invalid_target_profile() -> None:
    """Ensure supported_backends rejects invalid target profiles."""
    with pytest.raises(ConfigurationError, match="Profile"):
        supported_backends("not-a-real-target-profile")


def test_require_torch_module_rejects_missing_torch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure torch module inputs fail clearly when torch is unavailable."""
    monkeypatch.setitem(sys.modules, "torch", None)

    with pytest.raises(ConfigurationError, match="torch' package is required"):
        _require_torch_module(object())


def test_require_torch_module_rejects_wrong_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure non-Module objects are rejected after torch import succeeds."""
    fake_torch = types.SimpleNamespace(
        nn=types.SimpleNamespace(Module=type("FakeModule", (), {}))
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    with pytest.raises(
        ConfigurationError, match="Model must be a path or torch.nn.Module"
    ):
        _require_torch_module(object())


def test_target_supports_torch_module_missing_target() -> None:
    """Missing targets should report no torch-module support."""
    assert _target_supports_torch_module("missing-target") is False


def test_torch_module_backend_config_missing_target_info(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unknown targets should fall back to default quantization option name."""
    monkeypatch.setattr("mlia.api.get_target", lambda _profile: "missing-target")
    assert _torch_module_backend_config("profile") == (None, "enable_quantization")


def test_validate_module_input_invalid_target_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Target-profile resolution errors should map to ConfigurationError."""
    fake_module_type = type("FakeModule", (), {})
    fake_torch = types.SimpleNamespace(
        nn=types.SimpleNamespace(Module=fake_module_type)
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(
        "mlia.api.get_target", MagicMock(side_effect=ValueError("bad profile"))
    )

    with pytest.raises(ConfigurationError, match="bad profile"):
        _validate_module_input(fake_module_type(), (), False, "bad-profile")


def test_validate_module_input_requires_tuple_example_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Module inputs require tuple example_inputs."""
    fake_module_type = type("FakeModule", (), {})
    fake_torch = types.SimpleNamespace(
        nn=types.SimpleNamespace(Module=fake_module_type)
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    with pytest.raises(ConfigurationError, match="example_inputs must be a tuple"):
        _validate_module_input(fake_module_type(), cast(Any, []), False, "profile")


def test_validate_module_input_requires_example_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Module inputs require example_inputs to be provided."""
    fake_module_type = type("FakeModule", (), {})
    fake_torch = types.SimpleNamespace(
        nn=types.SimpleNamespace(Module=fake_module_type)
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    with pytest.raises(ConfigurationError, match="example_inputs is required"):
        _validate_module_input(fake_module_type(), None, False, "profile")


def test_validate_module_input_rejects_unsupported_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Module inputs should reject targets without declared support."""
    fake_module_type = type("FakeModule", (), {})
    fake_torch = types.SimpleNamespace(
        nn=types.SimpleNamespace(Module=fake_module_type)
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr("mlia.api.get_target", lambda _profile: "unsupported-target")

    with pytest.raises(UnsupportedConfigurationError, match="Unsupported target"):
        _validate_module_input(fake_module_type(), (), False, "profile")


def test_resolve_model_for_run_rejects_non_path_without_module() -> None:
    """Internal path resolution requires a path-like model when no module exists."""
    with pytest.raises(InternalError, match="Expected model path"):
        _resolve_model_for_run(cast(Any, object()), None, None, None)


def test_resolve_model_for_run_requires_output_dir_for_module() -> None:
    """Module export requires an output directory."""
    with pytest.raises(InternalError, match="Output directory is required"):
        _resolve_model_for_run("ignored", cast(Any, object()), (), None)


def test_export_module_to_pt2_rejects_missing_torch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Export should fail clearly when torch is unavailable."""
    monkeypatch.setitem(sys.modules, "torch", None)

    with pytest.raises(
        ConfigurationError, match="torch' package is required when exporting"
    ):
        _export_module_to_pt2(cast(Any, object()), (), tmp_path)


def test_export_module_to_pt2_wraps_export_failures(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """torch.export failures should map to ConfigurationError."""

    class FakeExport:  # pylint: disable=too-few-public-methods
        @staticmethod
        def export(_module: object, args: tuple[object, ...]) -> object:
            raise RuntimeError("export failed")

    fake_torch = types.SimpleNamespace(export=FakeExport())
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    with pytest.raises(ConfigurationError, match="export failed"):
        _export_module_to_pt2(cast(Any, object()), (), tmp_path)


def test_export_module_to_pt2_success(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Successful exports should write to model.pt2 under the output dir."""
    saved: dict[str, object] = {}

    class FakeExport:  # pylint: disable=too-few-public-methods
        @staticmethod
        def export(_module: object, args: tuple[object, ...]) -> object:
            saved["args"] = args
            return "exported-program"

        @staticmethod
        def save(exported: object, output_path: Path) -> None:
            saved["exported"] = exported
            saved["output_path"] = output_path

    fake_torch = types.SimpleNamespace(export=FakeExport())
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    output_path = _export_module_to_pt2(cast(Any, object()), (1, 2), tmp_path)

    assert output_path == tmp_path / "model.pt2"
    assert saved == {
        "args": (1, 2),
        "exported": "exported-program",
        "output_path": tmp_path / "model.pt2",
    }


def test_resolve_model_for_run_exports_module(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Module inputs should resolve through the PT2 export helper."""
    monkeypatch.setattr(
        "mlia.api._export_module_to_pt2",
        lambda _module, _inputs, output_dir: output_dir / "model.pt2",
    )

    assert _resolve_model_for_run("ignored", cast(Any, object()), (), tmp_path) == str(
        tmp_path / "model.pt2"
    )


def test_normalize_validation_mode_rejects_invalid_values() -> None:
    """Invalid validation inputs should fail fast."""
    with pytest.raises(ConfigurationError, match="validation must be one of"):
        _normalize_validation_mode("bad-mode")
    with pytest.raises(
        ConfigurationError, match="validation must be a ValidationMode or string"
    ):
        _normalize_validation_mode(cast(Any, 123))


def test_run_advisor_sets_module_backend_option_when_quantization_disabled(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Module inputs auto-disable PTQ through backend options when needed."""

    class FakeModule:  # pylint: disable=too-few-public-methods
        pass

    fake_torch = types.SimpleNamespace(nn=types.SimpleNamespace(Module=FakeModule))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr("mlia.api.get_target", lambda _profile: "supported-target")
    monkeypatch.setitem(
        target_registry.items,
        "supported-target",
        TargetInfo(
            supported_backends=["backend-a"],
            default_backends=["backend-a"],
            advisor_factory_func=cast(Any, MagicMock()),
            target_profile_cls=cast(Any, MagicMock()),
            supports_torch_module=True,
            torch_module_backend="backend-a",
            torch_module_quantization_option="enable_quantization",
        ),
    )
    monkeypatch.setattr("mlia.api.validate_backend", lambda _tp, _b: ["backend-a"])
    monkeypatch.setattr(
        "mlia.api._get_api_event_handler", lambda _target, _output_dir: _FakeHandler()
    )
    monkeypatch.setattr(
        "mlia.api._resolve_model_for_run", lambda *_args, **_kwargs: "model.pt2"
    )
    monkeypatch.setattr("mlia.api.collect_validation_errors", lambda _data: [])

    captured: dict[str, object] = {}

    def fake_get_advice(*_args: object, **kwargs: object) -> None:
        captured["backend_options"] = kwargs["backend_options"]

    monkeypatch.setattr("mlia.api.get_advice", fake_get_advice)

    output = run_advisor(
        "compatibility",
        "profile",
        cast(Any, FakeModule()),
        example_inputs=(),
        write_output_files=True,
        output_dir=tmp_path,
    )

    assert output == _FakeHandler.output
    assert captured["backend_options"] == {"backend-a": {"enable_quantization": False}}


def test_get_advice_updates_context_and_runs_advisor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """get_advice should resolve the category and run the advisor."""
    advisor = MagicMock()
    context = ExecutionContext()
    monkeypatch.setattr("mlia.api.get_advisor", lambda *args, **kwargs: advisor)

    get_advice(
        "profile",
        "model.tflite",
        {"compatibility"},
        context=context,
        backends=["backend-a"],
        backend_options={"backend-a": {"flag": True}},
    )

    assert context.advice_category == {AdviceCategory.COMPATIBILITY}
    advisor.run.assert_called_once_with(context)


def test_get_advice_creates_context_when_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """get_advice should create an execution context when one is not provided."""
    advisor = MagicMock()
    captured: dict[str, object] = {}

    def fake_get_advisor(
        context: ExecutionContext, *_args: object, **_kwargs: object
    ) -> object:
        captured["context"] = context
        return advisor

    monkeypatch.setattr("mlia.api.get_advisor", fake_get_advisor)

    get_advice("profile", "model.tflite", {"compatibility"})

    context = cast(ExecutionContext, captured["context"])
    assert context.advice_category == {AdviceCategory.COMPATIBILITY}
    advisor.run.assert_called_once_with(context)


def test_get_advisor_resolves_optimization_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """get_advisor should normalize optimization profiles before factory dispatch."""
    context = ExecutionContext()
    factory = MagicMock(return_value="advisor")
    monkeypatch.setattr(
        "mlia.api.get_optimization_profile",
        lambda value: {"name": value, "resolved": True},
    )
    monkeypatch.setattr(
        "mlia.api.profile",
        lambda _target_profile: types.SimpleNamespace(target="target-a"),
    )
    monkeypatch.setitem(
        target_registry.items,
        "target-a",
        TargetInfo(
            supported_backends=[],
            default_backends=[],
            advisor_factory_func=factory,
            target_profile_cls=cast(Any, MagicMock()),
        ),
    )

    assert (
        get_advisor(
            context,
            "profile-a",
            "model.tflite",
            optimization_profile="opt-a",
        )
        == "advisor"
    )
    factory.assert_called_once_with(
        context,
        "profile-a",
        "model.tflite",
        optimization_profile={"name": "opt-a", "resolved": True},
    )


def test_get_api_event_handler_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """Registered API event handler factories should be returned."""
    handler = _FakeHandler()
    handler.collect_only = True
    monkeypatch.setitem(
        target_registry.items,
        "ok-target",
        TargetInfo(
            supported_backends=[],
            default_backends=[],
            advisor_factory_func=MagicMock(),
            target_profile_cls=cast(Any, MagicMock()),
            event_handler_factory=lambda _output_dir: cast(
                WorkflowEventsHandler, handler
            ),
        ),
    )

    assert _get_api_event_handler("ok-target", None) is handler


def test_get_api_event_handler_requires_collect_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """API handlers must be configured for collect-only execution."""
    handler = _FakeHandler()
    handler.collect_only = False
    monkeypatch.setitem(
        target_registry.items,
        "bad-target",
        TargetInfo(
            supported_backends=[],
            default_backends=[],
            advisor_factory_func=MagicMock(),
            target_profile_cls=cast(Any, MagicMock()),
            event_handler_factory=lambda _output_dir: cast(
                WorkflowEventsHandler, handler
            ),
        ),
    )

    with pytest.raises(
        ConfigurationError,
        match="must be created with collect_only=True",
    ):
        _get_api_event_handler("bad-target", None)


def test_run_advisor_calls_setup_logging_when_logs_dir_provided(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    test_tflite_model: Path,
) -> None:
    """Providing logs_dir should configure API file logging."""
    setup_logging = MagicMock()
    _patch_common_run_advisor_dependencies(monkeypatch)
    monkeypatch.setattr("mlia.api.get_advice", lambda *args, **kwargs: None)
    monkeypatch.setattr("mlia.api.collect_validation_errors", lambda _data: [])
    monkeypatch.setattr("mlia.api.setup_logging", setup_logging)

    run_advisor(
        "compatibility",
        "tosa",
        test_tflite_model,
        logs_dir=tmp_path / "logs",
    )

    setup_logging.assert_called_once()


def test_capture_external_output_ignores_unsupported_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Capture helper should ignore streams that cannot be redirected."""
    calls: list[object] = []

    @contextmanager
    def fake_process_raw_output(_consumer: object, output: object) -> Iterator[None]:
        calls.append(output)
        if len(calls) == 1:
            raise io.UnsupportedOperation("unsupported")
        yield

    monkeypatch.setattr("mlia.api.process_raw_output", fake_process_raw_output)

    with _capture_external_output():
        pass

    assert len(calls) == 2


def test_capture_external_output_ignores_none_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Capture helper should skip missing stdio streams safely."""
    calls: list[object] = []

    @contextmanager
    def fake_process_raw_output(_consumer: object, output: object) -> Iterator[None]:
        calls.append(output)
        yield

    monkeypatch.setattr("mlia.api.process_raw_output", fake_process_raw_output)
    monkeypatch.setattr(sys, "__stdout__", None)
    monkeypatch.setattr(sys, "__stderr__", cast(Any, object()))

    with _capture_external_output():
        pass

    assert calls == [sys.__stderr__]


@pytest.mark.parametrize("error", [AttributeError("missing"), ValueError("closed")])
def test_capture_external_output_ignores_non_redirectable_stream_errors(
    monkeypatch: pytest.MonkeyPatch, error: Exception
) -> None:
    """Capture helper should skip streams that cannot expose a file descriptor."""
    calls: list[object] = []

    @contextmanager
    def fake_process_raw_output(_consumer: object, output: object) -> Iterator[None]:
        calls.append(output)
        if len(calls) == 1:
            raise error
        yield

    monkeypatch.setattr("mlia.api.process_raw_output", fake_process_raw_output)

    with _capture_external_output():
        pass

    assert len(calls) == 2


def test_capture_external_output_reroutes_lines_to_logger(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Captured lines should be logged through mlia.api."""
    logger = MagicMock()

    @contextmanager
    def fake_process_raw_output(consumer: Any, _output: object) -> Iterator[None]:
        consumer("captured line\n")
        yield

    monkeypatch.setattr("mlia.api.process_raw_output", fake_process_raw_output)
    monkeypatch.setattr("mlia.api.logger", logger)

    with _capture_external_output():
        pass

    logger.debug.assert_called()


def test_override_model_name_for_module_non_dict_model_is_ignored() -> None:
    """Model-name override should skip malformed payloads safely."""
    from mlia.api import _override_model_name_for_module

    output: dict[str, object] = {"model": "not-a-dict"}
    _override_model_name_for_module(output, cast(Any, object()))
    assert output == {"model": "not-a-dict"}


def test_override_model_name_for_module_sets_module_name() -> None:
    """Model-name override should use the module class name."""
    from mlia.api import _override_model_name_for_module

    class DemoModule:  # pylint: disable=too-few-public-methods
        pass

    output: dict[str, object] = {"model": {}}
    _override_model_name_for_module(output, cast(Any, DemoModule()))

    model = cast(dict[str, str], output["model"])
    assert model["name"] == "DemoModule"


def test_resolve_logs_dir_branches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Logs-dir helper should cover relative, file, mkdir, and writeability paths."""
    monkeypatch.chdir(tmp_path)
    assert _resolve_logs_dir("logs") == tmp_path / "logs"

    file_path = tmp_path / "file"
    file_path.write_text("x", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="is not a directory"):
        _resolve_logs_dir(file_path)

    failing_path = tmp_path / "mkdir-fail"
    original_mkdir = Path.mkdir

    def mkdir_fail(
        self: Path,
        mode: int = 0o777,
        parents: bool = False,
        exist_ok: bool = False,
    ) -> None:
        if self == failing_path:
            raise OSError("boom")
        original_mkdir(self, mode=mode, parents=parents, exist_ok=exist_ok)

    monkeypatch.setattr(Path, "mkdir", mkdir_fail)
    with pytest.raises(ConfigurationError, match="Unable to create logs directory"):
        _resolve_logs_dir(failing_path)

    unwritable = tmp_path / "unwritable"
    Path.mkdir(unwritable)
    monkeypatch.setattr("mlia.api.os.access", lambda *_args, **_kwargs: False)
    with pytest.raises(ConfigurationError, match="is not writable"):
        _resolve_logs_dir(unwritable)


def test_validate_deprecated_backends_uses_default_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Deprecated backends without a custom message should use the default text."""
    monkeypatch.setitem(
        backend_registry.items,
        "legacy-default",
        BackendConfiguration(
            supported_advice=[],
            supported_systems=None,
            backend_type=BackendType.BUILTIN,
            installation=None,
            is_deprecated=True,
            deprecated_message=None,
        ),
    )

    with pytest.raises(FunctionalityNotSupportedError, match="Backend is deprecated"):
        _validate_deprecated_backends(
            ["legacy-default"], {AdviceCategory.COMPATIBILITY}
        )


def test_raise_if_deprecated_output_missing_multiple_backends(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Multiple deprecated backends should produce the combined error."""
    monkeypatch.setitem(
        backend_registry.items,
        "legacy-a",
        BackendConfiguration(
            supported_advice=[AdviceCategory.COMPATIBILITY],
            supported_systems=None,
            backend_type=BackendType.BUILTIN,
            installation=None,
            is_deprecated=True,
        ),
    )
    monkeypatch.setitem(
        backend_registry.items,
        "legacy-b",
        BackendConfiguration(
            supported_advice=[AdviceCategory.COMPATIBILITY],
            supported_systems=None,
            backend_type=BackendType.BUILTIN,
            installation=None,
            is_deprecated=True,
        ),
    )

    with pytest.raises(FunctionalityNotSupportedError, match="legacy-a, legacy-b"):
        _raise_if_deprecated_output_missing(["legacy-a", "legacy-b"], "profile")


def test_list_targets_success_with_mocked_registry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Discovery should build supported_advice entries in configured order."""
    monkeypatch.setattr(
        "mlia.api.profiles_by_target", lambda: {"target-a": ["profile-a"]}
    )
    monkeypatch.setattr(target_registry, "names", lambda: ["target-a"])
    monkeypatch.setattr(target_registry, "pretty_name", lambda _name: "Pretty Target")
    monkeypatch.setattr(
        "mlia.api.target_supported_advice",
        lambda _target: [AdviceCategory.OPTIMIZATION, AdviceCategory.COMPATIBILITY],
    )
    monkeypatch.setattr(
        "mlia.api.target_supported_backends",
        lambda _target: ["backend-b", "backend-a"],
    )

    assert list_targets() == [
        {
            "target": "target-a",
            "pretty_name": "Pretty Target",
            "profiles": ["profile-a"],
            "supported_backends": ["backend-a", "backend-b"],
            "supported_advice": ["compatibility", "optimization"],
        }
    ]


def test_list_backends_skips_non_selectable_entries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Non-selectable or advice-less backends should be omitted."""
    monkeypatch.setattr(
        "mlia.api.get_installation_manager", lambda noninteractive=True: MagicMock()
    )
    monkeypatch.setattr(
        backend_registry,
        "items",
        {
            "hidden": BackendConfiguration(
                supported_advice=[AdviceCategory.COMPATIBILITY],
                supported_systems=None,
                backend_type=BackendType.BUILTIN,
                installation=None,
                selectable=False,
            ),
            "empty": BackendConfiguration(
                supported_advice=[],
                supported_systems=None,
                backend_type=BackendType.BUILTIN,
                installation=None,
            ),
        },
    )

    assert list_backends() == []


def test_list_backends_covers_builtin_and_installation_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Discovery should cover builtin and installation-backed backends."""

    class FakeInstallation:  # pylint: disable=too-few-public-methods
        already_installed = False
        could_be_installed = True

    monkeypatch.setattr(
        "mlia.api.get_installation_manager", lambda noninteractive=True: MagicMock()
    )
    monkeypatch.setattr(
        backend_registry,
        "items",
        {
            "builtin": BackendConfiguration(
                supported_advice=[AdviceCategory.COMPATIBILITY],
                supported_systems=None,
                backend_type=BackendType.BUILTIN,
                installation=None,
            ),
            "installable": BackendConfiguration(
                supported_advice=[AdviceCategory.COMPATIBILITY],
                supported_systems=None,
                backend_type=BackendType.CUSTOM,
                installation=cast(Any, FakeInstallation()),
            ),
        },
    )
    monkeypatch.setattr(
        backend_registry,
        "pretty_name",
        lambda name: f"Pretty {name}",
    )

    assert list_backends() == [
        {
            "name": "builtin",
            "description": "Pretty builtin",
            "installed": True,
            "could_be_installed": True,
        },
        {
            "name": "installable",
            "description": "Pretty installable",
            "installed": False,
            "could_be_installed": True,
        },
    ]


def test_list_backend_options_type_name_variants(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Backend option discovery should map common Python types correctly."""
    monkeypatch.setattr(
        "mlia.api.discover_backend_option_specs",
        lambda: [
            {
                "module": "a",
                "backend": "backend-a",
                "config_key": "path",
                "cli_option": "--path",
                "full_cli_option": "--backend-a.path",
                "dest": "path",
                "type": Path,
                "help": "path help",
            },
            {
                "module": "a",
                "backend": "backend-a",
                "config_key": "flag",
                "cli_option": "--flag",
                "full_cli_option": "--backend-a.flag",
                "dest": "flag",
                "type": bool,
                "help": "flag help",
            },
            {
                "module": "a",
                "backend": "backend-a",
                "config_key": "ratio",
                "cli_option": "--ratio",
                "full_cli_option": "--backend-a.ratio",
                "dest": "ratio",
                "type": float,
                "help": "ratio help",
            },
            {
                "module": "a",
                "backend": "backend-a",
                "config_key": "name",
                "cli_option": "--name",
                "full_cli_option": "--backend-a.name",
                "dest": "name",
                "type": str,
                "help": "name help",
            },
            {
                "module": "a",
                "backend": "backend-a",
                "config_key": "fallback",
                "cli_option": "--fallback",
                "full_cli_option": "--backend-a.fallback",
                "dest": "fallback",
                "type": None,
                "help": "fallback help",
            },
            {
                "module": "a",
                "backend": "backend-a",
                "config_key": "custom",
                "cli_option": "--custom",
                "full_cli_option": "--backend-a.custom",
                "dest": "custom",
                "type": type("CustomType", (), {}),
                "help": "custom help",
            },
        ],
    )
    monkeypatch.setattr(
        backend_registry,
        "items",
        {
            "backend-a": BackendConfiguration(
                supported_advice=[AdviceCategory.COMPATIBILITY],
                supported_systems=None,
                backend_type=BackendType.BUILTIN,
                installation=None,
            )
        },
    )

    options = list_backend_options()
    assert options == [
        {
            "backend": "backend-a",
            "options": [
                {"config_key": "path", "type": "path", "description": "path help"},
                {"config_key": "flag", "type": "bool", "description": "flag help"},
                {"config_key": "ratio", "type": "float", "description": "ratio help"},
                {"config_key": "name", "type": "str", "description": "name help"},
                {
                    "config_key": "fallback",
                    "type": "str",
                    "description": "fallback help",
                },
                {
                    "config_key": "custom",
                    "type": "CustomType",
                    "description": "custom help",
                },
            ],
        }
    ]


def test_validate_backend_options_accepts_known_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Known backend-option keys should validate successfully."""
    monkeypatch.setattr(
        "mlia.api.list_backend_options",
        lambda: [{"backend": "backend-a", "options": [{"config_key": "known"}]}],
    )

    _validate_backend_options({"backend-a": {"known": "value"}})
