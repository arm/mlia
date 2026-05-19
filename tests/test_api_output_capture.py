# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Focused tests for API stdout/stderr capture behavior."""

from __future__ import annotations

import logging
import os
import subprocess  # nosec B404
import sys
from pathlib import Path

import pytest

from mlia.api import ValidationMode, run_advisor
from mlia.core.errors import InternalError


class _FakeHandler:
    """Minimal workflow handler stub for API capture tests."""

    output = {"schema_version": "1.0.0", "results": []}


def _patch_common_api_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("mlia.api.validate_backend", lambda _tp, _b: [])
    monkeypatch.setattr("mlia.api.get_target", lambda target_profile: target_profile)
    monkeypatch.setattr(
        "mlia.api._get_api_event_handler",
        lambda _target, _output_dir: _FakeHandler(),
    )


def test_run_advisor_captures_subprocess_stdio(
    monkeypatch: pytest.MonkeyPatch,
    test_tflite_model: Path,
    capfd: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """API mode captures child process stdout/stderr and reroutes to logger."""

    def fake_get_advice(*_args: object, **_kwargs: object) -> None:
        subprocess.run(  # nosec B603
            [
                sys.executable,
                "-c",
                (
                    "import os;"
                    "os.write(1, b'subprocess stdout line\\n');"
                    "os.write(2, b'subprocess stderr line\\n')"
                ),
            ],
            check=True,
        )

    _patch_common_api_dependencies(monkeypatch)
    monkeypatch.setattr("mlia.api.get_advice", fake_get_advice)
    monkeypatch.setattr("mlia.api.collect_validation_errors", lambda _d: [])

    caplog.set_level(logging.DEBUG, logger="mlia.api")
    run_advisor("compatibility", "tosa", test_tflite_model, validation="off")
    captured = capfd.readouterr()

    assert captured.out == ""
    assert captured.err == ""
    messages = [record.message for record in caplog.records]
    assert any("subprocess stdout line" in message for message in messages)
    assert any("subprocess stderr line" in message for message in messages)


def test_run_advisor_capture_window_is_scoped_to_get_advice(
    monkeypatch: pytest.MonkeyPatch,
    test_tflite_model: Path,
    capfd: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Only output produced during get_advice is captured by API wrapper."""

    def fake_get_advice(*_args: object, **_kwargs: object) -> None:
        os.write(1, b"inside-capture\n")

    _patch_common_api_dependencies(monkeypatch)
    monkeypatch.setattr("mlia.api.get_advice", fake_get_advice)
    monkeypatch.setattr("mlia.api.collect_validation_errors", lambda _d: [])

    os.write(1, b"outside-before\n")
    caplog.set_level(logging.DEBUG, logger="mlia.api")
    run_advisor("compatibility", "tosa", test_tflite_model, validation="off")
    os.write(2, b"outside-after\n")
    captured = capfd.readouterr()

    assert "outside-before" in captured.out
    assert "outside-after" in captured.err
    assert "inside-capture" not in captured.out
    messages = [record.message for record in caplog.records]
    assert any("inside-capture" in message for message in messages)


def test_run_advisor_capture_handles_partial_and_multiline_writes(
    monkeypatch: pytest.MonkeyPatch,
    test_tflite_model: Path,
    capfd: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Partial writes are combined and multiline output is preserved in logs."""

    def fake_get_advice(*_args: object, **_kwargs: object) -> None:
        os.write(1, b"line-part-a ")
        os.write(1, b"line-part-b\nline-two\n")

    _patch_common_api_dependencies(monkeypatch)
    monkeypatch.setattr("mlia.api.get_advice", fake_get_advice)
    monkeypatch.setattr("mlia.api.collect_validation_errors", lambda _d: [])

    caplog.set_level(logging.DEBUG, logger="mlia.api")
    run_advisor("compatibility", "tosa", test_tflite_model, validation="off")
    captured = capfd.readouterr()

    assert captured.out == ""
    assert captured.err == ""
    messages = [record.message for record in caplog.records]
    assert any("line-part-a line-part-b" in message for message in messages)
    assert any("line-two" in message for message in messages)


def test_run_advisor_capture_works_without_logs_dir(
    monkeypatch: pytest.MonkeyPatch,
    test_tflite_model: Path,
    capfd: pytest.CaptureFixture[str],
) -> None:
    """Capture does not depend on logs_dir/setup_logging being configured."""

    setup_logging_called = False

    def fake_setup_logging(*_args: object, **_kwargs: object) -> None:
        nonlocal setup_logging_called
        setup_logging_called = True

    def fake_get_advice(*_args: object, **_kwargs: object) -> None:
        os.write(2, b"external-no-logs-dir\n")

    _patch_common_api_dependencies(monkeypatch)
    monkeypatch.setattr("mlia.api.setup_logging", fake_setup_logging)
    monkeypatch.setattr("mlia.api.get_advice", fake_get_advice)
    monkeypatch.setattr("mlia.api.collect_validation_errors", lambda _d: [])

    run_advisor("compatibility", "tosa", test_tflite_model, validation="off")
    captured = capfd.readouterr()

    assert setup_logging_called is False
    assert captured.out == ""
    assert captured.err == ""


def test_run_advisor_capture_with_validation_warn(
    monkeypatch: pytest.MonkeyPatch,
    test_tflite_model: Path,
    capfd: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Capture remains active when validation warns but does not raise."""

    def fake_get_advice(*_args: object, **_kwargs: object) -> None:
        os.write(2, b"warn-validation-capture\n")

    _patch_common_api_dependencies(monkeypatch)
    monkeypatch.setattr("mlia.api.get_advice", fake_get_advice)
    monkeypatch.setattr("mlia.api.collect_validation_errors", lambda _d: ["bad schema"])

    caplog.set_level(logging.DEBUG, logger="mlia.api")
    output = run_advisor(
        "compatibility",
        "tosa",
        test_tflite_model,
        validation=ValidationMode.WARN,
    )
    captured = capfd.readouterr()

    assert output == _FakeHandler.output
    assert captured.out == ""
    assert captured.err == ""
    messages = [record.message for record in caplog.records]
    assert any("warn-validation-capture" in message for message in messages)
    assert any("Schema validation failed" in message for message in messages)


def test_run_advisor_capture_with_validation_strict(
    monkeypatch: pytest.MonkeyPatch,
    test_tflite_model: Path,
    capfd: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Capture remains active when strict validation raises InternalError."""

    def fake_get_advice(*_args: object, **_kwargs: object) -> None:
        os.write(1, b"strict-validation-capture\n")

    _patch_common_api_dependencies(monkeypatch)
    monkeypatch.setattr("mlia.api.get_advice", fake_get_advice)
    monkeypatch.setattr("mlia.api.collect_validation_errors", lambda _d: ["bad schema"])

    caplog.set_level(logging.DEBUG, logger="mlia.api")
    with pytest.raises(InternalError, match="Schema validation failed"):
        run_advisor(
            "compatibility",
            "tosa",
            test_tflite_model,
            validation=ValidationMode.STRICT,
        )
    captured = capfd.readouterr()

    assert captured.out == ""
    assert captured.err == ""
    messages = [record.message for record in caplog.records]
    assert any("strict-validation-capture" in message for message in messages)


@pytest.mark.parametrize(
    "target_profile",
    ["tosa", "cortex-a", "ethos-u55-256", "custom-target"],
)
def test_run_advisor_capture_smoke_across_targets(
    monkeypatch: pytest.MonkeyPatch,
    test_tflite_model: Path,
    capfd: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
    target_profile: str,
) -> None:
    """Capture behavior is stable across target-profile API handler selection."""

    def fake_get_advice(*_args: object, **_kwargs: object) -> None:
        os.write(2, f"external-line-{target_profile}\n".encode())

    _patch_common_api_dependencies(monkeypatch)
    monkeypatch.setattr("mlia.api.get_advice", fake_get_advice)
    monkeypatch.setattr("mlia.api.collect_validation_errors", lambda _d: [])

    caplog.set_level(logging.DEBUG, logger="mlia.api")
    run_advisor("compatibility", target_profile, test_tflite_model, validation="off")
    captured = capfd.readouterr()

    assert captured.out == ""
    assert captured.err == ""
    assert any(
        f"external-line-{target_profile}" in record.message for record in caplog.records
    )
