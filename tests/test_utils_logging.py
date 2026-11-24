# SPDX-FileCopyrightText: Copyright 2022-2023, 2025-2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the logging utility functions."""

from __future__ import annotations

import logging
import sys
import threading
import time
from contextlib import ExitStack as does_not_raise
from io import StringIO
from pathlib import Path
from typing import Any, Callable
from unittest.mock import MagicMock, call

import pytest
from rich.console import Console

from mlia.utils.logging import (
    LogFilter,
    LoggerWriter,
    ProgressStreamHandler,
    capture_raw_output,
    configure_interactive_progress,
    create_log_handler,
    log_action,
    progress_heartbeat,
    redirect_output,
    redirect_raw_output,
)


@pytest.mark.parametrize("fails", [False, True])
@pytest.mark.parametrize("interactive_flag", [False, True])
def test_progress_heartbeat_stops_after_work(
    fails: bool,
    interactive_flag: bool,
) -> None:
    """Text heartbeats stop on exit, including when output is redirected."""
    configure_interactive_progress(
        interactive_flag, MagicMock(is_terminal=False) if interactive_flag else None
    )
    logger = MagicMock()
    heartbeat = threading.Event()
    logger.info.side_effect = lambda *_args, **_kwargs: heartbeat.set()

    try:
        if fails:
            with pytest.raises(RuntimeError, match="failed"):
                with progress_heartbeat(logger, "Model run", interval=0.01):
                    assert heartbeat.wait(1)
                    raise RuntimeError("failed")
        else:
            with progress_heartbeat(logger, "Model run", interval=0.01):
                assert heartbeat.wait(1)
    finally:
        configure_interactive_progress(False)

    count = logger.info.call_count
    time.sleep(0.03)
    assert logger.info.call_count == count


@pytest.mark.parametrize("fails", [False, True])
def test_interactive_progress_is_transient(
    monkeypatch: pytest.MonkeyPatch, fails: bool
) -> None:
    """A terminal spinner clears on success and on an exception."""
    console = MagicMock(is_terminal=True)
    progress = MagicMock()
    progress.__enter__.return_value = progress
    create_progress = MagicMock(return_value=progress)
    monkeypatch.setattr("mlia.utils.logging.Progress", create_progress)
    logger = MagicMock()

    configure_interactive_progress(True, console)
    try:
        if fails:
            with pytest.raises(RuntimeError, match="failed"):
                with progress_heartbeat(logger, "Model run"):
                    raise RuntimeError("failed")
        else:
            with progress_heartbeat(logger, "Model run"):
                pass
    finally:
        configure_interactive_progress(False)

    assert create_progress.call_args.kwargs["transient"] is True
    progress.add_task.assert_called_once_with("Model run", total=None)
    progress.__exit__.assert_called_once()
    logger.info.assert_not_called()


def test_nested_progress_uses_one_spinner(monkeypatch: pytest.MonkeyPatch) -> None:
    """A check spinner names each runtime operation while it is running."""
    console = MagicMock(is_terminal=True)
    progress = MagicMock()
    progress.__enter__.return_value = progress
    progress.add_task.return_value = 1
    create_progress = MagicMock(return_value=progress)
    monkeypatch.setattr("mlia.utils.logging.Progress", create_progress)
    configure_interactive_progress(True, console)
    try:
        with progress_heartbeat(MagicMock(), "Checking performance"):
            with progress_heartbeat(MagicMock(), "Running PyTorch to TOSA conversion"):
                pass
            with progress_heartbeat(MagicMock(), "Running NX Performance Estimator"):
                pass
            with pytest.raises(RuntimeError, match="worker failed"):
                with progress_heartbeat(MagicMock(), "Running failed worker"):
                    raise RuntimeError("worker failed")
    finally:
        configure_interactive_progress(False)

    create_progress.assert_called_once()
    progress.add_task.assert_called_once_with("Checking performance", total=None)
    assert progress.update.call_args_list == [
        call(1, description="Running PyTorch to TOSA conversion", refresh=True),
        call(1, description="Checking performance", refresh=True),
        call(1, description="Running NX Performance Estimator", refresh=True),
        call(1, description="Checking performance", refresh=True),
        call(1, description="Running failed worker", refresh=True),
        call(1, description="Checking performance", refresh=True),
    ]


def test_nested_text_progress_uses_inner_label_then_restores_outer() -> None:
    """Nonterminal heartbeats name the operation currently running."""
    configure_interactive_progress(False)
    logger = MagicMock()
    inner_seen = threading.Event()
    outer_seen = threading.Event()
    labels: list[str] = []

    def record(_message: str, current: str, _elapsed: int, **_kwargs: object) -> None:
        labels.append(current)
        (inner_seen if current == "Running NX" else outer_seen).set()

    logger.info.side_effect = record
    with progress_heartbeat(logger, "Checking performance", interval=0.01):
        with progress_heartbeat(logger, "Running NX", interval=0.01):
            assert inner_seen.wait(1)
        assert outer_seen.wait(1)

    assert labels.index("Running NX") < labels.index("Checking performance")


def test_progress_uses_configured_no_color_console() -> None:
    """The spinner respects the check console's disabled color setting."""
    stream = StringIO()
    console = Console(file=stream, force_terminal=True, no_color=True)
    configure_interactive_progress(True, console)
    try:
        with progress_heartbeat(MagicMock(), "Checking performance"):
            console.print("Stage message")
    finally:
        configure_interactive_progress(False)

    assert "Stage message" in stream.getvalue()
    assert "\x1b[36m" not in stream.getvalue()


def test_stage_log_uses_live_console(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stage messages use one console before, during, and after a spinner."""
    console = MagicMock(is_terminal=True)
    progress = MagicMock()
    progress.__enter__.return_value = progress
    monkeypatch.setattr("mlia.utils.logging.Progress", MagicMock(return_value=progress))
    stream = StringIO()
    handler = ProgressStreamHandler(stream, console)
    record = logging.makeLogRecord({"msg": "Running NX Performance Estimator..."})
    configure_interactive_progress(True, console)
    try:
        handler.handle(record)
        thread = threading.Thread(target=handler.handle, args=(record,))
        thread.start()
        thread.join()
        with progress_heartbeat(MagicMock(), "Checking performance"):
            handler.handle(record)
            assert stream.getvalue() == ""
        handler.handle(record)
    finally:
        configure_interactive_progress(False)

    assert console.print.call_count == 4
    console.print.assert_any_call(
        "Running NX Performance Estimator...", markup=False, highlight=False
    )
    handler.handle(record)
    assert console.print.call_count == 5
    assert stream.getvalue() == ""


@pytest.mark.parametrize(
    "file_path, stream, log_filter, delay, expected_error, expected_class",
    [
        (
            "test.log",
            None,
            None,
            True,
            does_not_raise(),
            logging.FileHandler,
        ),
        (
            None,
            sys.stdout,
            None,
            None,
            does_not_raise(),
            logging.StreamHandler,
        ),
        (
            None,
            None,
            None,
            None,
            pytest.raises(Exception, match="Unable to create logging handler"),
            None,
        ),
        (
            None,
            sys.stdout,
            LogFilter(lambda log_record: True),
            None,
            does_not_raise(),
            logging.StreamHandler,
        ),
    ],
)
def test_create_log_handler(
    file_path: Path | None,
    stream: Any | None,
    log_filter: logging.Filter | None,
    delay: bool,
    expected_error: Any,
    expected_class: type,
) -> None:
    """Test function test_create_log_handler."""
    with expected_error:
        handler = create_log_handler(
            file_path=file_path,
            stream=stream,
            log_level=logging.INFO,
            log_format="%(name)s - %(message)s",
            log_filter=log_filter,
            delay=delay,
        )
        assert isinstance(handler, expected_class)


def test_create_log_handler_uses_classified_file_gateway(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """File-backed handlers are constructed through the boundary gateway."""
    handler = MagicMock(spec=logging.FileHandler)
    create_handler = MagicMock(return_value=handler)
    monkeypatch.setattr("mlia.utils.logging.create_file_handler", create_handler)

    assert create_log_handler(file_path=Path("output.log"), delay=False) is handler
    create_handler.assert_called_once_with(Path("output.log"), delay=False)


@pytest.mark.parametrize(
    "redirect_context_manager",
    [
        redirect_raw_output,
        redirect_output,
    ],
)
def test_output_redirection(redirect_context_manager: Callable) -> None:
    """Test output redirection via context manager."""
    print("before redirect")
    logger_mock = MagicMock()
    with redirect_context_manager(logger_mock):
        print("output redirected")
    print("after redirect")

    logger_mock.log.assert_called_once_with(logging.INFO, "output redirected")


def test_output_and_error_capture() -> None:
    """Test output/error capturing."""
    with (
        capture_raw_output(sys.stdout) as std_output,
        capture_raw_output(sys.stderr) as stderr_output,
    ):
        print("hello from stdout")
        print("hello from stderr", file=sys.stderr)

    assert std_output == ["hello from stdout\n"]
    assert stderr_output == ["hello from stderr\n"]


def test_log_filtration_by_equal(capfd: pytest.CaptureFixture) -> None:
    """Test LogFilter that filters non INFO and DEBUG logs."""
    handler = create_log_handler(
        stream=sys.stdout, log_level=0, log_filter=LogFilter.equals(logging.INFO)
    )

    test_logger = logging.getLogger("test")
    test_logger.addHandler(handler)

    test_logger.debug("Debug message")
    test_logger.info("Info message")
    test_logger.error("Error message")

    stdout, _ = capfd.readouterr()
    assert stdout == ""


def test_log_filtration_by_skip(capfd: pytest.CaptureFixture) -> None:
    """Test LogFilter that skips INFO and DEBUG logs."""
    handler = create_log_handler(
        stream=sys.stdout, log_filter=LogFilter.skip(logging.INFO | logging.DEBUG)
    )

    test_logger = logging.getLogger("test")
    test_logger.addHandler(handler)

    test_logger.debug("Debug message")
    test_logger.info("Info message")
    test_logger.error("Error message")

    stdout, _ = capfd.readouterr()
    assert stdout == "Error message\n"


def test_logger_writer_recursion_guard() -> None:
    """LoggerWriter should drop recursive writes instead of re-entering logging."""
    logger_mock = MagicMock()
    writer = LoggerWriter(logger_mock, logging.INFO)
    writer._write_state.active = True

    writer.write("recursive message")

    logger_mock.log.assert_not_called()


@pytest.mark.parametrize(
    ("action", "expected"),
    [
        ("Checking performance...", "Completed: Checking performance"),
        ("Getting the performance data...", "Completed: Getting the performance data"),
        ("Running converter...", "Completed: Running converter"),
    ],
)
def test_log_action_names_completed_work(
    caplog: pytest.LogCaptureFixture, action: str, expected: str
) -> None:
    """A finished action identifies the work instead of printing 'Done'."""
    with caplog.at_level(logging.INFO, logger="mlia.utils.logging"):
        with log_action(action):
            pass

    assert [record.message for record in caplog.records] == [
        action,
        expected,
    ]
