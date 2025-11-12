# SPDX-FileCopyrightText: Copyright 2022-2023, 2025, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the logging utility functions."""
from __future__ import annotations

import logging
import sys
from contextlib import ExitStack as does_not_raise
from pathlib import Path
from typing import Any
from typing import Callable
from unittest.mock import MagicMock

import pytest

from mlia.utils.logging import capture_raw_output
from mlia.utils.logging import create_log_handler
from mlia.utils.logging import LogFilter
from mlia.utils.logging import redirect_output
from mlia.utils.logging import redirect_raw_output


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
    with capture_raw_output(sys.stdout) as std_output, capture_raw_output(
        sys.stderr
    ) as stderr_output:
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
