# SPDX-FileCopyrightText: Copyright 2022, Arm Limited and/or its affiliates.
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

from mlia.utils.logging import create_log_handler
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
