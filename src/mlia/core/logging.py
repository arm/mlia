# SPDX-FileCopyrightText: Copyright 2022-2023, 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""CLI logging configuration."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Iterable

from mlia.core.typing import OutputFormat
from mlia.utils.logging import NoASCIIFormatter, create_log_handler

_CONSOLE_DEBUG_FORMAT = "%(name)s - %(levelname)s - %(message)s"
_FILE_DEBUG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
_CONFIGURED_HANDLERS: dict[logging.Logger, list[logging.Handler]] = {}


def _replace_configured_handlers(
    handlers: Iterable[logging.Handler], loggers: Iterable[logging.Logger]
) -> None:
    """Replace and close handlers installed by previous MLIA configuration."""
    new_handlers = list(handlers)
    target_loggers = list(loggers)
    previous_handlers = {
        handler
        for logger in target_loggers
        for handler in _CONFIGURED_HANDLERS.get(logger, [])
    }
    for logger in target_loggers:
        for handler in _CONFIGURED_HANDLERS.get(logger, []):
            logger.removeHandler(handler)
    for handler in previous_handlers:
        handler.close()

    for logger in target_loggers:
        for handler in new_handlers:
            logger.addHandler(handler)
        _CONFIGURED_HANDLERS[logger] = new_handlers


def close_configured_handlers() -> None:
    """Remove and close every handler installed by MLIA configuration."""
    configured_handlers = {
        handler for handlers in _CONFIGURED_HANDLERS.values() for handler in handlers
    }
    for logger, handlers in _CONFIGURED_HANDLERS.items():
        for handler in handlers:
            logger.removeHandler(handler)
    for handler in configured_handlers:
        handler.close()
    _CONFIGURED_HANDLERS.clear()


def setup_logging(
    logs_dir: str | Path | None = None,
    verbose: bool = False,
    output_format: OutputFormat = "plain_text",
    log_filename: str = "mlia.log",
) -> None:
    """Set up logging.

    MLIA uses module 'logging' when it needs to produce output.

    :param logs_dir: path to the directory where application will save logs with
           debug information. If the path is not provided then no log files will
           be created during execution
    :param verbose: enable extended logging for the tools loggers
    :param output_format: specify the out format needed for setting up the right
           logging system
    :param log_filename: name of the log file in the logs directory
    """
    mlia_logger = logging.getLogger("mlia")
    tensorflow_logger = logging.getLogger("tensorflow")
    py_warnings_logger = logging.getLogger("py.warnings")

    # enable debug output, actual message filtering depends on
    # the provided parameters and being done at the handlers level
    for logger in [mlia_logger, tensorflow_logger]:
        logger.setLevel(logging.DEBUG)

    mlia_handlers = _get_mlia_handlers(logs_dir, log_filename, verbose, output_format)
    _replace_configured_handlers(mlia_handlers, [mlia_logger])

    tools_handlers = _get_tools_handlers(logs_dir, log_filename, verbose)
    _replace_configured_handlers(
        tools_handlers,
        [tensorflow_logger, py_warnings_logger],
    )


def _get_mlia_handlers(
    logs_dir: str | Path | None,
    log_filename: str,
    verbose: bool,
    output_format: OutputFormat,
) -> Iterable[logging.Handler]:
    """Get handlers for the MLIA loggers."""
    # MLIA needs output to standard output via the logging system only when the
    # format is plain text. When the user specifies the "json" output format,
    # MLIA disables completely the logging system for the console output and it
    # relies on the print() function. This is needed because the output might
    # be corrupted with spurious messages in the standard output.
    if output_format == "plain_text":
        if verbose:
            log_level = logging.DEBUG
            log_format = _CONSOLE_DEBUG_FORMAT
        else:
            log_level = logging.INFO
            log_format = None

        # Create log handler for stdout
        yield create_log_handler(
            stream=sys.stdout, log_level=log_level, log_format=log_format
        )
    else:
        # In case of non plain text output, we need to inform the user if an
        # error happens during execution.
        yield create_log_handler(
            stream=sys.stderr,
            log_level=logging.ERROR,
        )

    # If the logs directory is specified, MLIA stores all output (according to
    # the logging level) into the file and removing the colouring of the
    # console output.
    if logs_dir:
        if verbose:
            log_level = logging.DEBUG
        else:
            log_level = logging.INFO

        yield create_log_handler(
            file_path=_get_log_file(logs_dir, log_filename),
            log_level=log_level,
            log_format=NoASCIIFormatter(fmt=_FILE_DEBUG_FORMAT),
            delay=False,
        )


def _get_tools_handlers(
    logs_dir: str | Path | None,
    log_filename: str,
    verbose: bool,
) -> Iterable[logging.Handler]:
    """Get handler for the tools loggers."""
    if verbose:
        yield create_log_handler(
            stream=sys.stdout,
            log_level=logging.DEBUG,
            log_format=_CONSOLE_DEBUG_FORMAT,
        )

    if logs_dir:
        yield create_log_handler(
            file_path=_get_log_file(logs_dir, log_filename),
            log_level=logging.DEBUG,
            log_format=_FILE_DEBUG_FORMAT,
            delay=False,
        )


def _get_log_file(logs_dir: str | Path, log_filename: str) -> Path:
    """Get the log file path."""
    logs_dir_path = Path(logs_dir)
    logs_dir_path.mkdir(parents=True, exist_ok=True)

    return logs_dir_path / log_filename
