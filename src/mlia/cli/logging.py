# Copyright 2021, Arm Ltd.
"""CLI logging configuration."""
import datetime
import logging
import sys
from pathlib import Path
from typing import Any
from typing import Optional


_TOOLS_LOGGERS = [
    "mlia.tools.aiet",
    "mlia.tools.vela",
    "tensorflow",
    "py.warnings",
]

_MLIA_LOGGERS = [
    "mlia.cli",
    "mlia.performance",
    "mlia.operators",
    "mlia.reporters",
]


def setup_logging(logs_dir: Optional[str] = None, verbose: bool = False) -> None:
    """Set up logging.

    IA uses module 'logging' when it needs to produce output.

    :param logs_dir: path to the directory where application will save logs with
           debug information. If the path is not provided then no log files will
           be created during execution
    :param verbose: enable extended logging for the tools loggers
    """
    now_timestamp = f"{datetime.datetime.now():%Y%m%d_%H%M%S}"
    logs_dir_path: Optional[Path] = None

    if logs_dir:
        logs_dir_path = Path(logs_dir)
        logs_dir_path.mkdir(exist_ok=True)

    for logger_name in _TOOLS_LOGGERS:
        module_name = logger_name.split(".")[-1]
        logger = logging.getLogger(logger_name)

        if logs_dir_path is not None:
            log_file_path = logs_dir_path / f"{now_timestamp}_{module_name}.log"

            file_handler = create_log_handler(
                file_path=log_file_path,
                log_level=logging.DEBUG,
                log_format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                delay=True,
            )
            logger.setLevel(logging.DEBUG)
            logger.addHandler(file_handler)

        if verbose:
            stdout_handler = create_log_handler(
                stream=sys.stdout,
                log_level=logging.DEBUG,
                log_format="%(name)s - %(message)s",
            )
            logger.setLevel(logging.DEBUG)
            logger.addHandler(stdout_handler)

    for logger_name in _MLIA_LOGGERS:
        logger = logging.getLogger(logger_name)

        logger.setLevel(logging.INFO)
        stdout_handler = create_log_handler(stream=sys.stdout)
        logger.addHandler(stdout_handler)


def create_log_handler(
    *,
    file_path: Optional[Path] = None,
    stream: Optional[Any] = None,
    log_level: Optional[int] = None,
    log_format: Optional[str] = None,
    delay: bool = True,
) -> logging.Handler:
    """Create logger handler."""
    handler: Optional[logging.Handler] = None

    if file_path is not None:
        handler = logging.FileHandler(file_path, delay=delay)
    elif stream is not None:
        handler = logging.StreamHandler(stream)

    if handler is None:
        raise Exception("Unable to create logging handler")

    if log_level:
        handler.setLevel(log_level)

    if log_format:
        handler.setFormatter(logging.Formatter(log_format))

    return handler
