# Copyright 2021, Arm Ltd.
"""CLI logging configuration."""
import logging
import sys
from pathlib import Path
from typing import Any
from typing import Callable
from typing import Optional
from typing import Union


class LogFilter(logging.Filter):
    """Configurable log filter."""

    def __init__(self, log_record_filter: Callable[[logging.LogRecord], bool]) -> None:
        """Init log filter instance."""
        super().__init__()
        self.log_record_filter = log_record_filter

    def filter(self, record: logging.LogRecord) -> bool:
        """Filter log messages."""
        return self.log_record_filter(record)

    @classmethod
    def equals(cls, log_level: int) -> "LogFilter":
        """Return log filter that filters messages by log level."""

        def filter_by_level(log_record: logging.LogRecord) -> bool:
            return log_record.levelno == log_level

        return cls(filter_by_level)

    @classmethod
    def skip(cls, log_level: int) -> "LogFilter":
        """Return log filter that skips messages with particular level."""

        def skip_by_level(log_record: logging.LogRecord) -> bool:
            return log_record.levelno != log_level

        return cls(skip_by_level)


def setup_logging(
    logs_dir: Optional[Union[str, Path]] = None,
    verbose: bool = False,
    log_filename: str = "mlia.log",
) -> None:
    """Set up logging.

    IA uses module 'logging' when it needs to produce output.

    :param logs_dir: path to the directory where application will save logs with
           debug information. If the path is not provided then no log files will
           be created during execution
    :param verbose: enable extended logging for the tools loggers
    :param log_filename: name of the log file in the logs directory
    """
    mlia_logger, tensorflow_logger, python_warnings_logger = [
        logging.getLogger(logger_name)
        for logger_name in ["mlia", "tensorflow", "py.warnings"]
    ]

    mlia_logger.setLevel(logging.DEBUG)

    stdout_handler = create_log_handler(
        stream=sys.stdout,
        log_level=logging.INFO,
    )
    mlia_logger.addHandler(stdout_handler)

    if verbose:
        mlia_verbose_handler = create_log_handler(
            stream=sys.stdout,
            log_level=logging.DEBUG,
            log_format="%(name)s - %(message)s",
            log_filter=LogFilter.equals(logging.DEBUG),
        )
        mlia_logger.addHandler(mlia_verbose_handler)

        verbose_stdout_handler = create_log_handler(
            stream=sys.stdout,
            log_level=logging.DEBUG,
            log_format="%(name)s - %(message)s",
        )
        for logger in [tensorflow_logger, python_warnings_logger]:
            logger.addHandler(verbose_stdout_handler)

    if logs_dir:
        logs_dir_path = Path(logs_dir)
        logs_dir_path.mkdir(exist_ok=True)
        log_file_path = logs_dir_path / log_filename

        mlia_file_handler = create_log_handler(
            file_path=log_file_path,
            log_level=logging.DEBUG,
            log_format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            log_filter=LogFilter.skip(logging.INFO),
            delay=True,
        )
        mlia_logger.addHandler(mlia_file_handler)

        file_handler = create_log_handler(
            file_path=log_file_path,
            log_level=logging.DEBUG,
            log_format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            delay=True,
        )

        for logger in [tensorflow_logger, python_warnings_logger]:
            logger.addHandler(file_handler)


def create_log_handler(
    *,
    file_path: Optional[Path] = None,
    stream: Optional[Any] = None,
    log_level: Optional[int] = None,
    log_format: Optional[str] = None,
    log_filter: Optional[logging.Filter] = None,
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

    if log_filter:
        handler.addFilter(log_filter)

    return handler
