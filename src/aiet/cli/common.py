# SPDX-FileCopyrightText: Copyright 2022, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Common functions for cli module."""
import enum
import logging
from functools import wraps
from signal import SIG_IGN
from signal import SIGINT
from signal import signal as signal_handler
from signal import SIGTERM
from typing import Any
from typing import Callable
from typing import cast
from typing import Dict

from click import ClickException
from click import Context
from click import UsageError

from aiet.backend.common import ConfigurationException
from aiet.backend.execution import AnotherInstanceIsRunningException
from aiet.backend.execution import ConnectionException
from aiet.backend.protocol import SSHConnectionException
from aiet.utils.proc import CommandFailedException


class MiddlewareExitCode(enum.IntEnum):
    """Middleware exit codes."""

    SUCCESS = 0
    # exit codes 1 and 2 are used by click
    SHUTDOWN_REQUESTED = 3
    BACKEND_ERROR = 4
    CONCURRENT_ERROR = 5
    CONNECTION_ERROR = 6
    CONFIGURATION_ERROR = 7
    MODEL_OPTIMISED_ERROR = 8
    INVALID_TFLITE_FILE_ERROR = 9


class CustomClickException(ClickException):
    """Custom click exception."""

    def show(self, file: Any = None) -> None:
        """Override show method."""
        super().show(file)

        logging.debug("Execution failed with following exception: ", exc_info=self)


class MiddlewareShutdownException(CustomClickException):
    """Exception indicates that user requested middleware shutdown."""

    exit_code = int(MiddlewareExitCode.SHUTDOWN_REQUESTED)


class BackendException(CustomClickException):
    """Exception indicates that command failed."""

    exit_code = int(MiddlewareExitCode.BACKEND_ERROR)


class ConcurrentErrorException(CustomClickException):
    """Exception indicates concurrent execution error."""

    exit_code = int(MiddlewareExitCode.CONCURRENT_ERROR)


class BackendConnectionException(CustomClickException):
    """Exception indicates that connection could not be established."""

    exit_code = int(MiddlewareExitCode.CONNECTION_ERROR)


class BackendConfigurationException(CustomClickException):
    """Exception indicates some configuration issue."""

    exit_code = int(MiddlewareExitCode.CONFIGURATION_ERROR)


class ModelOptimisedException(CustomClickException):
    """Exception indicates input file has previously been Vela optimised."""

    exit_code = int(MiddlewareExitCode.MODEL_OPTIMISED_ERROR)


class InvalidTFLiteFileError(CustomClickException):
    """Exception indicates input TFLite file is misformatted."""

    exit_code = int(MiddlewareExitCode.INVALID_TFLITE_FILE_ERROR)


def print_command_details(command: Dict) -> None:
    """Print command details including parameters."""
    command_strings = command["command_strings"]
    print("Commands: {}".format(command_strings))
    user_params = command["user_params"]
    for i, param in enumerate(user_params, 1):
        print("User parameter #{}".format(i))
        print("\tName: {}".format(param.get("name", "-")))
        print("\tDescription: {}".format(param["description"]))
        print("\tPossible values: {}".format(param.get("values", "-")))
        print("\tDefault value: {}".format(param.get("default_value", "-")))
        print("\tAlias: {}".format(param.get("alias", "-")))


def raise_exception_at_signal(
    signum: int, frame: Any  # pylint: disable=unused-argument
) -> None:
    """Handle signals."""
    # Disable both SIGINT and SIGTERM signals. Further SIGINT and SIGTERM
    # signals will be ignored as we allow a graceful shutdown.
    # Unused arguments must be present here in definition as used in signal handler
    # callback

    signal_handler(SIGINT, SIG_IGN)
    signal_handler(SIGTERM, SIG_IGN)
    raise MiddlewareShutdownException("Middleware shutdown requested")


def middleware_exception_handler(func: Callable) -> Callable:
    """Handle backend exceptions decorator."""

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return func(*args, **kwargs)
        except (MiddlewareShutdownException, UsageError, ClickException) as error:
            # click should take care of these exceptions
            raise error
        except ValueError as error:
            raise ClickException(str(error)) from error
        except AnotherInstanceIsRunningException as error:
            raise ConcurrentErrorException(
                "Another instance of the system is running"
            ) from error
        except (SSHConnectionException, ConnectionException) as error:
            raise BackendConnectionException(str(error)) from error
        except ConfigurationException as error:
            raise BackendConfigurationException(str(error)) from error
        except (CommandFailedException, Exception) as error:
            raise BackendException(
                "Execution failed. Please check output for the details."
            ) from error

    return wrapper


def middleware_signal_handler(func: Callable) -> Callable:
    """Handle signals decorator."""

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        # Set up signal handlers for SIGINT (ctrl-c) and SIGTERM (kill command)
        # The handler ignores further signals and it raises an exception
        signal_handler(SIGINT, raise_exception_at_signal)
        signal_handler(SIGTERM, raise_exception_at_signal)

        return func(*args, **kwargs)

    return wrapper


def set_format(ctx: Context, format_: str) -> None:
    """Save format in click context."""
    ctx_obj = ctx.ensure_object(dict)
    ctx_obj["format"] = format_


def get_format(ctx: Context) -> str:
    """Get format from click context."""
    ctx_obj = cast(Dict[str, str], ctx.ensure_object(dict))
    return ctx_obj["format"]
