# SPDX-FileCopyrightText: Copyright 2022, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Processes module.

This module contains all classes and functions for dealing with Linux
processes.
"""
from __future__ import annotations

import datetime
import logging
import shlex
import signal
import tempfile
import time
from pathlib import Path
from typing import Any

from sh import Command
from sh import CommandNotFound
from sh import ErrorReturnCode
from sh import RunningCommand

from mlia.backend.executor.fs import valid_for_filename

logger = logging.getLogger(__name__)


class CommandFailedException(Exception):
    """Exception for failed command execution."""


class ShellCommand:
    """Wrapper class for shell commands."""

    def run(
        self,
        cmd: str,
        *args: str,
        _cwd: Path | None = None,
        _tee: bool = True,
        _bg: bool = True,
        _out: Any = None,
        _err: Any = None,
        _search_paths: list[Path] | None = None,
    ) -> RunningCommand:
        """Run the shell command with the given arguments.

        There are special arguments that modify the behaviour of the process.
        _cwd: current working directory
        _tee: it redirects the stdout both to console and file
        _bg: if True, it runs the process in background and the command is not
        blocking.
        _out: use this object for stdout redirect,
        _err: use this object for stderr redirect,
        _search_paths: If presented used for searching executable
        """
        try:
            kwargs = {}
            if _cwd:
                kwargs["_cwd"] = str(_cwd)
            command = Command(cmd, _search_paths).bake(args, **kwargs)
        except CommandNotFound as error:
            logging.error("Command '%s' not found", error.args[0])
            raise error

        out, err = _out, _err
        if not _out and not _err:
            out, err = (str(item) for item in self.get_stdout_stderr_paths(cmd))

        return command(_out=out, _err=err, _tee=_tee, _bg=_bg, _bg_exc=False)

    @classmethod
    def get_stdout_stderr_paths(cls, cmd: str) -> tuple[Path, Path]:
        """Construct and returns the paths of stdout/stderr files."""
        timestamp = datetime.datetime.now().timestamp()
        base_path = Path(tempfile.mkdtemp(prefix="mlia-", suffix=f"{timestamp}"))
        base = base_path / f"{valid_for_filename(cmd, '_')}_{timestamp}"
        stdout = base.with_suffix(".out")
        stderr = base.with_suffix(".err")
        try:
            stdout.touch()
            stderr.touch()
        except FileNotFoundError as error:
            logging.error("File not found: %s", error.filename)
            raise error
        return stdout, stderr


def parse_command(command: str, shell: str = "bash") -> list[str]:
    """Parse command."""
    cmd, *args = shlex.split(command, posix=True)

    if is_shell_script(cmd):
        args = [cmd] + args
        cmd = shell

    return [cmd] + args


def execute_command(  # pylint: disable=invalid-name
    command: str,
    cwd: Path,
    bg: bool = False,
    shell: str = "bash",
    out: Any = None,
    err: Any = None,
) -> RunningCommand:
    """Execute shell command."""
    cmd, *args = parse_command(command, shell)

    search_paths = None
    if cmd != shell and (cwd / cmd).is_file():
        search_paths = [cwd]

    return ShellCommand().run(
        cmd, *args, _cwd=cwd, _bg=bg, _search_paths=search_paths, _out=out, _err=err
    )


def is_shell_script(cmd: str) -> bool:
    """Check if command is shell script."""
    return cmd.endswith(".sh")


def run_and_wait(
    command: str,
    cwd: Path,
    terminate_on_error: bool = False,
    out: Any = None,
    err: Any = None,
) -> tuple[int, bytearray, bytearray]:
    """
    Run command and wait while it is executing.

    Returns a tuple: (exit_code, stdout, stderr)
    """
    running_cmd: RunningCommand | None = None
    try:
        running_cmd = execute_command(command, cwd, bg=True, out=out, err=err)
        return running_cmd.exit_code, running_cmd.stdout, running_cmd.stderr
    except ErrorReturnCode as cmd_failed:
        raise CommandFailedException() from cmd_failed
    except Exception as error:
        is_running = running_cmd is not None and running_cmd.is_alive()
        if terminate_on_error and is_running:
            logger.debug("Terminating ...")
            terminate_command(running_cmd)

        raise error


def terminate_command(
    running_cmd: RunningCommand,
    wait: bool = True,
    wait_period: float = 0.5,
    number_of_attempts: int = 20,
) -> None:
    """Terminate running command."""
    try:
        running_cmd.process.signal_group(signal.SIGINT)
        if wait:
            for _ in range(number_of_attempts):
                time.sleep(wait_period)
                if not running_cmd.is_alive():
                    return
            logger.error(
                "Unable to terminate process %i. Sending SIGTERM...",
                running_cmd.process.pid,
            )
            running_cmd.process.signal_group(signal.SIGTERM)
    except ProcessLookupError:
        pass


def print_command_stdout(command: RunningCommand) -> None:
    """Print the stdout of a command.

    The command has 2 states: running and done.
    If the command is running, the output is taken by the running process.
    If the command has ended its execution, the stdout is taken from stdout
    property
    """
    if command.is_alive():
        while True:
            try:
                print(command.next(), end="")
            except StopIteration:
                break
    else:
        print(command.stdout)
