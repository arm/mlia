# SPDX-FileCopyrightText: Copyright 2022, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Processes module.

This module contains all classes and functions for dealing with Linux
processes.
"""
import csv
import datetime
import logging
import shlex
import signal
import time
from pathlib import Path
from typing import Any
from typing import List
from typing import NamedTuple
from typing import Optional
from typing import Tuple

import psutil
from sh import Command
from sh import CommandNotFound
from sh import ErrorReturnCode
from sh import RunningCommand

from aiet.utils.fs import valid_for_filename


class CommandFailedException(Exception):
    """Exception for failed command execution."""


class ShellCommand:
    """Wrapper class for shell commands."""

    def __init__(self, base_log_path: str = "/tmp") -> None:
        """Initialise the class.

        base_log_path: it is the base directory where logs will be stored
        """
        self.base_log_path = base_log_path

    def run(
        self,
        cmd: str,
        *args: str,
        _cwd: Optional[Path] = None,
        _tee: bool = True,
        _bg: bool = True,
        _out: Any = None,
        _err: Any = None,
        _search_paths: Optional[List[Path]] = None
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
            out, err = [
                str(item)
                for item in self.get_stdout_stderr_paths(self.base_log_path, cmd)
            ]

        return command(_out=out, _err=err, _tee=_tee, _bg=_bg, _bg_exc=False)

    @classmethod
    def get_stdout_stderr_paths(cls, base_log_path: str, cmd: str) -> Tuple[Path, Path]:
        """Construct and returns the paths of stdout/stderr files."""
        timestamp = datetime.datetime.now().timestamp()
        base_path = Path(base_log_path)
        base = base_path / "{}_{}".format(valid_for_filename(cmd, "_"), timestamp)
        stdout = base.with_suffix(".out")
        stderr = base.with_suffix(".err")
        try:
            stdout.touch()
            stderr.touch()
        except FileNotFoundError as error:
            logging.error("File not found: %s", error.filename)
            raise error
        return stdout, stderr


def parse_command(command: str, shell: str = "bash") -> List[str]:
    """Parse command."""
    cmd, *args = shlex.split(command, posix=True)

    if is_shell_script(cmd):
        args = [cmd] + args
        cmd = shell

    return [cmd] + args


def get_stdout_stderr_paths(
    command: str, base_log_path: str = "/tmp"
) -> Tuple[Path, Path]:
    """Construct and returns the paths of stdout/stderr files."""
    cmd, *_ = parse_command(command)

    return ShellCommand.get_stdout_stderr_paths(base_log_path, cmd)


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
) -> Tuple[int, bytearray, bytearray]:
    """
    Run command and wait while it is executing.

    Returns a tuple: (exit_code, stdout, stderr)
    """
    running_cmd: Optional[RunningCommand] = None
    try:
        running_cmd = execute_command(command, cwd, bg=True, out=out, err=err)
        return running_cmd.exit_code, running_cmd.stdout, running_cmd.stderr
    except ErrorReturnCode as cmd_failed:
        raise CommandFailedException() from cmd_failed
    except Exception as error:
        is_running = running_cmd is not None and running_cmd.is_alive()
        if terminate_on_error and is_running:
            print("Terminating ...")
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
            print(
                "Unable to terminate process {}. Sending SIGTERM...".format(
                    running_cmd.process.pid
                )
            )
            running_cmd.process.signal_group(signal.SIGTERM)
    except ProcessLookupError:
        pass


def terminate_external_process(
    process: psutil.Process,
    wait_period: float = 0.5,
    number_of_attempts: int = 20,
    wait_for_termination: float = 5.0,
) -> None:
    """Terminate external process."""
    try:
        process.terminate()
        for _ in range(number_of_attempts):
            if not process.is_running():
                return
            time.sleep(wait_period)

        if process.is_running():
            process.terminate()
            time.sleep(wait_for_termination)
    except psutil.Error:
        print("Unable to terminate process")


class ProcessInfo(NamedTuple):
    """Process information."""

    name: str
    executable: str
    cwd: str
    pid: int


def save_process_info(pid: int, pid_file: Path) -> None:
    """Save process information to file."""
    try:
        parent = psutil.Process(pid)
        children = parent.children(recursive=True)
        family = [parent] + children

        with open(pid_file, "w", encoding="utf-8") as file:
            csv_writer = csv.writer(file)
            for member in family:
                process_info = ProcessInfo(
                    name=member.name(),
                    executable=member.exe(),
                    cwd=member.cwd(),
                    pid=member.pid,
                )
                csv_writer.writerow(process_info)
    except psutil.NoSuchProcess:
        # if process does not exist or finishes before
        # function call then nothing could be saved
        # just ignore this exception and exit
        pass


def read_process_info(pid_file: Path) -> List[ProcessInfo]:
    """Read information about previous system processes."""
    if not pid_file.is_file():
        return []

    result = []
    with open(pid_file, encoding="utf-8") as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            name, executable, cwd, pid = row
            result.append(
                ProcessInfo(name=name, executable=executable, cwd=cwd, pid=int(pid))
            )

    return result


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
