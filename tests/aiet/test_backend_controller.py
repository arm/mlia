# SPDX-FileCopyrightText: Copyright 2022, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for system controller."""
import csv
import os
import time
from pathlib import Path
from typing import Any

import psutil
import pytest

from aiet.backend.common import ConfigurationException
from aiet.backend.controller import SystemController
from aiet.backend.controller import SystemControllerSingleInstance
from aiet.utils.proc import ShellCommand


def get_system_controller(**kwargs: Any) -> SystemController:
    """Get service controller."""
    single_instance = kwargs.get("single_instance", False)
    if single_instance:
        pid_file_path = kwargs.get("pid_file_path")
        return SystemControllerSingleInstance(pid_file_path)

    return SystemController()


def test_service_controller() -> None:
    """Test service controller functionality."""
    service_controller = get_system_controller()

    assert service_controller.get_output() == ("", "")
    with pytest.raises(ConfigurationException, match="Wrong working directory"):
        service_controller.start(["sleep 100"], Path("unknown"))

    service_controller.start(["sleep 100"], Path.cwd())
    assert service_controller.is_running()

    service_controller.stop(True)
    assert not service_controller.is_running()
    assert service_controller.get_output() == ("", "")

    service_controller.stop()

    with pytest.raises(
        ConfigurationException, match="System should have only one command to run"
    ):
        service_controller.start(["sleep 100", "sleep 101"], Path.cwd())

    with pytest.raises(ConfigurationException, match="No startup command provided"):
        service_controller.start([""], Path.cwd())


def test_service_controller_bad_configuration() -> None:
    """Test service controller functionality for bad configuration."""
    with pytest.raises(Exception, match="No pid file path presented"):
        service_controller = get_system_controller(
            single_instance=True, pid_file_path=None
        )
        service_controller.start(["sleep 100"], Path.cwd())


def test_service_controller_writes_process_info_correctly(tmpdir: Any) -> None:
    """Test that controller writes process info correctly."""
    pid_file = Path(tmpdir) / "test.pid"

    service_controller = get_system_controller(
        single_instance=True, pid_file_path=Path(tmpdir) / "test.pid"
    )

    service_controller.start(["sleep 100"], Path.cwd())
    assert service_controller.is_running()
    assert pid_file.is_file()

    with open(pid_file, "r", encoding="utf-8") as file:
        csv_reader = csv.reader(file)
        rows = list(csv_reader)
        assert len(rows) == 1

        name, *_ = rows[0]
        assert name == "sleep"

    service_controller.stop()
    assert pid_file.exists()


def test_service_controller_does_not_write_process_info_if_process_finishes(
    tmpdir: Any,
) -> None:
    """Test that controller does not write process info if process already finished."""
    pid_file = Path(tmpdir) / "test.pid"
    service_controller = get_system_controller(
        single_instance=True, pid_file_path=pid_file
    )
    service_controller.is_running = lambda: False  # type: ignore
    service_controller.start(["echo hello"], Path.cwd())

    assert not pid_file.exists()


def test_service_controller_searches_for_previous_instances_correctly(
    tmpdir: Any,
) -> None:
    """Test that controller searches for previous instances correctly."""
    pid_file = Path(tmpdir) / "test.pid"
    command = ShellCommand().run("sleep", "100")
    assert command.is_alive()

    pid = command.process.pid
    process = psutil.Process(pid)
    with open(pid_file, "w", encoding="utf-8") as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow(("some_process", "some_program", "some_cwd", os.getpid()))
        csv_writer.writerow((process.name(), process.exe(), process.cwd(), process.pid))
        csv_writer.writerow(("some_old_process", "not_running", "from_nowhere", 77777))

    service_controller = get_system_controller(
        single_instance=True, pid_file_path=pid_file
    )
    service_controller.start(["sleep 100"], Path.cwd())
    # controller should stop this process as it is currently running and
    # mentioned in pid file
    assert not command.is_alive()

    service_controller.stop()


@pytest.mark.parametrize(
    "executable", ["test_backend_run_script.sh", "test_backend_run"]
)
def test_service_controller_run_shell_script(
    executable: str, test_resources_path: Path
) -> None:
    """Test controller's ability to run shell scripts."""
    script_path = test_resources_path / "scripts"

    service_controller = get_system_controller()

    service_controller.start([executable], script_path)

    assert service_controller.is_running()
    # give time for the command to produce output
    time.sleep(2)
    service_controller.stop(wait=True)
    assert not service_controller.is_running()
    stdout, stderr = service_controller.get_output()
    assert stdout == "Hello from script\n"
    assert stderr == "Oops!\n"


def test_service_controller_does_nothing_if_not_started(tmpdir: Any) -> None:
    """Test that nothing happened if controller is not started."""
    service_controller = get_system_controller(
        single_instance=True, pid_file_path=Path(tmpdir) / "test.pid"
    )

    assert not service_controller.is_running()
    service_controller.stop()
    assert not service_controller.is_running()
