# SPDX-FileCopyrightText: Copyright 2022, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Contain protocol related classes and functions."""
from abc import ABC
from abc import abstractmethod
from contextlib import closing
from pathlib import Path
from typing import Any
from typing import cast
from typing import Iterable
from typing import Optional
from typing import Tuple
from typing import Union

import paramiko

from aiet.backend.common import ConfigurationException
from aiet.backend.config import LocalProtocolConfig
from aiet.backend.config import SSHConfig
from aiet.utils.proc import run_and_wait


# Redirect all paramiko thread exceptions to a file otherwise these will be
# printed to stderr.
paramiko.util.log_to_file("/tmp/main_paramiko_log.txt", level=paramiko.common.INFO)


class SSHConnectionException(Exception):
    """SSH connection exception."""


class SupportsClose(ABC):
    """Class indicates support of close operation."""

    @abstractmethod
    def close(self) -> None:
        """Close protocol session."""


class SupportsDeploy(ABC):
    """Class indicates support of deploy operation."""

    @abstractmethod
    def deploy(self, src: Path, dst: str, retry: bool = True) -> None:
        """Abstract method for deploy data."""


class SupportsConnection(ABC):
    """Class indicates that protocol uses network connections."""

    @abstractmethod
    def establish_connection(self) -> bool:
        """Establish connection with underlying system."""

    @abstractmethod
    def connection_details(self) -> Tuple[str, int]:
        """Return connection details (host, port)."""


class Protocol(ABC):
    """Abstract class for representing the protocol."""

    def __init__(self, iterable: Iterable = (), **kwargs: Any) -> None:
        """Initialize the class using a dict."""
        self.__dict__.update(iterable, **kwargs)
        self._validate()

    @abstractmethod
    def _validate(self) -> None:
        """Abstract method for config data validation."""

    @abstractmethod
    def run(
        self, command: str, retry: bool = False
    ) -> Tuple[int, bytearray, bytearray]:
        """
        Abstract method for running commands.

        Returns a tuple: (exit_code, stdout, stderr)
        """


class CustomSFTPClient(paramiko.SFTPClient):
    """Class for creating a custom sftp client."""

    def put_dir(self, source: Path, target: str) -> None:
        """Upload the source directory to the target path.

        The target directory needs to exists and the last directory of the
        source will be created under the target with all its content.
        """
        # Create the target directory
        self._mkdir(target, ignore_existing=True)
        # Create the last directory in the source on the target
        self._mkdir("{}/{}".format(target, source.name), ignore_existing=True)
        # Go through the whole content of source
        for item in sorted(source.glob("**/*")):
            relative_path = item.relative_to(source.parent)
            remote_target = target / relative_path
            if item.is_file():
                self.put(str(item), str(remote_target))
            else:
                self._mkdir(str(remote_target), ignore_existing=True)

    def _mkdir(self, path: str, mode: int = 511, ignore_existing: bool = False) -> None:
        """Extend mkdir functionality.

        This version adds an option to not fail if the folder exists.
        """
        try:
            super().mkdir(path, mode)
        except IOError as error:
            if ignore_existing:
                pass
            else:
                raise error


class LocalProtocol(Protocol):
    """Class for local protocol."""

    protocol: str
    cwd: Path

    def run(
        self, command: str, retry: bool = False
    ) -> Tuple[int, bytearray, bytearray]:
        """
        Run command locally.

        Returns a tuple: (exit_code, stdout, stderr)
        """
        if not isinstance(self.cwd, Path) or not self.cwd.is_dir():
            raise ConfigurationException("Wrong working directory {}".format(self.cwd))

        stdout = bytearray()
        stderr = bytearray()

        return run_and_wait(
            command, self.cwd, terminate_on_error=True, out=stdout, err=stderr
        )

    def _validate(self) -> None:
        """Validate protocol configuration."""
        assert hasattr(self, "protocol") and self.protocol == "local"
        assert hasattr(self, "cwd")


class SSHProtocol(Protocol, SupportsClose, SupportsDeploy, SupportsConnection):
    """Class for SSH protocol."""

    protocol: str
    username: str
    password: str
    hostname: str
    port: int

    def __init__(self, iterable: Iterable = (), **kwargs: Any) -> None:
        """Initialize the class using a dict."""
        super().__init__(iterable, **kwargs)
        # Internal state to store if the system is connectable. It will be set
        # to true at the first connection instance
        self.client: Optional[paramiko.client.SSHClient] = None
        self.port = int(self.port)

    def run(self, command: str, retry: bool = True) -> Tuple[int, bytearray, bytearray]:
        """
        Run command over SSH.

        Returns a tuple: (exit_code, stdout, stderr)
        """
        transport = self._get_transport()
        with closing(transport.open_session()) as channel:
            # Enable shell's .profile settings and execute command
            channel.exec_command("bash -l -c '{}'".format(command))
            exit_status = -1
            stdout = bytearray()
            stderr = bytearray()
            while True:
                if channel.exit_status_ready():
                    exit_status = channel.recv_exit_status()
                    # Call it one last time to read any leftover in the channel
                    self._recv_stdout_err(channel, stdout, stderr)
                    break
                self._recv_stdout_err(channel, stdout, stderr)

        return exit_status, stdout, stderr

    def deploy(self, src: Path, dst: str, retry: bool = True) -> None:
        """Deploy src to remote dst over SSH.

        src and dst should be path to a file or directory.
        """
        transport = self._get_transport()
        sftp = cast(CustomSFTPClient, CustomSFTPClient.from_transport(transport))

        with closing(sftp):
            if src.is_dir():
                sftp.put_dir(src, dst)
            elif src.is_file():
                sftp.put(str(src), dst)
            else:
                raise Exception("Deploy error: file type not supported")

        # After the deployment of files, sync the remote filesystem to flush
        # buffers to hard disk
        self.run("sync")

    def close(self) -> None:
        """Close protocol session."""
        if self.client is not None:
            print("Try syncing remote file system...")
            # Before stopping the system, we try to run sync to make sure all
            # data are flushed on disk.
            self.run("sync", retry=False)
            self._close_client(self.client)

    def establish_connection(self) -> bool:
        """Establish connection with underlying system."""
        if self.client is not None:
            return True

        self.client = self._connect()
        return self.client is not None

    def _get_transport(self) -> paramiko.transport.Transport:
        """Get transport."""
        self.establish_connection()

        if self.client is None:
            raise SSHConnectionException(
                "Couldn't connect to '{}:{}'.".format(self.hostname, self.port)
            )

        transport = self.client.get_transport()
        if not transport:
            raise Exception("Unable to get transport")

        return transport

    def connection_details(self) -> Tuple[str, int]:
        """Return connection details of underlying system."""
        return (self.hostname, self.port)

    def _connect(self) -> Optional[paramiko.client.SSHClient]:
        """Try to establish connection."""
        client: Optional[paramiko.client.SSHClient] = None
        try:
            client = paramiko.client.SSHClient()
            client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            client.connect(
                self.hostname,
                self.port,
                self.username,
                self.password,
                # next parameters should be set to False to disable authentication
                # using ssh keys
                allow_agent=False,
                look_for_keys=False,
            )
            return client
        except (
            # OSError raised on first attempt to connect when running inside Docker
            OSError,
            paramiko.ssh_exception.NoValidConnectionsError,
            paramiko.ssh_exception.SSHException,
        ):
            # even if connection is not established socket could be still
            # open, it should be closed
            self._close_client(client)

            return None

    @staticmethod
    def _close_client(client: Optional[paramiko.client.SSHClient]) -> None:
        """Close ssh client."""
        try:
            if client is not None:
                client.close()
        except Exception:  # pylint: disable=broad-except
            pass

    @classmethod
    def _recv_stdout_err(
        cls, channel: paramiko.channel.Channel, stdout: bytearray, stderr: bytearray
    ) -> None:
        """Read from channel to stdout/stder."""
        chunk_size = 512
        if channel.recv_ready():
            stdout_chunk = channel.recv(chunk_size)
            stdout.extend(stdout_chunk)
        if channel.recv_stderr_ready():
            stderr_chunk = channel.recv_stderr(chunk_size)
            stderr.extend(stderr_chunk)

    def _validate(self) -> None:
        """Check if there are all the info for establishing the connection."""
        assert hasattr(self, "protocol") and self.protocol == "ssh"
        assert hasattr(self, "username")
        assert hasattr(self, "password")
        assert hasattr(self, "hostname")
        assert hasattr(self, "port")


class ProtocolFactory:
    """Factory class to return the appropriate Protocol class."""

    @staticmethod
    def get_protocol(
        config: Optional[Union[SSHConfig, LocalProtocolConfig]],
        **kwargs: Union[str, Path, None]
    ) -> Union[SSHProtocol, LocalProtocol]:
        """Return the right protocol instance based on the config."""
        if not config:
            raise ValueError("No protocol config provided")

        protocol = config["protocol"]
        if protocol == "ssh":
            return SSHProtocol(config)

        if protocol == "local":
            cwd = kwargs.get("cwd")
            return LocalProtocol(config, cwd=cwd)

        raise ValueError("Protocol not supported: '{}'".format(protocol))
