# SPDX-FileCopyrightText: Copyright 2022, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=no-self-use,attribute-defined-outside-init,protected-access
"""Tests for the protocol backend module."""
from contextlib import ExitStack as does_not_raise
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import paramiko
import pytest

from mlia.backend.common import ConfigurationException
from mlia.backend.config import LocalProtocolConfig
from mlia.backend.protocol import CustomSFTPClient
from mlia.backend.protocol import LocalProtocol
from mlia.backend.protocol import ProtocolFactory
from mlia.backend.protocol import SSHProtocol


class TestProtocolFactory:
    """Test ProtocolFactory class."""

    @pytest.mark.parametrize(
        "config, expected_class, exception",
        [
            (
                {
                    "protocol": "ssh",
                    "username": "user",
                    "password": "pass",
                    "hostname": "hostname",
                    "port": "22",
                },
                SSHProtocol,
                does_not_raise(),
            ),
            ({"protocol": "local"}, LocalProtocol, does_not_raise()),
            (
                {"protocol": "something"},
                None,
                pytest.raises(Exception, match="Protocol not supported"),
            ),
            (None, None, pytest.raises(Exception, match="No protocol config provided")),
        ],
    )
    def test_get_protocol(
        self, config: Any, expected_class: type, exception: Any
    ) -> None:
        """Test get_protocol method."""
        factory = ProtocolFactory()
        with exception:
            protocol = factory.get_protocol(config)
            assert isinstance(protocol, expected_class)


class TestLocalProtocol:
    """Test local protocol."""

    def test_local_protocol_run_command(self) -> None:
        """Test local protocol run command."""
        config = LocalProtocolConfig(protocol="local")
        protocol = LocalProtocol(config, cwd=Path("/tmp"))
        ret, stdout, stderr = protocol.run("pwd")
        assert ret == 0
        assert stdout.decode("utf-8").strip() == "/tmp"
        assert stderr.decode("utf-8") == ""

    def test_local_protocol_run_wrong_cwd(self) -> None:
        """Execution should fail if wrong working directory provided."""
        config = LocalProtocolConfig(protocol="local")
        protocol = LocalProtocol(config, cwd=Path("unknown_directory"))
        with pytest.raises(
            ConfigurationException, match="Wrong working directory unknown_directory"
        ):
            protocol.run("pwd")


class TestSSHProtocol:
    """Test SSH protocol."""

    @pytest.fixture(autouse=True)
    def setup_method(self, monkeypatch: Any) -> None:
        """Set up protocol mocks."""
        self.mock_ssh_client = MagicMock(spec=paramiko.client.SSHClient)

        self.mock_ssh_channel = (
            self.mock_ssh_client.get_transport.return_value.open_session.return_value
        )
        self.mock_ssh_channel.mock_add_spec(spec=paramiko.channel.Channel)
        self.mock_ssh_channel.exit_status_ready.side_effect = [False, True]
        self.mock_ssh_channel.recv_exit_status.return_value = True
        self.mock_ssh_channel.recv_ready.side_effect = [False, True]
        self.mock_ssh_channel.recv_stderr_ready.side_effect = [False, True]

        monkeypatch.setattr(
            "mlia.backend.protocol.paramiko.client.SSHClient",
            MagicMock(return_value=self.mock_ssh_client),
        )

        self.mock_sftp_client = MagicMock(spec=CustomSFTPClient)
        monkeypatch.setattr(
            "mlia.backend.protocol.CustomSFTPClient.from_transport",
            MagicMock(return_value=self.mock_sftp_client),
        )

        ssh_config = {
            "protocol": "ssh",
            "username": "user",
            "password": "pass",
            "hostname": "hostname",
            "port": "22",
        }
        self.protocol = SSHProtocol(ssh_config)

    def test_unable_create_ssh_client(self, monkeypatch: Any) -> None:
        """Test that command should fail if unable to create ssh client instance."""
        monkeypatch.setattr(
            "mlia.backend.protocol.paramiko.client.SSHClient",
            MagicMock(side_effect=OSError("Error!")),
        )

        with pytest.raises(Exception, match="Couldn't connect to 'hostname:22'"):
            self.protocol.run("command_example", retry=False)

    def test_ssh_protocol_run_command(self) -> None:
        """Test that command run via ssh successfully."""
        self.protocol.run("command_example")
        self.mock_ssh_channel.exec_command.assert_called_once()

    def test_ssh_protocol_run_command_connect_failed(self) -> None:
        """Test that if connection is not possible then correct exception is raised."""
        self.mock_ssh_client.connect.side_effect = OSError("Unable to connect")
        self.mock_ssh_client.close.side_effect = Exception("Error!")

        with pytest.raises(Exception, match="Couldn't connect to 'hostname:22'"):
            self.protocol.run("command_example", retry=False)

    def test_ssh_protocol_run_command_bad_transport(self) -> None:
        """Test that command should fail if unable to get transport."""
        self.mock_ssh_client.get_transport.return_value = None

        with pytest.raises(Exception, match="Unable to get transport"):
            self.protocol.run("command_example", retry=False)

    def test_ssh_protocol_deploy_command_file(
        self, test_applications_path: Path
    ) -> None:
        """Test that files could be deployed over ssh."""
        file_for_deploy = test_applications_path / "readme.txt"
        dest = "/tmp/dest"

        self.protocol.deploy(file_for_deploy, dest)
        self.mock_sftp_client.put.assert_called_once_with(str(file_for_deploy), dest)

    def test_ssh_protocol_deploy_command_unknown_file(self) -> None:
        """Test that deploy will fail if file does not exist."""
        with pytest.raises(Exception, match="Deploy error: file type not supported"):
            self.protocol.deploy(Path("unknown_file"), "/tmp/dest")

    def test_ssh_protocol_deploy_command_bad_transport(self) -> None:
        """Test that deploy should fail if unable to get transport."""
        self.mock_ssh_client.get_transport.return_value = None

        with pytest.raises(Exception, match="Unable to get transport"):
            self.protocol.deploy(Path("some_file"), "/tmp/dest")

    def test_ssh_protocol_deploy_command_directory(
        self, test_resources_path: Path
    ) -> None:
        """Test that directory could be deployed over ssh."""
        directory_for_deploy = test_resources_path / "scripts"
        dest = "/tmp/dest"

        self.protocol.deploy(directory_for_deploy, dest)
        self.mock_sftp_client.put_dir.assert_called_once_with(
            directory_for_deploy, dest
        )

    @pytest.mark.parametrize("establish_connection", (True, False))
    def test_ssh_protocol_close(self, establish_connection: bool) -> None:
        """Test protocol close operation."""
        if establish_connection:
            self.protocol.establish_connection()
        self.protocol.close()

        call_count = 1 if establish_connection else 0
        assert self.mock_ssh_channel.exec_command.call_count == call_count

    def test_connection_details(self) -> None:
        """Test getting connection details."""
        assert self.protocol.connection_details() == ("hostname", 22)


class TestCustomSFTPClient:
    """Test CustomSFTPClient class."""

    @pytest.fixture(autouse=True)
    def setup_method(self, monkeypatch: Any) -> None:
        """Set up mocks for CustomSFTPClient instance."""
        self.mock_mkdir = MagicMock()
        self.mock_put = MagicMock()
        monkeypatch.setattr(
            "mlia.backend.protocol.paramiko.SFTPClient.__init__",
            MagicMock(return_value=None),
        )
        monkeypatch.setattr(
            "mlia.backend.protocol.paramiko.SFTPClient.mkdir", self.mock_mkdir
        )
        monkeypatch.setattr(
            "mlia.backend.protocol.paramiko.SFTPClient.put", self.mock_put
        )

        self.sftp_client = CustomSFTPClient(MagicMock())

    def test_put_dir(self, test_systems_path: Path) -> None:
        """Test deploying directory to remote host."""
        directory_for_deploy = test_systems_path / "system1"

        self.sftp_client.put_dir(directory_for_deploy, "/tmp/dest")
        assert self.mock_put.call_count == 3
        assert self.mock_mkdir.call_count == 3

    def test_mkdir(self) -> None:
        """Test creating directory on remote host."""
        self.mock_mkdir.side_effect = IOError("Cannot create directory")

        self.sftp_client._mkdir("new_directory", ignore_existing=True)

        with pytest.raises(IOError, match="Cannot create directory"):
            self.sftp_client._mkdir("new_directory", ignore_existing=False)
