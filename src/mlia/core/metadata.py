# SPDX-FileCopyrightText: Copyright 2023, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Classes for metadata."""
from pathlib import Path

from mlia.utils.misc import get_file_checksum
from mlia.utils.misc import get_pkg_version


class Metadata:  # pylint: disable=too-few-public-methods
    """Base class metadata."""

    def __init__(self, data_name: str) -> None:
        """Init Metadata."""
        self.version = self.get_version(data_name)

    def get_version(self, data_name: str) -> str:
        """Get version of the python package."""
        return get_pkg_version(data_name)


class MLIAMetadata(Metadata):  # pylint: disable=too-few-public-methods
    """MLIA metadata."""


class ModelMetadata:  # pylint: disable=too-few-public-methods
    """Model metadata."""

    def __init__(self, path_name: Path) -> None:
        """Init ModelMetadata."""
        self.model_name = path_name.name
        self.path_name = path_name
        self.checksum = self.get_checksum()

    def get_checksum(self) -> str:
        """Get checksum of the model."""
        return get_file_checksum(self.path_name)
