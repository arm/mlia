# SPDX-FileCopyrightText: Copyright 2023, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Classes for metadata."""
from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from pathlib import Path

from mlia.utils.misc import get_file_checksum
from mlia.utils.misc import get_pkg_version


class Metadata(ABC):  # pylint: disable=too-few-public-methods
    """Base class for possbile metadata."""

    def __init__(self, name: str) -> None:
        """Init Metadata."""
        self.name = name
        self.data_dict = self.get_metadata()

    @abstractmethod
    def get_metadata(self) -> dict:
        """Fill and return the metadata dictionary."""


class ModelMetadata(Metadata):
    """Model metadata."""

    def __init__(self, model_path: Path, name: str = "Model") -> None:
        """Metadata for model zoo."""
        self.model_path = model_path
        super().__init__(name)

    def get_metadata(self) -> dict:
        """Fill in metadata for model file."""
        return {
            "model_name": self.model_path.name,
            "model_checksum": get_file_checksum(self.model_path),
        }


class MLIAMetadata(Metadata):  # pylint: disable=too-few-public-methods
    """MLIA metadata."""

    def __init__(self, name: str = "MLIA") -> None:
        """Init MLIAMetadata."""
        super().__init__(name)

    def get_metadata(self) -> dict:
        """Get mlia version."""
        return {"mlia_version": get_pkg_version("mlia")}
