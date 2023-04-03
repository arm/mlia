# SPDX-FileCopyrightText: Copyright 2023, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""TOSA package metadata."""
from mlia.core.metadata import Metadata
from mlia.utils.misc import get_pkg_version


class TOSAMetadata(Metadata):  # pylint: disable=too-few-public-methods
    """TOSA metadata."""

    def __init__(self) -> None:
        """Init TOSAMetadata."""
        super().__init__("tosa-checker")

    def get_metadata(self) -> dict:
        """Return TOSA checker version."""
        return {"tosa_checker_version": get_pkg_version(self.name)}
