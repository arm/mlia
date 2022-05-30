# SPDX-FileCopyrightText: Copyright 2022, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""MLIA exceptions module."""


class ConfigurationError(Exception):
    """Configuration error."""


class FunctionalityNotSupportedError(Exception):
    """Functionality is not supported error."""

    def __init__(self, reason: str, description: str) -> None:
        """Init exception."""
        super().__init__(f"{reason}: {description}")

        self.reason = reason
        self.description = description
