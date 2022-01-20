# Copyright 2021, Arm Ltd.
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
