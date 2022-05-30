# SPDX-FileCopyrightText: Copyright 2022, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""IP configuration module."""


class IPConfiguration:  # pylint: disable=too-few-public-methods
    """Base class for IP configuration."""

    def __init__(self, target: str) -> None:
        """Init IP configuration instance."""
        self.target = target
