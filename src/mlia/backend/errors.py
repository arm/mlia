# SPDX-FileCopyrightText: Copyright 2022, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Backend errors."""


class BackendUnavailableError(Exception):
    """Backend unavailable error."""

    def __init__(self, msg: str, backend: str) -> None:
        """Init error."""
        super().__init__(msg)
        self.backend = backend
