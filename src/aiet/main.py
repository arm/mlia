# SPDX-FileCopyrightText: Copyright 2022, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Entry point module of AIET."""
from aiet.cli import cli


def main() -> None:
    """Entry point of aiet application."""
    cli()  # pylint: disable=no-value-for-parameter


if __name__ == "__main__":
    main()
