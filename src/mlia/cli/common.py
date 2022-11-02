# SPDX-FileCopyrightText: Copyright 2022, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""CLI common module."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Callable


@dataclass
class CommandInfo:
    """Command description."""

    func: Callable
    aliases: list[str]
    opt_groups: list[Callable[[argparse.ArgumentParser], None]]
    is_default: bool = False
    name: str | None = None

    @property
    def command_name(self) -> str:
        """Return command name."""
        return self.name or self.func.__name__

    @property
    def command_name_and_aliases(self) -> list[str]:
        """Return list of command name and aliases."""
        return [self.command_name, *self.aliases]

    @property
    def command_help(self) -> str:
        """Return help message for the command."""
        assert self.func.__doc__, "Command function does not have a docstring"
        func_help = self.func.__doc__.splitlines()[0].rstrip(".")

        if self.is_default:
            func_help = f"{func_help} [default]"

        return func_help
