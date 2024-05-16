# SPDX-FileCopyrightText: Copyright 2024, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""NGP operator compatibility module."""
from __future__ import annotations

import logging
import re


logger = logging.getLogger(__name__)


class VMCCompatibilityLogReader:
    """Read log from VMC and extract low-level NGP compatibility information."""

    _lowered_ops: dict[str, str]
    _lowering_errors: dict[str, str]

    def __init__(self) -> None:
        """Initialize cumulative fields and patterns."""
        self._lowered_ops = {}
        self._lowering_errors = {}
        self._loc_pattern = re.compile(r'loc\((?:"(\S+)"|fused\[(.*?)\])\)')
        self._success_pattern = re.compile(r"^Successfully lowered: (\S*)\s+(at)? (.*)")
        self._error_pattern = re.compile(r"^\S+: error: (.*?): (.*?)$")

    def __call__(self, line: str) -> None:
        """Redirect output to the logger."""
        if match := self._success_pattern.match(line):
            lowered_op, filling, rest = match.group(1, 2, 3)
            if filling != "at":
                raise RuntimeError("Unrecognized log line: '{line}'")

            loc_string = self.parse_loc(rest)
            self._lowered_ops[loc_string] = lowered_op
        elif match := self._error_pattern.match(line):
            loc_part, error = match.group(1, 2)
            loc_string = self.parse_loc(loc_part)
            self._lowering_errors[loc_string] = error

    @property
    def lowered_ops(self) -> dict[str, str]:
        """Return a mapping of source(TF) to lowered(TOSA) operators."""
        return self._lowered_ops

    @property
    def lowering_errors(self) -> dict[str, str]:
        """Return a mapping of source(TF) to error messages of affected operators."""
        return self._lowering_errors

    def parse_loc(self, line: str) -> str:
        """Parse loc() expression in log strings."""
        if loc_match := self._loc_pattern.match(line):
            loc_string, fused_list = loc_match.group(1, 2)
            if loc_string:
                return loc_string
            if fused_list:
                locs = fused_list.split(", ")
                loc_strings = [loc.strip('"') for loc in locs]
                return loc_strings[0]
        raise RuntimeError(f"Can't find a valid location string in {line}")
