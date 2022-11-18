# SPDX-FileCopyrightText: Copyright 2022, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Output consumers module."""
from __future__ import annotations

import base64
import json
import re
from typing import Protocol
from typing import runtime_checkable


@runtime_checkable
class OutputConsumer(Protocol):
    """Protocol to consume output."""

    def feed(self, line: str) -> bool:
        """
        Feed a new line to be parsed.

        Return True if the line should be removed from the output.
        """


class Base64OutputConsumer(OutputConsumer):
    """
    Parser to extract base64-encoded JSON from tagged standard output.

    Example of the tagged output:
    ```
        # Encoded JSON: {"test": 1234}
        <metrics>eyJ0ZXN0IjogMTIzNH0</metrics>
    ```
    """

    TAG_NAME = "metrics"

    def __init__(self) -> None:
        """Set up the regular expression to extract tagged strings."""
        self._regex = re.compile(rf"<{self.TAG_NAME}>(.*)</{self.TAG_NAME}>")
        self.parsed_output: list = []

    def feed(self, line: str) -> bool:
        """
        Parse the output line and save the decoded output.

        Returns True if the line contains tagged output.

        Example:
        Using the tagged output from the class docs the parser should collect
        the following:
        ```
            [
                {"test": 1234}
            ]
        ```
        """
        res_b64 = self._regex.search(line)
        if res_b64:
            res_json = base64.b64decode(res_b64.group(1), validate=True)
            res = json.loads(res_json)
            self.parsed_output.append(res)
            # Remove this line from the output, i.e. consume it, as it
            # does not contain any human readable content.
            return True

        return False
