# SPDX-FileCopyrightText: Copyright 2022, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the output parsing."""
import base64
import json
from typing import Any
from typing import Dict

import pytest

from mlia.backend.output_consumer import Base64OutputConsumer
from mlia.backend.output_consumer import OutputConsumer


OUTPUT_MATCH_ALL = bytearray(
    """
String1: My awesome string!
String2: STRINGS_ARE_GREAT!!!
Int: 12
Float: 3.14
""",
    encoding="utf-8",
)

OUTPUT_NO_MATCH = bytearray(
    """
This contains no matches...
Test1234567890!"£$%^&*()_+@~{}[]/.,<>?|
""",
    encoding="utf-8",
)

OUTPUT_PARTIAL_MATCH = bytearray(
    "String1: My awesome string!",
    encoding="utf-8",
)

REGEX_CONFIG = {
    "FirstString": {"pattern": r"String1.*: (.*)", "type": "str"},
    "SecondString": {"pattern": r"String2.*: (.*)!!!", "type": "str"},
    "IntegerValue": {"pattern": r"Int.*: (.*)", "type": "int"},
    "FloatValue": {"pattern": r"Float.*: (.*)", "type": "float"},
}

EMPTY_REGEX_CONFIG: Dict[str, Dict[str, Any]] = {}

EXPECTED_METRICS_ALL = {
    "FirstString": "My awesome string!",
    "SecondString": "STRINGS_ARE_GREAT",
    "IntegerValue": 12,
    "FloatValue": 3.14,
}

EXPECTED_METRICS_PARTIAL = {
    "FirstString": "My awesome string!",
}


@pytest.mark.parametrize(
    "expected_metrics",
    [
        EXPECTED_METRICS_ALL,
        EXPECTED_METRICS_PARTIAL,
    ],
)
def test_base64_output_consumer(expected_metrics: Dict) -> None:
    """
    Make sure the Base64OutputConsumer yields valid results.

    I.e. return an empty dict if either the input or the config is empty and
    return the parsed metrics otherwise.
    """
    parser = Base64OutputConsumer()
    assert isinstance(parser, OutputConsumer)

    def create_base64_output(expected_metrics: Dict) -> bytearray:
        json_str = json.dumps(expected_metrics, indent=4)
        json_b64 = base64.b64encode(json_str.encode("utf-8"))
        return (
            OUTPUT_MATCH_ALL  # Should not be matched by the Base64OutputConsumer
            + f"<{Base64OutputConsumer.TAG_NAME}>".encode("utf-8")
            + bytearray(json_b64)
            + f"</{Base64OutputConsumer.TAG_NAME}>".encode("utf-8")
            + OUTPUT_NO_MATCH  # Just to add some difficulty...
        )

    output = create_base64_output(expected_metrics)

    consumed = False
    for line in output.splitlines():
        if parser.feed(line.decode("utf-8")):
            consumed = True
    assert consumed  # we should have consumed at least one line

    res = parser.parsed_output
    assert len(res) == 1
    assert isinstance(res, list)
    for val in res:
        assert val == expected_metrics
