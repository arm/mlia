# SPDX-FileCopyrightText: Copyright 2022, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the output parsing."""
import base64
import json
from typing import Any
from typing import Dict

import pytest

from aiet.backend.output_parser import Base64OutputParser
from aiet.backend.output_parser import OutputParser
from aiet.backend.output_parser import RegexOutputParser


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


class TestRegexOutputParser:
    """Collect tests for the RegexOutputParser."""

    @staticmethod
    @pytest.mark.parametrize(
        ["output", "config", "expected_metrics"],
        [
            (OUTPUT_MATCH_ALL, REGEX_CONFIG, EXPECTED_METRICS_ALL),
            (OUTPUT_MATCH_ALL + OUTPUT_NO_MATCH, REGEX_CONFIG, EXPECTED_METRICS_ALL),
            (OUTPUT_MATCH_ALL + OUTPUT_NO_MATCH, REGEX_CONFIG, EXPECTED_METRICS_ALL),
            (
                OUTPUT_MATCH_ALL + OUTPUT_PARTIAL_MATCH,
                REGEX_CONFIG,
                EXPECTED_METRICS_ALL,
            ),
            (OUTPUT_NO_MATCH, REGEX_CONFIG, {}),
            (OUTPUT_MATCH_ALL, EMPTY_REGEX_CONFIG, {}),
            (bytearray(), EMPTY_REGEX_CONFIG, {}),
            (bytearray(), REGEX_CONFIG, {}),
        ],
    )
    def test_parsing(output: bytearray, config: Dict, expected_metrics: Dict) -> None:
        """
        Make sure the RegexOutputParser yields valid results.

        I.e. return an empty dict if either the input or the config is empty and
        return the parsed metrics otherwise.
        """
        parser = RegexOutputParser(name="Test", regex_config=config)
        assert parser.name == "Test"
        assert isinstance(parser, OutputParser)
        res = parser(output)
        assert res == expected_metrics

    @staticmethod
    def test_unsupported_type() -> None:
        """An unsupported type in the regex_config must raise an exception."""
        config = {"BrokenMetric": {"pattern": "(.*)", "type": "UNSUPPORTED_TYPE"}}
        with pytest.raises(TypeError):
            RegexOutputParser(name="Test", regex_config=config)

    @staticmethod
    @pytest.mark.parametrize(
        "config",
        (
            {"TooManyGroups": {"pattern": r"(\w)(\d)", "type": "str"}},
            {"NoGroups": {"pattern": r"\W", "type": "str"}},
        ),
    )
    def test_invalid_pattern(config: Dict) -> None:
        """Exactly one capturing parenthesis is allowed in the regex pattern."""
        with pytest.raises(ValueError):
            RegexOutputParser(name="Test", regex_config=config)


@pytest.mark.parametrize(
    "expected_metrics",
    [
        EXPECTED_METRICS_ALL,
        EXPECTED_METRICS_PARTIAL,
    ],
)
def test_base64_output_parser(expected_metrics: Dict) -> None:
    """
    Make sure the Base64OutputParser yields valid results.

    I.e. return an empty dict if either the input or the config is empty and
    return the parsed metrics otherwise.
    """
    parser = Base64OutputParser(name="Test")
    assert parser.name == "Test"
    assert isinstance(parser, OutputParser)

    def create_base64_output(expected_metrics: Dict) -> bytearray:
        json_str = json.dumps(expected_metrics, indent=4)
        json_b64 = base64.b64encode(json_str.encode("utf-8"))
        return (
            OUTPUT_MATCH_ALL  # Should not be matched by the Base64OutputParser
            + f"<{Base64OutputParser.TAG_NAME}>".encode("utf-8")
            + bytearray(json_b64)
            + f"</{Base64OutputParser.TAG_NAME}>".encode("utf-8")
            + OUTPUT_NO_MATCH  # Just to add some difficulty...
        )

    output = create_base64_output(expected_metrics)
    res = parser(output)
    assert len(res) == 1
    assert isinstance(res, dict)
    for val in res.values():
        assert val == expected_metrics

    output = parser.filter_out_parsed_content(output)
    assert output == (OUTPUT_MATCH_ALL + OUTPUT_NO_MATCH)
