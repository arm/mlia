# SPDX-FileCopyrightText: Copyright 2022, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Definition of output parsers (including base class OutputParser)."""
import base64
import json
import re
from abc import ABC
from abc import abstractmethod
from typing import Any
from typing import Dict
from typing import Union


class OutputParser(ABC):
    """Abstract base class for output parsers."""

    def __init__(self, name: str) -> None:
        """Set up the name of the parser."""
        super().__init__()
        self.name = name

    @abstractmethod
    def __call__(self, output: bytearray) -> Dict[str, Any]:
        """Parse the output and return a map of names to metrics."""
        return {}

    # pylint: disable=no-self-use
    def filter_out_parsed_content(self, output: bytearray) -> bytearray:
        """
        Filter out the parsed content from the output.

        Does nothing by default. Can be overridden in subclasses.
        """
        return output


class RegexOutputParser(OutputParser):
    """Parser of standard output data using regular expressions."""

    _TYPE_MAP = {"str": str, "float": float, "int": int}

    def __init__(
        self,
        name: str,
        regex_config: Dict[str, Dict[str, str]],
    ) -> None:
        """
        Set up the parser with the regular expressions.

        The regex_config is mapping from a name to a dict with keys 'pattern'
        and 'type':
        - The 'pattern' holds the regular expression that must contain exactly
        one capturing parenthesis
        - The 'type' can be one of ['str', 'float', 'int'].

        Example:
        ```
            {"Metric1": {"pattern": ".*= *(.*)", "type": "str"}}
        ```

        The different regular expressions from the config are combined using
        non-capturing parenthesis, i.e. regular expressions must not overlap
        if more than one match per line is expected.
        """
        super().__init__(name)

        self._verify_config(regex_config)
        self._regex_cfg = regex_config

        # Compile regular expression to match in the output
        self._regex = re.compile(
            "|".join("(?:{0})".format(x["pattern"]) for x in self._regex_cfg.values())
        )

    def __call__(self, output: bytearray) -> Dict[str, Union[str, float, int]]:
        """
        Parse the output and return a map of names to metrics.

        Example:
        Assuming a regex_config as used as example in `__init__()` and the
        following output:
        ```
            Simulation finished:
            SIMULATION_STATUS = SUCCESS
            Simulation DONE
        ```
        Then calling the parser should return the following dict:
        ```
            {
                "Metric1": "SUCCESS"
            }
        ```
        """
        metrics = {}
        output_str = output.decode("utf-8")
        results = self._regex.findall(output_str)
        for line_result in results:
            for idx, (name, cfg) in enumerate(self._regex_cfg.items()):
                # The result(s) returned by findall() are either a single string
                # or a tuple (depending on the number of groups etc.)
                result = (
                    line_result if isinstance(line_result, str) else line_result[idx]
                )
                if result:
                    mapped_result = self._TYPE_MAP[cfg["type"]](result)
                    metrics[name] = mapped_result
        return metrics

    def _verify_config(self, regex_config: Dict[str, Dict[str, str]]) -> None:
        """Make sure we have a valid regex_config.

        I.e.
        - Exactly one capturing parenthesis per pattern
        - Correct types
        """
        for name, cfg in regex_config.items():
            # Check that there is one capturing group defined in the pattern.
            regex = re.compile(cfg["pattern"])
            if regex.groups != 1:
                raise ValueError(
                    f"Pattern for metric '{name}' must have exactly one "
                    f"capturing parenthesis, but it has {regex.groups}."
                )
            # Check if type is supported
            if not cfg["type"] in self._TYPE_MAP:
                raise TypeError(
                    f"Type '{cfg['type']}' for metric '{name}' is not "
                    f"supported. Choose from: {list(self._TYPE_MAP.keys())}."
                )


class Base64OutputParser(OutputParser):
    """
    Parser to extract base64-encoded JSON from tagged standard output.

    Example of the tagged output:
    ```
        # Encoded JSON: {"test": 1234}
        <metrics>eyJ0ZXN0IjogMTIzNH0</metrics>
    ```
    """

    TAG_NAME = "metrics"

    def __init__(self, name: str) -> None:
        """Set up the regular expression to extract tagged strings."""
        super().__init__(name)
        self._regex = re.compile(rf"<{self.TAG_NAME}>(.*)</{self.TAG_NAME}>")

    def __call__(self, output: bytearray) -> Dict[str, Any]:
        """
        Parse the output and return a map of index (as string) to decoded JSON.

        Example:
        Using the tagged output from the class docs the parser should return
        the following dict:
        ```
            {
                "0": {"test": 1234}
            }
        ```
        """
        metrics = {}
        output_str = output.decode("utf-8")
        results = self._regex.findall(output_str)
        for idx, result_base64 in enumerate(results):
            result_json = base64.b64decode(result_base64, validate=True)
            result = json.loads(result_json)
            metrics[str(idx)] = result

        return metrics

    def filter_out_parsed_content(self, output: bytearray) -> bytearray:
        """Filter out base64-encoded content from the output."""
        output_str = self._regex.sub("", output.decode("utf-8"))
        return bytearray(output_str.encode("utf-8"))
