# SPDX-FileCopyrightText: Copyright 2023, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: LicenseRef-LICENSE
"""Backend module for parsinng NGP Graph Compiler's output."""
from __future__ import annotations

import csv
import re
from abc import abstractmethod
from itertools import islice
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List


class ColumnParser:
    """Parsing the contents of the cell in a column."""

    def __init__(self, title: str):
        """Initialize a parser with a title."""
        self.title = title

    @abstractmethod
    def __call__(self, cell: str) -> Any:
        """Parse the cell, return the extract data."""


class NumberColumnParser(ColumnParser):
    """Parser for number values."""

    def __init__(self, title: str, converter: type):
        """Initialize parser."""
        super().__init__(title)
        self.converter = converter

    def __call__(self, cell: str) -> Any:
        """Run parser."""
        try:
            return self.converter(cell)
        except ValueError:
            return f"Invalid:{cell}"


class SubtableColumnParser(ColumnParser):
    """Parse nested columns found in NGP compiler csv output."""

    def __init__(self, title: str, header: str):
        """Initialize parser with header."""
        super().__init__(title)
        self.sub_columns = header.split(";")

    def __call__(self, cell: str) -> Any:
        """Parse a cell based on header data."""
        raw_list = cell.split(";")

        iter_raw = iter(raw_list)
        size = len(self.sub_columns)

        mappings = []
        while chunk := list(islice(iter_raw, size)):
            if len(chunk) < size:
                if chunk == [""]:
                    continue
                raise RuntimeError("Unmatched column entries: {chunk}")
            mapping = dict(zip(self.sub_columns, chunk))
            mappings.append(mapping)
        return mappings

    def __eq__(self, other: Any) -> bool:
        """Override the default implementation."""
        if isinstance(other, SubtableColumnParser):
            return self.sub_columns == other.sub_columns
        return False


PerformanceDatabaseContentsType = List[Dict[str, Any]]


class NGPPerformanceDatabase:
    """Access the performance database produced by the NGP graph compiler."""

    def __init__(self) -> None:
        """Initialize the database."""
        self._column_parsers: dict[str, ColumnParser] = {}
        self.records: PerformanceDatabaseContentsType = []
        self.register_sub_table(
            "Memory", "memoryName;readBytes;writeBytes;trafficCycles"
        )
        self.register_sub_table("Utilization", "sectionName;hwUtil")

    def register_sub_table(self, title: str, header: str) -> None:
        """Add subtable column."""
        self._column_parsers[header] = SubtableColumnParser(title, header)

    def parse_contents(self, perf_db_strings: str) -> PerformanceDatabaseContentsType:
        """Parse contents of the performance DB.

        Returns a list of dicts (records), which in turn contain key value pairs
        to represent performance statistics for various operators.
        Values under a record can be numberical, text, or list of other nested
        dicts (for various memory and HW utilization stats).

        Example:
        [
            {
                "id": 26,
                "opCycles": 18,
                "totalCycles": 212,
                "Memory": [
                    {
                        "memoryName": "DRAM",
                        "readBytes": 124,
                        "trafficCycles": 4,
                        "writeBytes": 4
                    }
                ...
            },
            ..
        ]
        """
        reader = csv.reader(perf_db_strings.splitlines())
        header = [extract_field(column) for column in next(reader)]
        column_parsers = [
            self._column_parsers.get(col, NumberColumnParser(col, int))
            for col in header
        ]

        records = []
        for row in reader:
            parsed_row = [
                (cparser.title, cparser(extract_field(cell)))
                for hcell, cparser, cell in zip(header, column_parsers, row)
            ]
            record = dict(parsed_row)
            records.append(record)
        return records

    def load(self, db_path: Path) -> PerformanceDatabaseContentsType:
        """Parse NGP compiler's output: performance database.

        The file format is XML, but the only meaningful content within it
        is embedded in a CDATA secion. Current versions of the
        graph compiler produce invalid XML (eg an closing <table> element
        without the starting one), so we are trying to be tolerant and
        extract only the CDATA section without expecting any other elements.
        """
        with open(db_path, encoding="utf-8") as file:
            xmlish = file.read()

        cdata_content = extract_cdata(xmlish)
        self.records = self.parse_contents(cdata_content)
        return self.records


def extract_field(field: str) -> str:
    """Extract contents of text field in a CSV table."""
    field = field.strip()
    field = re.sub(r"^(['\"])(.*?)\1$", r"\2", field)
    return field


def extract_cdata(xml_string: str) -> str:
    """Extract CDATA section from xml."""
    matches = re.findall(r"<!\[CDATA\[(.*?)\]\]>", xml_string, re.DOTALL)
    if len(matches) != 1:
        raise RuntimeError(f"No single CDATA section in:\n{xml_string}")

    return str(matches[0]).strip()
