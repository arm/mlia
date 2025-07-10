# SPDX-FileCopyrightText: Copyright 2023-2025, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: LicenseRef-LICENSE
"""Backend module for parsing the Neural Accellerator Graph Compiler's output."""
from __future__ import annotations

import csv
import re
from abc import abstractmethod
from itertools import islice
from pathlib import Path
from typing import Any
from typing import Dict
from typing import Iterator
from typing import List

from mlia.utils.misc import list_to_dict

PerformanceDatabaseContentsType = List[Dict[str, Any]]
DebugDatabaseContentsType = Dict[str, Dict[str, List]]
# All of the tables in the debug database have two columns,
# except for one table that has three. This constant helps
# us handle that case and error catching.
MAX_NUM_DEBUG_DB_HEADERS = 3


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
    """Parse nested columns found in the Neural Accellerator compiler csv output."""

    def __init__(self, title: str, header: str):
        """Initialize parser with header."""
        super().__init__(title)
        self.sub_columns = header.split(";")

    def safe_convert_to_float(self, value: str) -> int | float | str:
        """Convert strings to float or int if possible."""
        try:
            # Try to convert to integer first
            if "." not in value:
                return int(value)
            # If it contains a dot, attempt to convert to float
            return float(value)
        except ValueError:
            # If conversion fails, return the original value
            return value

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

            typed_chunk = [self.safe_convert_to_float(item) for item in chunk]
            mapping = dict(zip(self.sub_columns, typed_chunk))
            mappings.append(mapping)
        return mappings

    def __eq__(self, other: Any) -> bool:
        """Override the default implementation."""
        if isinstance(other, SubtableColumnParser):
            return self.sub_columns == other.sub_columns
        return False


class SubtableColumnParserDict(SubtableColumnParser):
    """Parse nested columns found in the Neural Accellerator compiler csv output into dict."""

    def __init__(self, title: str, header: str, key_field: str | None):
        """Initialize dict parser with header."""
        super().__init__(title, header)
        self.key_field = key_field

    def __call__(self, cell: str) -> Any:
        """Parse a cell based on header data."""
        mappings = super().__call__(cell)
        return list_to_dict(mappings, self.key_field)


class NXOutputParser:
    """Parser for Neural Accellerator output files with a .dat extension.

    Current versions of the graph compiler produce an
    invalid XML (eg. a closing <table> element without the
    starting one), so we read the file into string first.
    Then we can use the string to extract relevant fields,
    without expecting strict XML syntax or structure.

    There is an open ticket to modify the format of these files.
    """

    def __init__(self, db_path: Path | None = None) -> None:
        """Initialize output parser for the Neural Accellerator files."""
        self.db_path: Path
        self.raw_xmlish: str = ""

        if db_path:
            self.db_path = db_path
            self.raw_xmlish = self.load(self.db_path)

    def load(self, db_path: Path) -> str:
        """Read the Neural Accellerator compiler's output into a string."""
        with open(db_path, encoding="utf-8") as file:
            self.raw_xmlish = file.read()

        return self.raw_xmlish

    def get_csv_reader(self, table_data: str) -> Iterator[list[str]]:
        """Get file content into csv reader object."""
        return csv.reader(table_data.splitlines())

    def get_csv_headers(self, csv_reader: Iterator[list[str]]) -> list:
        """Get headers of csv reader."""
        return [self.extract_field(column) for column in next(csv_reader)]

    def extract_field(self, field: str) -> str:
        """Extract contents of text field in a CSV table."""
        field = field.strip()
        field = re.sub(r"^(['\"])(.*?)\1$", r"\2", field)
        return field

    def extract_cdata(self, xml_string: str) -> str:
        """Extract CDATA section from a string."""
        matches = re.findall(r"<!\[CDATA\[(.*?)\]\]>", xml_string, re.DOTALL)
        if len(matches) != 1:
            raise RuntimeError(f"No single CDATA section in:\n{xml_string}")

        return str(matches[0]).strip()


class NXPerformanceDatabaseParser(NXOutputParser):
    """Parser for the Neural Accellerator performance database."""

    def __init__(self, db_path: Path | None = None) -> None:
        """Initialise the performance database parser."""
        super().__init__(db_path)
        self.performance_db: PerformanceDatabaseContentsType = []
        self.column_parsers: dict[str, ColumnParser] = {}
        self.register_sub_table(
            "Memory", "memoryName;readBytes;writeBytes;trafficCycles", "memoryName"
        )
        self.register_sub_table("Utilization", "sectionName;hwUtil", None)
        self._column_parsers: dict[str, ColumnParser] = {}

    def register_sub_table(
        self, title: str, header: str, key_field: str | None
    ) -> dict[str, ColumnParser]:
        """Add subtable column."""
        self.column_parsers[header] = SubtableColumnParserDict(title, header, key_field)
        return self.column_parsers

    def parse_performance_database(self) -> PerformanceDatabaseContentsType:
        """Parse contents of the performance DB.

        Returns a list of dicts (records), which in turn contain key value pairs
        to represent performance statistics for various operators.
        Values under a record can be numerical, text, or list of other nested
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
        table_data = self.extract_cdata(self.raw_xmlish)
        reader = self.get_csv_reader(table_data=table_data)
        headers = self.get_csv_headers(csv_reader=reader)
        int_column_parsers = self.set_column_parsers(headers=headers, content_type=int)
        self.performance_db = self.make_parsed_db(
            csv_reader=reader, headers=headers, column_parsers=int_column_parsers
        )
        return self.performance_db

    def make_parsed_db(
        self,
        csv_reader: Iterator[list[str]],
        headers: list,
        column_parsers: list,
    ) -> list:
        """Parse database into list."""
        for row in csv_reader:
            parsed_row = [
                (cparser.title, cparser(self.extract_field(cell)))
                for _, cparser, cell in zip(headers, column_parsers, row)
            ]
            record = dict(parsed_row)
            self.performance_db.append(record)
        return self.performance_db

    def set_column_parsers(
        self, headers: list, content_type: type
    ) -> list[ColumnParser]:
        """Make column parsers."""
        return [
            self.column_parsers.get(col, NumberColumnParser(col, content_type))
            for col in headers
        ]


class NXDebugDatabaseParser(NXOutputParser):
    """Parser for Neural Accellerator debug database."""

    def __init__(self, db_path: Path | None = None) -> None:
        """Initialise the debug database parser."""
        super().__init__(db_path)
        self.debug_db: dict = {}

    def parse_debug_database(self) -> DebugDatabaseContentsType:
        """Parse the contents of the debug DB.

        Returns a dict. Values and keys represent graph compiler-level operation ids.

        Every key is a string explaining the
        operation mapping: e.g. "fused_op_id_to_tosa_op_ids" maps
        fused ops to tosa ops.
        Every value is a dictionary of one such mapping e.g.
        {'1537': ['1305', '1417']} means operation 1537 maps to both
        operations 1305 and 1417.

        Example:
        {'tosa_op_id_to_api_id': {},
        'fused_op_id_to_tosa_op_ids':
            {'1305': ['1001'],
            '1417': ['1002'],
            '1307': ['1003']
            ...
            '1299': ['1116', '1114', '1117', '1118'],
            '1383': ['1119']
            ...}
            ...
        'chain_op_id_to_fused_op_ids':
            {'1537': ['1305', '1417'],
            '1539': ['1307', '1419'],
            '1541': ['1309', '1421'],
            ...}
        }
        """
        table_elements = self.raw_xmlish.split('<table name="')[1:]

        for table_element in table_elements:
            table_name = table_element.split('">')[0]
            table_data = self.extract_cdata(table_element)
            reader = self.get_csv_reader(table_data=table_data)
            headers = self.get_csv_headers(csv_reader=reader)
            if len(headers) > MAX_NUM_DEBUG_DB_HEADERS:
                raise RuntimeError(
                    f"Unsupported number of headers. Found {len(headers)} headers but "
                    f"max {MAX_NUM_DEBUG_DB_HEADERS} headers supported."
                )
            headers[0] = table_name + "_to_" + headers[1]
            if len(headers) == MAX_NUM_DEBUG_DB_HEADERS:
                headers[1] = table_name + "_to_" + headers[2]

            self.make_parsed_db(csv_reader=reader, headers=headers)

        return self.debug_db

    def make_parsed_db(self, csv_reader: Iterator[list[str]], headers: list) -> None:
        """Parse database into dictionary."""
        self.debug_db[headers[0]] = {}

        if len(headers) > MAX_NUM_DEBUG_DB_HEADERS:
            raise RuntimeError(
                f"Unsupported number of headers. Found {len(headers)} headers but "
                f"max {MAX_NUM_DEBUG_DB_HEADERS} headers supported."
            )

        if len(headers) == MAX_NUM_DEBUG_DB_HEADERS:
            self.debug_db[headers[1]] = {}

        for row in csv_reader:
            op_ids_list = [
                op_id for op_id in row[1].strip().split(";") if op_id.strip()
            ]
            self.debug_db[headers[0]][row[0]] = op_ids_list
            if len(headers) == MAX_NUM_DEBUG_DB_HEADERS:
                op_ids_list = [row[2].strip(";").strip()]
                self.debug_db[headers[1]][row[0]] = op_ids_list
