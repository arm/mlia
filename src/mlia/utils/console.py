# SPDX-FileCopyrightText: Copyright 2022, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Console output utility functions."""
from __future__ import annotations

from typing import Iterable

from rich.console import Console
from rich.console import RenderableType
from rich.table import box
from rich.table import Table
from rich.text import Text


def create_section_header(
    section_name: str | None = None, length: int = 80, sep: str = "-"
) -> str:
    """Return section header."""
    if not section_name:
        content = sep * length
    else:
        before = 3
        spaces = 2
        after = length - (len(section_name) + before + spaces)
        if after < 0:
            raise ValueError("Section name too long")
        content = f"{sep * before} {section_name} {sep * after}"

    return f"\n{content}\n"


def apply_style(value: str, style: str) -> str:
    """Apply style to the value."""
    return f"[{style}]{value}"


def style_improvement(result: bool) -> str:
    """Return different text style based on result."""
    return "green" if result else "yellow"


def produce_table(
    rows: Iterable,
    headers: list[str] | None = None,
    table_style: str = "default",
) -> str:
    """Represent data in tabular form."""
    table = _get_table(table_style)

    if headers:
        table.show_header = True
        for header in headers:
            table.add_column(header)

    for row in rows:
        table.add_row(*row)

    return _convert_to_text(table)


def _get_table(table_style: str) -> Table:
    """Get Table instance for the provided style."""
    if table_style == "default":
        return Table(
            show_header=False,
            show_lines=True,
            box=box.SQUARE_DOUBLE_HEAD,
        )

    if table_style == "nested":
        return Table(
            show_header=False,
            box=None,
            padding=(0, 1, 1, 0),
        )

    if table_style == "no_borders":
        return Table(show_header=False, box=None)

    raise Exception(f"Unsupported table style {table_style}")


def _convert_to_text(*renderables: RenderableType) -> str:
    """Convert renderable object to text."""
    console = Console()
    with console.capture() as capture:
        for item in renderables:
            console.print(item)

    text = capture.get()
    return text.rstrip()


def remove_ascii_codes(value: str) -> str:
    """Decode and remove ASCII codes."""
    text = Text.from_ansi(value)
    return text.plain
