# SPDX-FileCopyrightText: Copyright 2022, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Utils for files downloading."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import requests
from rich.progress import BarColumn
from rich.progress import DownloadColumn
from rich.progress import FileSizeColumn
from rich.progress import Progress
from rich.progress import ProgressColumn
from rich.progress import TextColumn

from mlia.utils.filesystem import sha256
from mlia.utils.types import parse_int


def download_progress(
    content_chunks: Iterable[bytes], content_length: int | None, label: str | None
) -> Iterable[bytes]:
    """Show progress info while reading content."""
    columns: list[ProgressColumn] = [TextColumn("{task.description}")]

    if content_length is None:
        total = float("inf")
        columns.append(FileSizeColumn())
    else:
        total = content_length
        columns.extend([BarColumn(), DownloadColumn(binary_units=True)])

    with Progress(*columns) as progress:
        task = progress.add_task(label or "Downloading", total=total)

        for chunk in content_chunks:
            progress.update(task, advance=len(chunk))
            yield chunk


def download(
    url: str,
    dest: Path,
    show_progress: bool = False,
    label: str | None = None,
    chunk_size: int = 8192,
) -> None:
    """Download the file."""
    with requests.get(url, stream=True) as resp:
        resp.raise_for_status()
        content_chunks = resp.iter_content(chunk_size=chunk_size)

        if show_progress:
            content_length = parse_int(resp.headers.get("Content-Length"))
            content_chunks = download_progress(content_chunks, content_length, label)

        with open(dest, "wb") as file:
            for chunk in content_chunks:
                file.write(chunk)


@dataclass
class DownloadArtifact:
    """Download artifact attributes."""

    name: str
    url: str
    filename: str
    version: str
    sha256_hash: str

    def download_to(self, dest_dir: Path, show_progress: bool = True) -> Path:
        """Download artifact into destination directory."""
        if (dest := dest_dir / self.filename).exists():
            raise ValueError(f"{dest} already exists")

        download(
            self.url,
            dest,
            show_progress=show_progress,
            label=f"Downloading {self.name} ver. {self.version}",
        )

        if sha256(dest) != self.sha256_hash:
            raise ValueError("Digests do not match")

        return dest
