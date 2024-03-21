# SPDX-FileCopyrightText: Copyright 2023, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Utils for files downloading."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable
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


@dataclass
class DownloadConfig:
    """Parameters to download an artifact."""

    url: str
    sha256_hash: str
    header_gen_fn: Callable[[], dict[str, str]] | None = None

    @property
    def filename(self) -> str:
        """Get the filename from the URL."""
        return self.url.rsplit("/", 1)[-1]

    @property
    def headers(self) -> dict[str, str]:
        """Get the headers using the header_gen_fn."""
        return self.header_gen_fn() if self.header_gen_fn else {}


def download(
    dest: Path,
    cfg: DownloadConfig,
    show_progress: bool = False,
    label: str | None = None,
    chunk_size: int = 8192,
    timeout: int = 30,
) -> None:
    """Download the file."""
    if dest.exists():
        raise FileExistsError(f"{dest} already exists.")

    with requests.get(
        cfg.url, stream=True, timeout=timeout, headers=cfg.headers
    ) as resp:
        resp.raise_for_status()
        content_chunks = resp.iter_content(chunk_size=chunk_size)

        if show_progress:
            if not label:
                label = f"Downloading to {dest}."
            content_length = parse_int(resp.headers.get("Content-Length"))
            content_chunks = download_progress(content_chunks, content_length, label)

        with open(dest, "wb") as file:
            for chunk in content_chunks:
                file.write(chunk)

    if cfg.sha256_hash and sha256(dest) != cfg.sha256_hash:
        raise ValueError("Hashes do not match.")
