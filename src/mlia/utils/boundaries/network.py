# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Classified network boundary gateways."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable
from urllib.parse import urlsplit, urlunsplit

import requests
from rich.progress import (
    BarColumn,
    DownloadColumn,
    FileSizeColumn,
    Progress,
    ProgressColumn,
    TextColumn,
)

from mlia.utils.filesystem import sha256
from mlia.utils.logging import log_boundary_action
from mlia.utils.types import parse_int

logger = logging.getLogger(__name__)


def _url_for_logging(url: str) -> str:
    """Return a URL with credential-bearing components removed."""
    try:
        parsed = urlsplit(url)
        hostname = parsed.hostname

        if not parsed.scheme or hostname is None:
            return "<redacted URL>"

        host = f"[{hostname}]" if ":" in hostname else hostname
        netloc = f"{host}:{parsed.port}" if parsed.port is not None else host
        return urlunsplit((parsed.scheme, netloc, parsed.path, "", ""))
    except ValueError:
        return "<redacted URL>"


def download_progress(
    content_chunks: Iterable[bytes], content_length: int | None, label: str | None
) -> Iterable[bytes]:
    """Show progress information while reading content."""
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
    """Parameters used to download an artifact."""

    url: str
    sha256_hash: str
    header_gen_fn: Callable[[], dict[str, str]] | None = None

    @property
    def filename(self) -> str:
        """Get the filename from the URL."""
        return self.url.rsplit("/", 1)[-1]

    @property
    def headers(self) -> dict[str, str]:
        """Get request headers."""
        return self.header_gen_fn() if self.header_gen_fn else {}


def download_with_notice(
    dest: Path,
    cfg: DownloadConfig,
    show_progress: bool = False,
    label: str | None = None,
    chunk_size: int = 8192,
    timeout: int = 30,
) -> None:
    """Announce and download a file."""
    if dest.exists():
        raise FileExistsError(f"{dest} already exists.")
    log_boundary_action(
        logger,
        "Downloading '%s' to '%s'.",
        _url_for_logging(cfg.url),
        dest,
    )
    with requests.get(
        cfg.url, stream=True, timeout=timeout, headers=cfg.headers
    ) as response:
        response.raise_for_status()
        content_chunks = response.iter_content(chunk_size=chunk_size)

        if show_progress:
            if not label:
                label = f"Downloading to {dest}."
            content_length = parse_int(response.headers.get("Content-Length"))
            content_chunks = download_progress(content_chunks, content_length, label)

        with open(dest, "wb") as file:
            for chunk in content_chunks:
                file.write(chunk)

    if cfg.sha256_hash and sha256(dest) != cfg.sha256_hash:
        raise ValueError("Hashes do not match.")
