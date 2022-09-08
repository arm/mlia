# SPDX-FileCopyrightText: Copyright 2022, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for download functionality."""
from __future__ import annotations

from contextlib import ExitStack as does_not_raise
from pathlib import Path
from typing import Any
from typing import Iterable
from unittest.mock import MagicMock
from unittest.mock import PropertyMock

import pytest
import requests

from mlia.utils.download import download
from mlia.utils.download import DownloadArtifact


def response_mock(
    content_length: str | None, content_chunks: Iterable[bytes]
) -> MagicMock:
    """Mock response object."""
    mock = MagicMock(spec=requests.Response)
    mock.__enter__.return_value = mock

    type(mock).headers = PropertyMock(return_value={"Content-Length": content_length})
    mock.iter_content.return_value = content_chunks

    return mock


@pytest.mark.parametrize("show_progress", [True, False])
@pytest.mark.parametrize(
    "content_length, content_chunks, label",
    [
        [
            "5",
            [bytes(range(5))],
            "Downloading artifact",
        ],
        [
            "10",
            [bytes(range(5)), bytes(range(5))],
            None,
        ],
        [
            None,
            [bytes(range(5))],
            "Downlading no size",
        ],
        [
            "abc",
            [bytes(range(5))],
            "Downloading wrong size",
        ],
    ],
)
def test_download(
    show_progress: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    content_length: str | None,
    content_chunks: Iterable[bytes],
    label: str | None,
) -> None:
    """Test function download."""
    monkeypatch.setattr(
        "mlia.utils.download.requests.get",
        MagicMock(return_value=response_mock(content_length, content_chunks)),
    )

    dest = tmp_path / "sample.bin"
    download("some_url", dest, show_progress=show_progress, label=label)

    assert dest.is_file()
    assert dest.read_bytes() == bytes(
        byte for chunk in content_chunks for byte in chunk
    )


@pytest.mark.parametrize(
    "content_length, content_chunks, sha256_hash, expected_error",
    [
        [
            "10",
            [bytes(range(10))],
            "1f825aa2f0020ef7cf91dfa30da4668d791c5d4824fc8e41354b89ec05795ab3",
            does_not_raise(),
        ],
        [
            "10",
            [bytes(range(10))],
            "bad_hash",
            pytest.raises(ValueError, match="Digests do not match"),
        ],
    ],
)
def test_download_artifact_download_to(
    monkeypatch: pytest.MonkeyPatch,
    content_length: str | None,
    content_chunks: Iterable[bytes],
    sha256_hash: str,
    expected_error: Any,
    tmp_path: Path,
) -> None:
    """Test artifact downloading."""
    monkeypatch.setattr(
        "mlia.utils.download.requests.get",
        MagicMock(return_value=response_mock(content_length, content_chunks)),
    )

    with expected_error:
        artifact = DownloadArtifact(
            "test_artifact",
            "some_url",
            "artifact_filename",
            "1.0",
            sha256_hash,
        )

        dest = artifact.download_to(tmp_path)
        assert isinstance(dest, Path)
        assert dest.name == "artifact_filename"


def test_download_artifact_unable_to_overwrite(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test that download process cannot overwrite file."""
    monkeypatch.setattr(
        "mlia.utils.download.requests.get",
        MagicMock(return_value=response_mock("10", [bytes(range(10))])),
    )

    artifact = DownloadArtifact(
        "test_artifact",
        "some_url",
        "artifact_filename",
        "1.0",
        "sha256_hash",
    )

    existing_file = tmp_path / "artifact_filename"
    existing_file.touch()

    with pytest.raises(ValueError, match=f"{existing_file} already exists"):
        artifact.download_to(tmp_path)
