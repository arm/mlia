# SPDX-FileCopyrightText: Copyright 2022, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Test for module setup."""
from __future__ import annotations

from unittest.mock import Mock
from unittest.mock import patch

import pytest

from setup import replace_markdown_relative_paths


@pytest.mark.parametrize(
    "linked_file_found, file_content, expected_result",
    [
        [
            True,
            "[Test](README.md)",
            "[Test](https://review.mlplatform.org/plugins/gitiles/"
            "ml/mlia/+/refs/tags/0.1.0/README.md)",
        ],
        [
            True,
            "![Test](image.png)",
            "![Test](https://git.mlplatform.org/ml/mlia.git/plain/image.png?h=0.1.0)",
        ],
        [
            False,
            "[Test](http://review.mlplatform.org)",
            "[Test](http://review.mlplatform.org)",
        ],
        [False, "[Test](README.md)", "[Test](README.md)"],
        [
            True,
            "Lorem ipsum dolor sit amet, consectetur adipiscing elit",
            "Lorem ipsum dolor sit amet, consectetur adipiscing elit",
        ],
    ],
)
@patch("pathlib.Path")
def test_replace_markdown_relative_paths(
    path_mock: Mock,
    linked_file_found: bool,
    file_content: str,
    expected_result: str,
) -> None:
    """Test replacement of relative md paths with links to review.mlplatform.org."""
    # Set a mock setuptools scm version for testing
    tag = "0.1.0"
    path_mock.read_text.return_value = file_content
    path_mock.exists.return_value = linked_file_found
    path_mock.joinpath.return_value = path_mock

    result = replace_markdown_relative_paths(path_mock, "test.md", tag)
    assert result == expected_result
